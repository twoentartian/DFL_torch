import argparse
import torch
import os
import sys
import random
import copy
import json
import numpy as np
from datetime import datetime
import concurrent.futures
from torch.utils.data import DataLoader
import logging
from PIL import Image
import lightning as L

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, complete_ml_setup, util, cuda
from py_src.service import record_model_stat
from py_src.ml_setup import ModelType

logger = logging.getLogger("generate_high_accuracy_model")

def set_logging(target_logger, task_name, log_file_path=None):
    class ExitOnExceptionHandler(logging.StreamHandler):
        def emit(self, record):
            if record.levelno == logging.CRITICAL:
                raise SystemExit(-1)

    formatter = logging.Formatter(f"[%(asctime)s] [%(levelname)8s] [{task_name}] --- %(message)s (%(filename)s:%(lineno)s)")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    target_logger.setLevel(logging.DEBUG)
    target_logger.addHandler(console)
    target_logger.addHandler(ExitOnExceptionHandler())

    if log_file_path is not None:
        file = logging.FileHandler(log_file_path)
        file.setLevel(logging.DEBUG)
        file.setFormatter(formatter)
        target_logger.addHandler(file)

    del console, formatter

def set_seed(seed: int, logger=None) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    if logger is not None:
        logger.info(f"Random seed set as {seed}")

def manually_define_optimizer(arg_ml_setup: ml_setup.MlSetup, model):
    # lr = 0.1
    # epochs = 50
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    # steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
    # return optimizer, lr_scheduler, epochs

    return None, None, None

def training_model(output_folder, index, arg_number_of_models, arg_ml_setup: ml_setup.MlSetup, arg_use_cpu: bool, random_seed,
                   arg_worker_count, arg_total_cpu_count, arg_save_format, arg_amp, arg_preset, arg_epoch_override, transfer_learn_model_path):
    thread_per_process = arg_total_cpu_count // arg_worker_count
    torch.set_num_threads(thread_per_process)

    child_logger = logging.getLogger(f"find_high_accuracy_path.{index}")
    set_logging(child_logger, f"{index}")

    if random_seed is not None:
        set_seed(random_seed, child_logger)

    if arg_use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    digit_number_of_models = len(str(arg_number_of_models))
    model: torch.nn.Module = copy.deepcopy(arg_ml_setup.model)
    model.to(device)
    dataset = copy.deepcopy(arg_ml_setup.training_data)
    batch_size = arg_ml_setup.training_batch_size
    num_worker = 16 if thread_per_process > 16 else thread_per_process
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=arg_ml_setup.collate_fn,
                            pin_memory=True, num_workers=num_worker, persistent_workers=True, prefetch_factor=4)
    criterion = arg_ml_setup.criterion

    if arg_epoch_override is not None:
        epochs = arg_epoch_override

    # services
    if arg_save_format != 'none':
        record_model_service = record_model_stat.ModelStatRecorder(1, arg_ml_setup.model_name, arg_ml_setup.dataset_name)
        record_model_service.initialize_without_runtime_parameters([0], output_folder, save_format=arg_save_format, lmdb_db_name=f"{str(index).zfill(digit_number_of_models)}")
    else:
        record_model_service = None

    # init weights and optimizer
    optimizer, lr_scheduler, epochs = manually_define_optimizer(arg_ml_setup, model)

    if transfer_learn_model_path is None:
        # reset random weights
        arg_ml_setup.re_initialize_model(model)
        if optimizer is None:
            child_logger.info(f"mode: ||||||||    TRAIN FROM INITIALIZATION    ||||||||")
            if isinstance(model, L.LightningModule):
                # this is a model in lightning pytorch framework
                optimizer_from_lighting, lr_scheduler_from_lighting = model.configure_optimizers()
                optimizer_from_config, lr_scheduler_from_config, epochs_from_config = complete_ml_setup.FastTrainingSetup.get_optimizer_lr_scheduler_epoch(arg_ml_setup, model, arg_preset)
                optimizer = optimizer_from_lighting if optimizer_from_config is None else optimizer_from_config
                lr_scheduler = lr_scheduler_from_lighting if lr_scheduler_from_config is None else lr_scheduler_from_config
                epochs = epochs_from_config
            else:
                # this is a normal pytorch model
                optimizer, lr_scheduler, epochs = complete_ml_setup.FastTrainingSetup.get_optimizer_lr_scheduler_epoch(arg_ml_setup, model, arg_preset)
    else:
        # load model weights and apply it (transfer learning)
        existing_model_state, existing_model_name, existing_dataset_name = util.load_model_state_file(transfer_learn_model_path)
        child_logger.info(f"load model weights for transfer learning, original model type: {existing_model_name}, dataset type: {existing_dataset_name}")
        model.load_state_dict(existing_model_state)
        model.to(device)
        if optimizer is None:
            child_logger.info(f"mode: ||||||||    TRANSFER TRAINING    ||||||||")
            optimizer, lr_scheduler, epochs = complete_ml_setup.TransferTrainingSetup.get_optimizer_lr_scheduler_epoch(existing_dataset_name, arg_ml_setup, model, arg_preset)

    epoch_loss_lr_log_file = open(os.path.join(output_folder, f"{str(index).zfill(digit_number_of_models)}.log"), "w")
    epoch_loss_lr_log_file.write("epoch,loss,lrs" + "\n")
    epoch_loss_lr_log_file.flush()

    child_logger.info(f"begin training")
    if hasattr(model, 'set_batches_per_epoch'):
        model.set_batches_per_epoch(len(dataloader))

    if arg_amp:
        scaler = torch.amp.GradScaler('cuda')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        count = 0

        """ Training procedure """
        if isinstance(model, L.LightningModule):
            """ Lighting model """
            for batch_idx, batch in enumerate(dataloader):
                batch = cuda.to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)

                loss = model.training_step(batch, batch_idx)
                loss.backward()

                model.optimizer_step(epoch, batch_idx, optimizer, optimizer_closure=None)

                if lr_scheduler is not None:
                    lr_scheduler.step()
                for func in arg_ml_setup.func_handler_post_training:
                    func(model=model)
                train_loss += loss.item()
                count += 1
        else:
            """ Normal PyTorch model """
            for data, label in dataloader:
                data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if arg_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(data)
                        if criterion == ml_setup.CriterionType.DiffusionModel:
                            loss = outputs
                        else:
                            loss = criterion(outputs, label)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    outputs = model(data)
                    if criterion == ml_setup.CriterionType.DiffusionModel:
                        loss = outputs
                    else:
                        loss = criterion(outputs, label)
                    loss.backward()
                    optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                for func in arg_ml_setup.func_handler_post_training:
                    func(model=model)
                train_loss += loss.item()
                count += 1

        """ print progress """
        lrs = []
        for param_group in optimizer.param_groups:
            lrs.append(param_group['lr'])
        child_logger.info(f"epoch[{epoch}] loss={train_loss/count} lrs={lrs}")
        epoch_loss_lr_log_file.write(f"{epoch},{train_loss/count},{lrs}" + "\n")
        epoch_loss_lr_log_file.flush()

        # services
        if record_model_service is not None:
            model_stat = model.state_dict()
            record_model_service.trigger_without_runtime_parameters(epoch, [0], [model_stat])

        # post epoch functions
        # ddpm
        if arg_ml_setup.model_type in [ModelType.ddpm_cifar10]:
            with torch.no_grad():
                model.eval()
                samples = model.sample(10, device)
                samples = ((samples + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy()
                for i, sample in enumerate(samples):
                    sample = (sample * 255).astype(np.uint8)
                    Image.fromarray(sample).save(os.path.join(output_folder, f"epoch{epoch}_{i}.png"))

    child_logger.info(f"finish training")
    epoch_loss_lr_log_file.flush()
    epoch_loss_lr_log_file.close()

    util.save_model_state(os.path.join(output_folder, f"{str(index).zfill(digit_number_of_models)}.model.pt"),
                          model.state_dict(), arg_ml_setup.model_name, arg_ml_setup.dataset_name)
    util.save_optimizer_state(os.path.join(output_folder, f"{str(index).zfill(digit_number_of_models)}.optimizer.pt"),
                              optimizer.state_dict(), arg_ml_setup.model_name, arg_ml_setup.dataset_name)

    del model, dataset, dataloader, criterion, optimizer, epoch_loss_lr_log_file
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Generate some high accuracy models')
    parser.add_argument("-n", "--number_of_models", type=int, default=1)
    parser.add_argument("-c", '--core', type=int, default=os.cpu_count(), help='specify the number of CPU cores to use')
    parser.add_argument("-w", "--worker", type=int, default=1, help='specify how many models to train in parallel')
    parser.add_argument("-m", "--model_type", type=str, default='lenet5')
    parser.add_argument("-d", "--dataset_type", type=str, default='default')
    parser.add_argument("--cpu", action='store_true', help='force using CPU for training')
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')
    parser.add_argument("--save_format", type=str, default='none', choices=['none', 'file', 'lmdb'], help='which format to save the training states')
    parser.add_argument("--amp", action='store_true', help='enable auto mixed precision')
    parser.add_argument("--random_seed", type=int, help='specify the random seed')
    parser.add_argument("-i", "--start_index", type=int, default=0, help='specify the start index for model names')
    parser.add_argument("-P", "--preset", type=int, default=0, help='specify the preset training hyperparameters')
    parser.add_argument("-e", "--epoch", type=int, default=None, help='override the epoch')
    parser.add_argument("-t", "--transfer_learn", type=str, default=None, help='specify a model weight file to perform transfer learning from.')

    args = parser.parse_args()

    number_of_models = args.number_of_models
    worker_count = args.worker
    total_cpu_cores = args.core
    model_type = args.model_type
    dataset_type = args.dataset_type
    use_cpu = args.cpu
    output_folder_name = args.output_folder_name
    save_format = args.save_format
    amp = args.amp
    random_seed = args.random_seed
    start_index = args.start_index
    preset = args.preset
    epoch_override = args.epoch
    transfer_learn_model_path = args.transfer_learn

    # logger
    set_logging(logger, "main")
    logger.info("logging setup complete")

    # prepare model and dataset
    current_ml_setup = ml_setup.get_ml_setup_from_config(model_type, dataset_type=dataset_type, pytorch_preset_version=preset)
    output_model_name = current_ml_setup.model_name
    logger.info(f"model name: {output_model_name}")

    # create output folder
    if output_folder_name is None:
        time_now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_folder_path = os.path.join(os.curdir, f"{__file__}_{time_now_str}")
    else:
        output_folder_path = os.path.join(os.curdir, output_folder_name)
    os.mkdir(output_folder_path)

    # write info file
    info_content = {}
    info_content['model_type'] = current_ml_setup.model_name
    info_content['model_count'] = number_of_models
    info_content['generated_by_cpu'] = use_cpu
    json_data = json.dumps(info_content)
    with open(os.path.join(output_folder_path, 'info.json'), 'w') as f:
        f.write(json_data)

    # training
    if worker_count > number_of_models:
        worker_count = number_of_models
    args = [(output_folder_path, i, number_of_models, current_ml_setup, use_cpu, random_seed, worker_count, total_cpu_cores, save_format, amp, preset, epoch_override, transfer_learn_model_path) for i in range(start_index, start_index+number_of_models, 1)]
    if worker_count == 1:
        for arg in args:
            training_model(*arg)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(training_model, *arg) for arg in args]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
            pass
