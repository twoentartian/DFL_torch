import argparse
import torch
import os
import math
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


def manually_define_optimizer(arg_ml_setup: ml_setup.MlSetup, model):
    # lr = 0.1
    # epochs = 50
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    # steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
    # return optimizer, lr_scheduler, epochs

    return None, None, None

def training_model(output_folder, index, arg_number_of_models, arg_ml_setup: ml_setup.MlSetup, arg_use_cpu: bool, random_seed,
                   arg_worker_count, arg_total_cpu_count, arg_save_format, arg_save_interval, arg_amp, arg_preset, arg_epoch_override,
                   transfer_learn_model_path, disable_reinit, enable_validation, inverse_train_val):
    thread_per_process = arg_total_cpu_count // arg_worker_count
    torch.set_num_threads(thread_per_process)

    child_logger = logging.getLogger(f"find_high_accuracy_path.{index}")
    util.set_logging(child_logger, f"{index}")

    if random_seed is not None:
        util.set_seed(random_seed, child_logger)

    if arg_use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    digit_number_of_models = len(str(arg_number_of_models))
    model: torch.nn.Module = copy.deepcopy(arg_ml_setup.model)
    model.to(device)

    criterion = arg_ml_setup.criterion

    training_data = arg_ml_setup.training_data
    testing_data = arg_ml_setup.testing_data
    if inverse_train_val:
        temp = training_data
        training_data = testing_data
        testing_data = temp

    if arg_ml_setup.override_training_dataset_loader is None:
        batch_size = arg_ml_setup.training_batch_size
        num_worker = 16 if thread_per_process > 16 else thread_per_process
        dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=arg_ml_setup.collate_fn,
                                pin_memory=True, num_workers=num_worker, persistent_workers=True, prefetch_factor=4)
    else:
        dataloader = arg_ml_setup.override_training_dataset_loader

    if enable_validation:
        if criterion == ml_setup.CriterionType.Diffusion:
            dataloader_test = None  # no val accuracy for diffusion models
        else:
            if arg_ml_setup.override_testing_dataset_loader is None:
                batch_size = arg_ml_setup.training_batch_size
                num_worker = 8 if thread_per_process > 8 else thread_per_process
                dataloader_test = DataLoader(testing_data, batch_size=batch_size, shuffle=False,
                                             pin_memory=True, num_workers=num_worker, persistent_workers=True, prefetch_factor=4)
            else:
                dataloader_test = arg_ml_setup.override_testing_dataset_loader
    else:
        dataloader_test = None

    epochs = None
    if arg_epoch_override is not None:
        epochs = arg_epoch_override

    # services
    if arg_save_format != 'none':
        record_model_service = record_model_stat.ModelStatRecorder(1, arg_ml_setup.model_name, arg_ml_setup.dataset_name)
        model_state_path = f"{output_folder}/{index}"
        os.makedirs(model_state_path)
        record_model_service.initialize_without_runtime_parameters([0], model_state_path, save_format=arg_save_format, lmdb_db_name=f"{str(index).zfill(digit_number_of_models)}")
    else:
        record_model_service = None

    # init weights and optimizer
    optimizer, lr_scheduler, epochs_new = manually_define_optimizer(arg_ml_setup, model)
    epochs = epochs_new if epochs is None else epochs

    if transfer_learn_model_path is None:
        # reset random weights
        if disable_reinit:
            child_logger.info(f"re-initialize model is disabled")
        else:
            child_logger.info(f"re-initialize model")
            arg_ml_setup.re_initialize_model(model)
        if optimizer is None:
            child_logger.info(f"mode: ||||||||    TRAIN FROM INITIALIZATION    ||||||||")
            if isinstance(model, L.LightningModule):
                # this is a model in lightning pytorch framework
                optimizer_from_lighting, lr_scheduler_from_lighting = model.configure_optimizers()
                optimizer_from_config, lr_scheduler_from_config, epochs_from_config = complete_ml_setup.FastTrainingSetup.get_optimizer_lr_scheduler_epoch(arg_ml_setup, model, arg_preset)
                optimizer = optimizer_from_lighting if optimizer_from_config is None else optimizer_from_config
                lr_scheduler = lr_scheduler_from_lighting if lr_scheduler_from_config is None else lr_scheduler_from_config
                epochs = epochs_from_config if epochs is None else epochs
            else:
                # this is a normal pytorch model
                optimizer, lr_scheduler, epochs_from_config = complete_ml_setup.FastTrainingSetup.get_optimizer_lr_scheduler_epoch(arg_ml_setup, model, arg_preset)
                epochs = epochs_from_config if epochs is None else epochs
    else:
        # load model weights and apply it (transfer learning)
        existing_model_state, existing_model_name, existing_dataset_name = util.load_model_state_file(transfer_learn_model_path)
        child_logger.info(f"load model weights for transfer learning, original model type: {existing_model_name}, dataset type: {existing_dataset_name}")
        model.load_state_dict(existing_model_state)
        model.to(device)
        if optimizer is None:
            child_logger.info(f"mode: ||||||||    TRANSFER TRAINING    ||||||||")
            optimizer, lr_scheduler, epochs_from_config = complete_ml_setup.TransferTrainingSetup.get_optimizer_lr_scheduler_epoch(existing_dataset_name, arg_ml_setup, model, arg_preset)
            epochs = epochs_from_config if epochs is None else epochs

    epoch_loss_lr_log_file = open(os.path.join(output_folder, f"{str(index).zfill(digit_number_of_models)}.log.csv"), "w")
    epoch_loss_lr_log_file.write("epoch,training_loss,training_accuracy,validation_loss,validation_accuracy,lrs" + "\n")
    epoch_loss_lr_log_file.flush()

    child_logger.info(f"begin training")
    if hasattr(model, 'set_batches_per_epoch'):
        model.set_batches_per_epoch(len(dataloader))

    if arg_amp:
        scaler = torch.amp.GradScaler('cuda')
    for epoch in range(epochs):
        model.train()
        train_correct = None
        train_loss = 0
        train_count = 0

        """ Training procedure """
        # user defined step function
        if arg_ml_setup.override_train_step_function is not None:
            for batch_idx, batch in enumerate(dataloader):
                batch = cuda.to_device(batch, device)
                output = arg_ml_setup.override_train_step_function(batch_idx, batch, model, optimizer, lr_scheduler, arg_ml_setup)
                loss = output.loss_value
                train_loss += loss * output.sample_count
                train_count += output.sample_count
                train_correct = 0 if train_correct is None else train_correct
                train_correct += output.correct_count
        # L.LightningModule
        elif isinstance(model, L.LightningModule):
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
                train_loss += loss.item() * batch.size(0)
                train_count += batch.size(0)
        else:
            """ Normal PyTorch model """
            for data, label in dataloader:
                data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if arg_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(data)
                        if criterion == ml_setup.CriterionType.Diffusion:
                            loss = outputs
                        elif isinstance(criterion, torch.nn.modules.loss.CrossEntropyLoss):
                            loss = criterion(outputs, label)
                        else:
                            raise NotImplementedError
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    outputs = model(data)
                    if criterion == ml_setup.CriterionType.Diffusion:
                        loss = outputs
                    elif isinstance(criterion, torch.nn.modules.loss.CrossEntropyLoss):
                        loss = criterion(outputs, label)
                    else:
                        raise NotImplementedError
                    loss.backward()
                    optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                for func in arg_ml_setup.func_handler_post_training:
                    func(model=model)

                if isinstance(criterion, torch.nn.modules.loss.CrossEntropyLoss):
                    _, predicted = torch.max(outputs, 1)
                    train_correct = 0 if train_correct is None else train_correct
                    train_correct += (predicted == label).sum().item()
                train_loss += loss.item() * label.size(0)
                train_count += label.size(0)

        """ print progress / validation """
        lrs = []
        for param_group in optimizer.param_groups:
            lrs.append(param_group['lr'])
        if dataloader_test is None:
            train_correct = math.nan if train_correct is None else train_correct
            child_logger.info(f"epoch[{epoch}] training loss={train_loss / train_count:.4} training accuracy={train_correct / train_count:.4} lrs={lrs}")
            epoch_loss_lr_log_file.write(f"{epoch},{train_loss / train_count:.4e},{train_correct / train_count:.4e},{math.nan},{math.nan},{lrs}" + "\n")
            epoch_loss_lr_log_file.flush()
        else:
            if arg_ml_setup.override_evaluation_step_function is not None:
                val_loss, val_correct, val_count = 0.0, 0.0, 0
                for batch_idx, batch in enumerate(dataloader_test):
                    output = arg_ml_setup.override_evaluation_step_function(batch_idx, batch, model, optimizer, lr_scheduler, arg_ml_setup)
                    val_loss += output.loss_value * output.sample_count
                    val_correct += output.correct_count
                    val_count += output.sample_count
                child_logger.info(f"epoch[{epoch}] loss,accuracy= (train) {train_loss / train_count:.4},{train_correct / train_count:.4} (val) {val_loss / val_count:.4},{val_correct / val_count:.4} lrs={lrs}")
                epoch_loss_lr_log_file.write(f"{epoch},{train_loss / train_count:.4e},{train_correct / train_count:.4e},{val_loss / val_count:.3e},{val_correct / val_count:.4e},{lrs}" + "\n")
                epoch_loss_lr_log_file.flush()
            else:
                val_loss, val_correct, val_count = 0.0, 0.0, 0
                for data, label in dataloader_test:
                    data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
                    outputs = model(data)
                    if isinstance(criterion, torch.nn.modules.loss.CrossEntropyLoss):
                        loss = criterion(outputs, label)
                    else:
                        raise NotImplementedError
                    val_loss += loss.item() * label.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == label).sum().item()
                    val_count += label.size(0)
                child_logger.info(f"epoch[{epoch}] loss,accuracy= (train) {train_loss / train_count:.4},{train_correct / train_count:.4} (val) {val_loss/val_count:.4},{val_correct/val_count:.4} lrs={lrs}")
                epoch_loss_lr_log_file.write(f"{epoch},{train_loss / train_count:.4e},{train_correct / train_count:.4e},{val_loss/val_count:.3e},{val_correct/val_count:.4e},{lrs}" + "\n")
                epoch_loss_lr_log_file.flush()

        # services
        if record_model_service is not None:
            model_stat = model.state_dict()
            if epoch % arg_save_interval == 0:
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
    parser.add_argument("--save_interval", type=int, default=1, help='save model state per n epoch')
    parser.add_argument("--amp", action='store_true', help='enable auto mixed precision')
    parser.add_argument("-s","--random_seed", type=int, help='specify the random seed')
    parser.add_argument("-i", "--start_index", type=int, default=0, help='specify the start index for model names')
    parser.add_argument("-P", "--preset", type=int, default=0, help='specify the preset training hyperparameters')
    parser.add_argument("-e", "--epoch", type=int, default=None, help='override the epoch')
    parser.add_argument("-t", "--transfer_learn", type=str, default=None, help='specify a model weight file to perform transfer learning from.')
    parser.add_argument("--disable_reinit", action='store_true', help='disable reinitialization')
    parser.add_argument("--enable_eval", action='store_true', help='enable measuring loss and accuracy on validation set')
    parser.add_argument("--inverse_train_val", action='store_true', help='inverse train and validation set')

    args = parser.parse_args()

    number_of_models = args.number_of_models
    worker_count = args.worker
    total_cpu_cores = args.core
    model_type = args.model_type
    dataset_type = args.dataset_type
    use_cpu = args.cpu
    output_folder_name = args.output_folder_name
    save_format = args.save_format
    save_interval = args.save_interval
    amp = args.amp
    random_seed = args.random_seed
    start_index = args.start_index
    preset = args.preset
    epoch_override = args.epoch
    transfer_learn_model_path = args.transfer_learn

    # logger
    util.set_logging(logger, "main")
    logger.info("logging setup complete")

    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare model and dataset
    current_ml_setup = ml_setup.get_ml_setup_from_config(model_type, dataset_type=dataset_type, pytorch_preset_version=preset, device=device)
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
    args = [(output_folder_path, i, number_of_models, current_ml_setup,
             use_cpu, random_seed, worker_count, total_cpu_cores, save_format, save_interval, amp,
             preset, epoch_override, transfer_learn_model_path, args.disable_reinit, args.enable_eval, args.inverse_train_val) for i in range(start_index, start_index+number_of_models, 1)]
    if worker_count == 1:
        for arg in args:
            training_model(*arg)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(training_model, *arg) for arg in args]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
            pass
