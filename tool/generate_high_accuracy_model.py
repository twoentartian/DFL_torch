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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, complete_ml_setup, util
from py_src.service import record_model_stat

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





def training_model(output_folder, index, arg_number_of_models, arg_ml_setup: ml_setup, arg_use_cpu: bool, arg_worker_count, arg_total_cpu_count, arg_save_format, arg_amp):
    thread_per_process = arg_total_cpu_count // arg_worker_count
    torch.set_num_threads(thread_per_process)

    child_logger = logging.getLogger(f"find_high_accuracy_path.{index}")
    set_logging(child_logger, f"{index}")

    if arg_use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    digit_number_of_models = len(str(arg_number_of_models))
    model = copy.deepcopy(arg_ml_setup.model)
    model.to(device)
    dataset = copy.deepcopy(arg_ml_setup.training_data)
    batch_size = arg_ml_setup.training_batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    criterion = arg_ml_setup.criterion
    optimizer, lr_scheduler, epochs = complete_ml_setup.FastTrainingSetup.get_optimizer_lr_scheduler_epoch(arg_ml_setup, model)

    # services
    if arg_save_format != 'none':
        record_model_service = record_model_stat.ModelStatRecorder(1)
        record_model_service.initialize_without_runtime_parameters([0], output_folder, save_format=arg_save_format, lmdb_db_name=f"{str(index).zfill(digit_number_of_models)}")
    else:
        record_model_service = None

    # reset random weights
    arg_ml_setup.re_initialize_model(model)

    log_file = open(os.path.join(output_folder, f"{str(index).zfill(digit_number_of_models)}.log"), "w")
    log_file.write("epoch,loss,lrs" + "\n")
    log_file.flush()

    model.train()
    child_logger.info(f"begin training")
    if arg_amp:
        scaler = torch.cuda.amp.GradScaler()
    for epoch in range(epochs):
        train_loss = 0
        count = 0
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            if arg_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
                    loss = criterion(outputs, label)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                outputs = model(data)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            train_loss += loss.item()
            count += 1
        lrs = []
        for param_group in optimizer.param_groups:
            lrs.append(param_group['lr'])
        child_logger.info(f"epoch[{epoch}] loss={train_loss/count} lrs={lrs}")
        log_file.write(f"{epoch},{train_loss/count},{lrs}" + "\n")
        log_file.flush()

        # services
        if record_model_service is not None:
            model_stat = model.state_dict()
            record_model_service.trigger_without_runtime_parameters(epoch, [0], [model_stat])
    child_logger.info(f"finish training")
    log_file.flush()
    log_file.close()

    util.save_model_state(os.path.join(output_folder, f"{str(index).zfill(digit_number_of_models)}.model.pt"),
                          model.state_dict(), arg_ml_setup.model_name)
    util.save_optimizer_state(os.path.join(output_folder, f"{str(index).zfill(digit_number_of_models)}.optimizer.pt"),
                              optimizer.state_dict(), arg_ml_setup.model_name)

    del model, dataset, dataloader, criterion, optimizer, log_file
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Generate some high accuracy models')
    parser.add_argument("-n", "--number_of_models", type=int, default=1)
    parser.add_argument("-c", '--core', type=int, default=os.cpu_count(), help='specify the number of CPU cores to use')
    parser.add_argument("-t", "--thread", type=int, default=1, help='specify how many models to train in parallel')
    parser.add_argument("-m", "--model_type", type=str, default='lenet5',
                        choices=['lenet4', 'lenet5', 'resnet18', 'simplenet', 'cct7', 'lenet5_large_fc', 'vgg11_mnist', 'vgg11_cifar10', 'mobilenet_v3_small', 'mobilenet_v3_large', 'mobilenet_v2'])
    parser.add_argument("-d", "--dataset_type", type=str, default='default',
                        choices=['default', 'mnist', 'cifar10', 'cifar100'])
    parser.add_argument("--norm_method", type=str, default='auto', choices=['auto', 'bn', 'ln', 'gn'])
    parser.add_argument("--cpu", action='store_true', help='force using CPU for training')
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')
    parser.add_argument("--save_format", type=str, default='none', choices=['none', 'file', 'lmdb'], help='which format to save the training states')
    parser.add_argument("--amp", action='store_true', help='enable auto mixed precision')


    args = parser.parse_args()

    number_of_models = args.number_of_models
    worker_count = args.thread
    total_cpu_cores = args.core
    model_type = args.model_type
    dataset_type = args.dataset_type
    use_cpu = args.cpu
    output_folder_name = args.output_folder_name
    save_format = args.save_format
    norm_method = args.norm_method
    amp = args.amp

    # logger
    set_logging(logger, "main")
    logger.info("logging setup complete")

    # prepare model and dataset
    current_ml_setup = ml_setup.get_ml_setup_from_config(model_type, norm_method, dataset_type=dataset_type)
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
    if current_ml_setup.has_normalization_layer:
        info_content['norm_method'] = norm_method
    json_data = json.dumps(info_content)
    with open(os.path.join(output_folder_path, 'info.json'), 'w') as f:
        f.write(json_data)

    # training
    if worker_count > number_of_models:
        worker_count = number_of_models
    args = [(output_folder_path, i, number_of_models, current_ml_setup, use_cpu, worker_count, total_cpu_cores, save_format, amp) for i in range(number_of_models)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(training_model, *arg) for arg in args]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
