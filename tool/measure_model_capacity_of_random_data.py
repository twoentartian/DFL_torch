import os, sys, argparse, logging, copy
from datetime import datetime
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, util, dataset_random, complete_ml_setup


logger = logging.getLogger("measure_model_capacity_of_random_data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_number_of_sample(sample_count_per_label, random_dataset_type, random_dataset_func, output_dir, current_ml_setup, use_amp=False, core=os.cpu_count()):
    dataset_path = os.path.join(output_dir, f"random_dataset_count_{sample_count_per_label}")
    dataset_random.save_random_images(sample_count_per_label, random_dataset_type, dataset_path)

    # try to train
    train_path = os.path.join(output_dir, f"train_on_random_dataset_count_{sample_count_per_label}")
    os.makedirs(train_path)

    epoch_loss_lr_log_file = open(os.path.join(train_path, f"train.log"), "w")
    epoch_loss_lr_log_file.write("epoch,loss,lrs" + "\n")
    epoch_loss_lr_log_file.flush()

    model: torch.nn.Module = copy.deepcopy(current_ml_setup.model)
    model.to(device)

    dataset_setup = random_dataset_func(override_dataset_path=dataset_path)
    batch_size = len(dataset_setup.training_data) if len(dataset_setup.training_data) < current_ml_setup.training_batch_size else current_ml_setup.training_batch_size

    current_ml_setup.re_initialize_model(model)
    # optimizer, lr_scheduler, epochs = complete_ml_setup.FastTrainingSetup.get_optimizer_lr_scheduler_epoch(current_ml_setup, model, 0, override_dataset=dataset_setup.training_data, override_batch_size=batch_size)

    steps_per_epoch = len(dataset_setup.training_data) // batch_size + 1
    lr = 0.01
    epochs = 100
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)

    criterion = current_ml_setup.criterion
    num_worker = 8 if core > 8 else core
    dataloader = DataLoader(dataset_setup.training_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_worker, persistent_workers=True, prefetch_factor=4)

    logger.info(f"begin training (count: {sample_count_per_label})")
    model.train()
    final_loss = None
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
    for epoch in range(epochs):
        train_loss = 0
        count = 0
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda'):
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
        final_loss = train_loss / count
        logger.info(f"epoch[{epoch}] loss={final_loss} lrs={lrs}")
        epoch_loss_lr_log_file.write(f"{epoch},{final_loss},{lrs}" + "\n")
        epoch_loss_lr_log_file.flush()

    logger.info(f"finish training")
    epoch_loss_lr_log_file.flush()
    epoch_loss_lr_log_file.close()

    return final_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure the model capacity in terms of how many random samples can be memorized')
    parser.add_argument("-m", "--model", type=str, required=True, help="specify the model type")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="specify the dataset type")
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')
    parser.add_argument("-c", '--core', type=int, default=os.cpu_count(), help='specify the number of CPU cores to use')
    parser.add_argument("--amp", action='store_true', help='enable auto mixed precision')
    parser.add_argument("-l","--loss_threshold", type=float, default=0.1, help="specify the loss threshold of treating as not low enough")

    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset
    amp = args.amp
    core = args.core
    loss_threshold = args.loss_threshold

    util.set_logging(logger, "main")

    if args.output_folder_name is None:
        time_now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_folder_path = os.path.join(os.curdir, f"{__file__}_{time_now_str}")
    else:
        output_folder_path = os.path.join(os.curdir, args.output_folder_name)

    current_ml_setup = ml_setup.get_ml_setup_from_config(model_name, dataset_type=dataset_name)
    random_dataset_type = ml_setup.dataset_type_to_random(current_ml_setup.dataset_type)
    try:
        random_dataset_func = ml_setup.dataset_type_to_setup[random_dataset_type]
    except KeyError:
        logger.fatal(f"The dataset type({random_dataset_type.name}) is not found in the ml_setup.name_to_dataset_setup mapping.")
        exit(-1)
    logger.info(f"Random dataset type: {random_dataset_type.name}")

    low = 10
    loss = check_number_of_sample(low, random_dataset_type, random_dataset_func, output_folder_path, current_ml_setup, use_amp=amp, core=core)
    if loss > loss_threshold:
        logger.fatal(f"The loss of random_dataset_count_{low} is larger than {loss_threshold}. Stopped.")
        exit(-1)

    high = 20
    loss = check_number_of_sample(high, random_dataset_type, random_dataset_func, output_folder_path, current_ml_setup, use_amp=amp, core=core)
    while loss < loss_threshold:
        low = high
        high *= 2
        loss = check_number_of_sample(high, random_dataset_type, random_dataset_func, output_folder_path, current_ml_setup, use_amp=amp, core=core)

    while True:
        mid = (low + high) // 2
        if mid==low or mid==high:
            logger.info(f"the maximum sample count is {mid}.")
        loss = check_number_of_sample(mid, random_dataset_type, random_dataset_func, output_folder_path, current_ml_setup, use_amp=amp, core=core)

        if loss <= loss_threshold:
            low=mid
        elif loss > loss_threshold:
            high=mid
        else:
            raise NotImplementedError

