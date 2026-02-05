import os, sys, argparse, logging, copy, json
from datetime import datetime
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, util, complete_ml_setup
from py_src.ml_setup_base import dataset_random


logger = logging.getLogger("measure_model_capacity_of_random_data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_lr_wd(model: torch.nn.modules, optimizer: torch.optim.Optimizer , file_path: str):
    param2name = {p: n for n, p in model.named_parameters()}
    decays = {}
    for g in optimizer.param_groups:
        wd = g.get("weight_decay", 0.0)
        names = [param2name.get(p, "<unnamed>") for p in g["params"]]
        for name in names:
            decays[name] = wd

    with open(file_path, "w") as f:
        f.write(json.dumps(decays, indent=4))

def check_number_of_sample(sample_count_per_label, random_dataset_type, random_dataset_func, output_dir, current_ml_setup, accuracy_threshold,
                           use_amp=False, core=os.cpu_count(), dataset_gen_mp=None, dataset_gen_reset_seed_per_label=False, dataset_gen_reset_seed_per_sample=False,
                           override_epoch=None, override_weight_decay=None):
    dataset_path = os.path.join(output_dir, f"random_dataset_count_{sample_count_per_label}")
    dataset_random.save_random_images(sample_count_per_label, random_dataset_type, dataset_path,
                                      num_workers=dataset_gen_mp, reset_random_seeds_per_label=dataset_gen_reset_seed_per_label, reset_random_seeds_per_sample=dataset_gen_reset_seed_per_sample)

    # try to train
    train_path = os.path.join(output_dir, f"train_on_random_dataset_count_{sample_count_per_label}")
    os.makedirs(train_path)

    epoch_loss_lr_log_file = open(os.path.join(train_path, f"train.log"), "w")
    epoch_loss_lr_log_file.write("epoch,loss,accuracy,lrs" + "\n")
    epoch_loss_lr_log_file.flush()

    model: torch.nn.Module = copy.deepcopy(current_ml_setup.model)
    model.to(device)

    dataset_setup = random_dataset_func(override_dataset_path=dataset_path)
    batch_size = len(dataset_setup.training_data) if len(dataset_setup.training_data) < current_ml_setup.training_batch_size else current_ml_setup.training_batch_size

    current_ml_setup.re_initialize_model(model)
    optimizer, lr_scheduler, epochs = complete_ml_setup.RandomDatasetTrainingSetup.get_optimizer_lr_scheduler_epoch(current_ml_setup, model, 0,
                                                                                                                    override_dataset=dataset_setup.training_data,
                                                                                                                    override_batch_size=batch_size, override_epoch=override_epoch,
                                                                                                                    override_weight_decay=override_weight_decay)
    save_lr_wd(model, optimizer, os.path.join(train_path, "lr_wd.txt"))
    criterion = current_ml_setup.criterion
    num_worker = 8 if core > 8 else core
    dataloader = DataLoader(dataset_setup.training_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_worker, persistent_workers=True, prefetch_factor=4)

    logger.info(f"begin training (count: {sample_count_per_label})")
    model.train()
    final_loss = None
    final_accuracy = None
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
    for epoch in range(epochs):
        train_loss = 0
        count = 0
        training_correct_val = 0
        training_total_val = 0

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
            _, predicted = torch.max(outputs, 1)
            training_correct_val += (predicted == label).sum().item()
            training_total_val += label.size(0)
            if lr_scheduler is not None:
                lr_scheduler.step()
            train_loss += loss.item()
            count += 1
        lrs = []
        for param_group in optimizer.param_groups:
            lrs.append(param_group['lr'])
        final_loss = train_loss / count
        final_accuracy = training_correct_val / training_total_val
        util.save_model_state(os.path.join(train_path, f"{epoch}.model.pt"), model.state_dict(), current_ml_setup.model_name, dataset_setup.dataset_name)
        logger.info(f"epoch[{epoch}] loss={final_loss} accuracy={final_accuracy} lrs={lrs}")
        epoch_loss_lr_log_file.write(f"{epoch},{final_loss},{final_accuracy},{lrs}" + "\n")
        epoch_loss_lr_log_file.flush()
        if final_accuracy > accuracy_threshold:
            logger.info(f"early stopping at epoch {epoch}, {final_accuracy}(accuracy) > {accuracy_threshold}(threshold)")
            break

    logger.info(f"finish training")
    epoch_loss_lr_log_file.flush()
    epoch_loss_lr_log_file.close()

    return final_loss, final_accuracy


def measure_one_configuration(random_dataset_type, random_dataset_func, output_folder_path, current_ml_setup, accuracy_threshold, override_epoch, override_weight_decay):
    child_logger = logging.getLogger(f"measure_one_configuration_{override_weight_decay}")
    util.set_logging(child_logger, "single_task", log_file_path=os.path.join(output_folder_path, "log.txt"))

    low = 1
    child_logger.info(f"try sample_count per label: {low}.")
    loss, accuracy = check_number_of_sample(low, random_dataset_type, random_dataset_func, output_folder_path, current_ml_setup, accuracy_threshold,
                                            use_amp=amp, core=core, dataset_gen_mp=dataset_gen_worker, override_epoch=override_epoch, override_weight_decay=override_weight_decay,
                                            dataset_gen_reset_seed_per_label=dataset_gen_reset_seed_per_label, dataset_gen_reset_seed_per_sample=dataset_gen_reset_seed_per_sample)
    if accuracy < accuracy_threshold:
        logger.fatal(f"The accuracy of random_dataset_count_{low} is smaller than {accuracy_threshold}. Stopped.")
        exit(-1)

    high = 2
    child_logger.info(f"try sample_count per label: {high}.")
    loss, accuracy = check_number_of_sample(high, random_dataset_type, random_dataset_func, output_folder_path, current_ml_setup, accuracy_threshold,
                                            use_amp=amp, core=core, dataset_gen_mp=dataset_gen_worker, override_epoch=override_epoch, override_weight_decay=override_weight_decay,
                                            dataset_gen_reset_seed_per_label=dataset_gen_reset_seed_per_label, dataset_gen_reset_seed_per_sample=dataset_gen_reset_seed_per_sample)
    while accuracy >= accuracy_threshold:
        low = high
        high *= 2
        child_logger.info(f"try sample_count per label: {high}.")
        loss, accuracy = check_number_of_sample(high, random_dataset_type, random_dataset_func, output_folder_path, current_ml_setup, accuracy_threshold,
                                                use_amp=amp, core=core, dataset_gen_mp=dataset_gen_worker, override_epoch=override_epoch, override_weight_decay=override_weight_decay,
                                                dataset_gen_reset_seed_per_label=dataset_gen_reset_seed_per_label, dataset_gen_reset_seed_per_sample=dataset_gen_reset_seed_per_sample)

    while True:
        mid = (low + high) // 2
        if mid == low or mid == high:
            logger.info(f"the maximum sample count is {mid}.")
            child_logger.info(f"the maximum sample count is {mid}.")
            return
        child_logger.info(f"try sample_count per label: {mid}.")
        loss, accuracy = check_number_of_sample(mid, random_dataset_type, random_dataset_func, output_folder_path, current_ml_setup, accuracy_threshold,
                                                use_amp=amp, core=core, dataset_gen_mp=dataset_gen_worker, override_epoch=override_epoch, override_weight_decay=override_weight_decay,
                                                dataset_gen_reset_seed_per_label=dataset_gen_reset_seed_per_label, dataset_gen_reset_seed_per_sample=dataset_gen_reset_seed_per_sample)

        if accuracy < accuracy_threshold:
            high = mid
        elif accuracy >= accuracy_threshold:
            low = mid
        else:
            raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure the model capacity in terms of how many random samples can be memorized')
    parser.add_argument("-m", "--model", type=str, required=True, help="specify the model type")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="specify the dataset type")
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')
    parser.add_argument("-c", '--core', type=int, default=os.cpu_count(), help='specify the number of CPU cores to use')
    parser.add_argument("--amp", action='store_true', help='enable auto mixed precision')
    parser.add_argument("-t","--accuracy_threshold", type=float, default=0.5, help="specify the accuracy threshold of treating as not low enough")
    parser.add_argument("--dataset_gen_worker", type=int, default=None, help='enable multiprocessing during dataset generation')
    parser.add_argument("--dataset_gen_reset_seed_per_label", action='store_true', help='reset the random seed after generating for each label')
    parser.add_argument("--dataset_gen_reset_seed_per_sample", action='store_true', help='reset the random seed after generating for each sample')
    parser.add_argument("-e", "--epoch", type=int, default=None, help="specify the number of epochs, None=default")
    parser.add_argument("-w", "--weight_decay", nargs="+", type=float, default=None, help="specify the weight decay, None=default, can be a list of float.")
    parser.add_argument("-s", "--sample_size", type=int, default=None, help="only measure whether the neural network model can memorize the dataset with that sample size or not.")

    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset
    amp = args.amp
    core = args.core
    accuracy_threshold = args.accuracy_threshold
    dataset_gen_worker = args.dataset_gen_worker
    dataset_gen_reset_seed_per_label = args.dataset_gen_reset_seed_per_label
    dataset_gen_reset_seed_per_sample = args.dataset_gen_reset_seed_per_sample
    override_epoch = args.epoch
    override_weight_decay = args.weight_decay
    sample_size = args.sample_size

    if args.output_folder_name is None:
        time_now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_folder_path = os.path.join(os.curdir, f"{__file__}_{time_now_str}")
    else:
        output_folder_path = os.path.join(os.curdir, args.output_folder_name)
    os.makedirs(output_folder_path)

    util.set_logging(logger, "main", log_file_path=os.path.join(output_folder_path, "log.txt"))

    current_ml_setup = ml_setup.get_ml_setup_from_config(model_name, dataset_type=dataset_name)
    random_dataset_type = ml_setup.dataset_type_to_random(current_ml_setup.dataset_type)
    try:
        random_dataset_func = ml_setup.dataset_type_to_setup[random_dataset_type]
    except KeyError:
        logger.fatal(f"The dataset type({random_dataset_type.name}) is not found in the ml_setup.name_to_dataset_setup mapping.")
        exit(-1)
    logger.info(f"Random dataset type: {random_dataset_type.name}")

    if sample_size is not None:
        logger.info(f"sample size is {sample_size}")
        if isinstance(override_weight_decay, list):
            logger.info(f"mode: measure multiple weights decay for single sample size")
            logger.info(f"wd is a list: {override_weight_decay}")
            for wd in override_weight_decay:
                output_folder_path_wd = os.path.join(output_folder_path, f"wd_{wd}")
                os.mkdir(output_folder_path_wd)
                check_number_of_sample(sample_size, random_dataset_type, random_dataset_func, output_folder_path, current_ml_setup, accuracy_threshold,
                                       use_amp=amp, core=core, dataset_gen_mp=dataset_gen_worker, override_epoch=override_epoch, override_weight_decay=wd,
                                       dataset_gen_reset_seed_per_label=dataset_gen_reset_seed_per_label, dataset_gen_reset_seed_per_sample=dataset_gen_reset_seed_per_sample)

        else:
            logger.info(f"mode: measure single weights decay for single sample size")
            logger.info(f"wd is a single value: {override_weight_decay}")
            check_number_of_sample(sample_size, random_dataset_type, random_dataset_func, output_folder_path, current_ml_setup, accuracy_threshold,
                                   use_amp=amp, core=core, dataset_gen_mp=dataset_gen_worker, override_epoch=override_epoch, override_weight_decay=override_weight_decay,
                                   dataset_gen_reset_seed_per_label=dataset_gen_reset_seed_per_label, dataset_gen_reset_seed_per_sample=dataset_gen_reset_seed_per_sample)
    else:
        if isinstance(override_weight_decay, list):
            logger.info(f"measure multiple weights decay mode")
            logger.info(f"wd is a list: {override_weight_decay}")
            for wd in override_weight_decay:
                output_folder_path_wd = os.path.join(output_folder_path, f"wd_{wd}")
                os.mkdir(output_folder_path_wd)
                measure_one_configuration(random_dataset_type, random_dataset_func, output_folder_path_wd, current_ml_setup, accuracy_threshold, override_epoch, wd)
        else:
            logger.info(f"measure singe weights decay mode")
            logger.info(f"wd is a single value: {override_weight_decay}")
            measure_one_configuration(random_dataset_type, random_dataset_func, output_folder_path, current_ml_setup, accuracy_threshold, override_epoch, override_weight_decay)



