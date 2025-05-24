import torch
import os
import sys
import random
import copy
import hashlib
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, complete_ml_setup, util
from py_src.service import record_model_stat

logger = logging.getLogger("generate_high_accuracy_model_from_script")
save_model_stat_lmdb = True

def hash_model_state_dict(model):
    state_dict = model.state_dict()
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].cpu().numpy().tobytes()
        hasher.update(key.encode('utf-8'))
        hasher.update(tensor)
    return hasher.hexdigest()

if __name__ == "__main__":
    values_wd = [2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4]
    value_ml_setup = ml_setup.resnet18_cifar100()
    random_seed = 42

    """random seed"""
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    use_amp = True

    # logger
    util.set_logging(logger, "main")
    logger.info("logging setup complete")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = copy.deepcopy(value_ml_setup.training_data)

    batch_size = value_ml_setup.training_batch_size
    num_worker = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_worker, persistent_workers=True)
    criterion = value_ml_setup.criterion

    time_now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    main_output_folder_path = os.path.join(os.curdir, f"{__file__}_{time_now_str}")
    os.makedirs(main_output_folder_path, exist_ok=True)

    for wd in values_wd:
        """random seed"""
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(random_seed)

        model = copy.deepcopy(value_ml_setup.model)
        print(f"hash of init model: {hash_model_state_dict(model)}")
        model.to(device)

        name = f"{value_ml_setup.model_name}_{value_ml_setup.dataset_name}_rs{random_seed}_wd{wd:.1e}"
        output_folder_path = os.path.join(main_output_folder_path, name)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        else:
            print(f"{name} already exists")
            exit(-1)

        lr = 0.1
        epochs = 50
        # epochs = 50
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        steps_per_epoch = len(value_ml_setup.training_data) // value_ml_setup.training_batch_size + 1
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)

        value_ml_setup.re_initialize_model(model)

        log_file = open(os.path.join(output_folder_path, f"0.log"), "w")
        log_file.write("epoch,loss,lrs" + "\n")
        log_file.flush()

        model.train()
        logger.info(f"begin training")
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if save_model_stat_lmdb:
            record_model_service = record_model_stat.ModelStatRecorder(1)
            record_model_service.initialize_without_runtime_parameters([0], output_folder_path, save_format='lmdb', lmdb_db_name=f"0")
        else:
            record_model_service = None

        for epoch in range(epochs):
            train_loss = 0
            count = 0
            for data, label in dataloader:
                data, label = data.to(device), label.to(device)
                optimizer.zero_grad()
                if use_amp:
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
            logger.info(f"epoch[{epoch}] loss={train_loss / count} lrs={lrs}")
            log_file.write(f"{epoch},{train_loss / count},{lrs}" + "\n")
            log_file.flush()
            if record_model_service is not None:
                model_stat = model.state_dict()
                record_model_service.trigger_without_runtime_parameters(epoch, [0], [model_stat])
        logger.info(f"finish training")
        log_file.flush()
        log_file.close()

        util.save_model_state(os.path.join(output_folder_path, f"0.model.pt"),
                              model.state_dict(), value_ml_setup.model_name)
        util.save_optimizer_state(os.path.join(output_folder_path, f"0.optimizer.pt"),
                                  optimizer.state_dict(), value_ml_setup.model_name)

        torch.cuda.empty_cache()
