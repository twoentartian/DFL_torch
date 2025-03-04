import io
import torch
from torch.utils.data import DataLoader
import sys
import os
import pandas
import logging

import lmdb
import argparse
import concurrent.futures

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, util

logger = logging.getLogger("measure_test_accuracy")

def measure_model_in_lmdb(db_path, arg_ml_setup: ml_setup.MlSetup, arg_test_batch_size, arg_dataloader_worker, output_path, task_name):
    # logger
    child_logger = logging.getLogger(f"measure_test_accuracy.{task_name}")
    util.set_logging(child_logger, task_name)

    # Open the LMDB environment
    env = lmdb.open(db_path, readonly=True)

    with env.begin() as txn:
        # Get the total number of entries
        total_entries = txn.stat()['entries']
        child_logger.info(f"Total number of models: {total_entries}")
        cursor = txn.cursor()

        model = arg_ml_setup.model
        testing_dataset = current_ml_setup.testing_data
        criterion = current_ml_setup.criterion
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if arg_dataloader_worker is None:
            dataloader_test = DataLoader(testing_dataset, batch_size=arg_test_batch_size, shuffle=True)
        else:
            dataloader_test = DataLoader(testing_dataset, batch_size=arg_test_batch_size, shuffle=True, num_workers=arg_dataloader_worker, persistent_workers=True)
        model.eval()
        model.to(device)
        with torch.no_grad():
            record_node_name = None
            keys = sorted(txn.cursor().iternext(values=False))
            all_ticks = []
            for key in keys:
                items = key.decode("utf-8").split("/")
                node_name = items[0]
                tick = items[1].replace(".model.pt", "")
                if record_node_name is None:
                    record_node_name = node_name
                else:
                    assert record_node_name == node_name, f"node name changes: {record_node_name} -> {node_name}"
                all_ticks.append(int(tick))
            all_ticks = sorted(all_ticks)

            test_accuracy_file_path = os.path.join(output_path, f"full_test_accuracy.csv")
            test_loss_file_path = os.path.join(output_path, f"full_test_loss.csv")
            if os.path.exists(test_accuracy_file_path) and os.path.exists(test_loss_file_path):
                accuracy_df = pandas.read_csv(test_accuracy_file_path)
                loss_df = pandas.read_csv(test_loss_file_path)
                accuracy_ticks = accuracy_df["tick"].sort_values().tolist()
                loss_ticks = loss_df["tick"].sort_values().tolist()
                if all_ticks == accuracy_ticks and all_ticks == loss_ticks:
                    child_logger.info(f"Loss and accuracy already generated")
                    return

            # create output files
            test_accuracy_file = open(test_accuracy_file_path, "w+")
            header = ",".join(["tick", f"{record_node_name}"])
            test_accuracy_file.write(header+ "\n")
            test_loss_file = open(test_loss_file_path, "w+")
            header = ",".join(["tick", f"{record_node_name}"])
            test_loss_file.write(header+ "\n")

            for tick in all_ticks:
                key = f"{record_node_name}/{tick}.model.pt"
                key_b = key.encode("utf-8")
                value = cursor.get(key_b)

                child_logger.info(f"Model: {key}")
                buffer = io.BytesIO(value)
                state_dict = torch.load(buffer, map_location=device)
                arg_ml_setup.model.load_state_dict(state_dict)

                test_loss = 0
                correct = 0
                total = 0
                for batch_idx, (data, label) in enumerate(dataloader_test):
                    data, label = data.to(device), label.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, label)
                    test_loss += loss.item() * data.size(0)
                    _, predicted = outputs.max(1)
                    total += label.size(0)
                    correct += predicted.eq(label).sum().item()
                test_loss = test_loss / total
                test_accuracy = correct / total

                row_accuracy_str = ",".join([str(tick), str(test_accuracy)])
                row_loss_str = ",".join([str(tick), str(test_loss)])

                test_accuracy_file.write(row_accuracy_str + "\n")
                test_accuracy_file.flush()
                test_loss_file.write(row_loss_str + "\n")
                test_loss_file.flush()

            # close files
            test_accuracy_file.flush()
            test_accuracy_file.close()
            test_loss_file.flush()
            test_loss_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="measure the model test accuracy in a LMDB file")
    parser.add_argument("path", help="High accuracy trajectory folder path")

    parser.add_argument("-w", "--worker", type=int, default=1, help='specify how many models to train in parallel')
    parser.add_argument("model_type", type=str)
    parser.add_argument("-d", "--dataset_type", type=str, default=None)
    parser.add_argument("--test_batch_size", type=int, default=100)
    parser.add_argument("--dataloader_worker", default=None)
    args = parser.parse_args()

    if args.dataset_type is None:
        current_ml_setup = ml_setup.get_ml_setup_from_config(args.model_type)
    else:
        current_ml_setup = ml_setup.get_ml_setup_from_config(args.model_type, dataset_type=args.dataset_type)

    path = os.path.abspath(args.path)
    worker = args.worker
    test_batch_size = args.test_batch_size
    data_loader_worker = args.dataloader_worker
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and "-" in f]
    print(f"all folders: {folders}")

    output_base_path = os.path.join(path, "temp_test_accuracy")
    if not os.path.exists(output_base_path):
        os.mkdir(output_base_path)

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker) as executor:
        futures = []
        for folder in folders:
            task_name = folder
            db_path = os.path.join(path, folder, "model_stat.lmdb")
            output_path = os.path.join(output_base_path, folder)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            futures.append(executor.submit(measure_model_in_lmdb, db_path, current_ml_setup, test_batch_size, data_loader_worker, output_path, task_name))

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
    executor.shutdown(wait=True)