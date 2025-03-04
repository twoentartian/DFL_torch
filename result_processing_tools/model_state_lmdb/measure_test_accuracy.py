import io
import torch
from torch.utils.data import DataLoader
import sys
import os

import lmdb
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup

def measure_model_in_lmdb(db_path, arg_ml_setup: ml_setup.MlSetup, arg_test_batch_size, output_path):
    # Open the LMDB environment
    env = lmdb.open(db_path, readonly=True)

    with env.begin() as txn:
        # Get the total number of entries
        total_entries = txn.stat()['entries']
        print(f"Total number of models: {total_entries}")
        cursor = txn.cursor()

        model = arg_ml_setup.model
        testing_dataset = current_ml_setup.testing_data
        criterion = current_ml_setup.criterion
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataloader_test = DataLoader(testing_dataset, batch_size=arg_test_batch_size, shuffle=True, num_workers=8, persistent_workers=True)

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

            # create output files
            test_accuracy_file = open(os.path.join(output_path, f"full_test_accuracy_{record_node_name}.csv"), "w+")
            header = ",".join(["tick", f"{record_node_name}"])
            test_accuracy_file.write(header)
            test_loss_file = open(os.path.join(output_path, f"full_test_loss_{record_node_name}.csv"), "w+")
            header = ",".join(["tick", f"{record_node_name}"])
            test_loss_file.write(header)

            for tick in all_ticks:
                key = f"{record_node_name}/{tick}.model.pt"
                key_b = key.encode("utf-8")
                value = cursor.get(key_b)

                print(f"Model: {key}")
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
    parser.add_argument("db_path", help="Path to the LMDB database")
    parser.add_argument("model_type", type=str)
    parser.add_argument("-d", "--dataset_type", type=str, default=None)
    parser.add_argument("--test_batch_size", type=int, default=100)
    args = parser.parse_args()

    if args.dataset_type is None:
        current_ml_setup = ml_setup.get_ml_setup_from_config(args.model_type)
    else:
        current_ml_setup = ml_setup.get_ml_setup_from_config(args.model_type, dataset_type=args.dataset_type)
    print(f"Current ML model: {current_ml_setup.model_name}, dataset name: {current_ml_setup.dataset_name}")
    test_batch_size = args.test_batch_size
    db_path = args.db_path
    output_path = os.path.dirname(db_path)
    measure_model_in_lmdb(db_path, current_ml_setup, test_batch_size, output_path)
