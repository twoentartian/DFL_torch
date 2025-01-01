import os
import sys
import json
import argparse
import torch
from torch.utils.data.dataloader import DataLoader
import copy
import concurrent.futures
import numpy as np
from decimal import Decimal
import random


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, util


MAX_CPU_COUNT = 32
RANDOM_SEED = 42

def fuse_two_model_states(model_a, weight_a, model_b, weight_b):
    output = {}
    for layer_name in model_a.keys():
        tensor_a = model_a[layer_name]
        tensor_b = model_b[layer_name]
        output[layer_name] = tensor_a * weight_a + tensor_b * weight_b
    return output

def process_file_func(model_states, task_list, ml_parameters, worker_info):
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    result = {}
    worker_index, total_worker, cores = worker_info
    model_state_a, model_state_b = model_states
    arg_ml_setup, test_batch_size, test_batch_count, test_dataset_type = ml_parameters

    thread_per_process = cores // total_worker
    torch.set_num_threads(thread_per_process)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = copy.deepcopy(arg_ml_setup.model)
    criterion = arg_ml_setup.criterion
    if test_dataset_type == "train":
        whole_training_dataset = arg_ml_setup.training_data
        indices = torch.randperm(len(whole_training_dataset))[:test_batch_size*test_batch_count]
        sub_training_dataset = torch.utils.data.Subset(whole_training_dataset, indices.tolist())
        dataloader = DataLoader(sub_training_dataset, batch_size=test_batch_size, shuffle=True)
    elif test_dataset_type == "test":
        whole_testing_dataset = arg_ml_setup.testing_data
        indices = torch.randperm(len(whole_testing_dataset))[:test_batch_size*test_batch_count]
        sub_testing_dataset = torch.utils.data.Subset(whole_testing_dataset, indices.tolist())
        dataloader = DataLoader(sub_testing_dataset, batch_size=test_batch_size, shuffle=True)
    else:
        raise NotImplementedError

    progress = 0
    for index, (model_a_weight, model_b_weight) in enumerate(task_list):
        output_model = fuse_two_model_states(model_state_a, model_a_weight, model_state_b, model_b_weight)
        model.load_state_dict(output_model)
        model.to(device)
        model.eval()

        all_loss = []
        total_predictions = 0
        correct_predictions = 0
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)
            outputs = model(data)
            loss = criterion(outputs, label)
            all_loss.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == label).sum().item()
            total_predictions += len(label)
        accuracy = correct_predictions / total_predictions
        result[(model_a_weight, model_b_weight)] = (accuracy, np.mean(all_loss))

        current_progress = index*100 // len(task_list)
        if current_progress > progress:
            print(f"worker{worker_index} progress = {current_progress} ")
            progress = current_progress
    return result


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Calculate the fused model accuracy and loss.')
    parser.add_argument("model_a", type=str, help="path to model a")
    parser.add_argument("model_b", type=str, help="path to model b")
    parser.add_argument("-p", "--precision", type=int, default=100, help="how many points to calculate for each model, default is 100")
    parser.add_argument("-s", "--scale", type=float, default=1.0, help="the upper limit for model scale, default is 1, indicating same scale as input model")
    parser.add_argument("-b", "--batch", type=int, default=100, help="test batch size,  default is 100")
    parser.add_argument("-k", "--batch_count", type=int, default=1, help="the number of test batches, default is 1")

    parser.add_argument("-c", '--core', type=int, default=os.cpu_count(), help='specify the number of CPU cores to use')
    parser.add_argument("-w", "--worker", type=int, default=1, help='specify how many workers to run in parallel')
    parser.add_argument("-t", "--test_dataset_type", type=str, default='test', choices=['test', 'train'])

    args = parser.parse_args()

    model_a_path = args.model_a
    model_b_path = args.model_b
    scale = args.scale
    precision = args.precision + 1 # +1 because we count the final ending point
    cores = args.core
    worker_count = args.worker
    test_batch_size = args.batch
    test_batch_count = args.batch_count
    test_dataset_type = args.test_dataset_type
    print(f"cores: {cores}    worker: {worker_count}")

    # output_path
    model_a_folder = os.path.dirname(model_a_path)
    model_b_folder = os.path.dirname(model_b_path)
    assert model_a_folder == model_b_folder, "please put input models in the same folder"
    model_a_file_name = os.path.basename(model_a_path).replace('.model.pt', '')
    model_b_file_name = os.path.basename(model_b_path).replace('.model.pt', '')
    output_folder = model_a_folder

    # load models
    model_state_a, model_a_name = util.load_model_state_file(model_a_path)
    model_state_b, model_b_name = util.load_model_state_file(model_b_path)
    util.assert_if_both_not_none(model_a_name, model_b_name)

    current_ml_setup = ml_setup.get_ml_setup_from_model_type(model_a_name)

    todo_list = []
    for p0 in np.linspace(0, scale, num=precision, endpoint=True):
        for p1 in np.linspace(0, scale, num=precision, endpoint=True):
            todo_list.append((p0, p1))
    todo_list_for_each_worker = util.split_list(todo_list, worker_count)

    total_cpu_count = args.core
    if total_cpu_count > MAX_CPU_COUNT:
        total_cpu_count = MAX_CPU_COUNT

    args = [((model_state_a, model_state_b), todo_list_for_each_worker[worker], (current_ml_setup, test_batch_size, test_batch_count, test_dataset_type), (worker, worker_count, cores)) for worker in range(0, worker_count)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(process_file_func, *arg) for arg in args]
        final_output = {}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            final_output = final_output | result

    # save to csv
    csv_path = os.path.join(output_folder, f'fused_model_accuracy_loss_on_{test_dataset_type}_{model_a_file_name}_{model_b_file_name}.csv')
    with open(csv_path, "w") as f:
        # write header
        f.write(f"0,1,accuracy,loss\n")
        for (p0, p1), (accuracy, loss) in final_output.items():
            f.write(f"{p0:.3f},{p1:.3f},{accuracy:.3f},{loss:.3f}\n")
        f.flush()
        f.close()


