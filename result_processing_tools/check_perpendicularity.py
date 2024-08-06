import os
import argparse
import torch


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Generate some high accuracy models')
    parser.add_argument("start_folder", type=str, help="folder containing starting models")
    parser.add_argument("end_folder", type=str, help="folder containing destination models")
    parser.add_argument("--mapping_mode", type=str, default='auto', choices=['auto', 'all_to_all', 'each_to_each', 'one_to_all', 'all_to_one'])
    parser.add_argument("-c", '--parallel', type=int, default=os.cpu_count(), help='specify how many models to train in parallel')
    parser.add_argument("-m", "--model_type", type=str, default='lenet5', choices=['lenet5', 'resnet18'])
    parser.add_argument("-t", "--max_tick", type=int, default=10000)
    parser.add_argument("-s", "--step_size", type=float, default=0.001)
    parser.add_argument("-a", "--adoptive_step_size", type=float, default=0.0005)
    parser.add_argument("--training_round", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_format", type=str, default='lmdb', choices=['file', 'lmdb'])
    parser.add_argument("--cpu", action='store_true', help='force using CPU for training')

