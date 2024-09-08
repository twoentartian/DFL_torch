import os
import argparse
import sys
import shutil
import concurrent.futures

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import lmdb_pack

def process_func(arg_model_stat_folder, output_path=None):
    if os.path.exists(arg_model_stat_folder) and os.path.isdir(arg_model_stat_folder):
        if output_path is None:
            output_path = f"{arg_model_stat_folder}.lmdb"
        print(f"processing {arg_model_stat_folder}")
        lmdb_pack.store_folder_in_lmdb(arg_model_stat_folder, output_path)
        shutil.rmtree(arg_model_stat_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Move all model stats in the output folder of find_high_accuracy_path to a lmdb database')
    parser.add_argument("folder", type=str, help="the output folder of find_high_accuracy_path")
    parser.add_argument("-c", '--parallel', type=int, default=os.cpu_count(), help='specify how many models to train in parallel')

    args = parser.parse_args()

    root_folder = args.folder
    worker_count = args.parallel

    all_entries = os.listdir(root_folder)
    folders = [entry for entry in all_entries if os.path.isdir(os.path.join(root_folder, entry))]
    folders = sorted(folders)
    print(f"all {len(folders)} folders: {folders}")
    model_stat_folder_name = "model_stat"

    args = []
    for folder in folders:
        model_stat_folder = os.path.join(root_folder, folder, model_stat_folder_name)
        args.append((model_stat_folder,))

    if worker_count > len(folders):
        worker_count = len(folders)
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(process_func, *arg) for arg in args]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
