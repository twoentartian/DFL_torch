import os
import argparse
from py_src import lmdb_pack

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Move all model stats in the output folder of find_high_accuracy_path to a lmdb database')
    parser.add_argument("folder", type=str, help="the output folder of find_high_accuracy_path")

    args = parser.parse_args()

    root_folder = args.folder

    all_entries = os.listdir(root_folder)
    folders = [entry for entry in all_entries if os.path.isdir(os.path.join(root_folder, entry))]
    print(f"all {len(folders)} folders: {folders}")
    model_stat_folder_name = "model_stat"
    for folder in folders:
        model_stat_folder = os.path.join(root_folder, folder, model_stat_folder_name)
        print(f"processing {folder}/{model_stat_folder_name}")
        if os.path.exists(model_stat_folder):
            lmdb_pack.store_folder_in_lmdb(model_stat_folder, f"{model_stat_folder}.lmdb")

