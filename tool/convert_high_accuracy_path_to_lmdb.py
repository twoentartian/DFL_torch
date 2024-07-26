import os
from py_src import lmdb_pack

if __name__ == '__main__':
    current_path = os.getcwd()
    all_entries = os.listdir(current_path)
    folders = [entry for entry in all_entries if os.path.isdir(os.path.join(current_path, entry))]
    model_stat_folder_name = "model_stat"
    for folder in folders:
        model_stat_folder = os.path.join(current_path, folder, model_stat_folder_name)
        print(f"processing {folder}/{model_stat_folder_name}")
        if os.path.exists(model_stat_folder):
            lmdb_pack.store_folder_in_lmdb(model_stat_folder, f"{model_stat_folder}.lmdb")

