import os
import argparse
import shutil

import pandas as pd

def load_mpi_result_files(working_path, file_name):
    folders = [f for f in os.listdir(working_path) if os.path.isdir(f) and f.startswith("rank_")]
    folders.sort()
    merged_df_dict = {}
    for folder in folders:
        file_path = os.path.join(folder, file_name)
        if not os.path.exists(file_path):
            print(f"{file_path} does not exist")
            return None
        print(f"loading {file_path}")
        df = pd.read_csv(file_path, header=0)

        for col in df.columns:
            if col in merged_df_dict.keys():
                pass
                if not merged_df_dict[col].equals(df[col]):
                    raise ValueError(f"Column '{col}' in '{file_path}' has different content across files.")
            else:
                merged_df_dict[col] = df[col]
    merged_df = pd.DataFrame(merged_df_dict)
    print(merged_df)
    info_columns = []
    for col in merged_df.columns:
        if not col.isdigit():
            info_columns.append(col)
    tick_phase = merged_df[info_columns]
    numerical_sorted = merged_df.drop(columns=info_columns).sort_index(axis=1)
    sorted_df = pd.concat([tick_phase, numerical_sorted], axis=1)
    return sorted_df

def merge_topology_file(working_path):
    dest_dir = os.path.join(working_path, "topology")
    os.makedirs(dest_dir, exist_ok=True)
    folders = [f for f in os.listdir(working_path) if os.path.isdir(f) and f.startswith("rank_")]
    for folder in folders:
        topology_path = os.path.join(folder, "topology")
        if os.path.isdir(topology_path):
            for filename in os.listdir(topology_path):
                src_file = os.path.join(topology_path, filename)
                dest_file = os.path.join(dest_dir, filename)
                shutil.copy2(src_file, dest_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge MPI results to follow normal result style')
    parser.add_argument("-f", "--mpi_folder", type=str, help="folder containing MPI results", default=".")

    args = parser.parse_args()

    path = args.mpi_folder

    merge_file_list = ["loss.csv", "accuracy.csv", "training_loss.csv", "weight_difference_l1.csv", "weight_difference_l2.csv"]
    for file_name in merge_file_list:
        merged_df = load_mpi_result_files(path, file_name)
        if merged_df is None:
            continue
        merged_df.to_csv(os.path.join(path, file_name), index=False)
    merge_topology_file(path)
