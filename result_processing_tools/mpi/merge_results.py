import os
import argparse
import shutil
import pandas as pd

def load_mpi_result_files(working_path, file_name):
    folders = [f for f in os.listdir(working_path) if os.path.isdir(f) and f.startswith("rank_")]
    merged_df = pd.DataFrame()
    for folder in folders:
        file_path = os.path.join(folder, file_name)
        print(f"loading {file_path}")
        df = pd.read_csv(file_path, header=0, index_col='tick')

        for col in df.columns:
            if col in merged_df.columns:
                if not merged_df[col].equals(df[col]):
                    raise ValueError(f"Column '{col}' in '{file_path}' has different content across files.")

        merged_df = pd.concat([merged_df, df], axis=1, ignore_index=True, sort=True)
    return merged_df

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
    parser.add_argument("mpi_folder", type=str, help="folder containing MPI results", default=".")

    args = parser.parse_args()

    path = args.mpi_folder

    merge_file_list = ["loss.csv", "accuracy.csv", "training_loss.csv", "weight_difference_l1.csv", "weight_difference_l2.csv"]
    for file_name in merge_file_list:
        merged_df = load_mpi_result_files(path, file_name)
        merged_df.to_csv(os.path.join(path, file_name))
    merge_topology_file(path)
