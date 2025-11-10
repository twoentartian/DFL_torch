import os
import re
import lmdb

def store_folder_in_lmdb(root_folder, lmdb_path, map_size=1099511627776):
    """
    Store the entire folder, including subfolders and files, into an LMDB database.

    Args:
    - root_folder (str): The path to the root folder to store in LMDB.
    - lmdb_path (str): The path to the LMDB database file.
    - map_size (int): The maximum size of the database in bytes.
    """
    env = lmdb.open(lmdb_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, root_folder)

                with open(file_path, 'rb') as f:
                    file_content = f.read()

                txn.put(relative_path.encode(), file_content)

    env.close()


def load_folder_from_lmdb(lmdb_path, output_folder):
    """
    Load the entire folder from an LMDB database and reconstruct it on the filesystem.

    Args:
    - lmdb_path (str): The path to the LMDB database file.
    - output_folder (str): The path to the folder where the contents will be restored.
    """
    env = lmdb.open(lmdb_path, readonly=True)

    with env.begin() as txn:
        cursor = txn.cursor()

        for key, value in cursor:
            relative_path = key.decode()
            file_path = os.path.join(output_folder, relative_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'wb') as f:
                f.write(value)

    env.close()


def generate_lmdb_index_from_node_name_and_tick(node_name: int, tick: int) -> str:
    lmdb_tx_name = f"{node_name}/{tick}.model.pt"
    return lmdb_tx_name

def get_node_name_and_tick_from_lmdb_index(lmdb_index) -> (int, int):
    match = re.match(rb"(\d+)/(\d+)\.model\.pt", lmdb_index)
    node_name = int(match.group(1))
    tick = int(match.group(2))
    return node_name, tick
