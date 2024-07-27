import lmdb
import argparse


def print_lmdb_entries(db_path):
    # Open the LMDB environment
    env = lmdb.open(db_path, readonly=True)

    with env.begin() as txn:
        # Get the total number of entries
        total_entries = txn.stat()['entries']
        print(f"Total number of entries: {total_entries}")

        # Create a cursor to iterate through the entries
        cursor = txn.cursor()

        # Iterate through all entries in the database
        for key, value in cursor:
            print(f"Key: {key}, Value length: {len(value)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LMDB Reader")
    parser.add_argument("db_path", help="Path to the LMDB database")
    args = parser.parse_args()

    print_lmdb_entries(args.db_path)
