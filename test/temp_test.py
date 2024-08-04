import os
from mpi4py import MPI
from mpi4py.util import pkl5

CHUNK_SIZE = 1024 * 1024  # 1MB per chunk


def isend_large_data(data, dest, tag=0, chunk_size=CHUNK_SIZE):
    """
    Send large data in chunks using non-blocking MPI.

    Parameters:
    - data (bytes): The data to send.
    - dest (int): The rank of the destination process.
    - tag (int): The starting tag for sending chunks.
    - chunk_size (int): The size of each chunk in bytes.
    """
    comm = MPI.COMM_WORLD
    data_size = len(data)
    num_chunks = (data_size + chunk_size - 1) // chunk_size  # Calculate the number of chunks
    requests = []

    # Send the total number of chunks and the size of the data first
    comm.send((num_chunks, data_size), dest=dest, tag=tag)

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        chunk = data[start_index:end_index]
        req = comm.isend(chunk, dest=dest, tag=tag + 1 + i)
        requests.append(req)

    # Wait for all non-blocking sends to complete
    MPI.Request.Waitall(requests)


def ireceive_large_data(source, tag=0, chunk_size=CHUNK_SIZE):
    """
    Receive large data in chunks using non-blocking MPI.

    Parameters:
    - source (int): The rank of the source process.
    - tag (int): The starting tag for receiving chunks.
    - chunk_size (int): The size of each chunk in bytes.

    Returns:
    - bytes: The received data.
    """
    comm = MPI.COMM_WORLD

    # Receive the total number of chunks and the size of the data first
    num_chunks, data_size = comm.recv(source=source, tag=tag)

    received_data = bytearray(data_size)
    requests = []

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        if end_index > data_size:
            end_index = data_size
        req = comm.irecv(source=source, tag=tag + 1 + i)
        requests.append((req, start_index, end_index))

    # Wait for all non-blocking receives to complete and place them in the correct position
    for req, start_index, end_index in requests:
        chunk = req.wait()
        received_data[start_index:end_index] = chunk

    return bytes(received_data)


def generate_random_data(size):
    """
    Generate random data of a given size.

    Parameters:
    - size (int): The size of the random data to generate in bytes.

    Returns:
    - bytes: The generated random data.
    """
    return os.urandom(size)


# Example usage
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        data_size = 28 * 1024 * 1024  # 28MB
        random_data = generate_random_data(data_size)
        for i in range(1, size):
            isend_large_data(random_data, dest=i, tag=0)
        print(f"Process {rank} sent 28MB of random data to all processes.")
    else:
        received_data = ireceive_large_data(source=0, tag=0)
        print(f"Process {rank} received 28MB of random data.")
        assert len(received_data) == 28 * 1024 * 1024
        print(f"Process {rank} verified the received data size is correct.")
