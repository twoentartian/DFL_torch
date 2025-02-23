import io
import torch
import unittest
import os
from mpi4py import MPI
from enum import IntEnum
from typing import Final

DEFAULT_CHUNk_SIZE: Final[int] = 1024*1024
MAX_CHUNK_SIZE = 1000


class MpiMessageTag(IntEnum):
    """take care: the value should at least differ for MAX_CHUNK_SIZE to ensure the tag does not overlap"""
    ModelStateData = 0,



class MpiData(object):
    def __init__(self, src_node, serialized_model_stat, dst_node):
        self.src_node = src_node
        self.dst_node = dst_node
        self.model_stat_data = serialized_model_stat

    def get_model_stat(self):
        return deserialize_model_stat(self.model_stat_data)

    def get_src_node(self):
        return self.src_node

    def get_dst_node(self):
        return self.dst_node

class MpiDataPack(object):
    def __init__(self, sender_rank: int):
        self.elements = []
        self.pack_sender_rank = sender_rank

    def get_pack_sender_rank(self):
        return self.pack_sender_rank

    def add_mpi_data(self, src_node, model_stat_ser, dst_node):
        new_data = MpiData(src_node, model_stat_ser, dst_node)
        self.elements.append(new_data)

    def get_mpi_data(self):
        output_list = []
        for element in self.elements:
            output_list.append((element.get_src_node(), element.get_model_stat(), element.get_dst_node()))
        return output_list

def serialize_model_stat(model_stat):
    # buffer = io.BytesIO()
    # torch.save(model_stat, buffer)
    # buffer.seek(0)
    # data = buffer.read()
    # return data

    output = {}
    for layer_name, layer_tensor in model_stat.items():
        output[layer_name] = layer_tensor.detach().cpu().numpy()
    return output

def deserialize_model_stat(data):
    # buffer = io.BytesIO(data)
    # buffer.seek(0)
    # model_stat = torch.load(buffer)
    # return model_stat

    model_stat = {}
    for layer_name, layer_numpy in data.items():
        model_stat[layer_name] = torch.from_numpy(layer_numpy)
    return model_stat

def mpi_isend_large_data(data, dest, MPI_comm, tag=0, chunk_size=DEFAULT_CHUNk_SIZE):
    """
    Send large data in chunks using non-blocking MPI.

    Parameters:
    - data (bytes): The data to send.
    - dest (int): The rank of the destination process.
    - tag (int): The starting tag for sending chunks.
    - chunk_size (int): The size of each chunk in bytes.
    """
    data_size = len(data)
    num_chunks = (data_size + chunk_size - 1) // chunk_size
    requests = []

    assert tag + 2 + num_chunks <= MAX_CHUNK_SIZE

    # Send the total number of chunks first
    req = MPI_comm.isend(num_chunks, dest=dest, tag=tag)
    requests.append(req)

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        end_index = min(end_index, data_size - 1)
        chunk = data[start_index:end_index]
        req = MPI_comm.isend(chunk, dest=dest, tag=tag + 1 + i)
        requests.append(req)

    return requests

def mpi_irece_large_data(source, MPI_comm, tag=0, chunk_size=DEFAULT_CHUNk_SIZE):
    """
    Receive large data in chunks using non-blocking MPI.

    Parameters:
    - source (int): The rank of the source process.
    - tag (int): The starting tag for receiving chunks.
    - chunk_size (int): The size of each chunk in bytes.

    Returns:
    - bytes: The received data.
    """
    requests = []

    # Receive the total number of chunks first
    num_chunks_req = MPI_comm.irecv(source=source, tag=tag)
    num_chunks = num_chunks_req.wait()
    for i in range(num_chunks):
        req = MPI_comm.irecv(source=source, tag=tag + 1 + i)
        requests.append(req)
    return requests

def mpi_irece_wait(requests):
    received_data = bytearray()
    chunks = [req.wait() for req in requests]
    for chunk in chunks:
        if chunk is not None:
            received_data.extend(chunk)
    return bytes(received_data)


"""MPI test entry"""
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    data_size = 30 * 1024 * 1024  # 30MB in bytes
    random_data = os.urandom(data_size)
    if rank == 0:
        reqs = mpi_isend_large_data(random_data, 1, comm)
        for req in reqs:
            req.wait()
        print(f"{rank} isend: {len(random_data)} bytes to 1")

    elif rank == 1:
        reqs = mpi_irece_large_data(0, comm)
        received_data = mpi_irece_wait(reqs)
        print(f"{rank} ireceive: {len(received_data)} bytes")
