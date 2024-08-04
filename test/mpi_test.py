import os
from mpi4py import MPI
from datetime import datetime
import numpy as np
from mpi4py.util import pkl5

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

"""test 1"""
if rank == 0:
    data = {'a': 7, 'b': 3.14}
    comm.send(data, dest=1, tag=11)
    now = datetime.now()
    now_str = now.strftime("%H:%M:%S.%f")
    print(f"{rank} send: {now_str} to 1")
elif rank == 1:
    data = comm.recv(source=0, tag=11)
    now = datetime.now()
    now_str = now.strftime("%H:%M:%S.%f")
    print(f"{rank} receive: {data} at {now_str}")
comm.barrier()

if rank == 0:
    data = {'a': 7, 'b': 3.14}
    req = comm.isend(data, dest=1, tag=11)
    now = datetime.now()
    now_str = now.strftime("%H:%M:%S.%f")
    req.wait()
    print(f"{rank} isend: {now_str} to 1")
elif rank == 1:
    req = comm.irecv(source=0, tag=11)
    data = req.wait()
    now = datetime.now()
    now_str = now.strftime("%H:%M:%S.%f")
    print(f"{rank} ireceive: {data} at {now_str}")
comm.barrier()

"""test 2 Broadcasting"""
if rank == 0:
    data = {'key1': [7, 2.72, 2+3j],
            'key2': ('abc', 'xyz')}
else:
    data = None
data = comm.bcast(data, root=0)
print(f"{rank}:{data}")
comm.barrier()

"""test 3 Scattering"""
if rank == 0:
    data = [(i+1)**2 for i in range(size)]
    print(f"{rank}:{data}")
else:
    data = None
data = comm.scatter(data, root=0)
print(f"{rank}:{data}")
assert data == (rank+1)**2
comm.barrier()

"""test 4 Gathering"""
data = (rank+1)**2
data = comm.gather(data, root=0)
if rank == 0:
    for i in range(size):
        assert data[i] == (i+1)**2
else:
    assert data is None
print(f"{rank}:{data}")
comm.barrier()

"""test 5 super large data"""
data_size = 30 * 1024 * 1024  # 30MB in bytes
CHUNK_SIZE = 1024 * 1024  # 1MB per chunk
random_data = os.urandom(data_size)
if rank == 0:
    num_chunks = data_size // CHUNK_SIZE
    for i in range(1, size):
        for j in range(num_chunks):
            start_index = j * CHUNK_SIZE
            end_index = start_index + CHUNK_SIZE
            comm.send(random_data[start_index:end_index], dest=i, tag=j)
        # Send any remaining data
        if data_size % CHUNK_SIZE != 0:
            comm.send(random_data[num_chunks * CHUNK_SIZE:], dest=i, tag=num_chunks)
    print(f"{rank} isend: {len(random_data)} bytes to 1")
elif rank == 1:
    received_data = bytearray(data_size)
    num_chunks = data_size // CHUNK_SIZE
    for j in range(num_chunks):
        chunk = comm.recv(source=0, tag=j)
        start_index = j * CHUNK_SIZE
        received_data[start_index:start_index + CHUNK_SIZE] = chunk
    # Receive any remaining data
    if data_size % CHUNK_SIZE != 0:
        chunk = comm.recv(source=0, tag=num_chunks)
        received_data[num_chunks * CHUNK_SIZE:] = chunk
    print(f"{rank} ireceive: {len(received_data)} bytes")
comm.barrier()

"""test 6 mpi4py pickle5"""
large_comm = pkl5.Intracomm(MPI.COMM_WORLD)
dst = (rank + 1) % size
src = (rank - 1) % size
sobj = np.full(500*1024**2, rank, dtype='i4')  # > 4 GiB
sreq = large_comm.isend(sobj, dst, tag=42)
sreq.Free()

status = MPI.Status()
rmsg = large_comm.mprobe(status=status)
assert status.Get_source() == src
assert status.Get_tag() == 42
rreq = rmsg.irecv()
robj = rreq.wait()


assert np.max(robj) == src
assert np.min(robj) == src

print("test pass")