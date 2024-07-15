from mpi4py import MPI
from datetime import datetime

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

"""test 2 Broadcasting"""
if rank == 0:
    data = {'key1': [7, 2.72, 2+3j],
            'key2': ('abc', 'xyz')}
else:
    data = None
data = comm.bcast(data, root=0)
print(f"{rank}:{data}")

"""test 3 Scattering"""
if rank == 0:
    data = [(i+1)**2 for i in range(size)]
    print(f"{rank}:{data}")
else:
    data = None
data = comm.scatter(data, root=0)
print(f"{rank}:{data}")
assert data == (rank+1)**2

"""test 4 Gathering"""
data = (rank+1)**2
data = comm.gather(data, root=0)
if rank == 0:
    for i in range(size):
        assert data[i] == (i+1)**2
else:
    assert data is None
print(f"{rank}:{data}")