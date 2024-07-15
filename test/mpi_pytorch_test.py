import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from datetime import datetime
from mpi4py import MPI
import io
import pydevd_pycharm

# Initialize the MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
port_mapping = [65110, 65111]
pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)

# Ensure we are running with exactly 2 processes
if size != 2:
    raise ValueError("This example requires exactly 2 MPI processes.")

# Root process (rank 0) creates and sends the model
if rank == 0:
    # Create the model and set state_dict
    model = models.resnet18(progress=False, num_classes=10, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Sample state_dict modification (for demonstration purposes)
    state_dict = model.state_dict()
    # for key in state_dict:
    #     state_dict[key] = state_dict[key] + 1

    # Serialize the state_dict to a bytes buffer
    start = datetime.now()
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    data = buffer.read()
    end = datetime.now()
    elapsed_str = (end - start).microseconds
    print(f"serialization costs: {elapsed_str}us")


    # Send the serialized state_dict to the other process
    comm.send(data, dest=1, tag=0)
    print("Root process (rank 0) sent the model state_dict.")

# Other process (rank 1) receives the model
elif rank == 1:
    # Receive the serialized state_dict
    data = comm.recv(source=0, tag=0)

    # Deserialize the state_dict
    buffer = io.BytesIO(data)
    buffer.seek(0)
    state_dict = torch.load(buffer)

    # Create a model instance and load the received state_dict
    model = models.resnet18(progress=False, num_classes=10, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None)
    model.load_state_dict(state_dict)

    print("Process (rank 1) received the model state_dict.")
    # print("Model state_dict:", model.state_dict())

# Finalize the MPI environment
MPI.Finalize()
