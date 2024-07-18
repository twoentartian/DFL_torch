import platform
import logging
from enum import Enum
from py_src import cuda, internal_names, util

logger = logging.getLogger(f"{internal_names.logger_simulator_base_name}.{util.basename_without_extension(__file__)}")


def collect_netbios_name():
    return platform.node()

def collect_netbios_cuda_info():
    # collect netbios name
    netbios_name = collect_netbios_name()
    # collect free GPU memory
    gpu_infos = cuda.CudaEnv.get_gpu_memory_info()
    return netbios_name, gpu_infos

class MpiGpuMemStrategy(Enum):
    AllocateAllModels = 0,
    ShareSingleModel = 1,

    Unknown = 2

class MpiGpu(object):
    def __init__(self, name):
        self.name = name
        self.total_mem = None
        self.free_mem = None
        self.used_mem = None
        self.gpu_index = None

class MpiProcess(object):
    def __init__(self, rank):
        self.rank = rank
        self.nodes = None
        self.allocated_gpu = None

class MpiHost(object):
    def __init__(self, hostname):
        self.hostname = hostname
        self.gpus = {}
        self.mpi_process = {}

    def add_gpu(self, gpu_index: int, gpu_name: str, total_mem: int, free_mem: int):
        temp_gpu = MpiGpu(gpu_name)
        temp_gpu.gpu_index = gpu_index
        temp_gpu.total_mem = total_mem
        temp_gpu.free_mem = free_mem
        temp_gpu.used_mem = total_mem - free_mem
        self.gpus[gpu_index] = temp_gpu

    def add_mpi_process_rank(self, rank, nodes):
        temp_process = MpiProcess(rank)
        temp_process.nodes = nodes
        assert rank not in self.mpi_process.keys()
        self.mpi_process[rank] = temp_process

    def print_info(self):
        logger.info(f"Host: '{self.hostname}' contains {len(self.gpus)} GPUs and {len(self.mpi_process)} MPI processes({set(self.mpi_process.keys())}).")
        for gpu_index, gpu in self.gpus.items():
            logger.info(f"GPU '{gpu_index}': {gpu.name} total mem: {gpu.total_mem} free mem: {gpu.free_mem}")
        for mpi_process_rank, mpi_process in self.mpi_process.items():
            logger.info(f"MPI Process {mpi_process.rank} on '{mpi_process.allocated_gpu.name}': {mpi_process.nodes}")

class MpiWorld(object):
    def __init__(self):
        self.all_hosts = {}
        self.gpu_mem_strategy = MpiGpuMemStrategy.AllocateAllModels

    def add_mpi_host(self, host):
        self.all_hosts[host.hostname] = host

    def print_info(self):
        for _, host in self.all_hosts.items():
            host.print_info()

    def determine_mem_strategy(self, model_size, dataset_size, override_use_model_stat: None | bool = None, override_allocate_all_models: None | bool = None):
        if override_use_model_stat is None:
            override_use_model_stat = False
        if override_allocate_all_models is None:
            override_allocate_all_models = False
        if override_allocate_all_models:
            self.gpu_mem_strategy = MpiGpuMemStrategy.AllocateAllModels
        else:
            if not override_use_model_stat:
                allocate_all_models = True
                for host in self.all_hosts.values():
                    total_gpu_free_mem = 0
                    for gpu in host.gpus.values():
                        total_gpu_free_mem += gpu.free_mem
                    total_nodes = 0
                    for mpi_process in host.mpi_process.values():
                        total_nodes += len(mpi_process.nodes)
                    allocate_all_model_require_memory = total_nodes * model_size + dataset_size
                    if total_gpu_free_mem*(1-cuda.GPU_RESERVED_MEMORY_RATIO) <= allocate_all_model_require_memory:
                        allocate_all_models = False
                    logger.info(f"For host: '{host.hostname}'. Total memory required: {allocate_all_model_require_memory:.2f}MB, available: {total_gpu_free_mem}MB.")
                if allocate_all_models:
                    self.gpu_mem_strategy = MpiGpuMemStrategy.AllocateAllModels
                else:
                    self.gpu_mem_strategy = MpiGpuMemStrategy.ShareSingleModel
                logger.info(f"GPU memory strategy: {self.gpu_mem_strategy.name}.")
            else:
                self.gpu_mem_strategy = MpiGpuMemStrategy.ShareSingleModel


    def allocate_nodes_to_gpu(self):
        for host in self.all_hosts.values():
            gpu_index = 0
            total_gpu_count = len(host.gpus)
            for single_mpi_process in host.mpi_process.values():
                single_mpi_process.allocated_gpu = host.gpus[gpu_index]
                gpu_index += 1
                if gpu_index == total_gpu_count:
                    gpu_index = 0
