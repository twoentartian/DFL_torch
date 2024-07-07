import torch
import copy
import logging
import gc
import subprocess
import math

from torch.utils.data import Dataset, DataLoader
from typing import List, Final
from concurrent.futures import ProcessPoolExecutor
from py_src import internal_names, util, ml_setup

logger = logging.getLogger(f"{internal_names.logger_simulator_base_name}.{util.basename_without_extension(__file__)}")

GPU_RESERVED_MEMORY_RATIO: Final[float] = 0.2
GPU_MAX_WORKERS: Final[int] = 4

class CudaDevice:
    def __init__(self, device_index):
        self.device_name = torch.cuda.get_device_name(device_index)
        self.id = f"cuda:{device_index}"
        self.device = torch.device(self.id)
        self.executor = None
        self.executor_size = None
        self.executor_busy_count = None
        self.total_memory_MB = None
        self.free_memory_MB = None
        self.used_memory_MB = None


class CudaEnv:
    def __init__(self):
        self._initialized = False
        self.cuda_available = torch.cuda.is_available()
        self.cuda_device_list: List[CudaDevice] = []
        self.memory_consumption_dataset = None
        self.memory_consumption_model = None
        self.initialize()

    def initialize(self):
        if self._initialized:
            return
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            self.cuda_device_list.append(CudaDevice(i))

    def print_ml_info(self):
        logger.info(f"available GPUs: {[device.device_name for device in self.cuda_device_list]}")
        if self.memory_consumption_model is not None:
            logger.info(f"memory consumption model: {self.memory_consumption_model:.2f}MB")
        if self.memory_consumption_dataset is not None:
            logger.info(f"memory consumption dataset: {self.memory_consumption_dataset:.2f}MB")

    def print_gpu_info(self):
        self.__update_gpu_free_memory__()
        for device in self.cuda_device_list:
            logger.info(f"GPU {device.device_name}, free memory {device.used_memory_MB:.2f}/{device.total_memory_MB}MB")
            if device.executor_size is not None:
                logger.info(f"GPU {device.device_name} has {device.executor_size} workers")

    def measure_memory_consumption_for_performing_ml(self, ml_setup: ml_setup.MlSetup):
        if not self.cuda_available:
            return

        gpu_device = self.cuda_device_list[0].device
        # dataset
        initial_memory = torch.cuda.memory_allocated(device=gpu_device)
        temp_training_data = copy.deepcopy(ml_setup.training_data)
        temp_testing_data = copy.deepcopy(ml_setup.testing_data)
        temp_training_data.data = torch.from_numpy(temp_training_data.data)
        temp_testing_data.data = torch.from_numpy(temp_testing_data.data)
        temp_training_data.data = temp_training_data.data.to(device=gpu_device)
        temp_testing_data.data = temp_testing_data.data.to(device=gpu_device)
        final_memory = torch.cuda.memory_allocated(device=gpu_device)
        self.memory_consumption_dataset = (final_memory - initial_memory) / 1024 / 1024  # convert to MB
        del temp_training_data, temp_testing_data
        # model
        initial_memory = torch.cuda.memory_allocated(device=gpu_device)
        temp_model = copy.deepcopy(ml_setup.model)
        temp_training_data = copy.deepcopy(ml_setup.training_data)
        temp_model.cuda(device=gpu_device)
        temp_train_loader = DataLoader(temp_training_data, batch_size=ml_setup.training_batch_size, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(temp_model.parameters(), lr=0.001)
        for data, labels in temp_train_loader:
            data, labels = data.to(device=gpu_device), labels.to(device=gpu_device)
            temp_model.train()
            optimizer.zero_grad()
            output = temp_model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            break
        final_memory = torch.cuda.memory_allocated(device=gpu_device)
        self.memory_consumption_model = (final_memory - initial_memory) / 1024 / 1024  # convert to MB
        del temp_model, temp_training_data, temp_train_loader

        torch.cuda.empty_cache()
        gc.collect()

    def __update_gpu_free_memory__(self):
        nvidia_smi_result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free,gpu_name', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
        output = nvidia_smi_result.stdout.decode('utf-8').strip().split('\n')
        nvidia_smi_gpu_info = []
        for i, line in enumerate(output):
            total_memory, used_memory, free_memory, gpu_name = map(str, line.split(', '))
            total_memory, used_memory, free_memory = map(int, [total_memory, used_memory, free_memory])
            nvidia_smi_gpu_info.append({'gpu_id': i, 'total_memory': total_memory, 'used_memory': used_memory, 'free_memory': free_memory, 'gpu_name': gpu_name})

        assert len(self.cuda_device_list) == len(nvidia_smi_gpu_info)
        for index, gpu in enumerate(self.cuda_device_list):
            current_nvidia_smi_info = nvidia_smi_gpu_info[index]
            assert gpu.device_name == current_nvidia_smi_info['gpu_name']
            gpu.total_memory_MB = current_nvidia_smi_info['total_memory']
            gpu.used_memory_MB = current_nvidia_smi_info['used_memory']
            gpu.free_memory_MB = gpu.total_memory_MB - gpu.used_memory_MB

    def allocate_executors(self):
        self.__update_gpu_free_memory__()
        for gpu in self.cuda_device_list:
            available_for_model = gpu.total_memory_MB * (1-GPU_RESERVED_MEMORY_RATIO) - gpu.used_memory_MB - self.memory_consumption_dataset
            gpu.executor_size = math.floor(available_for_model / self.memory_consumption_model)
            if gpu.executor_size > GPU_MAX_WORKERS:
                gpu.executor_size = GPU_MAX_WORKERS
            gpu.executor = ProcessPoolExecutor(gpu.executor_size, mp_context=torch.multiprocessing)
            gpu.executor_busy_count = 0

    def select_gpu_with_balance(self) -> CudaDevice:
        return self.cuda_device_list[0]

    def submit_training_job(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.CrossEntropyLoss, training_data: torch.Tensor, training_label: torch.Tensor):
        cuda_device = self.select_gpu_with_balance()
        model.cuda(device=cuda_device.device)
        model.train()
        data, labels = training_data.cuda(device=cuda_device.device), training_label.cuda(device=cuda_device.device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # clean GPU memory
        model.cpu()
        del data, labels
        return loss.item()


current_env_info = CudaEnv()
