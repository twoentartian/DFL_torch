import torch
import copy
import logging
import gc
import subprocess

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from typing import List, Final
from py_src import internal_names, util, ml_setup
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase

logger = logging.getLogger(f"{internal_names.logger_simulator_base_name}.{util.basename_without_extension(__file__)}")

GPU_RESERVED_MEMORY_RATIO: Final[float] = 0.1
GPU_MAX_WORKERS: Final[int] = 1
GPU_SINGLE_THREAD_MODE: Final[bool] = True


class CudaDevice:
    def __init__(self, device_index):
        self.device_name = torch.cuda.get_device_name(device_index)
        self.id = f"cuda:{device_index}"
        self.device = torch.device(self.id)
        self.executor = None
        self.executor_size = None
        self.total_memory_MB = None
        self.free_memory_MB = None
        self.used_memory_MB = None

        # parameters for worker on gpu, nodes only have model_stat
        self.model = None
        self.optimizer = None
        self.nodes_allocated = None


def _measure_memory_consumption_for_performing_ml_proc_func(cuda_device_list, setup: ml_setup.MlSetup, shared_namespace):
    if len(cuda_device_list) == 0:
        return

    gpu_device = cuda_device_list[0].device
    # dataset
    initial_memory = torch.cuda.memory_allocated(device=gpu_device)
    temp_training_data = copy.deepcopy(setup.training_data)
    temp_testing_data = copy.deepcopy(setup.testing_data)
    temp_training_data.data = torch.from_numpy(temp_training_data.data)
    temp_testing_data.data = torch.from_numpy(temp_testing_data.data)
    temp_training_data.data = temp_training_data.data.to(device=gpu_device)
    temp_testing_data.data = temp_testing_data.data.to(device=gpu_device)
    final_memory = torch.cuda.memory_allocated(device=gpu_device)
    shared_namespace.memory_consumption_dataset_MB = (final_memory - initial_memory) / 1024 ** 2  # convert to MB
    del temp_training_data, temp_testing_data
    # model
    initial_memory = torch.cuda.memory_allocated(device=gpu_device)
    temp_model = copy.deepcopy(setup.model)
    temp_training_data = copy.deepcopy(setup.training_data)
    temp_model = temp_model.to(device=gpu_device)
    temp_train_loader = DataLoader(temp_training_data, batch_size=setup.training_batch_size, shuffle=True)
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
    shared_namespace.memory_consumption_model_MB = (final_memory - initial_memory) / 1024 ** 2  # convert to MB
    del temp_model, temp_training_data, temp_train_loader, criterion, optimizer

    gc.collect()
    torch.cuda.empty_cache()


class CudaEnv:
    def __init__(self):
        self._initialized = False
        self.cuda_available = torch.cuda.is_available()
        self.cuda_device_list: List[CudaDevice] = []
        self.memory_consumption_dataset_MB = None
        self.memory_consumption_model_MB = None
        self.initialize()
        self.mp_manager = mp.Manager()
        self.use_model_stat = None
        self.global_worker_model = None

    def initialize(self):
        if self._initialized:
            return
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            self.cuda_device_list.append(CudaDevice(i))

    def print_ml_info(self):
        logger.info(f"available GPUs: {[device.device_name for device in self.cuda_device_list]}")
        if self.memory_consumption_model_MB is not None:
            logger.info(f"memory consumption model: {self.memory_consumption_model_MB:.2f}MB")
        if self.memory_consumption_dataset_MB is not None:
            logger.info(f"memory consumption dataset: {self.memory_consumption_dataset_MB:.2f}MB")

    def print_gpu_info(self):
        self.__update_gpu_free_memory__()
        for device in self.cuda_device_list:
            logger.info(f"GPU {device.device_name}, used memory {device.used_memory_MB:.2f}/{device.total_memory_MB}MB")
            if device.executor_size is not None:
                logger.info(f"GPU {device.device_name} has {device.executor_size} workers")

    def measure_memory_consumption_for_performing_ml(self, setup: ml_setup.MlSetup):
        ns = self.mp_manager.Namespace()
        ns.memory_consumption_dataset_MB = 0
        ns.memory_consumption_model_MB = 0
        p = mp.Process(target=_measure_memory_consumption_for_performing_ml_proc_func, args=(self.cuda_device_list, setup, ns), )
        p.start()
        p.join()
        self.memory_consumption_dataset_MB = ns.memory_consumption_dataset_MB
        self.memory_consumption_model_MB = ns.memory_consumption_model_MB

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

    def generate_execution_strategy(self, model, config_file, config_ml_setup, node_count, override_use_model_stat: None | bool = None):
        self.__update_gpu_free_memory__()
        if GPU_SINGLE_THREAD_MODE:
            model_capacity_per_gpu = []
            if override_use_model_stat is None:
                # for GPU_SINGLE_THREAD_MODE, can we put all models to GPU memory?
                for gpu in self.cuda_device_list:
                    model_capacity_for_this_gpu = int((gpu.total_memory_MB * (1 - GPU_RESERVED_MEMORY_RATIO) - gpu.used_memory_MB - self.memory_consumption_dataset_MB) // self.memory_consumption_model_MB)
                    model_capacity_per_gpu.append(model_capacity_for_this_gpu)
                use_model_stat = (sum(model_capacity_per_gpu) < node_count)
            else:
                use_model_stat = override_use_model_stat
            if use_model_stat:
                # it's impossible to allocate all model to GPU memory, so we only allocate one model to the first gpu and each node keep model_stat
                logger.info(f"GPU execution strategy: SHARE_MODEL_ON_GPU -- keep model stat in memory and all nodes share one model on GPU")
                self.use_model_stat = True
                gpu = self.cuda_device_list[0]
                gpu.model = copy.deepcopy(model)
                gpu.model = gpu.model.to(device=gpu.device)
                para = RuntimeParameters()
                para.phase = SimulationPhase.INITIALIZING
                gpu.optimizer = config_file.get_optimizer(None, gpu.model, para, config_ml_setup)
                gpu.nodes_allocated = set(range(node_count))
            else:
                # it's possible to allocate all model to GPU memory
                logger.info(f"GPU execution strategy: DEDICATED_MODEL_ON_GPU each node has their own model on GPU")
                self.use_model_stat = False
                node_allocated = 0
                for index, gpu in enumerate(self.cuda_device_list):
                    gpu.nodes_allocated = set(range(node_allocated, node_allocated + model_capacity_per_gpu[index]))
        else:
            raise NotImplementedError("multiprocess is not implemented yet")

    @staticmethod
    def __optimizer_to(optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device, non_blocking=True)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device, non_blocking=True)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device, non_blocking=True)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device, non_blocking=True)


    def submit_training_jobs(self, training_nodes, criteria: list[torch.nn.CrossEntropyLoss], training_data: list[torch.Tensor], training_label: list[torch.Tensor]):
        assert len(training_nodes) == len(criteria) == len(training_data) == len(training_label)

        output_loss = []
        if GPU_SINGLE_THREAD_MODE:
            for index, target_node in enumerate(training_nodes):
                criterion = criteria[index]
                single_batch_data = training_data[index]
                single_batch_label = training_label[index]
                if target_node.is_using_model_stat:
                    """use model stat (share model on gpu)"""
                    gpu = target_node.allocated_gpu
                    shared_model_on_gpu = gpu.model
                    shared_optimizer_on_gpu = gpu.optimizer
                    shared_model_on_gpu.load_state_dict(target_node.model_status)
                    shared_optimizer_on_gpu.load_state_dict(target_node.optimizer_status)

                    shared_model_on_gpu.train()
                    data, labels = single_batch_data.cuda(device=gpu.device), single_batch_label.cuda(device=gpu.device)
                    shared_optimizer_on_gpu.zero_grad(set_to_none=True)
                    output = shared_model_on_gpu(data)
                    loss = criterion(output, labels)
                    loss.backward()
                    shared_optimizer_on_gpu.step()
                    target_node.model_status = shared_model_on_gpu.state_dict()
                    CudaEnv.__optimizer_to(shared_optimizer_on_gpu, torch.device('cpu')) # move optimizer data back to memory
                    target_node.optimizer_status = shared_optimizer_on_gpu.state_dict()
                    output_loss.append(loss.item())
                else:
                    """use dedicated model on gpu"""
                    model = target_node.model
                    gpu = target_node.allocated_gpu
                    optimizer = target_node.optimizer
                    data, labels = single_batch_data.cuda(device=gpu.device), single_batch_label.cuda(device=gpu.device)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(data)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    output_loss.append(loss.item())
        else:
            raise NotImplementedError("multiprocess is not implemented yet")

        return output_loss
