import torch
import copy
import logging
import gc
import subprocess
import numpy as np
import multiprocessing as python_mp
import torch.multiprocessing as pytorch_mp
from torch.utils.data import DataLoader
from typing import List, Final
from py_src import internal_names, util, ml_setup
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase

logger = logging.getLogger(f"{internal_names.logger_simulator_base_name}.{util.basename_without_extension(__file__)}")

GPU_RESERVED_MEMORY_RATIO: Final[float] = 0.1
GPU_MAX_WORKERS: Final[int] = 1
GPU_SINGLE_THREAD_MODE: Final[bool] = True
MEASURE_MODEL_MEMORY_CONSUMPTION_IN_A_NEW_PROCESS: Final[bool] = True


class CudaDevice:
    def __init__(self, device_index):
        self.device_name = torch.cuda.get_device_name(device_index)
        self.id = f"cuda:{device_index}"
        self.device = torch.device(self.id)
        self.total_memory_MB = None
        self.free_memory_MB = None
        self.used_memory_MB = None

        # parameters for worker on gpu, nodes only have model_stat
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.nodes_allocated = None

"""torch.cuda.mem_get_info()"""
def _measure_memory_consumption_for_performing_ml_proc_func(cuda_device_list, setup: ml_setup.MlSetup, return_queue=None):
    if len(cuda_device_list) == 0:
        return

    gpu_device = cuda_device_list[0].device
    # dataset
    initial_memory = torch.cuda.memory_allocated(device=gpu_device)
    temp_training_data = copy.deepcopy(setup.training_data)
    temp_testing_data = copy.deepcopy(setup.testing_data)
    # we don't use data loader here, so we should take care of whether they are tensor or ndarray
    if isinstance(temp_training_data.data, np.ndarray):
        temp_training_data.data = torch.from_numpy(temp_training_data.data)
    if isinstance(temp_testing_data.data, np.ndarray):
        temp_testing_data.data = torch.from_numpy(temp_testing_data.data)
    temp_training_data.data = temp_training_data.data.to(device=gpu_device)
    temp_testing_data.data = temp_testing_data.data.to(device=gpu_device)
    final_memory = torch.cuda.memory_allocated(device=gpu_device)
    memory_consumption_dataset_MB = (final_memory - initial_memory) / 1024 ** 2  # convert to MB
    del temp_training_data.data, temp_testing_data.data
    del temp_training_data, temp_testing_data
    # model
    initial_memory = torch.cuda.memory_allocated(device=gpu_device)
    temp_model = copy.deepcopy(setup.model)
    temp_training_data = copy.deepcopy(setup.training_data)
    temp_model = temp_model.to(device=gpu_device)
    temp_train_loader = DataLoader(temp_training_data, batch_size=setup.training_batch_size, shuffle=True)
    criterion = copy.deepcopy(setup.criterion)
    optimizer = torch.optim.Adam(temp_model.parameters(), lr=0.001)
    for index, (data, labels) in enumerate(temp_train_loader):
        data, labels = data.to(device=gpu_device), labels.to(device=gpu_device)
        temp_model.train()
        optimizer.zero_grad()
        output = temp_model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if index > 10:
            break
    final_memory = torch.cuda.memory_allocated(device=gpu_device)
    memory_consumption_model_MB = (final_memory - initial_memory) / 1024 ** 2  # convert to MB
    del temp_model, temp_training_data, temp_train_loader, criterion, optimizer

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    if return_queue is not None:
        return_queue.put((memory_consumption_dataset_MB, memory_consumption_model_MB))
    else:
        return memory_consumption_dataset_MB, memory_consumption_model_MB

class CudaEnv:
    def __init__(self):
        self._initialized = False
        self.cuda_available = torch.cuda.is_available()
        self.cuda_device_list: List[CudaDevice] = []
        self.memory_consumption_dataset_MB = None
        self.memory_consumption_model_MB = None
        self.initialize()
        self.use_model_stat = None
        self.global_worker_model = None
        self.model_capacity_per_gpu = None

    def initialize(self):
        if self._initialized:
            return
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            self.cuda_device_list.append(CudaDevice(i))

    def print_ml_info(self):
        logger.info(f"available GPUs: {[device.device_name for device in self.cuda_device_list]}")
        if self.memory_consumption_model_MB is not None:
            logger.info(f"memory consumption model and optimizer: {self.memory_consumption_model_MB:.2f}MB")
        if self.memory_consumption_dataset_MB is not None:
            logger.info(f"memory consumption dataset: {self.memory_consumption_dataset_MB:.2f}MB")

    def print_gpu_info(self):
        self.__update_gpu_free_memory__()
        for device in self.cuda_device_list:
            logger.info(f"GPU {device.device_name}, used memory {device.used_memory_MB:.2f}/{device.total_memory_MB}MB")

    def measure_memory_consumption_for_performing_ml(self, setup: ml_setup.MlSetup, measure_in_new_process=MEASURE_MODEL_MEMORY_CONSUMPTION_IN_A_NEW_PROCESS):
        if measure_in_new_process:
            """start a new process to measure GPU memory consumption"""
            queue = python_mp.Queue()
            p = pytorch_mp.Process(target=_measure_memory_consumption_for_performing_ml_proc_func, args=(self.cuda_device_list, setup, queue), )
            p.start()
            p.join()
            (dataset_memory_MB, model_memory_MB) = queue.get()
            self.memory_consumption_dataset_MB = dataset_memory_MB
            self.memory_consumption_model_MB = model_memory_MB
        else:
            """run in current process"""
            dataset_memory_MB, model_memory_MB = _measure_memory_consumption_for_performing_ml_proc_func(self.cuda_device_list, setup)
            self.memory_consumption_dataset_MB = dataset_memory_MB
            self.memory_consumption_model_MB = model_memory_MB

    @staticmethod
    def get_gpu_memory_info():
        nvidia_smi_result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free,gpu_name', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
        output = nvidia_smi_result.stdout.decode('utf-8').strip().split('\n')
        nvidia_smi_gpu_info = []
        for i, line in enumerate(output):
            total_memory, used_memory, free_memory, gpu_name = map(str, line.split(', '))
            total_memory, used_memory, free_memory = map(int, [total_memory, used_memory, free_memory])
            nvidia_smi_gpu_info.append({'gpu_id': i, 'total_memory': total_memory, 'used_memory': used_memory, 'free_memory': free_memory, 'gpu_name': gpu_name})

        output_list = []
        for index, gpu in enumerate(nvidia_smi_gpu_info):
            current_nvidia_smi_info = nvidia_smi_gpu_info[index]
            device_name = current_nvidia_smi_info['gpu_name']
            total_memory_MB = current_nvidia_smi_info['total_memory']
            used_memory_MB = current_nvidia_smi_info['used_memory']
            free_memory_MB = total_memory_MB - used_memory_MB
            output_list.append((device_name, total_memory_MB, used_memory_MB, free_memory_MB))
        return output_list

    def __update_gpu_free_memory__(self):
        gpu_infos = self.get_gpu_memory_info()
        assert len(gpu_infos) == len(self.cuda_device_list)
        for index, gpu_info in enumerate(gpu_infos):
            pytorch_gpu_info = self.cuda_device_list[index]
            (gpu_name, total_memory_MB, used_memory_MB, free_memory_MB) = gpu_info
            assert pytorch_gpu_info.device_name == gpu_name
            pytorch_gpu_info.free_memory_MB = free_memory_MB
            pytorch_gpu_info.used_memory_MB = used_memory_MB
            pytorch_gpu_info.total_memory_MB = total_memory_MB

    def generate_execution_strategy(self, node_set, override_use_model_stat: None | bool = None, override_allocate_all_models: None | bool = None):
        self.__update_gpu_free_memory__()
        if GPU_SINGLE_THREAD_MODE:
            model_capacity_per_gpu = []
            if override_use_model_stat is None:
                override_use_model_stat = False
            if override_allocate_all_models is None:
                override_allocate_all_models = False
            if override_allocate_all_models:
                for gpu in self.cuda_device_list:
                    model_capacity_for_this_gpu = int((gpu.total_memory_MB * (1 - GPU_RESERVED_MEMORY_RATIO) - gpu.used_memory_MB - self.memory_consumption_dataset_MB) // self.memory_consumption_model_MB)
                    model_capacity_per_gpu.append(model_capacity_for_this_gpu)
                use_model_stat = False
            else:
                if not override_use_model_stat:
                    # for GPU_SINGLE_THREAD_MODE, can we put all models to GPU memory?
                    for gpu in self.cuda_device_list:
                        model_capacity_for_this_gpu = int((gpu.total_memory_MB * (1 - GPU_RESERVED_MEMORY_RATIO) - gpu.used_memory_MB - self.memory_consumption_dataset_MB) // self.memory_consumption_model_MB)
                        model_capacity_per_gpu.append(model_capacity_for_this_gpu)
                    use_model_stat = (sum(model_capacity_per_gpu) < len(node_set))
                else:
                    use_model_stat = True
            self.use_model_stat = use_model_stat
            self.model_capacity_per_gpu = model_capacity_per_gpu
        else:
            raise NotImplementedError("multiprocess is not implemented yet")

    def prepare_gpu_memory(self, model, config_file, config_ml_setup, node_set):
        assert self.use_model_stat is not None
        if GPU_SINGLE_THREAD_MODE:
            if self.use_model_stat:
                # it's impossible to allocate all model to GPU memory, so we only allocate one model to the first gpu and each node keep model_stat
                logger.info(f"GPU execution strategy: SHARE_MODEL_ON_GPU -- keep model stat in memory and all nodes share one model on GPU")

                gpu = self.cuda_device_list[0]
                gpu.model = copy.deepcopy(model)
                gpu.model = gpu.model.to(device=gpu.device)
                para = RuntimeParameters()
                para.phase = SimulationPhase.INITIALIZING
                gpu.optimizer, lr_scheduler = config_file.get_optimizer(None, gpu.model, para, config_ml_setup)
                if lr_scheduler is not None:
                    gpu.lr_scheduler = lr_scheduler
                gpu.nodes_allocated = node_set
            else:
                assert self.model_capacity_per_gpu is not None
                # it's possible to allocate all model to GPU memory
                logger.info(f"GPU execution strategy: DEDICATED_MODEL_ON_GPU each node has their own model on GPU")
                node_allocated = 0
                node_set_list = list(node_set)
                for index, gpu in enumerate(self.cuda_device_list):
                    gpu.nodes_allocated = set(node_set_list[node_allocated: node_allocated + self.model_capacity_per_gpu[index]])
        else:
            raise NotImplementedError("multiprocess is not implemented yet")

    def mpi_prepare_gpu_memory(self, model, config_file, config_ml_setup, nodes, gpu_index):
        assert self.use_model_stat is not None
        gpu = self.cuda_device_list[gpu_index]
        gpu.nodes_allocated = set(nodes)
        if self.use_model_stat:
            gpu.model = copy.deepcopy(model)
            gpu.model = gpu.model.to(device=gpu.device)
            para = RuntimeParameters()
            para.phase = SimulationPhase.INITIALIZING
            optimizer, lr_scheduler = config_file.get_optimizer(None, gpu.model, para, config_ml_setup)
            gpu.optimizer = optimizer
            gpu.lr_scheduler = lr_scheduler


    @staticmethod
    def optimizer_to(optim, device):
        for param in optim.state.values():
            # Not sure are there any global tensors in the state dict
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

    @staticmethod
    def model_state_dict_to(stat_dict, device):
        for k, v in stat_dict.items():
            stat_dict[k] = v.to(device, non_blocking=True)

    def submit_training_job(self, training_node, criterion, training_data: torch.Tensor, training_label: torch.Tensor, use_amp=True):
        if training_node.is_using_model_stat:
            """use model stat (share model on gpu)"""
            gpu = training_node.allocated_gpu
            shared_model_on_gpu = gpu.model
            shared_optimizer_on_gpu = gpu.optimizer

            shared_model_on_gpu.load_state_dict(training_node.model_status)
            shared_optimizer_on_gpu.load_state_dict(training_node.optimizer_status)
            if training_node.lr_scheduler is not None:
                shared_lr_scheduler_on_gpu = gpu.lr_scheduler
                shared_lr_scheduler_on_gpu.load_state_dict(training_node.lr_scheduler_status)
            else:
                shared_lr_scheduler_on_gpu = None

            shared_model_on_gpu.train()

            data, labels = training_data.cuda(device=gpu.device), training_label.cuda(device=gpu.device)
            shared_optimizer_on_gpu.zero_grad(set_to_none=True)

            if use_amp:
                scaler = torch.cuda.amp.GradScaler()
                with torch.cuda.amp.autocast():
                    outputs = shared_model_on_gpu(data)
                    loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(shared_optimizer_on_gpu)
                    scaler.update()
            else:
                outputs = shared_model_on_gpu(data)
                loss = criterion(outputs, labels)
                loss.backward()
                shared_optimizer_on_gpu.step()

            if shared_lr_scheduler_on_gpu is not None:
                shared_lr_scheduler_on_gpu.step()

            stat_dict = shared_model_on_gpu.state_dict()
            CudaEnv.model_state_dict_to(stat_dict, torch.device('cpu'))
            training_node.set_model_stat(stat_dict)

            CudaEnv.optimizer_to(shared_optimizer_on_gpu, torch.device('cpu'))  # move optimizer data back to memory
            training_node.set_optimizer_stat(shared_optimizer_on_gpu.state_dict())

            if shared_lr_scheduler_on_gpu is not None:
                training_node.set_lr_scheduler_stat(shared_lr_scheduler_on_gpu.state_dict())
        else:
            """use dedicated model on gpu"""
            model = training_node.model
            gpu = training_node.allocated_gpu
            optimizer = training_node.optimizer
            data, labels = training_data.cuda(device=gpu.device), training_label.cuda(device=gpu.device)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler = torch.cuda.amp.GradScaler()
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                output = model(data)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        loss_val = float(loss.item())
        return loss_val
