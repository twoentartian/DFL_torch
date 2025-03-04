import torch
import logging
import sys
from pathlib import Path
from collections import deque

def basename_without_extension(name: str) -> str:
    return Path(name).stem


def check_for_nans_in_state_dict(state_dict):
    for key, tensor in state_dict.items():
        if torch.isnan(tensor).any():
            raise ValueError(f"find nan value in {key}")


class MovingAverage:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.sum = 0.0

    def add(self, value):
        if len(self.window) == self.window_size:
            self.sum -= self.window[0]
        self.window.append(value)
        self.sum += value
        return self.get_average()

    def get_average(self):
        if not self.window:
            return 0
        return self.sum / len(self.window)


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

class MovingMax:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.window = deque()
        self.max_deque = deque()

    def add(self, value):
        self.window.append(value)
        while self.max_deque and self.max_deque[-1] < value:
            self.max_deque.pop()
        self.max_deque.append(value)
        if len(self.window) > self.window_size:
            oldest = self.window.popleft()
            if oldest == self.max_deque[0]:
                self.max_deque.popleft()
        return self.get_max()

    def get_max(self):
        if not self.max_deque:
            return None
        return self.max_deque[0]

def expand_int_args(input_int_str: str):
    result = []
    parts = input_int_str.split(',')

    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))

    return result

def load_model_state_file(path: str):
    cpu_device = torch.device("cpu")
    raw_model_state = torch.load(path, map_location=cpu_device)
    if raw_model_state["state_dict"] is not None:
        # this is a model state with model name
        model_state = raw_model_state["state_dict"]
        model_name = raw_model_state["model_name"]
    else:
        model_state = raw_model_state["state_dict"]
        model_name = None
    return model_state, model_name

def load_optimizer_state_file(path: str):
    cpu_device = torch.device("cpu")
    raw_optimizer_state = torch.load(path, map_location=cpu_device)
    if raw_optimizer_state["state_dict"] is not None:
        # this is a model state with model name
        optimizer_state = raw_optimizer_state["state_dict"]
        model_name = raw_optimizer_state["model_name"]
    else:
        optimizer_state = raw_optimizer_state["state_dict"]
        model_name = None
    return optimizer_state, model_name

def save_model_state(path, model_state, model_name=None):
    info = {}
    info["state_dict"] = model_state
    info["model_name"] = model_name
    torch.save(info, path)

def save_optimizer_state(path, optimizer_state, model_name=None):
    info = {}
    info["state_dict"] = optimizer_state
    info["model_name"] = model_name
    torch.save(info, path)

def assert_if_both_not_none(a, b):
    if a is not None and b is not None:
        assert a == b

def set_logging(target_logger, task_name, log_file_path=None):
    class ExitOnExceptionHandler(logging.StreamHandler):
        def emit(self, record):
            if record.levelno == logging.CRITICAL:
                raise SystemExit(-1)

    formatter = logging.Formatter(f"[%(asctime)s] [%(levelname)8s] [{task_name}] --- %(message)s (%(filename)s:%(lineno)s)")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    target_logger.setLevel(logging.DEBUG)
    target_logger.addHandler(console)
    target_logger.addHandler(ExitOnExceptionHandler())

    if log_file_path is not None:
        file = logging.FileHandler(log_file_path)
        file.setLevel(logging.DEBUG)
        file.setFormatter(formatter)
        target_logger.addHandler(file)

    del console, formatter