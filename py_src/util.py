import torch
import logging
import sys
from pathlib import Path
from collections import deque
import numpy as np

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

def get_layer_info(state_dict, layer_name: str = None):
    layer_info = {}
    for name, tensor in state_dict.items():
        if layer_name is not None and layer_name not in name:
            continue
        if isinstance(tensor, torch.Tensor):

            # Convert to numpy for easier computation
            tensor_np = tensor.detach().cpu().numpy()
            flattened_tensor = tensor_np.flatten()

            # Calculate various variance statistics
            stats = {
                'layer_name': name,
                'original_shape': list(tensor.shape),
                'total_parameters': tensor.numel(),
                'variance': float(np.var(flattened_tensor)),
                'std_deviation': float(np.std(flattened_tensor)),
                'mean': float(np.mean(flattened_tensor)),
                'min_value': float(np.min(flattened_tensor)),
                'max_value': float(np.max(flattened_tensor))
            }
            layer_info[name] = stats
    return layer_info


def prompt_selection(options, prompt_message="Please make a selection:", allow_quit=True):
    """
    Display a list of options and prompt user to make one selection.

    Args:
        options (list): List of strings to choose from
        prompt_message (str): Custom prompt message
        allow_quit (bool): Whether to allow 'q' to quit

    Returns:
        str: Selected option, or None if user quits
    """
    if not options:
        print("No options provided.")
        return None

    while True:
        print(f"\n{prompt_message}")
        print("-" * len(prompt_message))

        # Display numbered options
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        if allow_quit:
            print("q. Quit")

        # Get user input
        choice = input(f"\nEnter your choice (1-{len(options)}" + ("or 'q'" if allow_quit else "") + "): ").strip().lower()

        # Handle quit
        if allow_quit and choice in ['q', 'quit']:
            return None

        # Handle numeric selection
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(options):
                selected = options[choice_num - 1]
                print(f"\nYou selected: {selected}")
                return selected
            else:
                print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")


def geodesic_distance(a: torch.Tensor, b: torch.Tensor) -> None | torch.Tensor:
    """
    Compute the geodesic (spherical) distance between two multi-dimensional points a and b
    on a sphere centered at the origin. Skips computation if dtype is not floating point.

    Args:
        a (torch.Tensor): Tensor of any shape and type representing the first point.
        b (torch.Tensor): Tensor of the same shape and type representing the second point.

    Returns:
        torch.Tensor: Scalar tensor representing the geodesic distance.
                      Returns torch.tensor(0.0) if types are not float.
    """
    # Skip if types are not float
    if not a.dtype.is_floating_point or not b.dtype.is_floating_point:
        return None

    # Flatten the tensors
    a_flat = a.flatten()
    b_flat = b.flatten()

    # Compute norms
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)

    if norm_a == 0 or norm_b == 0:
        return torch.tensor(0.0, dtype=torch.float32)  # Avoid division by zero

    # Use average radius
    r = (norm_a + norm_b) / 2

    # Compute cosine of angle
    cos_theta = torch.dot(a_flat, b_flat) / (norm_a * norm_b)

    # Clamp cosine to avoid numerical issues
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # Compute angle and geodesic distance
    theta = torch.acos(cos_theta)
    distance = r * theta

    return distance


