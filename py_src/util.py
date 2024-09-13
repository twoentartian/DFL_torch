import torch
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
