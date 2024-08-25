import torch
from pathlib import Path


def basename_without_extension(name: str) -> str:
    return Path(name).stem


def check_for_nans_in_state_dict(state_dict):
    for key, tensor in state_dict.items():
        if torch.isnan(tensor).any():
            raise ValueError(f"find nan value in {key}")
