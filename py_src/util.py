from pathlib import Path


def basename_without_extension(name: str) -> str:
    return Path(name).stem
