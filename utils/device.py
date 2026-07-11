import shutil
import subprocess

import torch


def _has_nvidia_gpu():
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and "GPU" in result.stdout
    except OSError:
        return False


def get_device(verbose=True):
    """
    Prefer NVIDIA CUDA when usable; otherwise fall back to CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            name = torch.cuda.get_device_name(0)
            print(f"Device: GPU ({name})")
        return device

    if verbose:
        if _has_nvidia_gpu():
            print(
                "Device: CPU "
                "(NVIDIA GPU detected, but current PyTorch cannot use CUDA; "
                "check driver / torch CUDA build compatibility)"
            )
        else:
            print("Device: CPU (no usable NVIDIA GPU)")

    return torch.device("cpu")
