import torch
import sys
import platform
import subprocess
import os

def check_cuda_toolkit():
    try:
        nvcc_output = subprocess.check_output(['nvcc', '--version']).decode()
        return nvcc_output.split('release ')[-1].split(',')[0]
    except:
        return "Not installed"

def check_environment():
    info = {
        "Python Version": sys.version,
        "PyTorch Version": torch.__version__,
        "CUDA Available": torch.cuda.is_available(),
        "PyTorch CUDA Version": torch.version.cuda if torch.cuda.is_available() else "Not available",
        "CUDA Toolkit Version": check_cuda_toolkit(),
        "GPU Device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available",
        "Platform": platform.platform(),
    }
    
    try:
        import torch_tensorrt
        info["TensorRT Version"] = torch_tensorrt.__version__
    except ImportError:
        info["TensorRT Version"] = "Not installed"
    
    print("\nEnvironment Information:")
    print("------------------------")
    for k, v in info.items():
        print(f"{k}: {v}")
    
    if info["CUDA Toolkit Version"] == "Not installed":
        print("\nWarning: CUDA Toolkit not found. Required for TensorRT optimization and profiling.")
        print("Install from: https://developer.nvidia.com/cuda-downloads")

if __name__ == "__main__":
    check_environment() 