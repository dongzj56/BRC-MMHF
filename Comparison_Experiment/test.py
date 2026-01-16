#!/usr/bin/env python3
# pip install nvidia-ml-py3 torch
import os
import torch
import pynvml

def main():
    # 1. 初始化 NVML
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()
    print(f"Found {gpu_count} GPU(s)\n")

    # 2. 遍历每张卡
    for i in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle).decode()
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        print(f"GPU {i} : {name}")
        print(f"  Total memory : {mem_info.total / 1024**3:.2f} GB")
        print(f"  Used  memory : {mem_info.used  / 1024**3:.2f} GB")
        print(f"  Free  memory : {mem_info.free  / 1024**3:.2f} GB")

        # 3. PyTorch 视角（该卡）
        if torch.cuda.is_available():
            try:
                # 把当前程序视角切到这张卡
                with torch.cuda.device(i):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved  = torch.cuda.memory_reserved(i)  / 1024**3
                    free_torch = (mem_info.total - mem_info.used) / 1024**3
                    print(f"  PyTorch allocated : {allocated:.3f} GB")
                    print(f"  PyTorch reserved  : {reserved:.3f}  GB")
                    print(f"  PyTorch max free  : {free_torch:.3f} GB")
            except RuntimeError as e:
                print(f"  PyTorch query fail: {e}")
        print("-" * 50)

    pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()