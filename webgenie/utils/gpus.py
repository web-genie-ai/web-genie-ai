import torch

def get_gpu_info():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        cached_memory = torch.cuda.memory_reserved(0)
        total_memory_mb = total_memory / (1024 ** 2)
        allocated_memory_mb = allocated_memory / (1024 ** 2)
        cached_memory_mb = cached_memory / (1024 ** 2)
        return total_memory_mb, allocated_memory_mb, cached_memory_mb
    else:
        return None, None, None