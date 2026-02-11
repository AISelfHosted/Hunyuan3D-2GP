import os
import time
import psutil
try:
    import torch
except ImportError:
    torch = None

_start_time = time.time()

def get_system_metrics():
    """Get system metrics including uptime, memory, and GPU usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    metrics = {
        "uptime_seconds": round(time.time() - _start_time, 1),
        "process": {
            "pid": os.getpid(),
            "rss_mb": round(mem_info.rss / 1024**2, 1),
            "vms_mb": round(mem_info.vms / 1024**2, 1),
            "threads": process.num_threads(),
            "cpu_percent": process.cpu_percent(),
        },
        "gpu": {},
    }
    
    if torch and torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            metrics["gpu"] = {
                "name": torch.cuda.get_device_name(device),
                "memory_allocated_mb": round(torch.cuda.memory_allocated(device) / 1024**2, 1),
                "memory_reserved_mb": round(torch.cuda.memory_reserved(device) / 1024**2, 1),
                "memory_total_mb": round(props.total_memory / 1024**2, 1),
            }
        except Exception:
            pass
            
    return metrics
