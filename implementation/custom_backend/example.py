import os

import torch
import dummy_collectives

import torch.distributed as dist
import time
import math

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])

dist.init_process_group("cpu:gloo,cuda:dummy", rank=rank, world_size=world_size)

size = 1024**3//4  # size for Ring AR
size = math.ceil(size * 4/(1024 * 7)) * 1024 * 7 // 4  # size for StragglAR, comment out if using Ring AR

print(size)

x = torch.full((size,), rank, dtype=torch.float)

if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    y = x.cuda()
    print(f"Device ID of CUDA tensor: {y.get_device()} | size: {y.size()}")
    dist.all_reduce(y)
    print(f"cuda allreduce: {y}")

    # performance testing
    total_time = 0
    for itr in range(10):
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        t1.record()
        dist.all_reduce(y)
        t2.record()
        torch.cuda.synchronize()
        total_time = total_time + t1.elapsed_time(t2)

    print(f"Rank: {rank} | Avg latency: {total_time / 10}")