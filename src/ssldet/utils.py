from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return DistributedContext(rank, local_rank, world_size, device)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int = 0) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def cosine_momentum(
    step: int,
    total_steps: int,
    start: float,
    end: float,
) -> float:
    progress = min(1.0, step / max(1, total_steps - 1))
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0


def reduce_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())

