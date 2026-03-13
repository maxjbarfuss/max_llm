"""Distributed training utilities for multi-GPU support.

Provides utilities for PyTorch Distributed Data Parallel (DDP) and
Fully Sharded Data Parallel (FSDP) training:
- Initialization and cleanup
- Rank/world_size management
- Barrier synchronization
- Distributed samplers
- DDP and FSDP model wrapping
"""

import os
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: str = "nccl") -> dict[str, Any]:
    """Initialize distributed training.

    Expects environment variables:
    - MASTER_ADDR: Address of rank 0 node (default: localhost)
    - MASTER_PORT: Port for communication (default: 29500)
    - RANK: Rank of this process
    - WORLD_SIZE: Total number of processes

    Or automatically detects from CUDA_VISIBLE_DEVICES and spawns processes.

    Args:
        backend: Distributed backend (nccl for GPU, gloo for CPU). Default: nccl.

    Returns:
        Dict with keys: rank, world_size, local_rank, device
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    # Check if environment is already set up (e.g., via torchrun)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # Single-node multi-GPU: auto-detect from CUDA_VISIBLE_DEVICES
        if torch.cuda.is_available():
            world_size = torch.cuda.device_count()
            if world_size > 1:
                rank = int(os.environ.get("RANK", 0))
                local_rank = rank % world_size
            else:
                raise RuntimeError(
                    "Distributed training requested but only 1 GPU available. "
                    "Set CUDA_VISIBLE_DEVICES to enable multiple GPUs."
                )
        else:
            raise RuntimeError("Distributed training requested but CUDA not available")

    # Set default master address/port if not set
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    # Initialize process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # Set device for this process
    if backend == "nccl":
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
    }


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronize all processes; no-op if not distributed."""
    if is_distributed():
        dist.barrier()  # type: ignore[misc, unused-ignore]


def wrap_model_ddp(
    model: torch.nn.Module,
    device_ids: list[int] | None = None,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """Wrap a model with DistributedDataParallel.

    Args:
        model: Model to wrap (should already be on correct device).
        device_ids: List of device IDs (default: [local_rank]).
        find_unused_parameters: Whether to find unused parameters (default: False).

    Returns:
        DDP-wrapped model if distributed, original model otherwise.
    """
    if not is_distributed():
        return model

    if device_ids is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_ids = [local_rank]

    return DDP(
        model,
        device_ids=device_ids,
        output_device=device_ids[0],
        find_unused_parameters=find_unused_parameters,
    )


def create_distributed_sampler(
    dataset: Any,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = False,
) -> DistributedSampler[Any] | None:
    """Create a DistributedSampler for the dataset.

    Args:
        dataset: Dataset to sample from.
        shuffle: Whether to shuffle the dataset.
        seed: Random seed for shuffling.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        DistributedSampler if distributed, None otherwise.
    """
    if not is_distributed():
        return None

    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


def print_once(*args: Any, **kwargs: Any) -> None:
    if is_main_process():
        print(*args, **kwargs)


def reduce_dict(data: dict[str, float]) -> dict[str, float]:
    """Average values in a dict across all processes.

    Args:
        data: Dict with scalar float values.

    Returns:
        Dict with averaged values if distributed, original dict otherwise.
    """
    if not is_distributed():
        return data

    world_size = get_world_size()
    reduced: dict[str, float] = {}

    for key, value in data.items():
        tensor = torch.tensor(value, device=torch.cuda.current_device())
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)  # type: ignore[misc, unused-ignore]
        reduced[key] = (tensor / world_size).item()

    return reduced


def wrap_model_fsdp(
    model: torch.nn.Module,
    transformer_layer_cls: type,
    sharding_strategy: str = "full_shard",
) -> torch.nn.Module:
    """Wrap a model with FullyShardedDataParallel.

    Parameters are wrapped at the TransformerBlock granularity so FSDP can
    overlap communication with the per-layer forward/backward passes.

    Args:
        model: Model to wrap (should already be on the correct device).
        transformer_layer_cls: The class used for individual transformer blocks
            (used by the auto-wrap policy to decide wrapping boundaries).
        sharding_strategy: One of "full_shard" (ZeRO-3, maximum savings) or
            "shard_grad_op" (ZeRO-2, grad+optimizer only, less communication).
            Default: "full_shard".

    Returns:
        FSDP-wrapped model if distributed, original model otherwise.
    """
    if not is_distributed():
        return model

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    _strategies = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    }
    strategy = _strategies.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

    wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_cls},
    )
    return FSDP(
        model,
        sharding_strategy=strategy,
        auto_wrap_policy=wrap_policy,
        device_id=torch.cuda.current_device(),
    )


def collect_fsdp_state_dicts(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Gather full (un-sharded) state dicts from an FSDP-wrapped model.

    This is a **collective** — every rank must call it simultaneously.
    With rank0_only=True, only rank 0 receives non-empty dicts; all other
    ranks return ({}, {}).  Call barrier() before saving if needed.

    Args:
        model: FSDP-wrapped model.
        optimizer: Optimizer whose state should also be gathered.

    Returns:
        (model_state_dict, optimizer_state_dict) — full on rank 0, empty elsewhere.
    """
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        model_sd = model.state_dict()
        opt_sd = FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)
    return model_sd, opt_sd
