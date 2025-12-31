"""Rank and distributed environment utilities.

Provides a backend-agnostic way to get rank information that works with:
- PyTorch DDP (torch.distributed)
- PyTorch FSDP (uses torch.distributed)
- DeepSpeed (has own comm, but also sets env vars)
- Hugging Face Accelerate (wraps torch.distributed)
- Single-process training (graceful fallback)

Detection priority:
1. torch.distributed (if initialized)
2. Environment variables (RANK, WORLD_SIZE, LOCAL_RANK)
3. DeepSpeed-specific env vars
4. Fallback to single-process defaults

Example:
    from post_training_toolkit.models.distributed.rank import (
        get_rank,
        is_main_process,
        barrier,
    )
    
    # Only rank 0 saves checkpoints
    if is_main_process():
        save_checkpoint()
    
    # Synchronize all ranks before evaluation
    barrier()
    run_evaluation()
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _get_torch_distributed():
    """Safely import torch.distributed."""
    try:
        import torch.distributed as dist
        return dist
    except ImportError:
        return None


def _get_deepspeed_comm():
    """Safely import deepspeed.comm."""
    try:
        import deepspeed.comm as ds_comm
        return ds_comm
    except ImportError:
        return None


def is_initialized() -> bool:
    """Check if distributed training is initialized.
    
    Returns:
        True if torch.distributed is initialized, False otherwise.
    """
    dist = _get_torch_distributed()
    if dist is not None and dist.is_initialized():
        return True
    return False


def is_distributed() -> bool:
    """Check if running in distributed mode (world_size > 1).
    
    Returns:
        True if world_size > 1 (either via torch.distributed or env vars).
    """
    return get_world_size() > 1


def get_rank() -> int:
    """Get global rank of current process.
    
    Returns:
        Global rank (0 to world_size-1), or 0 if not distributed.
    """
    # Priority 1: torch.distributed
    dist = _get_torch_distributed()
    if dist is not None and dist.is_initialized():
        return dist.get_rank()
    
    # Priority 2: Standard env vars (set by torchrun, accelerate, etc.)
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    
    # Priority 3: DeepSpeed-specific
    if "DEEPSPEED_RANK" in os.environ:
        return int(os.environ["DEEPSPEED_RANK"])
    
    # Fallback: single process
    return 0


def get_local_rank() -> int:
    """Get local rank (rank within this node/machine).
    
    Returns:
        Local rank (0 to num_gpus_per_node-1), or 0 if not distributed.
    """
    dist = _get_torch_distributed()
    
    # torch.distributed doesn't directly expose local_rank, use env vars
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    
    # DeepSpeed-specific
    if "DEEPSPEED_LOCAL_RANK" in os.environ:
        return int(os.environ["DEEPSPEED_LOCAL_RANK"])
    
    # Fallback: assume local_rank == global_rank (single node)
    # This is correct for single-node multi-GPU
    if dist is not None and dist.is_initialized():
        # Best effort: use global rank mod GPUs per node
        # This works for most single-node setups
        try:
            import torch
            if torch.cuda.is_available():
                return dist.get_rank() % torch.cuda.device_count()
        except ImportError:
            pass
        return dist.get_rank()
    
    return 0


def get_world_size() -> int:
    """Get total number of processes in distributed group.
    
    Returns:
        World size, or 1 if not distributed.
    """
    # Priority 1: torch.distributed
    dist = _get_torch_distributed()
    if dist is not None and dist.is_initialized():
        return dist.get_world_size()
    
    # Priority 2: Standard env vars
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    
    # Priority 3: DeepSpeed-specific
    if "DEEPSPEED_WORLD_SIZE" in os.environ:
        return int(os.environ["DEEPSPEED_WORLD_SIZE"])
    
    # Fallback: single process
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0).
    
    Use this to guard operations that should only happen once:
    - Saving checkpoints
    - Logging to files
    - Printing progress
    
    Returns:
        True if rank == 0.
    """
    return get_rank() == 0


def get_backend() -> Optional[str]:
    """Get the distributed backend name.
    
    Returns:
        Backend name ("nccl", "gloo", "mpi"), or None if not distributed.
    """
    dist = _get_torch_distributed()
    if dist is not None and dist.is_initialized():
        return dist.get_backend()
    return None


def get_device() -> str:
    """Get the device string for this rank.
    
    Returns:
        Device string like "cuda:0", "cuda:3", or "cpu".
    """
    try:
        import torch
        if torch.cuda.is_available():
            local_rank = get_local_rank()
            return f"cuda:{local_rank}"
    except ImportError:
        pass
    return "cpu"


def barrier(timeout_sec: Optional[float] = None) -> None:
    """Synchronize all processes.
    
    Blocks until all ranks reach this point. Use before operations
    that require all ranks to be synchronized (e.g., evaluation).
    
    Args:
        timeout_sec: Optional timeout in seconds. If None, waits indefinitely.
                     Only supported when torch.distributed is initialized.
    """
    dist = _get_torch_distributed()
    if dist is not None and dist.is_initialized():
        if timeout_sec is not None:
            try:
                import datetime
                dist.barrier(timeout=datetime.timedelta(seconds=timeout_sec))
            except TypeError:
                # Older PyTorch versions don't support timeout
                dist.barrier()
        else:
            dist.barrier()
    # If not distributed, barrier is a no-op


@dataclass
class DistributedInfo:
    """Snapshot of distributed environment information."""
    
    is_distributed: bool
    is_initialized: bool
    rank: int
    local_rank: int
    world_size: int
    backend: Optional[str]
    device: str
    
    # Node information
    num_nodes: Optional[int] = None
    node_rank: Optional[int] = None
    gpus_per_node: Optional[int] = None
    
    def __str__(self) -> str:
        if not self.is_distributed:
            return "DistributedInfo(single process)"
        
        lines = [
            f"DistributedInfo(",
            f"  rank={self.rank}/{self.world_size-1},",
            f"  local_rank={self.local_rank},",
            f"  device={self.device},",
            f"  backend={self.backend},",
        ]
        if self.num_nodes is not None:
            lines.append(f"  nodes={self.num_nodes}, gpus_per_node={self.gpus_per_node},")
        lines.append(")")
        return "\n".join(lines)


def get_distributed_info() -> DistributedInfo:
    """Get comprehensive distributed environment information.
    
    Returns:
        DistributedInfo dataclass with all distributed details.
    """
    world_size = get_world_size()
    local_rank = get_local_rank()
    
    # Try to determine node configuration
    num_nodes = None
    node_rank = None
    gpus_per_node = None
    
    # Check env vars for node info (set by SLURM, torchrun, etc.)
    if "NNODES" in os.environ or "SLURM_NNODES" in os.environ:
        num_nodes = int(os.environ.get("NNODES", os.environ.get("SLURM_NNODES", 1)))
    if "NODE_RANK" in os.environ or "SLURM_NODEID" in os.environ:
        node_rank = int(os.environ.get("NODE_RANK", os.environ.get("SLURM_NODEID", 0)))
    
    # Infer GPUs per node
    if num_nodes is not None and world_size > 1:
        gpus_per_node = world_size // num_nodes
    else:
        try:
            import torch
            if torch.cuda.is_available():
                gpus_per_node = torch.cuda.device_count()
                if world_size > 1 and gpus_per_node > 0:
                    num_nodes = world_size // gpus_per_node
                    node_rank = get_rank() // gpus_per_node
        except ImportError:
            pass
    
    return DistributedInfo(
        is_distributed=is_distributed(),
        is_initialized=is_initialized(),
        rank=get_rank(),
        local_rank=local_rank,
        world_size=world_size,
        backend=get_backend(),
        device=get_device(),
        num_nodes=num_nodes,
        node_rank=node_rank,
        gpus_per_node=gpus_per_node,
    )
