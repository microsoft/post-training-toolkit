
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

def _get_torch_distributed():
    try:
        import torch.distributed as dist
        return dist
    except ImportError:
        return None

def _get_deepspeed_comm():
    try:
        import deepspeed.comm as ds_comm
        return ds_comm
    except ImportError:
        return None

def is_initialized() -> bool:
    dist = _get_torch_distributed()
    if dist is not None and dist.is_initialized():
        return True
    return False

def is_distributed() -> bool:
    return get_world_size() > 1

def get_rank() -> int:
    dist = _get_torch_distributed()
    if dist is not None and dist.is_initialized():
        return dist.get_rank()
    
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    
    if "DEEPSPEED_RANK" in os.environ:
        return int(os.environ["DEEPSPEED_RANK"])
    
    return 0

def get_local_rank() -> int:
    dist = _get_torch_distributed()
    
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    
    if "DEEPSPEED_LOCAL_RANK" in os.environ:
        return int(os.environ["DEEPSPEED_LOCAL_RANK"])
    
    if dist is not None and dist.is_initialized():
        try:
            import torch
            if torch.cuda.is_available():
                return dist.get_rank() % torch.cuda.device_count()
        except ImportError:
            pass
        return dist.get_rank()
    
    return 0

def get_world_size() -> int:
    dist = _get_torch_distributed()
    if dist is not None and dist.is_initialized():
        return dist.get_world_size()
    
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    
    if "DEEPSPEED_WORLD_SIZE" in os.environ:
        return int(os.environ["DEEPSPEED_WORLD_SIZE"])
    
    return 1

def is_main_process() -> bool:
    return get_rank() == 0

def get_backend() -> Optional[str]:
    dist = _get_torch_distributed()
    if dist is not None and dist.is_initialized():
        return dist.get_backend()
    return None

def get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            local_rank = get_local_rank()
            return f"cuda:{local_rank}"
    except ImportError:
        pass
    return "cpu"

def barrier(timeout_sec: Optional[float] = None) -> None:
    dist = _get_torch_distributed()
    if dist is not None and dist.is_initialized():
        if timeout_sec is not None:
            try:
                import datetime
                dist.barrier(timeout=datetime.timedelta(seconds=timeout_sec))
            except TypeError:
                dist.barrier()
        else:
            dist.barrier()

@dataclass
class DistributedInfo:
    
    is_distributed: bool
    is_initialized: bool
    rank: int
    local_rank: int
    world_size: int
    backend: Optional[str]
    device: str
    
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
    world_size = get_world_size()
    local_rank = get_local_rank()
    
    num_nodes = None
    node_rank = None
    gpus_per_node = None
    
    if "NNODES" in os.environ or "SLURM_NNODES" in os.environ:
        num_nodes = int(os.environ.get("NNODES", os.environ.get("SLURM_NNODES", 1)))
    if "NODE_RANK" in os.environ or "SLURM_NODEID" in os.environ:
        node_rank = int(os.environ.get("NODE_RANK", os.environ.get("SLURM_NODEID", 0)))
    
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
