
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from post_training_toolkit.models.distributed.rank import (
    get_rank,
    get_world_size,
    get_local_rank,
    is_distributed,
    is_initialized,
)

def _get_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None

def _get_torch_distributed():
    try:
        import torch.distributed as dist
        return dist
    except ImportError:
        return None

ReduceOp = Literal["mean", "sum", "max", "min"]

def gather_scalar(
    value: float,
    op: ReduceOp = "mean",
    dst: Optional[int] = None,
) -> float:
    if not is_distributed() or not is_initialized():
        return value
    
    torch = _get_torch()
    dist = _get_torch_distributed()
    
    if torch is None or dist is None:
        return value
    
    device = f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    
    if op == "sum":
        reduce_op = dist.ReduceOp.SUM
    elif op == "max":
        reduce_op = dist.ReduceOp.MAX
    elif op == "min":
        reduce_op = dist.ReduceOp.MIN
    elif op == "mean":
        reduce_op = dist.ReduceOp.SUM
    else:
        raise ValueError(f"Unknown op: {op}. Expected 'mean', 'sum', 'max', or 'min'.")
    
    if dst is not None:
        dist.reduce(tensor, dst=dst, op=reduce_op)
    else:
        dist.all_reduce(tensor, op=reduce_op)
    
    if op == "mean":
        tensor = tensor / get_world_size()
    
    return tensor.item()

def reduce_tensor(
    tensor,
    op: ReduceOp = "mean",
) -> "torch.Tensor":
    if not is_distributed() or not is_initialized():
        return tensor
    
    torch = _get_torch()
    dist = _get_torch_distributed()
    
    if torch is None or dist is None:
        return tensor
    
    result = tensor.clone()
    
    if op == "sum":
        reduce_op = dist.ReduceOp.SUM
    elif op == "max":
        reduce_op = dist.ReduceOp.MAX
    elif op == "min":
        reduce_op = dist.ReduceOp.MIN
    elif op == "mean":
        reduce_op = dist.ReduceOp.SUM
    else:
        raise ValueError(f"Unknown op: {op}")
    
    dist.all_reduce(result, op=reduce_op)
    
    if op == "mean":
        result = result / get_world_size()
    
    return result

def gather_dict(
    metrics: Dict[str, float],
    ops: Optional[List[ReduceOp]] = None,
) -> Dict[str, float]:
    if ops is None:
        ops = ["mean", "max", "min"]
    
    if not is_distributed() or not is_initialized():
        result = {}
        for key, value in metrics.items():
            for op in ops:
                result[f"{key}_{op}"] = value
        return result
    
    result = {}
    
    for key, value in metrics.items():
        for op in ops:
            aggregated = gather_scalar(value, op=op)
            result[f"{key}_{op}"] = aggregated
    
    return result

def all_gather_object(obj: Any) -> List[Any]:
    if not is_distributed() or not is_initialized():
        return [obj]
    
    dist = _get_torch_distributed()
    if dist is None:
        return [obj]
    
    world_size = get_world_size()
    output = [None] * world_size
    
    dist.all_gather_object(output, obj)
    
    return output

def broadcast_object(obj: Any, src: int = 0) -> Any:
    if not is_distributed() or not is_initialized():
        return obj
    
    dist = _get_torch_distributed()
    if dist is None:
        return obj
    
    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    
    return object_list[0]

def gather_per_rank(
    metrics: Dict[str, float],
) -> Dict[str, List[float]]:
    if not is_distributed() or not is_initialized():
        return {key: [value] for key, value in metrics.items()}
    
    all_metrics = all_gather_object(metrics)
    
    result = {}
    for key in metrics.keys():
        result[key] = [m.get(key, 0.0) for m in all_metrics]
    
    return result
