"""Cross-rank metric aggregation utilities.

Provides functions to collect and aggregate metrics across all ranks
in a distributed training run. Uses the existing torch.distributed
backend that FSDP/DDP/DeepSpeed already initialized.

Key functions:
- gather_scalar: Aggregate a single value across ranks
- gather_dict: Aggregate a dictionary of metrics
- all_gather_object: Collect arbitrary Python objects from all ranks
- broadcast_object: Send an object from rank 0 to all ranks

Example:
    from post_training_toolkit.models.distributed.aggregation import (
        gather_scalar,
        gather_dict,
    )
    
    # Aggregate step time
    local_time = 0.8  # seconds
    mean_time = gather_scalar(local_time, op="mean")  # e.g., 0.85
    max_time = gather_scalar(local_time, op="max")    # e.g., 1.2
    
    # Aggregate multiple metrics at once
    local_metrics = {"step_time": 0.8, "memory_mb": 15000}
    global_metrics = gather_dict(local_metrics)
    # Returns: {
    #     "step_time_mean": 0.85, "step_time_max": 1.2, "step_time_min": 0.75,
    #     "memory_mb_mean": 14500, "memory_mb_max": 16000, ...
    # }
"""

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
    """Safely import torch."""
    try:
        import torch
        return torch
    except ImportError:
        return None


def _get_torch_distributed():
    """Safely import torch.distributed."""
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
    """Gather a scalar value from all ranks and reduce.
    
    Args:
        value: Local scalar value to aggregate.
        op: Reduction operation - "mean", "sum", "max", or "min".
        dst: If specified, only this rank receives the result (others get input).
             If None, all ranks receive the result (all_reduce).
    
    Returns:
        Reduced value across all ranks.
        If not distributed, returns input unchanged.
    """
    if not is_distributed() or not is_initialized():
        return value
    
    torch = _get_torch()
    dist = _get_torch_distributed()
    
    if torch is None or dist is None:
        return value
    
    # Create tensor on the correct device
    device = f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    
    # Map op string to ReduceOp
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
    
    # Perform reduction
    if dst is not None:
        dist.reduce(tensor, dst=dst, op=reduce_op)
    else:
        dist.all_reduce(tensor, op=reduce_op)
    
    # For mean, divide by world size after sum
    if op == "mean":
        tensor = tensor / get_world_size()
    
    return tensor.item()


def reduce_tensor(
    tensor,  # torch.Tensor
    op: ReduceOp = "mean",
) -> "torch.Tensor":
    """Reduce a tensor across all ranks.
    
    Args:
        tensor: Local tensor to reduce. Must be on the correct device.
        op: Reduction operation.
    
    Returns:
        Reduced tensor. All ranks receive the result.
        If not distributed, returns input unchanged.
    """
    if not is_distributed() or not is_initialized():
        return tensor
    
    torch = _get_torch()
    dist = _get_torch_distributed()
    
    if torch is None or dist is None:
        return tensor
    
    # Clone to avoid modifying input
    result = tensor.clone()
    
    # Map op string to ReduceOp
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
    """Gather a dictionary of metrics from all ranks.
    
    For each metric, computes mean, max, and min across ranks.
    
    Args:
        metrics: Dictionary of metric_name -> value.
        ops: List of operations to compute. Defaults to ["mean", "max", "min"].
    
    Returns:
        Dictionary with aggregated metrics. For each input key "foo":
        - "foo_mean": mean across ranks
        - "foo_max": max across ranks
        - "foo_min": min across ranks
        - "foo_rank0": value from rank 0 (for reference)
        
        If not distributed, returns input with "_mean" suffix added.
    
    Example:
        >>> gather_dict({"step_time": 0.8, "memory_mb": 15000})
        {
            "step_time_mean": 0.85,
            "step_time_max": 1.2,
            "step_time_min": 0.75,
            "memory_mb_mean": 14500,
            ...
        }
    """
    if ops is None:
        ops = ["mean", "max", "min"]
    
    if not is_distributed() or not is_initialized():
        # Return input with requested op suffixes for API consistency
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
    """Gather arbitrary Python objects from all ranks.
    
    Unlike gather_scalar which only works with numbers, this can gather
    any picklable Python object (dicts, lists, dataclasses, etc.).
    
    Args:
        obj: Any picklable Python object.
    
    Returns:
        List of objects, one from each rank (ordered by rank).
        If not distributed, returns [obj].
    
    Example:
        >>> all_gather_object({"rank": get_rank(), "data": [1, 2, 3]})
        [
            {"rank": 0, "data": [1, 2, 3]},
            {"rank": 1, "data": [1, 2, 3]},
            ...
        ]
    """
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
    """Broadcast an object from source rank to all ranks.
    
    Args:
        obj: Object to broadcast (only matters on src rank).
        src: Source rank that has the object to broadcast.
    
    Returns:
        The object from src rank (all ranks receive the same object).
        If not distributed, returns input unchanged.
    
    Example:
        >>> # On rank 0: obj = {"config": "value"}
        >>> # On other ranks: obj = None
        >>> result = broadcast_object(obj, src=0)
        >>> # Now all ranks have: result = {"config": "value"}
    """
    if not is_distributed() or not is_initialized():
        return obj
    
    dist = _get_torch_distributed()
    if dist is None:
        return obj
    
    # Create a list to hold the object (broadcast_object_list modifies in place)
    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    
    return object_list[0]


def gather_per_rank(
    metrics: Dict[str, float],
) -> Dict[str, List[float]]:
    """Gather metrics and return per-rank values.
    
    Unlike gather_dict which returns aggregated stats, this returns
    the raw value from each rank.
    
    Args:
        metrics: Dictionary of metric_name -> value.
    
    Returns:
        Dictionary with per-rank values:
        {"step_time": [0.8, 0.9, 1.2, 0.85], ...}  # one per rank
        
        If not distributed, returns single-element lists.
    """
    if not is_distributed() or not is_initialized():
        return {key: [value] for key, value in metrics.items()}
    
    # Gather the whole dict from each rank
    all_metrics = all_gather_object(metrics)
    
    # Transpose: from list of dicts to dict of lists
    result = {}
    for key in metrics.keys():
        result[key] = [m.get(key, 0.0) for m in all_metrics]
    
    return result
