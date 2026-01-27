from __future__ import annotations

import atexit
import os
import signal
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from post_training_toolkit.models.artifacts import (
    RunArtifactManager,
    Postmortem,
    get_environment_info,
)

class ExitReason:
    COMPLETED = "completed"
    OOM = "oom"
    SIGTERM = "sigterm"
    SIGINT = "sigint"
    NAN = "nan"
    DIVERGENCE = "divergence"
    EXCEPTION = "exception"
    UNKNOWN = "unknown"

class RingBuffer:
    
    def __init__(self, maxlen: int = 50):
        self.maxlen = maxlen
        self._items: List[Any] = []
    
    def append(self, item: Any) -> None:
        self._items.append(item)
        if len(self._items) > self.maxlen:
            self._items.pop(0)
    
    def get_all(self) -> List[Any]:
        return self._items.copy()
    
    def clear(self) -> None:
        self._items.clear()

class PostmortemRecorder:
    
    def __init__(
        self,
        artifact_manager: RunArtifactManager,
        max_recent_events: int = 50,
        max_recent_metrics: int = 20,
    ):
        self.artifact_manager = artifact_manager
        self._recent_events = RingBuffer(max_recent_events)
        self._recent_metrics = RingBuffer(max_recent_metrics)
        self._last_step = 0
        self._installed = False
        self._finalized = False
        self._exit_reason: Optional[str] = None
        self._exception_info: Optional[tuple] = None
        self._original_excepthook = None
        self._original_sigterm = None
        self._original_sigint = None
        self._checkpoint_callback: Optional[Callable[[], None]] = None
    
    def install(self) -> None:
        if self._installed or not self.artifact_manager.is_main_process:
            return
        
        self._original_excepthook = sys.excepthook
        sys.excepthook = self._exception_handler
        
        try:
            self._original_sigterm = signal.signal(signal.SIGTERM, self._sigterm_handler)
        except (ValueError, OSError):
            pass
        
        try:
            self._original_sigint = signal.signal(signal.SIGINT, self._sigint_handler)
        except (ValueError, OSError):
            pass
        
        atexit.register(self._atexit_handler)
        
        self._installed = True
    
    def set_checkpoint_callback(self, callback: Callable[[], None]) -> None:
        self._checkpoint_callback = callback
    
    def _save_preemption_checkpoint(self, reason: str) -> bool:
        if self._checkpoint_callback is None:
            return False
        
        try:
            self.record_event("preemption_checkpoint_start", {
                "reason": reason,
                "step": self._last_step,
            })
            self._checkpoint_callback()
            self.record_event("preemption_checkpoint_saved", {
                "reason": reason,
                "step": self._last_step,
            })
            return True
        except Exception as e:
            self.record_event("preemption_checkpoint_failed", {
                "reason": reason,
                "step": self._last_step,
                "error": str(e),
            })
            return False
    
    def uninstall(self) -> None:
        if not self._installed:
            return
        
        if self._original_excepthook is not None:
            sys.excepthook = self._original_excepthook
        
        if self._original_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, self._original_sigterm)
            except (ValueError, OSError):
                pass
        
        if self._original_sigint is not None:
            try:
                signal.signal(signal.SIGINT, self._original_sigint)
            except (ValueError, OSError):
                pass
        
        self._installed = False
    
    def record_step(self, step: int) -> None:
        self._last_step = step
    
    def record_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        self._recent_metrics.append({
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        })
        self._last_step = max(self._last_step, step)
    
    def record_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        self._recent_events.append({
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": self._last_step,
            "data": data or {},
        })
    
    def check_for_nan(self, metrics: Dict[str, Any]) -> bool:
        import math
        
        nan_keys = []
        for key, value in metrics.items():
            if isinstance(value, float) and math.isnan(value):
                nan_keys.append(key)
        
        if nan_keys:
            self.record_event("nan_detected", {"keys": nan_keys})
            self._exit_reason = ExitReason.NAN
            return True
        return False
    
    def check_for_divergence(
        self,
        metrics: Dict[str, Any],
        loss_threshold: float = 100.0,
        kl_threshold: float = 50.0,
    ) -> bool:
        diverged = False
        reasons = []
        
        for loss_key in ["loss", "dpo_loss", "ppo_loss", "sft_loss"]:
            if loss_key in metrics:
                val = metrics[loss_key]
                if isinstance(val, (int, float)) and val > loss_threshold:
                    reasons.append(f"{loss_key}={val}")
                    diverged = True
        
        if "kl" in metrics:
            kl = metrics["kl"]
            if isinstance(kl, (int, float)) and kl > kl_threshold:
                reasons.append(f"kl={kl}")
                diverged = True
        
        if diverged:
            self.record_event("divergence_detected", {"reasons": reasons})
            self._exit_reason = ExitReason.DIVERGENCE
        
        return diverged
    
    def _exception_handler(
        self,
        exc_type: type,
        exc_value: BaseException,
        exc_tb,
    ) -> None:
        self._exception_info = (exc_type, exc_value, exc_tb)
        
        if self._is_oom_error(exc_type, exc_value):
            self._exit_reason = ExitReason.OOM
        else:
            self._exit_reason = ExitReason.EXCEPTION
        
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        self.record_event("exception", {
            "type": exc_type.__name__,
            "message": str(exc_value),
            "traceback": tb_str[:5000],
        })
        
        self._write_postmortem()
        
        if self._original_excepthook:
            self._original_excepthook(exc_type, exc_value, exc_tb)
    
    def _sigterm_handler(self, signum: int, frame) -> None:
        self._exit_reason = ExitReason.SIGTERM
        self.record_event("sigterm_received", {"signal": signum, "step": self._last_step})
        
        self._save_preemption_checkpoint("sigterm")
        
        self._write_postmortem()
        
        if self._original_sigterm and callable(self._original_sigterm):
            self._original_sigterm(signum, frame)
        else:
            sys.exit(128 + signum)
    
    def _sigint_handler(self, signum: int, frame) -> None:
        self._exit_reason = ExitReason.SIGINT
        self.record_event("sigint_received", {"signal": signum, "step": self._last_step})
        
        self._save_preemption_checkpoint("sigint")
        
        self._write_postmortem()
        
        if self._original_sigint and callable(self._original_sigint):
            self._original_sigint(signum, frame)
        else:
            raise KeyboardInterrupt
    
    def _atexit_handler(self) -> None:
        if self._finalized:
            return
        
        if self._exit_reason is None:
            self._exit_reason = ExitReason.UNKNOWN
            self.record_event("unexpected_exit", {})
        
        self._write_postmortem()
    
    def _is_oom_error(self, exc_type: type, exc_value: BaseException) -> bool:
        exc_str = str(exc_value).lower()
        if "cuda" in exc_str and ("out of memory" in exc_str or "oom" in exc_str):
            return True
        
        if exc_type.__name__ == "RuntimeError":
            if "cuda" in exc_str and "memory" in exc_str:
                return True
        
        if exc_type.__name__ == "OutOfMemoryError":
            return True
        
        if exc_type.__name__ == "MemoryError":
            return True
        
        return False
    
    def _extract_cuda_error(self) -> Optional[str]:
        if self._exception_info is None:
            return None
        
        exc_type, exc_value, _ = self._exception_info
        exc_str = str(exc_value)
        
        if "cuda" in exc_str.lower():
            return exc_str[:1000]
        
        return None
    
    def _write_postmortem(self) -> None:
        if not self.artifact_manager.is_main_process:
            return
        
        if not self.artifact_manager.run_dir.exists():
            return
        
        tb_str = None
        if self._exception_info:
            tb_str = "".join(traceback.format_exception(*self._exception_info))
        
        recent_metrics = self._recent_metrics.get_all()
        last_metrics = recent_metrics[-1]["metrics"] if recent_metrics else {}
        
        recent_events = [
            f"[{e['timestamp']}] {e['type']}: {e.get('data', {})}"
            for e in self._recent_events.get_all()[-20:]
        ]
        
        postmortem = Postmortem(
            exit_reason=self._exit_reason or ExitReason.UNKNOWN,
            last_step=self._last_step,
            timestamp=datetime.now(timezone.utc).isoformat(),
            traceback=tb_str,
            last_metrics=last_metrics,
            recent_events=recent_events,
            environment=get_environment_info(),
            cuda_error=self._extract_cuda_error(),
        )
        
        self.artifact_manager.save_postmortem(postmortem)
    
    def finalize(self) -> None:
        self._finalized = True
        self.uninstall()
    
    @property
    def last_step(self) -> int:
        return self._last_step
    
    @property
    def exit_reason(self) -> Optional[str]:
        return self._exit_reason

def format_postmortem_report(postmortem: Postmortem) -> str:
    lines = [
        "## Postmortem Report",
        f"**Exit Reason:** {postmortem.exit_reason.upper()}",
        f"**Last Step:** {postmortem.last_step}",
        f"**Timestamp:** {postmortem.timestamp}",
        "",
    ]
    
    if postmortem.cuda_error:
        lines.append("### CUDA Error")
        lines.append(f"```\n{postmortem.cuda_error}\n```")
        lines.append("")
    
    if postmortem.traceback:
        lines.append("### Traceback")
        tb = postmortem.traceback
        if len(tb) > 3000:
            tb = tb[-3000:]
            tb = "... (truncated)\n" + tb
        lines.append(f"```\n{tb}\n```")
        lines.append("")
    
    if postmortem.last_metrics:
        lines.append("### Last Metrics")
        for key, value in sorted(postmortem.last_metrics.items()):
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
        lines.append("")
    
    if postmortem.recent_events:
        lines.append("### Recent Events")
        for event in postmortem.recent_events[-10:]:
            lines.append(f"- {event}")
        lines.append("")
    
    lines.append("### Environment")
    env = postmortem.environment
    if "python_version" in env:
        lines.append(f"- Python: {env['python_version'].split()[0]}")
    if "torch_version" in env:
        lines.append(f"- PyTorch: {env['torch_version']}")
    if "cuda_version" in env:
        lines.append(f"- CUDA: {env['cuda_version']}")
    if "hostname" in env:
        lines.append(f"- Host: {env['hostname']}")
    
    return "\n".join(lines)
