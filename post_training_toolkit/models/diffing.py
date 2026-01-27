from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from post_training_toolkit.models.artifacts import (
    RunArtifactManager,
    Snapshot,
    SnapshotEntry,
    SnapshotDiff,
    DiffEntry,
)

@dataclass
class DiffThresholds:
    length_pct_significant: float = 0.25
    length_abs_significant: int = 50
    
    entropy_delta_significant: float = 0.5
    
    logprob_delta_significant: float = 0.3
    
    refusal_rate_delta_significant: float = 0.1

@dataclass
class DiffSummary:
    step_a: int
    step_b: int
    num_prompts: int
    
    length_increased: int = 0
    length_decreased: int = 0
    length_stable: int = 0
    
    refusal_gained: int = 0
    refusal_lost: int = 0
    refusal_rate_before: float = 0.0
    refusal_rate_after: float = 0.0
    
    entropy_increased: int = 0
    entropy_decreased: int = 0
    
    significant_changes: int = 0
    flagged_prompt_ids: List[str] = None
    
    drift_severity: str = "minimal"
    
    def __post_init__(self):
        if self.flagged_prompt_ids is None:
            self.flagged_prompt_ids = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_a": self.step_a,
            "step_b": self.step_b,
            "num_prompts": self.num_prompts,
            "length_increased": self.length_increased,
            "length_decreased": self.length_decreased,
            "length_stable": self.length_stable,
            "refusal_gained": self.refusal_gained,
            "refusal_lost": self.refusal_lost,
            "refusal_rate_before": self.refusal_rate_before,
            "refusal_rate_after": self.refusal_rate_after,
            "entropy_increased": self.entropy_increased,
            "entropy_decreased": self.entropy_decreased,
            "significant_changes": self.significant_changes,
            "flagged_prompt_ids": self.flagged_prompt_ids,
            "drift_severity": self.drift_severity,
        }

def diff_entries(
    entry_a: SnapshotEntry,
    entry_b: SnapshotEntry,
    thresholds: DiffThresholds,
) -> Tuple[DiffEntry, bool]:
    length_delta = entry_b.output_length - entry_a.output_length
    length_pct = length_delta / max(entry_a.output_length, 1)
    
    refusal_changed = entry_a.is_refusal != entry_b.is_refusal
    
    entropy_delta = None
    logprob_delta = None
    
    if entry_a.entropy_mean is not None and entry_b.entropy_mean is not None:
        entropy_delta = entry_b.entropy_mean - entry_a.entropy_mean
    
    if entry_a.logprob_mean is not None and entry_b.logprob_mean is not None:
        logprob_delta = entry_b.logprob_mean - entry_a.logprob_mean
    
    is_significant = False
    
    if abs(length_pct) >= thresholds.length_pct_significant:
        is_significant = True
    if abs(length_delta) >= thresholds.length_abs_significant:
        is_significant = True
    
    if refusal_changed:
        is_significant = True
    
    if entropy_delta is not None and abs(entropy_delta) >= thresholds.entropy_delta_significant:
        is_significant = True
    
    if logprob_delta is not None and abs(logprob_delta) >= thresholds.logprob_delta_significant:
        is_significant = True
    
    diff = DiffEntry(
        prompt_id=entry_a.prompt_id,
        length_delta=length_delta,
        length_pct_change=length_pct,
        refusal_changed=refusal_changed,
        refusal_before=entry_a.is_refusal,
        refusal_after=entry_b.is_refusal,
        entropy_delta=entropy_delta,
        logprob_delta=logprob_delta,
    )
    
    return diff, is_significant

def diff_snapshots(
    snapshot_a: Snapshot,
    snapshot_b: Snapshot,
    thresholds: Optional[DiffThresholds] = None,
) -> Tuple[SnapshotDiff, DiffSummary]:
    if thresholds is None:
        thresholds = DiffThresholds()
    
    entries_a = {e.prompt_id: e for e in snapshot_a.entries}
    entries_b = {e.prompt_id: e for e in snapshot_b.entries}
    
    common_ids = set(entries_a.keys()) & set(entries_b.keys())
    
    diff_entries_list: List[DiffEntry] = []
    flagged_ids: List[str] = []
    
    length_increased = 0
    length_decreased = 0
    length_stable = 0
    entropy_increased = 0
    entropy_decreased = 0
    refusal_gained = 0
    refusal_lost = 0
    
    for pid in sorted(common_ids):
        entry_a = entries_a[pid]
        entry_b = entries_b[pid]
        
        diff, is_significant = diff_entries(entry_a, entry_b, thresholds)
        diff_entries_list.append(diff)
        
        if is_significant:
            flagged_ids.append(pid)
        
        if diff.length_delta > thresholds.length_abs_significant:
            length_increased += 1
        elif diff.length_delta < -thresholds.length_abs_significant:
            length_decreased += 1
        else:
            length_stable += 1
        
        if diff.entropy_delta is not None:
            if diff.entropy_delta > thresholds.entropy_delta_significant:
                entropy_increased += 1
            elif diff.entropy_delta < -thresholds.entropy_delta_significant:
                entropy_decreased += 1
        
        if diff.refusal_changed:
            if diff.refusal_after and not diff.refusal_before:
                refusal_gained += 1
            elif diff.refusal_before and not diff.refusal_after:
                refusal_lost += 1
    
    refusal_before = sum(1 for e in snapshot_a.entries if e.is_refusal) / max(len(snapshot_a.entries), 1)
    refusal_after = sum(1 for e in snapshot_b.entries if e.is_refusal) / max(len(snapshot_b.entries), 1)
    
    significant_count = len(flagged_ids)
    total_prompts = len(common_ids)
    
    if significant_count == 0:
        drift_severity = "minimal"
    elif significant_count / max(total_prompts, 1) < 0.2:
        drift_severity = "moderate"
    else:
        drift_severity = "significant"
    
    if refusal_gained > 0 or abs(refusal_after - refusal_before) > 0.15:
        drift_severity = "significant"
    
    summary = DiffSummary(
        step_a=snapshot_a.metadata.step,
        step_b=snapshot_b.metadata.step,
        num_prompts=total_prompts,
        length_increased=length_increased,
        length_decreased=length_decreased,
        length_stable=length_stable,
        refusal_gained=refusal_gained,
        refusal_lost=refusal_lost,
        refusal_rate_before=refusal_before,
        refusal_rate_after=refusal_after,
        entropy_increased=entropy_increased,
        entropy_decreased=entropy_decreased,
        significant_changes=significant_count,
        flagged_prompt_ids=flagged_ids,
        drift_severity=drift_severity,
    )
    
    diff = SnapshotDiff(
        step_a=snapshot_a.metadata.step,
        step_b=snapshot_b.metadata.step,
        timestamp=datetime.now(timezone.utc).isoformat(),
        entries=diff_entries_list,
        summary=summary.to_dict(),
    )
    
    return diff, summary

class DiffManager:
    
    def __init__(
        self,
        artifact_manager: RunArtifactManager,
        thresholds: Optional[DiffThresholds] = None,
    ):
        self.artifact_manager = artifact_manager
        self.thresholds = thresholds or DiffThresholds()
    
    def diff_steps(
        self,
        step_a: int,
        step_b: int,
        save: bool = True,
    ) -> Optional[Tuple[SnapshotDiff, DiffSummary]]:
        snapshot_a = self.artifact_manager.load_snapshot(step_a)
        snapshot_b = self.artifact_manager.load_snapshot(step_b)
        
        if snapshot_a is None or snapshot_b is None:
            return None
        
        diff, summary = diff_snapshots(snapshot_a, snapshot_b, self.thresholds)
        
        if save and self.artifact_manager.is_main_process:
            self.artifact_manager.save_diff(diff)
        
        return diff, summary
    
    def compute_all_diffs(
        self,
        save: bool = True,
    ) -> List[Tuple[SnapshotDiff, DiffSummary]]:
        steps = self.artifact_manager.list_snapshots()
        if len(steps) < 2:
            return []
        
        diffs = []
        for i in range(len(steps) - 1):
            result = self.diff_steps(steps[i], steps[i + 1], save=save)
            if result:
                diffs.append(result)
        
        return diffs
    
    def compute_drift_from_baseline(
        self,
        baseline_step: int = 0,
        save: bool = True,
    ) -> List[Tuple[SnapshotDiff, DiffSummary]]:
        steps = self.artifact_manager.list_snapshots()
        if baseline_step not in steps:
            return []
        
        diffs = []
        for step in steps:
            if step == baseline_step:
                continue
            result = self.diff_steps(baseline_step, step, save=save)
            if result:
                diffs.append(result)
        
        return diffs
    
    def get_drift_timeline(self) -> List[Dict[str, Any]]:
        diffs = self.compute_all_diffs(save=False)
        
        timeline = []
        for diff, summary in diffs:
            timeline.append({
                "step_a": summary.step_a,
                "step_b": summary.step_b,
                "drift_severity": summary.drift_severity,
                "significant_changes": summary.significant_changes,
                "refusal_delta": summary.refusal_rate_after - summary.refusal_rate_before,
            })
        
        return timeline

def format_diff_report(summary: DiffSummary, verbose: bool = False) -> str:
    lines = [
        f"## Behavior Diff: Step {summary.step_a} → {summary.step_b}",
        f"**Drift Severity:** {summary.drift_severity.upper()}",
        f"**Prompts Analyzed:** {summary.num_prompts}",
        "",
    ]
    
    refusal_delta = summary.refusal_rate_after - summary.refusal_rate_before
    lines.append(f"### Refusal Rate")
    lines.append(f"- Before: {summary.refusal_rate_before:.1%}")
    lines.append(f"- After: {summary.refusal_rate_after:.1%}")
    lines.append(f"- Delta: {refusal_delta:+.1%}")
    if summary.refusal_gained > 0:
        lines.append(f"- ⚠️ {summary.refusal_gained} prompts started refusing")
    if summary.refusal_lost > 0:
        lines.append(f"- ℹ️ {summary.refusal_lost} prompts stopped refusing")
    lines.append("")
    
    lines.append(f"### Length Changes")
    lines.append(f"- Increased: {summary.length_increased}")
    lines.append(f"- Decreased: {summary.length_decreased}")
    lines.append(f"- Stable: {summary.length_stable}")
    lines.append("")
    
    if summary.entropy_increased or summary.entropy_decreased:
        lines.append(f"### Entropy Changes")
        lines.append(f"- Increased: {summary.entropy_increased}")
        lines.append(f"- Decreased: {summary.entropy_decreased}")
        lines.append("")
    
    if summary.significant_changes > 0:
        lines.append(f"### Significant Changes")
        lines.append(f"- {summary.significant_changes} prompts flagged")
        if verbose and summary.flagged_prompt_ids:
            lines.append(f"- IDs: {', '.join(summary.flagged_prompt_ids[:10])}")
            if len(summary.flagged_prompt_ids) > 10:
                lines.append(f"  ... and {len(summary.flagged_prompt_ids) - 10} more")
    
    return "\n".join(lines)
