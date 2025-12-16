"""Main diagnostics engine for analyzing RLHF training logs.

This module provides the core analysis pipeline:
1. Load JSONL logs from any TRL trainer (supports both file and run directory)
2. Auto-detect trainer type from log metadata
3. Compute derived metrics (KL, drift proxies)
4. Run trainer-appropriate heuristics
5. Analyze behavior snapshots and diffs
6. Generate actionable reports with recommendations
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from importlib import resources

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from post_training_toolkit.diagnostics.heuristics import (
    Insight, 
    run_heuristics, 
    run_all_heuristics,
    TrainerType,
)
from post_training_toolkit.diagnostics import plotting


# Template directory packaged with the module
_TEMPLATES_DIR = resources.files("post_training_toolkit").joinpath("templates")


def load_metrics(path: Path) -> pd.DataFrame:
    """Load metrics from a JSONL log file.
    
    Convenience function to load just the DataFrame without trainer detection.
    
    Args:
        path: Path to JSONL log file OR run directory containing metrics.jsonl
        
    Returns:
        DataFrame with metrics from training run
    """
    df, _ = load_jsonl(path)
    return df


def load_jsonl(path: Path) -> Tuple[pd.DataFrame, str]:
    """Load a JSONL log file into a DataFrame and extract trainer type.
    
    Args:
        path: Path to JSONL log file OR run directory containing metrics.jsonl
        
    Returns:
        Tuple of (DataFrame with metrics, detected trainer type string)
    """
    # Support both file path and run directory
    if path.is_dir():
        path = path / "metrics.jsonl"
    
    records: List[Dict[str, Any]] = []
    trainer_type = TrainerType.UNKNOWN
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            
            # Check for header/footer metadata
            if obj.get("type") == "header":
                trainer_type = obj.get("trainer_type", TrainerType.UNKNOWN)
                continue
            if obj.get("type") == "footer":
                continue
            
            # Extract step-level metrics
            step = obj.get("step")
            ts = obj.get("timestamp")
            metrics = obj.get("metrics", {})
            
            # Also check for trainer_type in each record (for robustness)
            if trainer_type == TrainerType.UNKNOWN:
                trainer_type = obj.get("trainer_type", TrainerType.UNKNOWN)
            
            flat = {"step": step, "timestamp": ts}
            flat.update(metrics)
            records.append(flat)
    
    df = pd.DataFrame.from_records(records)
    if len(df) > 0:
        df = df.sort_values("step").reset_index(drop=True)
    
    return df, trainer_type


def compute_derived_metrics(df: pd.DataFrame, trainer_type: str = TrainerType.UNKNOWN,
                           dpo_beta: float = 0.1) -> pd.DataFrame:
    """Compute derived metrics from raw logged data.
    
    This function computes metrics like KL divergence and policy drift
    from available raw metrics, so they can be used by heuristics and plots.
    
    Args:
        df: DataFrame with raw logged metrics
        trainer_type: Detected trainer type for algorithm-specific derivations
        dpo_beta: Beta parameter for DPO (used if computing KL from rewards)
        
    Returns:
        DataFrame with additional computed columns
    """
    if len(df) == 0:
        return df
    
    # PPO-specific: compute advantage stats if we have values and returns
    if trainer_type == TrainerType.PPO:
        if "value_mean" in df.columns and "returns_mean" in df.columns:
            if "advantages_mean" not in df.columns:
                df["advantages_mean"] = df["returns_mean"] - df["value_mean"]
    
    # Compute perplexity from loss if missing (SFT)
    if trainer_type == TrainerType.SFT:
        if "perplexity" not in df.columns and "sft_loss" in df.columns:
            import numpy as np
            df["perplexity"] = np.exp(df["sft_loss"])
    
    return df


def summarize_run(df: pd.DataFrame, trainer_type: str = TrainerType.UNKNOWN) -> Dict[str, Any]:
    """Generate summary statistics for a training run.
    
    Args:
        df: DataFrame with training metrics
        trainer_type: Detected trainer type
        
    Returns:
        Dictionary of summary statistics
    """
    if len(df) == 0:
        return {"num_steps": 0, "trainer_type": trainer_type}
    
    last = df.iloc[-1]
    summary = {
        "trainer_type": trainer_type,
        "num_steps": int(df["step"].max()),
        "start_time": df["timestamp"].iloc[0] if "timestamp" in df.columns else None,
        "end_time": df["timestamp"].iloc[-1] if "timestamp" in df.columns else None,
    }
    
    # Common metrics
    if "reward_mean" in df.columns:
        summary["final_reward_mean"] = float(last.get("reward_mean", float("nan")))
    if "kl" in df.columns:
        summary["final_kl"] = float(last.get("kl", float("nan")))
    if "refusal_rate" in df.columns:
        summary["final_refusal_rate"] = float(last.get("refusal_rate", float("nan")))
    if "embedding_cosine_to_sft" in df.columns:
        summary["final_cosine"] = float(last.get("embedding_cosine_to_sft", float("nan")))
    
    # DPO-specific
    if trainer_type == TrainerType.DPO:
        if "dpo_loss" in df.columns:
            summary["final_dpo_loss"] = float(last.get("dpo_loss", float("nan")))
        if "win_rate" in df.columns:
            summary["final_win_rate"] = float(last.get("win_rate", float("nan")))
            summary["mean_win_rate"] = float(df["win_rate"].mean())
    
    # PPO-specific
    elif trainer_type == TrainerType.PPO:
        if "ppo_loss" in df.columns:
            summary["final_ppo_loss"] = float(last.get("ppo_loss", float("nan")))
        if "entropy" in df.columns:
            summary["final_entropy"] = float(last.get("entropy", float("nan")))
        if "value_loss" in df.columns:
            summary["final_value_loss"] = float(last.get("value_loss", float("nan")))
        if "clip_fraction" in df.columns:
            summary["mean_clip_fraction"] = float(df["clip_fraction"].mean())
    
    # SFT-specific
    elif trainer_type == TrainerType.SFT:
        if "sft_loss" in df.columns:
            summary["final_sft_loss"] = float(last.get("sft_loss", float("nan")))
        if "perplexity" in df.columns:
            summary["final_perplexity"] = float(last.get("perplexity", float("nan")))
    
    # ORPO-specific
    elif trainer_type == TrainerType.ORPO:
        if "orpo_loss" in df.columns:
            summary["final_orpo_loss"] = float(last.get("orpo_loss", float("nan")))
        if "log_odds_ratio" in df.columns:
            summary["final_log_odds_ratio"] = float(last.get("log_odds_ratio", float("nan")))
        if "win_rate" in df.columns:
            summary["final_win_rate"] = float(last.get("win_rate", float("nan")))
    
    # KTO-specific
    elif trainer_type == TrainerType.KTO:
        if "kto_loss" in df.columns:
            summary["final_kto_loss"] = float(last.get("kto_loss", float("nan")))
    
    return summary


def recommended_actions(insights: List[Insight], trainer_type: str = TrainerType.UNKNOWN) -> List[str]:
    """Generate recommended actions based on detected insights.
    
    Args:
        insights: List of detected insights
        trainer_type: Detected trainer type for trainer-specific recommendations
        
    Returns:
        List of actionable recommendation strings
    """
    actions: List[str] = []
    types = {ins.type for ins in insights}
    
    # KL-related actions
    if any(t in types for t in ("kl_instability", "kl_above_target", "kl_volatility")):
        if trainer_type == TrainerType.PPO:
            actions.append("Reduce learning rate or increase KL penalty coefficient (target_kl).")
        else:
            actions.append("Adjust KL schedule or reduce learning rate.")
    
    # Policy drift actions
    if any(t in types for t in ("policy_drift_alert", "policy_drift_warn")):
        actions.append("Increase KL strength or add anchor tasks to reduce drift from reference.")
    
    # Slice degradation
    if "slice_degradation" in types:
        actions.append("Resample/balance degraded slices; add targeted calibration examples.")
    
    # Length collapse
    if "length_collapse" in types:
        actions.append("Increase output length reward or add minimum length constraints.")
    
    # Refusal issues
    if any(t in types for t in ("refusal_alert", "refusal_warn", "refusal_uptick")):
        actions.append("Inject non-refusal positive examples and tune refusal penalties.")
    
    # Win rate plateau
    if "win_rate_plateau" in types:
        actions.append("Refresh curriculum or increase difficulty; adjust exploration.")
    
    # Reward instability
    if any(t in types for t in ("reward_variance_spike", "instability_hotspot")):
        actions.append("Increase batch size or use gradient clipping; consider reward smoothing.")
    
    # Reward model issues
    if "reward_model_imbalance" in types:
        actions.append("Audit reward model; consider smoothing, ensembling, or reweighting.")
    
    # DPO-specific actions
    if "dpo_loss_random" in types:
        actions.append("DPO loss at random chance: increase learning rate 2-5x, check data quality, or reduce beta.")
    if "dpo_loss_plateau" in types:
        actions.append("DPO loss plateaued: try learning rate warmup/decay or adjust beta parameter.")
    if "win_rate_instability" in types:
        actions.append("Win rate unstable: increase batch size for more stable gradient estimates.")
    if "margin_collapse" in types:
        actions.append("Chosen/rejected margin collapsed: check data quality, increase beta, or add margin-based regularization.")
    
    # PPO-specific actions
    if "value_head_divergence" in types:
        actions.append("Value head diverging: reduce value loss coefficient, increase value head training, or check reward scale.")
    if "entropy_collapse" in types:
        actions.append("Entropy collapsed: increase entropy bonus coefficient to encourage exploration.")
    if "advantage_explosion" in types:
        actions.append("Advantages exploding: enable advantage normalization, reduce learning rate, or clip rewards.")
    if "clip_fraction_high" in types:
        actions.append("High clip fraction: reduce learning rate or increase PPO clip range (epsilon).")
    if "ppo_approx_kl_spike" in types:
        actions.append("Approximate KL spiked: enable early stopping when KL exceeds threshold.")
    
    # SFT-specific actions
    if "sft_loss_plateau" in types:
        actions.append("SFT loss plateaued: training may have converged. Try LR decay or early stopping.")
    if "perplexity_spike" in types:
        actions.append("Perplexity spiked: check for bad data batches, reduce learning rate, or add gradient clipping.")
    
    # ORPO-specific actions
    if "odds_ratio_instability" in types:
        actions.append("ORPO odds ratio unstable: try smaller learning rate or increase batch size.")
    
    # KTO-specific actions
    if "kto_loss_imbalance" in types:
        actions.append("KTO loss imbalanced: rebalance desirable/undesirable examples in dataset.")
    
    # Deduplicate and cap
    dedup = []
    for a in actions:
        if a not in dedup:
            dedup.append(a)
    return dedup[:10]


def analyze_behavior_drift(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Analyze behavior drift from snapshots.
    
    Args:
        run_dir: Run directory containing snapshots
        
    Returns:
        Behavior drift analysis dict or None if no snapshots
    """
    from post_training_toolkit.artifacts import RunArtifactManager
    from post_training_toolkit.diffing import DiffManager
    
    artifact_manager = RunArtifactManager(run_dir)
    steps = artifact_manager.list_snapshots()
    
    if len(steps) < 2:
        return None
    
    diff_manager = DiffManager(artifact_manager)
    diffs = diff_manager.compute_all_diffs(save=False)
    
    if not diffs:
        return None
    
    # Aggregate drift info
    first_snapshot = artifact_manager.load_snapshot(steps[0])
    last_snapshot = artifact_manager.load_snapshot(steps[-1])
    
    if not first_snapshot or not last_snapshot:
        return None
    
    # Overall drift from first to last
    _, overall_summary = diff_manager.diff_steps(steps[0], steps[-1], save=False)
    
    # Refusal trend
    initial_refusal = sum(1 for e in first_snapshot.entries if e.is_refusal) / max(len(first_snapshot.entries), 1)
    final_refusal = sum(1 for e in last_snapshot.entries if e.is_refusal) / max(len(last_snapshot.entries), 1)
    
    # Length trend
    initial_lengths = [e.output_length for e in first_snapshot.entries]
    final_lengths = [e.output_length for e in last_snapshot.entries]
    mean_length_delta = (sum(final_lengths) / len(final_lengths)) - (sum(initial_lengths) / len(initial_lengths)) if initial_lengths and final_lengths else 0
    
    # Flagged prompts
    flagged = []
    if overall_summary:
        for pid in overall_summary.flagged_prompt_ids[:10]:
            # Find the prompt
            for e in first_snapshot.entries:
                if e.prompt_id == pid:
                    reason = "significant change"
                    # Check specific changes
                    for fe in last_snapshot.entries:
                        if fe.prompt_id == pid:
                            if e.is_refusal != fe.is_refusal:
                                reason = "refusal status changed"
                            elif abs(fe.output_length - e.output_length) > 100:
                                reason = f"length changed by {fe.output_length - e.output_length}"
                            break
                    flagged.append({"id": pid[:8], "reason": reason})
                    break
    
    return {
        "severity": overall_summary.drift_severity if overall_summary else "unknown",
        "refusal_trend": {
            "initial": initial_refusal,
            "final": final_refusal,
            "delta": final_refusal - initial_refusal,
        },
        "length_trend": {
            "mean_delta": mean_length_delta,
            "increased": overall_summary.length_increased if overall_summary else 0,
            "decreased": overall_summary.length_decreased if overall_summary else 0,
        },
        "flagged_prompts": flagged,
    }


def get_checkpoint_recommendation(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Get checkpoint recommendation for a run.
    
    Args:
        run_dir: Run directory
        
    Returns:
        Recommendation dict or None
    """
    from post_training_toolkit.checkpoints import recommend_checkpoint
    
    rec = recommend_checkpoint(run_dir)
    if rec:
        return rec.to_dict()
    return None


def load_postmortem(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load postmortem data if it exists.
    
    Args:
        run_dir: Run directory
        
    Returns:
        Postmortem dict or None
    """
    postmortem_path = run_dir / "postmortem.json"
    if postmortem_path.exists():
        with open(postmortem_path, "r") as f:
            return json.load(f)
    return None


def render_report(df: pd.DataFrame, insights: List[Insight], out_path: Path,
                  trainer_type: str = TrainerType.UNKNOWN,
                  plots_dir: Optional[Path] = None,
                  behavior_drift: Optional[Dict[str, Any]] = None,
                  checkpoint_recommendation: Optional[Dict[str, Any]] = None,
                  postmortem: Optional[Dict[str, Any]] = None) -> None:
    """Render a Markdown diagnostic report.
    
    Args:
        df: DataFrame with training metrics
        insights: List of detected insights
        out_path: Path to write the report
        trainer_type: Detected trainer type
        plots_dir: Optional path to plots directory
        behavior_drift: Optional behavior drift analysis
        checkpoint_recommendation: Optional checkpoint recommendation
        postmortem: Optional postmortem data
    """
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(enabled_extensions=("html", "xml"))
    )
    template = env.get_template("report_template.md")
    summary = summarize_run(df, trainer_type)
    
    # Status line heuristic
    status = "Stable"
    if postmortem:
        status = f"Crashed ({postmortem.get('exit_reason', 'unknown')})"
    elif any(ins.severity == "high" for ins in insights):
        status = "Unstable"
    elif any(ins.severity == "medium" for ins in insights):
        status = "Partially unstable"
    
    # Build context
    context: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "trainer_type": trainer_type.upper(),
        "status": status,
        "summary": summary,
        "insights": [asdict(i) for i in insights],
        "actions": recommended_actions(insights, trainer_type),
        "plots": {
            "reward": "plots/reward.png" if plots_dir else None,
            "kl": "plots/kl.png" if plots_dir else None,
            "drift": "plots/drift.png" if plots_dir else None,
            "slices": "plots/slices.png" if plots_dir else None,
        },
        "behavior_drift": behavior_drift,
        "checkpoint_recommendation": checkpoint_recommendation,
        "postmortem": postmortem,
    }
    
    md = template.render(**context)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")


def run_diagnostics(input_path: Path, reports_dir: Path, make_plots: bool = True,
                    trainer_type: Optional[str] = None) -> Path:
    """Run full diagnostics pipeline on a log file or run directory.
    
    This function:
    1. Loads the JSONL log file (or metrics.jsonl from run directory)
    2. Auto-detects the trainer type (DPO, PPO, SFT, etc.) unless overridden
    3. Computes derived metrics
    4. Runs trainer-appropriate heuristics
    5. Analyzes behavior snapshots (if available)
    6. Generates checkpoint recommendation (if snapshots available)
    7. Includes postmortem data (if run crashed)
    8. Generates plots and a diagnostic report
    
    Args:
        input_path: Path to JSONL log file
        reports_dir: Directory to write report and plots
        make_plots: Whether to generate plots
        trainer_type: Override auto-detected trainer type (dpo, ppo, sft, etc.)
        
    Returns:
        Path to the generated report
    """
    # Load logs and detect trainer type
    df, detected_trainer_type = load_jsonl(input_path)
    
    # Use override if provided, otherwise use detected
    effective_trainer_type = trainer_type if trainer_type else detected_trainer_type
    
    if len(df) == 0:
        raise ValueError(f"No valid log entries found in {input_path}")
    
    # Compute derived metrics
    df = compute_derived_metrics(df, effective_trainer_type)
    
    # Run trainer-aware heuristics
    insights = run_heuristics(df, effective_trainer_type)
    
    # Generate plots
    plots_dir: Optional[Path] = None
    if make_plots:
        plots_dir = reports_dir / "plots"
        plotting.plot_reward(df, plots_dir)
        plotting.plot_kl(df, plots_dir)
        plotting.plot_drift(df, plots_dir)
        plotting.plot_slices(df, plots_dir)
        
        # PPO-specific plots
        if effective_trainer_type == TrainerType.PPO:
            _plot_ppo_metrics(df, plots_dir)
    
    # Check if input is a run directory (has snapshots, postmortem, etc.)
    run_dir = input_path if input_path.is_dir() else input_path.parent
    
    # Analyze behavior drift (if snapshots available)
    behavior_drift = None
    if (run_dir / "snapshots").exists():
        try:
            behavior_drift = analyze_behavior_drift(run_dir)
        except Exception:
            pass  # Snapshots may be incomplete
    
    # Get checkpoint recommendation
    checkpoint_recommendation = None
    if behavior_drift:
        try:
            checkpoint_recommendation = get_checkpoint_recommendation(run_dir)
        except Exception:
            pass
    
    # Load postmortem if exists
    postmortem = load_postmortem(run_dir)
    
    # Generate report
    report_name = input_path.stem if input_path.is_file() else "run"
    out_path = reports_dir / f"{report_name}_report.md"
    render_report(
        df, insights, out_path, effective_trainer_type,
        plots_dir=plots_dir,
        behavior_drift=behavior_drift,
        checkpoint_recommendation=checkpoint_recommendation,
        postmortem=postmortem,
    )
    
    return out_path


def _plot_ppo_metrics(df: pd.DataFrame, plots_dir: Path) -> None:
    """Generate PPO-specific plots."""
    # Value loss and entropy
    if "value_loss" in df.columns or "entropy" in df.columns:
        ys = []
        if "value_loss" in df.columns:
            ys.append("value_loss")
        if "entropy" in df.columns:
            ys.append("entropy")
        if ys:
            plotting.plot_series(df, x="step", ys=ys,
                               title="PPO Value Loss & Entropy",
                               outfile=plots_dir / "ppo_value_entropy.png")
    
    # Clip fraction
    if "clip_fraction" in df.columns:
        plotting.plot_series(df, x="step", ys=["clip_fraction"],
                           title="PPO Clip Fraction",
                           outfile=plots_dir / "ppo_clip_fraction.png")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RLHF Diagnostics for TRL logs",
        epilog="Supports DPO, PPO, SFT, ORPO, KTO trainers with auto-detection."
    )
    parser.add_argument("--input", type=str, required=True, 
                       help="Path to run directory or JSONL log file")
    parser.add_argument("--reports-dir", type=str, default="reports",
                       help="Where to write reports (default: reports)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Disable plot generation")
    args = parser.parse_args()
    
    input_path = Path(args.input).absolute()
    reports_dir = Path(args.reports_dir).absolute()
    
    out = run_diagnostics(input_path, reports_dir, make_plots=not args.no_plots)
    print(f"Report written to: {out}")


if __name__ == "__main__":
    main()
