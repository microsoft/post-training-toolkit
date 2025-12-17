"""Heuristics library for detecting RLHF training issues.

This module provides algorithm-specific heuristics for detecting common failure
modes in post-training runs. Heuristics are organized by trainer type:

- **Common**: Apply to all trainer types
- **DPO-specific**: Loss at 0.693, margin collapse, win-rate instability
- **PPO-specific**: Value head divergence, entropy collapse, advantage explosion
- **SFT-specific**: Loss plateau, perplexity spikes
- **ORPO-specific**: Odds ratio instability
- **KTO-specific**: Desirable/undesirable imbalance

References:
- Rafailov et al. (2023) "Direct Preference Optimization"
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- Zheng et al. (2023) "Secrets of RLHF in Large Language Models"
- Hong et al. (2024) "ORPO: Monolithic Preference Optimization without Reference Model"
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from scipy.stats import linregress


# Trainer type constants (mirror callback)
class TrainerType:
    DPO = "dpo"
    PPO = "ppo"
    SFT = "sft"
    ORPO = "orpo"
    KTO = "kto"
    CPO = "cpo"
    GRPO = "grpo"
    UNKNOWN = "unknown"


@dataclass
class Insight:
    """A diagnostic insight detected by a heuristic.
    
    Attributes:
        type: Unique identifier for this insight type (e.g., "dpo_loss_random")
        severity: One of "high", "medium", "low"
        message: Human-readable description of the issue
        steps: List of training steps where this issue was detected
        data: Additional diagnostic data (thresholds, values, etc.)
        trainer_types: Set of trainer types this insight applies to
        reference: Optional citation or documentation link
    """
    type: str
    severity: str
    message: str
    steps: Optional[List[int]] = None
    data: Optional[Dict] = None
    trainer_types: Set[str] = field(default_factory=lambda: {TrainerType.UNKNOWN})
    reference: Optional[str] = None


def _rolling_std_ratio(series: pd.Series, window_short: int, window_long: int) -> pd.Series:
    """Compute ratio of short-term to long-term rolling standard deviation."""
    short = series.rolling(window_short, min_periods=max(2, window_short // 2)).std()
    long = series.rolling(window_long, min_periods=max(2, window_long // 2)).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = short / long
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =============================================================================
# COMMON HEURISTICS (all trainers)
# =============================================================================

def detect_reward_variance_spikes(df: pd.DataFrame, window_short: int = 10, window_long: int = 50,
                                  ratio_threshold: float = 2.5) -> List[Insight]:
    """Detect sudden spikes in reward variance.
    
    High reward variance can indicate:
    - Reward model instability
    - Distribution shift in training data
    - Learning rate too high
    
    Reference: Zheng et al. (2023) "Secrets of RLHF", Section 4.1
    """
    if "reward_mean" not in df.columns or "reward_std" not in df.columns:
        return []
    variance_signal = df["reward_std"].copy()
    if variance_signal.isna().all() or variance_signal.max() == 0:
        variance_signal = df["reward_mean"].rolling(5, min_periods=3).std().fillna(0.0)
    ratio = _rolling_std_ratio(variance_signal, window_short, window_long)
    spike_steps = df.loc[ratio > ratio_threshold, "step"].astype(int).tolist()
    if spike_steps:
        max_ratio = float(np.nanmax(ratio.values))
        return [Insight(
            type="reward_variance_spike",
            severity="high" if max_ratio > (ratio_threshold * 1.5) else "medium",
            message=f"Reward variance spiked up to {max_ratio:.2f}x relative to long-term baseline.",
            steps=spike_steps,
            data={"max_ratio": max_ratio, "ratio_threshold": ratio_threshold},
            trainer_types={TrainerType.DPO, TrainerType.PPO, TrainerType.ORPO, TrainerType.KTO},
            reference="Zheng et al. (2023) 'Secrets of RLHF', Section 4.1",
        )]
    return []


def detect_kl_instability(df: pd.DataFrame, kl_target: float = 0.12, hard_cap: float = 0.3,
                          roc_threshold: float = 0.05) -> List[Insight]:
    """Detect KL divergence exceeding safe thresholds.
    
    High KL indicates the policy is drifting too far from the reference,
    which can lead to reward hacking and capability loss.
    
    Reference: Schulman et al. (2017) "PPO"; Rafailov et al. (2023) "DPO"
    """
    if "kl" not in df.columns:
        return []
    kl = df["kl"].astype(float)
    steps = df["step"].astype(int)
    above_target = steps[kl > kl_target].tolist()
    above_cap = steps[kl > hard_cap].tolist()
    kl_diff = kl.diff().rolling(5, min_periods=2).mean().abs()
    roc_steps = steps[kl_diff > roc_threshold].tolist()
    
    insights: List[Insight] = []
    if above_cap:
        insights.append(Insight(
            type="kl_instability",
            severity="high",
            message=f"KL exceeded hard cap ({hard_cap:.2f}) at steps like {above_cap[:5]}...",
            steps=above_cap[:50],
            data={"kl_target": kl_target, "hard_cap": hard_cap},
            trainer_types={TrainerType.PPO, TrainerType.DPO, TrainerType.KTO},
            reference="Schulman et al. (2017) 'PPO', Section 4",
        ))
    elif above_target:
        insights.append(Insight(
            type="kl_above_target",
            severity="medium",
            message=f"KL persistently above target ({kl_target:.2f}) at steps like {above_target[:5]}...",
            steps=above_target[:50],
            data={"kl_target": kl_target},
            trainer_types={TrainerType.PPO, TrainerType.DPO, TrainerType.KTO},
        ))
    if roc_steps:
        insights.append(Insight(
            type="kl_volatility",
            severity="medium",
            message="KL shows high short-term volatility.",
            steps=roc_steps[:50],
            data={"roc_threshold": roc_threshold},
            trainer_types={TrainerType.PPO, TrainerType.DPO, TrainerType.KTO},
        ))
    return insights


def detect_policy_drift(df: pd.DataFrame, cosine_key: str = "embedding_cosine_to_sft",
                        warn_threshold: float = 0.92, alert_threshold: float = 0.88) -> List[Insight]:
    """Detect excessive policy drift from the SFT/reference model.
    
    Low cosine similarity indicates the policy has diverged significantly,
    risking capability regression and mode collapse.
    """
    if cosine_key not in df.columns:
        return []
    cos = df[cosine_key].astype(float)
    steps = df["step"].astype(int)
    low_warn = steps[cos < warn_threshold].tolist()
    low_alert = steps[cos < alert_threshold].tolist()
    
    recent = cos.tail(30)
    slope = linregress(np.arange(len(recent)), recent.values).slope if len(recent) >= 3 else 0.0
    
    if low_alert:
        return [Insight(
            type="policy_drift_alert",
            severity="high",
            message=f"Policy cosine to SFT dropped below {alert_threshold:.2f}.",
            steps=low_alert[:50],
            data={"min_cosine": float(cos.min()), "slope_recent": float(slope)},
            trainer_types={TrainerType.DPO, TrainerType.PPO, TrainerType.ORPO, TrainerType.KTO},
        )]
    elif low_warn:
        return [Insight(
            type="policy_drift_warn",
            severity="medium",
            message=f"Policy cosine to SFT dropped below {warn_threshold:.2f}.",
            steps=low_warn[:50],
            data={"min_cosine": float(cos.min()), "slope_recent": float(slope)},
            trainer_types={TrainerType.DPO, TrainerType.PPO, TrainerType.ORPO, TrainerType.KTO},
        )]
    return []


def detect_slice_degradation(df: pd.DataFrame, slice_prefix: str = "slice:", 
                             baseline_window: int = 50, recent_window: int = 50,
                             relative_drop_threshold: float = 0.08) -> List[Insight]:
    """Detect capability regression on specific evaluation slices.
    
    Monitors slice:medical, slice:coding, slice:safety etc. for degradation
    compared to early training baseline.
    """
    slice_cols = [c for c in df.columns if c.startswith(slice_prefix)]
    if not slice_cols:
        return []
    
    baseline = df.head(baseline_window)
    recent = df.tail(recent_window)
    insights: List[Insight] = []
    
    for col in slice_cols:
        base_mean = float(baseline[col].mean())
        recent_mean = float(recent[col].mean())
        if base_mean == 0:
            continue
        drop = (base_mean - recent_mean) / abs(base_mean)
        if drop >= relative_drop_threshold:
            slice_name = col.split("slice:")[-1]
            insights.append(Insight(
                type="slice_degradation",
                severity="high" if drop >= 0.15 else "medium",
                message=f"{slice_name.capitalize()} slice degraded {drop*100:.1f}%.",
                steps=None,
                data={"slice": slice_name, "baseline_mean": base_mean, 
                      "recent_mean": recent_mean, "relative_drop": drop},
                trainer_types={TrainerType.DPO, TrainerType.PPO, TrainerType.SFT, 
                              TrainerType.ORPO, TrainerType.KTO},
            ))
    return insights


def detect_output_length_collapse(df: pd.DataFrame, window: int = 30, 
                                  relative_drop_threshold: float = 0.2) -> List[Insight]:
    """Detect collapse in output length (often sign of mode collapse)."""
    if "output_length_mean" not in df.columns:
        return []
    baseline = float(df["output_length_mean"].head(window).mean())
    recent = float(df["output_length_mean"].tail(window).mean())
    
    if baseline > 0:
        drop = (baseline - recent) / baseline
        if drop >= relative_drop_threshold:
            return [Insight(
                type="length_collapse",
                severity="high" if drop > 0.35 else "medium",
                message=f"Output length collapsed by {drop*100:.1f}% (avg).",
                steps=None,
                data={"baseline_mean": baseline, "recent_mean": recent, "relative_drop": drop},
                trainer_types={TrainerType.DPO, TrainerType.PPO, TrainerType.ORPO, TrainerType.KTO},
            )]
    return []


def detect_refusal_regressions(df: pd.DataFrame, warn_threshold: float = 0.12, 
                               alert_threshold: float = 0.2, uptick_threshold: float = 0.05,
                               window: int = 40) -> List[Insight]:
    """Detect rising refusal rates (safety regression or over-refusal)."""
    if "refusal_rate" not in df.columns:
        return []
    refusal = df["refusal_rate"].astype(float)
    steps = df["step"].astype(int)
    recent = refusal.tail(window)
    baseline = refusal.head(window)
    
    insights: List[Insight] = []
    alert_steps = steps[refusal > alert_threshold].tolist()
    warn_steps = steps[refusal > warn_threshold].tolist()
    
    if alert_steps:
        insights.append(Insight(
            type="refusal_alert",
            severity="high",
            message=f"Refusal rate exceeded {alert_threshold:.2f}.",
            steps=alert_steps[:50],
            data={"max_refusal": float(refusal.max())},
            trainer_types={TrainerType.DPO, TrainerType.PPO, TrainerType.ORPO, TrainerType.KTO},
        ))
    elif warn_steps:
        insights.append(Insight(
            type="refusal_warn",
            severity="medium",
            message=f"Refusal rate exceeded {warn_threshold:.2f}.",
            steps=warn_steps[:50],
            data={"max_refusal": float(refusal.max())},
            trainer_types={TrainerType.DPO, TrainerType.PPO, TrainerType.ORPO, TrainerType.KTO},
        ))
    
    if len(recent) > 3 and len(baseline) > 3:
        if recent.mean() - baseline.mean() > uptick_threshold:
            insights.append(Insight(
                type="refusal_uptick",
                severity="medium",
                message="Refusal rate increased notably vs. early training.",
                steps=None,
                data={"baseline_mean": float(baseline.mean()), "recent_mean": float(recent.mean())},
                trainer_types={TrainerType.DPO, TrainerType.PPO, TrainerType.ORPO, TrainerType.KTO},
            ))
    return insights


def detect_instability_window(df: pd.DataFrame, fields: Tuple[str, ...] = ("reward_mean", "kl"),
                              window: int = 30, volatility_threshold: float = 1.0) -> List[Insight]:
    """Detect periods of high combined instability across multiple metrics."""
    missing = [f for f in fields if f not in df.columns]
    if missing:
        return []
    
    combined = np.zeros(len(df))
    for f in fields:
        s = df[f].astype(float)
        s_norm = (s - s.mean()) / (s.std() + 1e-8)
        combined += s_norm.values ** 2
    
    combined = pd.Series(combined, index=df.index)
    vol = combined.rolling(window, min_periods=max(2, window // 2)).mean()
    max_vol = float(vol.max()) if not math.isnan(vol.max()) else 0.0
    
    if max_vol > volatility_threshold:
        step = int(df.loc[vol.idxmax(), "step"])
        return [Insight(
            type="instability_hotspot",
            severity="high" if max_vol >= (volatility_threshold * 2.0) else "medium",
            message=f"Instability hotspot detected around step ~{step}.",
            steps=[step],
            data={"max_volatility": max_vol, "window": window},
            trainer_types={TrainerType.DPO, TrainerType.PPO, TrainerType.ORPO, TrainerType.KTO},
        )]
    return []


def detect_reward_hacking(df: pd.DataFrame, window: int = 50,
                          kl_reward_ratio_threshold: float = 3.0) -> List[Insight]:
    """Detect reward hacking (Goodhart's law) during RLHF training.
    
    Signal: reward increasing while KL divergence explodes disproportionately.
    This indicates the model is gaming the reward model rather than genuinely
    improving on the underlying task.
    
    Reference: Gao et al. (2022) 'Scaling Laws for Reward Model Overoptimization'
    """
    if "reward_mean" not in df.columns or "kl" not in df.columns:
        return []
    
    if len(df) < window:
        return []
    
    recent = df.tail(window)
    reward = recent["reward_mean"].astype(float)
    kl = recent["kl"].astype(float)
    
    # Compute slopes
    x = np.arange(len(reward))
    reward_slope = linregress(x, reward.values).slope
    kl_slope = linregress(x, kl.values).slope
    
    # Hacking signal: reward going up, KL going up much faster
    if reward_slope > 0 and kl_slope > 0.005:
        ratio = kl_slope / (abs(reward_slope) + 1e-8)
        if ratio > kl_reward_ratio_threshold:
            return [Insight(
                type="reward_hacking_suspected",
                severity="high",
                message=f"Possible reward hacking: KL growing {ratio:.1f}x faster than reward. "
                        "Model may be gaming the reward model.",
                steps=None,
                data={
                    "reward_slope": float(reward_slope),
                    "kl_slope": float(kl_slope),
                    "kl_reward_ratio": float(ratio),
                    "window": window,
                },
                trainer_types={TrainerType.PPO, TrainerType.DPO, TrainerType.GRPO},
                reference="Gao et al. (2022) 'Scaling Laws for Reward Model Overoptimization'",
            )]
    return []


def detect_mode_collapse(df: pd.DataFrame, window: int = 30,
                         length_variance_collapse_threshold: float = 0.2,
                         entropy_floor: float = 1.0,
                         entropy_drop_threshold: float = 0.5) -> List[Insight]:
    """Detect mode collapse (outputs becoming uniform/repetitive).
    
    Signals:
    - Output length variance collapsing (all outputs same length)
    - Entropy collapsing (PPO/GRPO - policy too deterministic)
    
    Mode collapse is a common failure mode where the model converges to
    generating nearly identical outputs regardless of input.
    """
    insights = []
    
    # Signal 1: Length variance collapse (from snapshots)
    if "output_length_mean" in df.columns and len(df) > window * 2:
        baseline_std = df["output_length_mean"].head(window).std()
        recent_std = df["output_length_mean"].tail(window).std()
        
        # Only flag if baseline had meaningful variance
        if baseline_std > 1.0 and recent_std < baseline_std * length_variance_collapse_threshold:
            insights.append(Insight(
                type="mode_collapse_length_variance",
                severity="high",
                message=f"Output length variance collapsed: {baseline_std:.1f} → {recent_std:.1f}. "
                        "Model may be stuck in a mode.",
                steps=None,
                data={
                    "baseline_length_std": float(baseline_std),
                    "recent_length_std": float(recent_std),
                    "collapse_ratio": float(recent_std / baseline_std) if baseline_std > 0 else 0,
                },
                trainer_types={TrainerType.DPO, TrainerType.PPO, TrainerType.GRPO, TrainerType.ORPO},
            ))
    
    # Signal 2: Entropy collapse (PPO/GRPO log this by default)
    if "entropy" in df.columns and len(df) > window:
        recent_entropy = df["entropy"].tail(window).mean()
        baseline_entropy = df["entropy"].head(window).mean()
        
        # Absolute floor check
        if recent_entropy < entropy_floor:
            insights.append(Insight(
                type="mode_collapse_entropy",
                severity="high",
                message=f"Policy entropy collapsed to {recent_entropy:.3f}. "
                        "Model is too deterministic.",
                steps=None,
                data={
                    "baseline_entropy": float(baseline_entropy),
                    "recent_entropy": float(recent_entropy),
                    "entropy_floor": entropy_floor,
                },
                trainer_types={TrainerType.PPO, TrainerType.GRPO},
            ))
        # Relative drop check
        elif baseline_entropy > 0 and recent_entropy < baseline_entropy * (1 - entropy_drop_threshold):
            drop_pct = (baseline_entropy - recent_entropy) / baseline_entropy * 100
            insights.append(Insight(
                type="mode_collapse_entropy",
                severity="medium",
                message=f"Policy entropy dropped {drop_pct:.0f}%: {baseline_entropy:.3f} → {recent_entropy:.3f}.",
                steps=None,
                data={
                    "baseline_entropy": float(baseline_entropy),
                    "recent_entropy": float(recent_entropy),
                    "drop_percentage": float(drop_pct),
                },
                trainer_types={TrainerType.PPO, TrainerType.GRPO},
            ))
    
    return insights


def detect_gradient_issues(df: pd.DataFrame, window: int = 50,
                           explosion_multiplier: float = 10.0,
                           vanishing_multiplier: float = 0.01) -> List[Insight]:
    """Detect gradient explosion or vanishing.
    
    Monitors grad_norm (logged by most Transformers trainers by default)
    to catch optimization issues early.
    
    - Gradient explosion: sudden large increase in gradient norm
    - Gradient vanishing: gradients dropping to near-zero (learning stalled)
    """
    if "grad_norm" not in df.columns:
        return []  # Gracefully skip if not logged
    
    grad_norm = df["grad_norm"].astype(float)
    if len(grad_norm) < window:
        return []
    
    # Use early training as baseline
    baseline_mean = grad_norm.head(min(window, len(grad_norm) // 3)).mean()
    baseline_std = grad_norm.head(min(window, len(grad_norm) // 3)).std()
    recent_mean = grad_norm.tail(window).mean()
    current = grad_norm.iloc[-1]
    
    insights = []
    
    # Gradient explosion: current >> baseline
    if baseline_mean > 0 and current > baseline_mean * explosion_multiplier:
        insights.append(Insight(
            type="gradient_explosion",
            severity="high",
            message=f"Gradient norm exploded: {current:.2f} (baseline: {baseline_mean:.2f}). "
                    "Consider reducing learning rate or adding gradient clipping.",
            steps=None,
            data={
                "current_grad_norm": float(current),
                "baseline_mean": float(baseline_mean),
                "baseline_std": float(baseline_std),
                "explosion_ratio": float(current / baseline_mean) if baseline_mean > 0 else 0,
            },
            trainer_types={TrainerType.DPO, TrainerType.PPO, TrainerType.SFT, 
                          TrainerType.GRPO, TrainerType.ORPO, TrainerType.KTO},
        ))
    
    # Gradient vanishing: current << baseline
    if baseline_mean > 0 and current < baseline_mean * vanishing_multiplier:
        insights.append(Insight(
            type="gradient_vanishing",
            severity="medium",
            message=f"Gradients nearly vanished: {current:.6f} (baseline: {baseline_mean:.2f}). "
                    "Learning may have stalled.",
            steps=None,
            data={
                "current_grad_norm": float(current),
                "baseline_mean": float(baseline_mean),
                "vanishing_ratio": float(current / baseline_mean) if baseline_mean > 0 else 0,
            },
            trainer_types={TrainerType.DPO, TrainerType.PPO, TrainerType.SFT,
                          TrainerType.GRPO, TrainerType.ORPO, TrainerType.KTO},
        ))
    
    # Also detect high variance (unstable training)
    recent_std = grad_norm.tail(window).std()
    if baseline_std > 0 and recent_std > baseline_std * 3:
        insights.append(Insight(
            type="gradient_instability",
            severity="medium",
            message=f"Gradient norm highly unstable (std: {recent_std:.2f} vs baseline {baseline_std:.2f}).",
            steps=None,
            data={
                "recent_std": float(recent_std),
                "baseline_std": float(baseline_std),
            },
            trainer_types={TrainerType.DPO, TrainerType.PPO, TrainerType.SFT,
                          TrainerType.GRPO, TrainerType.ORPO, TrainerType.KTO},
        ))
    
    return insights


# =============================================================================
# DPO-SPECIFIC HEURISTICS
# =============================================================================

def detect_dpo_loss_random(df: pd.DataFrame, window: int = 10) -> List[Insight]:
    """Detect DPO loss stuck at ~0.693 (random chance = ln(2)).
    
    When DPO loss stays near 0.693, the model cannot distinguish chosen from
    rejected responses. This indicates:
    - Learning rate too low
    - Beta too high (KL penalty dominates preference signal)
    - Chosen/rejected pairs too similar or mislabeled
    
    Reference: Rafailov et al. (2023) "DPO", Section 4.2
    """
    if "dpo_loss" not in df.columns:
        return []
    loss = df["dpo_loss"].astype(float)
    if len(loss) < window:
        return []
    
    recent = loss.tail(window)
    mean_loss = float(recent.mean())
    
    # ln(2) ≈ 0.693 is random chance for binary classification
    if abs(mean_loss - 0.693) < 0.02:
        return [Insight(
            type="dpo_loss_random",
            severity="high",
            message="DPO loss stuck at ~0.693 (random chance). Model may not be learning preferences.",
            steps=None,
            data={"mean_recent_loss": mean_loss, "expected_random": 0.693},
            trainer_types={TrainerType.DPO},
            reference="Rafailov et al. (2023) 'DPO', Section 4.2 - Loss at ln(2) indicates no preference signal",
        )]
    return []


def detect_dpo_loss_plateau(df: pd.DataFrame, window: int = 20, 
                            slope_threshold: float = 0.0001) -> List[Insight]:
    """Detect when DPO loss stops decreasing (plateau)."""
    if "dpo_loss" not in df.columns:
        return []
    loss = df["dpo_loss"].astype(float)
    if len(loss) < window:
        return []
    
    recent = loss.tail(window)
    if len(recent) >= 3:
        slope = linregress(np.arange(len(recent)), recent.values).slope
        if abs(slope) < slope_threshold:
            return [Insight(
                type="dpo_loss_plateau",
                severity="low",
                message="DPO loss has plateaued; consider adjusting learning rate or beta.",
                steps=None,
                data={"recent_slope": float(slope)},
                trainer_types={TrainerType.DPO},
            )]
    return []


def detect_dpo_win_rate_instability(df: pd.DataFrame, window: int = 5, 
                                    volatility_threshold: float = 0.3) -> List[Insight]:
    """Detect high variance in DPO win rate (rewards/accuracies).
    
    High win rate volatility indicates the model is inconsistently learning
    preferences, often due to:
    - Small batch sizes
    - Noisy preference data
    - Learning rate too high
    """
    if "win_rate" not in df.columns:
        return []
    win_rate = df["win_rate"].astype(float)
    if len(win_rate) < window:
        return []
    
    rolling_std = win_rate.rolling(window, min_periods=2).std()
    max_std = float(rolling_std.max()) if not rolling_std.isna().all() else 0.0
    
    if max_std > volatility_threshold:
        return [Insight(
            type="win_rate_instability",
            severity="medium",
            message=f"Win rate shows high volatility (std={max_std:.2f}), indicating inconsistent preference learning.",
            steps=None,
            data={"max_rolling_std": max_std, "window": window},
            trainer_types={TrainerType.DPO, TrainerType.ORPO, TrainerType.KTO, TrainerType.CPO},
        )]
    return []


def detect_dpo_margin_collapse(df: pd.DataFrame, window: int = 20, 
                               margin_threshold: float = 0.1) -> List[Insight]:
    """Detect collapse in chosen/rejected reward margin.
    
    When the margin between chosen and rejected rewards becomes too small,
    the model loses signal for preference learning.
    """
    if "reward_margin" not in df.columns and "rewards_chosen" not in df.columns:
        return []
    
    if "reward_margin" in df.columns:
        margin = df["reward_margin"].astype(float)
    else:
        # Compute margin from chosen/rejected
        chosen = df.get("rewards_chosen", df.get("logps_chosen", pd.Series()))
        rejected = df.get("rewards_rejected", df.get("logps_rejected", pd.Series()))
        if len(chosen) == 0 or len(rejected) == 0:
            return []
        margin = chosen - rejected
    
    if len(margin) < window:
        return []
    
    recent_margin = margin.tail(window).mean()
    early_margin = margin.head(window).mean()
    
    if abs(recent_margin) < margin_threshold and abs(early_margin) > margin_threshold * 2:
        return [Insight(
            type="margin_collapse",
            severity="high",
            message=f"Chosen/rejected margin collapsed from {early_margin:.3f} to {recent_margin:.3f}.",
            steps=None,
            data={"early_margin": float(early_margin), "recent_margin": float(recent_margin)},
            trainer_types={TrainerType.DPO, TrainerType.ORPO, TrainerType.CPO},
            reference="Margin collapse often precedes mode collapse in preference learning",
        )]
    return []


def detect_win_rate_plateau(df: pd.DataFrame, key: str = "win_rate", window: int = 80,
                            slope_threshold: float = 0.001) -> List[Insight]:
    """Detect when win rate stops improving (plateau)."""
    if key not in df.columns:
        return []
    series = df[key].astype(float)
    if len(series) < window:
        return []
    
    recent = series.tail(window)
    slope = linregress(np.arange(len(recent)), recent.values).slope
    
    if abs(slope) <= slope_threshold:
        return [Insight(
            type="win_rate_plateau",
            severity="low",
            message="Win-rate plateau detected in recent window.",
            steps=None,
            data={"recent_window": window, "slope": float(slope)},
            trainer_types={TrainerType.DPO, TrainerType.ORPO, TrainerType.KTO, TrainerType.CPO},
        )]
    return []


# =============================================================================
# PPO-SPECIFIC HEURISTICS
# =============================================================================

def detect_ppo_value_head_divergence(df: pd.DataFrame, window: int = 20,
                                     value_loss_threshold: float = 0.5) -> List[Insight]:
    """Detect value head divergence in PPO training.
    
    When value loss is too high, the critic cannot accurately estimate returns,
    leading to high-variance policy gradients and training instability.
    
    Reference: Schulman et al. (2017) "PPO", Section 4
    """
    if "value_loss" not in df.columns:
        return []
    
    value_loss = df["value_loss"].astype(float)
    steps = df["step"].astype(int)
    high_loss_steps = steps[value_loss > value_loss_threshold].tolist()
    
    if high_loss_steps:
        max_loss = float(value_loss.max())
        # Check if value loss is increasing
        recent = value_loss.tail(window)
        slope = linregress(np.arange(len(recent)), recent.values).slope if len(recent) >= 3 else 0.0
        
        return [Insight(
            type="value_head_divergence",
            severity="high" if slope > 0.01 or max_loss > value_loss_threshold * 2 else "medium",
            message=f"Value head diverging (loss={max_loss:.3f}, trend={'increasing' if slope > 0 else 'stable'}).",
            steps=high_loss_steps[:50],
            data={"max_value_loss": max_loss, "slope": float(slope), "threshold": value_loss_threshold},
            trainer_types={TrainerType.PPO},
            reference="Schulman et al. (2017) 'PPO' - High value loss indicates critic underfitting",
        )]
    return []


def detect_ppo_entropy_collapse(df: pd.DataFrame, window: int = 30,
                                entropy_floor: float = 0.1,
                                relative_drop_threshold: float = 0.5) -> List[Insight]:
    """Detect entropy collapse in PPO (policy becoming too deterministic too fast).
    
    Low entropy means the policy is very confident/deterministic. If entropy
    drops too quickly, the policy may be collapsing to a local optimum.
    
    Reference: Mnih et al. (2016) "A3C" - Entropy regularization prevents premature convergence
    """
    if "entropy" not in df.columns:
        return []
    
    entropy = df["entropy"].astype(float)
    if len(entropy) < window:
        return []
    
    initial_entropy = float(entropy.head(window).mean())
    recent_entropy = float(entropy.tail(window).mean())
    
    # Check for absolute floor
    if recent_entropy < entropy_floor:
        return [Insight(
            type="entropy_collapse",
            severity="high",
            message=f"Policy entropy collapsed to {recent_entropy:.3f} (floor: {entropy_floor}).",
            steps=None,
            data={"initial_entropy": initial_entropy, "recent_entropy": recent_entropy,
                  "entropy_floor": entropy_floor},
            trainer_types={TrainerType.PPO},
            reference="Low entropy indicates policy is too deterministic - consider increasing entropy bonus",
        )]
    
    # Check for relative drop
    if initial_entropy > 0:
        drop = (initial_entropy - recent_entropy) / initial_entropy
        if drop > relative_drop_threshold:
            return [Insight(
                type="entropy_collapse",
                severity="medium",
                message=f"Policy entropy dropped {drop*100:.1f}% (from {initial_entropy:.3f} to {recent_entropy:.3f}).",
                steps=None,
                data={"initial_entropy": initial_entropy, "recent_entropy": recent_entropy,
                      "relative_drop": drop},
                trainer_types={TrainerType.PPO},
            )]
    return []


def detect_ppo_advantage_explosion(df: pd.DataFrame, std_threshold: float = 5.0) -> List[Insight]:
    """Detect exploding advantages in PPO training.
    
    Large advantage values indicate the value function is poorly calibrated
    or the reward scale is unstable, leading to destructive policy updates.
    
    Reference: Engstrom et al. (2020) "Implementation Matters in Deep RL"
    """
    if "advantages_std" not in df.columns and "advantages_mean" not in df.columns:
        return []
    
    if "advantages_std" in df.columns:
        adv_std = df["advantages_std"].astype(float)
        steps = df["step"].astype(int)
        high_adv_steps = steps[adv_std > std_threshold].tolist()
        
        if high_adv_steps:
            max_std = float(adv_std.max())
            return [Insight(
                type="advantage_explosion",
                severity="high" if max_std > std_threshold * 2 else "medium",
                message=f"Advantage std exploded to {max_std:.2f} (threshold: {std_threshold}).",
                steps=high_adv_steps[:50],
                data={"max_advantage_std": max_std, "threshold": std_threshold},
                trainer_types={TrainerType.PPO},
                reference="Engstrom et al. (2020) - Normalize advantages to prevent update instability",
            )]
    return []


def detect_ppo_clip_fraction_high(df: pd.DataFrame, window: int = 20,
                                  clip_threshold: float = 0.3) -> List[Insight]:
    """Detect when PPO clip fraction is too high.
    
    High clip fraction means many policy updates are being clipped, indicating:
    - Learning rate too high
    - Policy changing too fast
    - Need for more conservative updates
    
    Reference: Schulman et al. (2017) "PPO" - Clip fraction should typically be < 0.2
    """
    if "clip_fraction" not in df.columns:
        return []
    
    clip_frac = df["clip_fraction"].astype(float)
    steps = df["step"].astype(int)
    high_clip_steps = steps[clip_frac > clip_threshold].tolist()
    
    if high_clip_steps:
        max_clip = float(clip_frac.max())
        mean_clip = float(clip_frac.mean())
        return [Insight(
            type="clip_fraction_high",
            severity="high" if mean_clip > clip_threshold else "medium",
            message=f"PPO clip fraction high (mean={mean_clip:.2f}, max={max_clip:.2f}). Policy changing too fast.",
            steps=high_clip_steps[:50],
            data={"max_clip_fraction": max_clip, "mean_clip_fraction": mean_clip,
                  "threshold": clip_threshold},
            trainer_types={TrainerType.PPO},
            reference="Schulman et al. (2017) 'PPO' - High clip fraction suggests LR too high",
        )]
    return []


def detect_ppo_approx_kl_spike(df: pd.DataFrame, kl_threshold: float = 0.02) -> List[Insight]:
    """Detect approximate KL divergence spikes in PPO.
    
    High approx KL between old and new policy indicates the policy is changing
    too aggressively, which can destabilize training.
    """
    if "approx_kl" not in df.columns:
        return []
    
    approx_kl = df["approx_kl"].astype(float)
    steps = df["step"].astype(int)
    high_kl_steps = steps[approx_kl > kl_threshold].tolist()
    
    if high_kl_steps:
        max_kl = float(approx_kl.max())
        return [Insight(
            type="ppo_approx_kl_spike",
            severity="high" if max_kl > kl_threshold * 3 else "medium",
            message=f"PPO approximate KL spiked to {max_kl:.4f} (threshold: {kl_threshold}).",
            steps=high_kl_steps[:50],
            data={"max_approx_kl": max_kl, "threshold": kl_threshold},
            trainer_types={TrainerType.PPO},
            reference="High approx KL suggests need for early stopping or lower LR",
        )]
    return []


# =============================================================================
# SFT-SPECIFIC HEURISTICS
# =============================================================================

def detect_sft_loss_plateau(df: pd.DataFrame, window: int = 50,
                            slope_threshold: float = 0.0001) -> List[Insight]:
    """Detect when SFT loss stops decreasing."""
    if "sft_loss" not in df.columns:
        return []
    
    loss = df["sft_loss"].astype(float)
    if len(loss) < window:
        return []
    
    recent = loss.tail(window)
    slope = linregress(np.arange(len(recent)), recent.values).slope if len(recent) >= 3 else 0.0
    
    if abs(slope) < slope_threshold:
        return [Insight(
            type="sft_loss_plateau",
            severity="low",
            message="SFT loss has plateaued. Training may have converged or needs LR adjustment.",
            steps=None,
            data={"recent_slope": float(slope), "recent_mean_loss": float(recent.mean())},
            trainer_types={TrainerType.SFT},
        )]
    return []


def detect_sft_perplexity_spike(df: pd.DataFrame, window: int = 10,
                                spike_threshold: float = 2.0) -> List[Insight]:
    """Detect sudden perplexity spikes in SFT training.
    
    Perplexity spikes often indicate:
    - Gradient explosion
    - Bad data batch
    - Learning rate too high
    """
    if "perplexity" not in df.columns:
        return []
    
    perp = df["perplexity"].astype(float)
    if len(perp) < window:
        return []
    
    rolling_mean = perp.rolling(window, min_periods=3).mean()
    rolling_std = perp.rolling(window, min_periods=3).std()
    
    with np.errstate(invalid="ignore"):
        z_scores = (perp - rolling_mean) / (rolling_std + 1e-8)
    
    spike_steps = df.loc[z_scores > spike_threshold, "step"].astype(int).tolist()
    
    if spike_steps:
        max_perp = float(perp.max())
        return [Insight(
            type="perplexity_spike",
            severity="high" if len(spike_steps) > 5 else "medium",
            message=f"Perplexity spiked at {len(spike_steps)} steps (max: {max_perp:.1f}).",
            steps=spike_steps[:50],
            data={"max_perplexity": max_perp, "spike_count": len(spike_steps)},
            trainer_types={TrainerType.SFT},
        )]
    return []


# =============================================================================
# ORPO-SPECIFIC HEURISTICS
# =============================================================================

def detect_orpo_odds_ratio_instability(df: pd.DataFrame, window: int = 10,
                                       volatility_threshold: float = 0.5) -> List[Insight]:
    """Detect instability in ORPO's odds ratio.
    
    Reference: Hong et al. (2024) "ORPO"
    """
    if "log_odds_ratio" not in df.columns:
        return []
    
    lor = df["log_odds_ratio"].astype(float)
    if len(lor) < window:
        return []
    
    rolling_std = lor.rolling(window, min_periods=2).std()
    max_std = float(rolling_std.max()) if not rolling_std.isna().all() else 0.0
    
    if max_std > volatility_threshold:
        return [Insight(
            type="odds_ratio_instability",
            severity="medium",
            message=f"ORPO log odds ratio shows high volatility (std={max_std:.2f}).",
            steps=None,
            data={"max_rolling_std": max_std, "window": window},
            trainer_types={TrainerType.ORPO},
            reference="Hong et al. (2024) 'ORPO' - Odds ratio instability affects preference learning",
        )]
    return []


# =============================================================================
# KTO-SPECIFIC HEURISTICS
# =============================================================================

def detect_kto_imbalance(df: pd.DataFrame, window: int = 20,
                         imbalance_threshold: float = 2.0) -> List[Insight]:
    """Detect imbalance between desirable and undesirable losses in KTO.
    
    Large imbalance may indicate the model is overfitting to one type of example.
    """
    if "desirable_loss" not in df.columns or "undesirable_loss" not in df.columns:
        return []
    
    des = df["desirable_loss"].astype(float).tail(window)
    und = df["undesirable_loss"].astype(float).tail(window)
    
    des_mean = float(des.mean())
    und_mean = float(und.mean())
    
    if und_mean > 0:
        ratio = des_mean / und_mean
        if ratio > imbalance_threshold or ratio < 1 / imbalance_threshold:
            return [Insight(
                type="kto_loss_imbalance",
                severity="medium",
                message=f"KTO desirable/undesirable loss ratio = {ratio:.2f} (imbalanced).",
                steps=None,
                data={"desirable_loss_mean": des_mean, "undesirable_loss_mean": und_mean,
                      "ratio": ratio},
                trainer_types={TrainerType.KTO},
            )]
    return []


# =============================================================================
# GRPO-SPECIFIC HEURISTICS (Group Relative Policy Optimization - DeepSeek)
# =============================================================================

def detect_grpo_group_reward_collapse(df: pd.DataFrame, window: int = 20,
                                       reward_threshold: float = 0.01) -> List[Insight]:
    """Detect collapse in group reward variance.
    
    GRPO relies on reward variance within groups to compute advantages.
    If group rewards become too uniform, learning signal diminishes.
    
    Reference: DeepSeek-R1 paper - GRPO uses group-relative advantages
    """
    if "group_reward_std" not in df.columns:
        return []
    
    reward_std = df["group_reward_std"].astype(float)
    if len(reward_std) < window:
        return []
    
    recent_std = float(reward_std.tail(window).mean())
    early_std = float(reward_std.head(window).mean()) if len(reward_std) > window else recent_std
    
    if recent_std < reward_threshold:
        return [Insight(
            type="grpo_reward_collapse",
            severity="high",
            message=f"GRPO group reward variance collapsed to {recent_std:.4f}. Learning signal may be lost.",
            steps=None,
            data={"recent_reward_std": recent_std, "early_reward_std": early_std,
                  "threshold": reward_threshold},
            trainer_types={TrainerType.GRPO},
            reference="DeepSeek-R1 - GRPO requires reward variance for group-relative advantages",
        )]
    return []


def detect_grpo_advantage_explosion(df: pd.DataFrame, window: int = 20,
                                     advantage_threshold: float = 10.0) -> List[Insight]:
    """Detect exploding advantages in GRPO.
    
    Large advantages can destabilize training and lead to policy collapse.
    """
    if "group_advantage_mean" not in df.columns:
        return []
    
    adv_mean = df["group_advantage_mean"].astype(float).abs()
    high_adv_steps = df.loc[adv_mean > advantage_threshold, "step"].astype(int).tolist()
    
    if high_adv_steps:
        max_adv = float(adv_mean.max())
        return [Insight(
            type="grpo_advantage_explosion",
            severity="high" if max_adv > advantage_threshold * 2 else "medium",
            message=f"GRPO advantages exploding (max={max_adv:.2f}). Consider reducing learning rate.",
            steps=high_adv_steps[:50],
            data={"max_advantage": max_adv, "threshold": advantage_threshold},
            trainer_types={TrainerType.GRPO},
            reference="Large advantages in policy gradient methods cause training instability",
        )]
    return []


def detect_grpo_entropy_collapse(df: pd.DataFrame, window: int = 30,
                                  entropy_floor: float = 0.1,
                                  relative_drop_threshold: float = 0.5) -> List[Insight]:
    """Detect entropy collapse in GRPO (similar to PPO).
    
    GRPO can suffer from the same entropy collapse issues as PPO.
    """
    if "entropy" not in df.columns:
        return []
    
    entropy = df["entropy"].astype(float)
    if len(entropy) < window:
        return []
    
    initial_entropy = float(entropy.head(window).mean())
    recent_entropy = float(entropy.tail(window).mean())
    
    if recent_entropy < entropy_floor:
        return [Insight(
            type="grpo_entropy_collapse",
            severity="high",
            message=f"GRPO policy entropy collapsed to {recent_entropy:.3f}. Model may be too deterministic.",
            steps=None,
            data={"initial_entropy": initial_entropy, "recent_entropy": recent_entropy,
                  "floor": entropy_floor},
            trainer_types={TrainerType.GRPO},
            reference="Entropy collapse in policy gradient methods leads to premature convergence",
        )]
    
    if initial_entropy > 0:
        relative_drop = (initial_entropy - recent_entropy) / initial_entropy
        if relative_drop > relative_drop_threshold:
            return [Insight(
                type="grpo_entropy_drop",
                severity="medium",
                message=f"GRPO entropy dropped {relative_drop*100:.1f}% from initial. Consider entropy bonus.",
                steps=None,
                data={"initial_entropy": initial_entropy, "recent_entropy": recent_entropy,
                      "relative_drop": relative_drop},
                trainer_types={TrainerType.GRPO},
            )]
    return []


def detect_grpo_kl_divergence(df: pd.DataFrame, kl_target: float = 0.1,
                               hard_cap: float = 0.3) -> List[Insight]:
    """Detect KL divergence issues in GRPO.
    
    GRPO can optionally use KL penalty. Monitor for excessive divergence.
    """
    if "kl" not in df.columns:
        return []
    
    kl = df["kl"].astype(float)
    steps = df["step"].astype(int)
    above_cap = steps[kl > hard_cap].tolist()
    
    if above_cap:
        return [Insight(
            type="grpo_kl_divergence",
            severity="high",
            message=f"GRPO KL divergence exceeded {hard_cap:.2f}. Policy may be diverging too fast.",
            steps=above_cap[:50],
            data={"kl_target": kl_target, "hard_cap": hard_cap, "max_kl": float(kl.max())},
            trainer_types={TrainerType.GRPO},
        )]
    return []


def detect_grpo_completion_length_drift(df: pd.DataFrame, window: int = 30,
                                         drift_threshold: float = 0.3) -> List[Insight]:
    """Detect significant drift in completion lengths.
    
    GRPO can cause models to generate increasingly long or short completions
    as a form of reward hacking.
    """
    if "completion_length" not in df.columns:
        return []
    
    lengths = df["completion_length"].astype(float)
    if len(lengths) < window * 2:
        return []
    
    early_mean = float(lengths.head(window).mean())
    recent_mean = float(lengths.tail(window).mean())
    
    if early_mean > 0:
        relative_change = abs(recent_mean - early_mean) / early_mean
        if relative_change > drift_threshold:
            direction = "increased" if recent_mean > early_mean else "decreased"
            return [Insight(
                type="grpo_length_drift",
                severity="medium",
                message=f"GRPO completion length {direction} by {relative_change*100:.1f}%. Check for reward hacking.",
                steps=None,
                data={"early_mean": early_mean, "recent_mean": recent_mean,
                      "relative_change": relative_change},
                trainer_types={TrainerType.GRPO},
                reference="Length manipulation is a common form of reward hacking",
            )]
    return []


def detect_grpo_reward_length_correlation(df: pd.DataFrame, window: int = 50,
                                           correlation_threshold: float = 0.6) -> List[Insight]:
    """Detect if GRPO rewards are correlated with output length.
    
    Strong correlation between reward and length indicates the model may be
    gaming the reward by producing longer/shorter outputs rather than
    genuinely improving quality.
    
    Reference: Singhal et al. (2023) 'A Long Way to Go' - length bias in RLHF
    """
    reward_col = None
    length_col = None
    
    # Find reward column
    for col in ["reward_mean", "group_reward_mean", "rewards"]:
        if col in df.columns:
            reward_col = col
            break
    
    # Find length column
    for col in ["completion_length", "output_length_mean", "response_length"]:
        if col in df.columns:
            length_col = col
            break
    
    if reward_col is None or length_col is None:
        return []
    
    if len(df) < window:
        return []
    
    recent = df.tail(window)
    rewards = recent[reward_col].astype(float)
    lengths = recent[length_col].astype(float)
    
    # Compute correlation
    if rewards.std() < 1e-8 or lengths.std() < 1e-8:
        return []
    
    correlation = rewards.corr(lengths)
    
    if abs(correlation) > correlation_threshold:
        direction = "longer" if correlation > 0 else "shorter"
        return [Insight(
            type="grpo_reward_length_correlation",
            severity="high" if abs(correlation) > 0.8 else "medium",
            message=f"Strong reward-length correlation ({correlation:.2f}). "
                    f"Model may be gaming reward by producing {direction} outputs.",
            steps=None,
            data={
                "correlation": float(correlation),
                "threshold": correlation_threshold,
                "reward_col": reward_col,
                "length_col": length_col,
            },
            trainer_types={TrainerType.GRPO, TrainerType.PPO},
            reference="Singhal et al. (2023) 'A Long Way to Go' - Length bias in RLHF",
        )]
    return []


def detect_grpo_clip_ratio(df: pd.DataFrame, window: int = 20,
                           clip_threshold: float = 0.25) -> List[Insight]:
    """Detect high clip ratio in GRPO (similar to PPO).
    
    GRPO uses importance sampling with clipping. High clip ratios indicate
    the policy is changing too aggressively between updates.
    
    Reference: DeepSeek-R1 - GRPO uses PPO-style clipping
    """
    clip_col = None
    for col in ["clip_fraction", "clip_ratio", "grpo_clip_fraction"]:
        if col in df.columns:
            clip_col = col
            break
    
    if clip_col is None:
        return []
    
    clip = df[clip_col].astype(float)
    if len(clip) < window:
        return []
    
    recent_mean = float(clip.tail(window).mean())
    max_clip = float(clip.max())
    
    if recent_mean > clip_threshold:
        return [Insight(
            type="grpo_clip_ratio_high",
            severity="high" if recent_mean > clip_threshold * 1.5 else "medium",
            message=f"GRPO clip ratio high (mean={recent_mean:.2f}, max={max_clip:.2f}). "
                    "Policy updates may be too aggressive.",
            steps=None,
            data={
                "recent_clip_mean": recent_mean,
                "max_clip": max_clip,
                "threshold": clip_threshold,
            },
            trainer_types={TrainerType.GRPO},
            reference="High clip ratios suggest learning rate may be too high",
        )]
    return []


def detect_grpo_loss_divergence(df: pd.DataFrame, window: int = 30,
                                 divergence_threshold: float = 0.5) -> List[Insight]:
    """Detect GRPO loss increasing (training diverging).
    
    If GRPO loss starts increasing after initial decrease, training may be
    diverging due to reward hacking or optimization instability.
    """
    loss_col = None
    for col in ["grpo_loss", "loss", "policy_loss"]:
        if col in df.columns:
            loss_col = col
            break
    
    if loss_col is None:
        return []
    
    loss = df[loss_col].astype(float)
    if len(loss) < window * 2:
        return []
    
    # Compare early vs recent loss trend
    early_mean = float(loss.head(window).mean())
    recent_mean = float(loss.tail(window).mean())
    
    # Check if loss is trending upward in recent window
    recent = loss.tail(window)
    slope = linregress(np.arange(len(recent)), recent.values).slope
    
    # Flag if recent loss is higher than early AND trending up
    if recent_mean > early_mean and slope > 0.001:
        increase_pct = (recent_mean - early_mean) / (abs(early_mean) + 1e-8) * 100
        if increase_pct > divergence_threshold * 100:
            return [Insight(
                type="grpo_loss_divergence",
                severity="high",
                message=f"GRPO loss increased {increase_pct:.1f}% and still rising. "
                        "Training may be diverging.",
                steps=None,
                data={
                    "early_loss": early_mean,
                    "recent_loss": recent_mean,
                    "increase_pct": increase_pct,
                    "recent_slope": float(slope),
                },
                trainer_types={TrainerType.GRPO},
                reference="Rising policy loss often indicates reward hacking or optimization issues",
            )]
    return []


def detect_grpo_response_diversity_collapse(df: pd.DataFrame, window: int = 30,
                                             diversity_threshold: float = 0.3) -> List[Insight]:
    """Detect collapse in response diversity within GRPO groups.
    
    GRPO generates multiple responses per prompt and ranks them. If responses
    become too similar (low diversity), the group-relative advantage signal weakens.
    
    Tracked via: unique n-grams, edit distance variance, or embedding variance.
    """
    diversity_col = None
    for col in ["response_diversity", "group_diversity", "intra_group_variance", 
                "unique_ngram_ratio", "embedding_variance"]:
        if col in df.columns:
            diversity_col = col
            break
    
    if diversity_col is None:
        return []
    
    diversity = df[diversity_col].astype(float)
    if len(diversity) < window * 2:
        return []
    
    early_diversity = float(diversity.head(window).mean())
    recent_diversity = float(diversity.tail(window).mean())
    
    if early_diversity > 0:
        relative_drop = (early_diversity - recent_diversity) / early_diversity
        if relative_drop > diversity_threshold:
            return [Insight(
                type="grpo_diversity_collapse",
                severity="high" if relative_drop > 0.5 else "medium",
                message=f"GRPO response diversity dropped {relative_drop*100:.1f}%. "
                        "Group-relative advantages may be weakening.",
                steps=None,
                data={
                    "early_diversity": early_diversity,
                    "recent_diversity": recent_diversity,
                    "relative_drop": relative_drop,
                },
                trainer_types={TrainerType.GRPO},
                reference="Response diversity is critical for GRPO's group-relative ranking",
            )]
    return []


# =============================================================================
# REWARD MODEL HEURISTICS
# =============================================================================

def detect_reward_model_imbalance(df: pd.DataFrame) -> List[Insight]:
    """Detect reward model imbalance (high reward with concerning signals)."""
    if "reward_mean" not in df.columns or "kl" not in df.columns:
        return []
    
    reward = df["reward_mean"].astype(float)
    kl = df["kl"].astype(float)
    refusal = df["refusal_rate"].astype(float) if "refusal_rate" in df.columns else pd.Series(np.zeros(len(df)))
    
    high_reward = reward > (reward.rolling(50, min_periods=20).mean() + reward.rolling(50, min_periods=20).std())
    high_kl = kl > 0.2
    rising_refusal = refusal.diff().rolling(10, min_periods=3).mean() > 0.0
    
    flags = (high_reward & (high_kl | rising_refusal))
    steps = df.loc[flags, "step"].astype(int).tolist()
    
    if steps:
        return [Insight(
            type="reward_model_imbalance",
            severity="medium",
            message="Signals of reward model imbalance (high reward with high KL or rising refusals).",
            steps=steps[:50],
            data={"num_flags": len(steps)},
            trainer_types={TrainerType.PPO, TrainerType.DPO},
        )]
    return []


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

# Heuristic registry by trainer type
COMMON_HEURISTICS = [
    detect_reward_variance_spikes,
    detect_kl_instability,
    detect_policy_drift,
    detect_slice_degradation,
    detect_output_length_collapse,
    detect_refusal_regressions,
    detect_instability_window,
    detect_reward_model_imbalance,
    detect_reward_hacking,
    detect_mode_collapse,
    detect_gradient_issues,
]

DPO_HEURISTICS = [
    detect_dpo_loss_random,
    detect_dpo_loss_plateau,
    detect_dpo_win_rate_instability,
    detect_dpo_margin_collapse,
    detect_win_rate_plateau,
]

PPO_HEURISTICS = [
    detect_ppo_value_head_divergence,
    detect_ppo_entropy_collapse,
    detect_ppo_advantage_explosion,
    detect_ppo_clip_fraction_high,
    detect_ppo_approx_kl_spike,
]

SFT_HEURISTICS = [
    detect_sft_loss_plateau,
    detect_sft_perplexity_spike,
]

ORPO_HEURISTICS = [
    detect_orpo_odds_ratio_instability,
    detect_dpo_win_rate_instability,  # Also applicable to ORPO
    detect_dpo_margin_collapse,  # Also applicable to ORPO
    detect_win_rate_plateau,
]

KTO_HEURISTICS = [
    detect_kto_imbalance,
    detect_dpo_win_rate_instability,  # Also applicable to KTO
    detect_win_rate_plateau,
]

CPO_HEURISTICS = [
    detect_dpo_win_rate_instability,  # Similar preference learning
    detect_dpo_margin_collapse,
    detect_win_rate_plateau,
]

GRPO_HEURISTICS = [
    detect_grpo_group_reward_collapse,
    detect_grpo_advantage_explosion,
    detect_grpo_entropy_collapse,
    detect_grpo_kl_divergence,
    detect_grpo_completion_length_drift,
    detect_grpo_reward_length_correlation,
    detect_grpo_clip_ratio,
    detect_grpo_loss_divergence,
    detect_grpo_response_diversity_collapse,
]


def run_heuristics(df: pd.DataFrame, trainer_type: str = TrainerType.UNKNOWN) -> List[Insight]:
    """Run heuristics appropriate for the detected trainer type.
    
    Args:
        df: DataFrame with training metrics
        trainer_type: One of 'dpo', 'ppo', 'sft', 'orpo', 'kto', 'cpo', 'unknown'
        
    Returns:
        List of Insight objects, sorted by severity (high → medium → low)
    """
    insights: List[Insight] = []
    
    # Always run common heuristics
    for heuristic in COMMON_HEURISTICS:
        insights.extend(heuristic(df))
    
    # Run trainer-specific heuristics
    if trainer_type == TrainerType.DPO:
        for heuristic in DPO_HEURISTICS:
            insights.extend(heuristic(df))
    elif trainer_type == TrainerType.PPO:
        for heuristic in PPO_HEURISTICS:
            insights.extend(heuristic(df))
    elif trainer_type == TrainerType.SFT:
        for heuristic in SFT_HEURISTICS:
            insights.extend(heuristic(df))
    elif trainer_type == TrainerType.ORPO:
        for heuristic in ORPO_HEURISTICS:
            insights.extend(heuristic(df))
    elif trainer_type == TrainerType.KTO:
        for heuristic in KTO_HEURISTICS:
            insights.extend(heuristic(df))
    elif trainer_type == TrainerType.CPO:
        for heuristic in CPO_HEURISTICS:
            insights.extend(heuristic(df))
    elif trainer_type == TrainerType.GRPO:
        for heuristic in GRPO_HEURISTICS:
            insights.extend(heuristic(df))
    else:
        # Unknown trainer - run all heuristics for best coverage
        for heuristic in DPO_HEURISTICS + PPO_HEURISTICS + SFT_HEURISTICS + GRPO_HEURISTICS:
            insights.extend(heuristic(df))
    
    # Sort by severity
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    insights.sort(key=lambda x: severity_rank.get(x.severity, 3))
    
    return insights


def run_all_heuristics(df: pd.DataFrame) -> List[Insight]:
    """Run all heuristics regardless of trainer type (backward compatibility).
    
    Deprecated: Use run_heuristics(df, trainer_type) for trainer-aware detection.
    """
    return run_heuristics(df, trainer_type=TrainerType.UNKNOWN)
