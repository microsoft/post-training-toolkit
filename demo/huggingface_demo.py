#!/usr/bin/env python3
"""
HuggingFace TRL Team Demo
=========================

This script demonstrates the Post-Training Toolkit's key features:
1. Zero-config integration (one line of code)
2. Easy contribution via YAML heuristics

Run: python demo/huggingface_demo.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# DEMO 1: Zero-Config Integration
# =============================================================================

print("=" * 70)
print("DEMO 1: Zero-Configuration Integration")
print("=" * 70)

print("""
Integration is ONE line of code:

    from post_training_toolkit import DiagnosticsCallback

    trainer = GRPOTrainer(
        model=model,
        callbacks=[DiagnosticsCallback()],  # <-- Just this
        ...
    )
    trainer.train()

That's it. No configuration needed. The callback:
- Auto-detects trainer type (DPO, PPO, SFT, ORPO, KTO, CPO, GRPO)
- Captures trainer-specific metrics automatically
- Runs heuristics during training and warns you
- Generates artifacts for post-hoc analysis
""")

# Show auto-detection
from post_training_toolkit.integrations.trl import TRAINER_CLASS_MAP
print("Supported trainers (auto-detected):")
for trainer_class, trainer_type in sorted(set((k, v) for k, v in TRAINER_CLASS_MAP.items() if "Trainer" in k)):
    print(f"  • {trainer_class} → {trainer_type}")

# =============================================================================
# DEMO 2: The GRPO Importance Sampling Ratio Issue
# =============================================================================

print("\n" + "=" * 70)
print("DEMO 2: Catching the GRPO Importance Sampling Ratio Bug")
print("=" * 70)

print("""
Quentin mentioned: "importance_sampling_ratio wasn't close to 1 (it was mostly ~0)"

This is a classic issue! The IS ratio (π_new/π_old) should be ~1.0.
When it's ~0, the policy has drifted too far between updates.

Let's simulate this scenario and see how the toolkit catches it:
""")

# Simulate the problematic GRPO run
np.random.seed(42)
steps = 100

# Scenario: IS ratio starts okay then collapses to ~0
is_ratio_good = np.random.normal(1.0, 0.1, 50)  # First 50 steps: healthy ~1.0
is_ratio_bad = np.random.normal(0.1, 0.05, 50)   # Last 50 steps: collapsed to ~0
is_ratio_bad = np.clip(is_ratio_bad, 0.01, 0.3)  # Keep it in problematic range

df_grpo = pd.DataFrame({
    "step": list(range(steps)),
    "importance_sampling_ratio": np.concatenate([is_ratio_good, is_ratio_bad]),
    "grpo_loss": np.random.normal(0.5, 0.1, steps),
    "reward_mean": np.linspace(0.1, 0.3, steps) + np.random.normal(0, 0.02, steps),
})

print("Simulated GRPO metrics:")
print(f"  Steps 0-50:  IS ratio mean = {df_grpo['importance_sampling_ratio'].iloc[:50].mean():.3f} (healthy)")
print(f"  Steps 50-100: IS ratio mean = {df_grpo['importance_sampling_ratio'].iloc[50:].mean():.3f} (PROBLEM!)")

# Run heuristics
from post_training_toolkit.models.heuristics import run_heuristics

print("\nRunning diagnostics...")
insights = run_heuristics(df_grpo, trainer_type="grpo")

print(f"\nDetected {len(insights)} issues:")
for insight in insights:
    icon = {"high": "🚨", "medium": "⚠️", "low": "ℹ️"}.get(insight.severity, "•")
    source = insight.data.get("source", "python") if insight.data else "python"
    print(f"  {icon} [{insight.severity.upper()}] {insight.type}")
    print(f"     {insight.message}")
    if insight.reference:
        print(f"     Ref: {insight.reference}")

# =============================================================================
# DEMO 3: Contributing a New Heuristic (YAML)
# =============================================================================

print("\n" + "=" * 70)
print("DEMO 3: Contributing Heuristics (No Python Required!)")
print("=" * 70)

print("""
Adding a new heuristic is as simple as creating a YAML file:

    # post_training_toolkit/heuristics/builtin/grpo/importance_sampling_ratio.yaml

    name: grpo_importance_sampling_ratio_anomaly
    description: Detect when IS ratio deviates from 1.0
    trainers: [grpo, ppo]
    metric: importance_sampling_ratio
    condition: "< 0.5"
    window: 20
    severity: high
    message: "IS ratio at {value:.3f} (should be ~1.0)"
    reference: "Schulman et al. (2017) 'PPO'"
    min_steps: 30
    enabled: true

That's it! The condition DSL supports:
  • Comparisons: < 0.5, > 2.0, <= 0.1, >= 0.9, == 0.693
  • Range: range(0.68, 0.71) - for "stuck in range" detection
  • Drop: drop(20%) - value dropped 20% from baseline
  • Spike: spike(3x) - value 3x above rolling average
""")

# Show the YAML heuristics that are loaded
from post_training_toolkit.heuristics.loader import HeuristicLoader

loader = HeuristicLoader()
grpo_heuristics = loader.load_for_trainer("grpo")

print(f"YAML heuristics loaded for GRPO ({len(grpo_heuristics)}):")
for h in grpo_heuristics:
    print(f"  • {h.name}: {h.condition}")

# =============================================================================
# DEMO 4: Inline Custom Alerts (For Quick Experiments)
# =============================================================================

print("\n" + "=" * 70)
print("DEMO 4: Inline Custom Alerts (No Files Needed)")
print("=" * 70)

print("""
For quick experiments, you can define alerts inline:

    DiagnosticsCallback(
        custom_alerts=[
            "grpo: importance_sampling_ratio < 0.5 -> high: IS ratio collapsed",
            "grpo: entropy drop(50%) for 30 steps -> medium: Entropy dropping",
        ]
    )

Syntax: trainer: metric condition [for N steps] -> severity: message
""")

# Demonstrate inline parsing
from post_training_toolkit.heuristics.inline import parse_inline_alert

examples = [
    "grpo: importance_sampling_ratio < 0.5 -> high: IS ratio collapsed",
    "dpo: dpo_loss == 0.693 -> high: Loss stuck at random chance",
    "ppo: entropy drop(50%) for 30 steps -> medium: Entropy collapsing",
]

print("Parsed inline alerts:")
for ex in examples:
    h = parse_inline_alert(ex)
    if h:
        print(f"  ✓ {h.metric} {h.condition} → {h.severity}")

# =============================================================================
# DEMO 5: Production Features
# =============================================================================

print("\n" + "=" * 70)
print("DEMO 5: Production-Ready Features")
print("=" * 70)

print("""
The toolkit includes production features out of the box:

  ✓ Resume Validation - Detects config changes when resuming from checkpoint
  ✓ Postmortem Recording - Automatic crash/interrupt diagnostics
  ✓ Safe Stopping - Optional auto-stop on NaN/Inf/critical issues
  ✓ Distributed Training - Straggler detection, memory imbalance tracking
  ✓ Experiment Trackers - WandB, MLflow, TensorBoard integration
  ✓ Behavior Snapshots - Track model outputs on fixed prompts over training
  ✓ Auto-Diff - Compare snapshots to detect drift, refusal changes

Example with all features:

    DiagnosticsCallback(
        run_dir="my_run",

        # Snapshots (track behavior drift)
        enable_snapshots=True,
        snapshot_interval=100,

        # Safety
        stop_on_critical=True,  # Stop on NaN/Inf
        enable_resume_validation=True,

        # Tracking
        experiment_tracker="wandb",
        experiment_project="my-grpo-run",

        # Custom monitoring
        custom_alerts=["grpo: importance_sampling_ratio < 0.5 -> high: IS collapsed"],
    )
""")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: Why First-Party Integration Makes Sense")
print("=" * 70)

print("""
1. ZERO FRICTION ADOPTION
   - One line to integrate
   - No configuration required
   - Works with all TRL trainers today

2. COMMUNITY CONTRIBUTIONS
   - YAML heuristics = no Python required
   - Anyone can contribute domain knowledge
   - Easy to maintain and review

3. TRAINER-AWARE DIAGNOSTICS
   - Not just generic metric alerts
   - Knows DPO loss at 0.693 = random chance
   - Knows GRPO needs IS ratio ~1.0
   - Domain knowledge encoded in heuristics

4. PRODUCTION READY
   - Resume validation prevents silent config drift
   - Postmortem recording for crash analysis
   - Distributed training support
   - Integrates with existing trackers

5. MINIMAL DEPENDENCIES
   - numpy, pandas, matplotlib, seaborn, scipy, pyyaml
   - No heavy ML dependencies in core
   - Optional: transformers, trl, torch for callback

The GRPO importance_sampling_ratio issue?
With this toolkit, you'd see a warning during training:

    🚨 [HIGH] grpo_importance_sampling_ratio_anomaly
       IS ratio at 0.102 (should be ~1.0). Policy may have drifted too far.

Before wasting more GPU hours.
""")

print("=" * 70)
print("Demo complete! Questions?")
print("=" * 70)
