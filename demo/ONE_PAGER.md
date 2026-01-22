# Post-Training Toolkit for TRL

**Production-grade diagnostics for RLHF training with Hugging Face TRL**

---

## The Problem

Training LLMs with RLHF (DPO, PPO, GRPO, etc.) is expensive and opaque:
- **Silent failures**: Training can look fine while the model collapses
- **Wasted GPU hours**: Issues caught late = compute burned
- **Tribal knowledge**: Knowing "DPO loss at 0.693 = random chance" isn't documented anywhere

## The Solution

```python
from post_training_toolkit import DiagnosticsCallback

trainer = GRPOTrainer(
    model=model,
    callbacks=[DiagnosticsCallback()],  # One line. Zero config.
    ...
)
```

**You get:**
- Live warnings during training when something looks wrong
- Trainer-specific diagnostics (DPO ≠ PPO ≠ GRPO)
- Domain knowledge encoded in heuristics
- Artifacts for post-hoc analysis

---

## Example: GRPO Importance Sampling Ratio

**The bug:** `importance_sampling_ratio` was ~0 instead of ~1. Noticed late, GPU hours wasted.

**With the toolkit:**
```
🚨 [HIGH] grpo_importance_sampling_ratio_anomaly
   IS ratio at 0.102 (should be ~1.0). Policy may have drifted too far.
```

Caught during training. Not after.

---

## Easy Contribution (YAML Heuristics)

Anyone can add a heuristic without writing Python:

```yaml
# heuristics/builtin/grpo/importance_sampling_ratio.yaml
name: grpo_importance_sampling_ratio_anomaly
trainers: [grpo, ppo]
metric: importance_sampling_ratio
condition: "< 0.5"
severity: high
message: "IS ratio at {value:.3f} (should be ~1.0)"
```

**Condition DSL:**
| Syntax | Meaning |
|--------|---------|
| `< 0.5` | Below threshold |
| `range(0.68, 0.71)` | Stuck in range |
| `drop(50%)` | Dropped 50% from baseline |
| `spike(3x)` | 3x above rolling average |

---

## What's Included

| Feature | Description |
|---------|-------------|
| **Auto-detection** | Knows DPO vs PPO vs GRPO automatically |
| **80+ heuristics** | DPO loss random, entropy collapse, margin collapse, etc. |
| **YAML contributions** | Add heuristics without Python |
| **Resume validation** | Detects config drift when resuming |
| **Postmortem recording** | Crash diagnostics |
| **Distributed training** | Straggler detection, memory tracking |
| **Tracker integration** | WandB, MLflow, TensorBoard |

---

## Why First-Party Integration?

1. **Zero friction** - One line to adopt
2. **Community scalable** - YAML heuristics = low barrier to contribute
3. **Domain knowledge** - Encodes tribal knowledge in code
4. **Already works** - Supports all TRL trainers today
5. **Minimal deps** - numpy, pandas, scipy, pyyaml

---

## Quick Start

```bash
pip install post-training-toolkit
```

```python
from post_training_toolkit import DiagnosticsCallback
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    callbacks=[DiagnosticsCallback(
        custom_alerts=["grpo: importance_sampling_ratio < 0.5 -> high: IS collapsed"]
    )],
    ...
)
trainer.train()
```

---

## Links

- **Repo:** [github link]
- **Demo:** `python demo/huggingface_demo.py`
- **Docs:** See README.md
