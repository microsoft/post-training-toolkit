# HuggingFace TRL Demo - Presentation Guide

## Quick Reference

**Duration:** ~10 minutes
**Format:** Screen share + recording
**Audience:** TRL team (async-friendly)

---

## Demo Script

### 0:00-0:30 - Hook with Their Pain Point

> "Quentin mentioned a GRPO experiment where `importance_sampling_ratio` was ~0 instead of ~1. By the time you noticed, you'd already burned GPU hours. This toolkit would have caught that during training."

Show the output:
```
🚨 [HIGH] grpo_importance_sampling_ratio_anomaly
   IS ratio at 0.102 (should be ~1.0). Policy may have drifted too far.
```

---

### 0:30-2:30 - Integration (Key Point #1)

**Message: "One line, zero config"**

```python
# This is the entire integration:
trainer = GRPOTrainer(
    model=model,
    callbacks=[DiagnosticsCallback()],  # <-- Just this
    ...
)
```

**What happens automatically:**
- Detects it's a GRPOTrainer
- Captures GRPO-specific metrics (group_reward, completion_length, etc.)
- Runs GRPO-specific heuristics
- Warns you during training if something looks wrong

**Show:** Run `python demo/huggingface_demo.py` - the first section

---

### 2:30-5:00 - The GRPO Example (Their Use Case)

**Message: "This would have saved you GPU hours"**

Walk through:
1. Simulate the problematic run (IS ratio drops from ~1.0 to ~0.1)
2. Show the toolkit detecting it
3. Explain WHY it's a problem (policy drift = high variance gradients)

**Key quote to use:**
> "The toolkit doesn't just alert on metrics crossing thresholds - it encodes domain knowledge. It knows that IS ratio ~0 means your policy has drifted so far that the old trajectories are useless."

---

### 5:00-7:30 - Contributing Heuristics (Key Point #2)

**Message: "Adding a heuristic = writing a YAML file"**

Show the YAML:
```yaml
name: grpo_importance_sampling_ratio_anomaly
trainers: [grpo, ppo]
metric: importance_sampling_ratio
condition: "< 0.5"
severity: high
message: "IS ratio at {value:.3f} (should be ~1.0)"
```

**Explain the condition DSL:**
- `< 0.5` - simple threshold
- `range(0.68, 0.71)` - stuck in range (DPO loss at random chance)
- `drop(50%)` - value dropped from baseline
- `spike(3x)` - sudden spike above average

**Why this matters for HuggingFace:**
> "Anyone in the community can contribute a heuristic. They don't need to understand the codebase. Just: 'I know this metric should be X, and if it's Y, that's bad.' Write YAML, open PR."

---

### 7:30-8:30 - Inline Alerts (For Experimentation)

**Message: "Even faster for quick experiments"**

```python
DiagnosticsCallback(
    custom_alerts=[
        "grpo: importance_sampling_ratio < 0.5 -> high: IS ratio collapsed",
    ]
)
```

> "No file needed. Just a string. Perfect for 'let me quickly monitor this thing I'm worried about.'"

---

### 8:30-10:00 - Production Features + Wrap-up

**Quick mentions (don't deep-dive, just list):**
- Resume validation (catches config drift when resuming)
- Postmortem recording (crash diagnostics)
- Distributed training support (straggler detection)
- WandB/MLflow/TensorBoard integration

**Closing:**
> "This is production-ready today. It works with all TRL trainers. And the YAML system makes it easy for the community to contribute domain knowledge without writing Python."

---

## Anticipated Questions & Answers

### "How does this compare to WandB alerts?"

> "WandB alerts are generic metric thresholds. This is trainer-aware. It knows DPO loss at 0.693 means random chance. It knows IS ratio should be ~1. It's domain knowledge encoded in code. And it works offline - you don't need an external service."

### "What's the overhead?"

> "Minimal. We're just logging metrics that TRL already computes, plus lightweight heuristics every N steps. In our benchmarks, <1% overhead. You can also disable features you don't need."

### "How would first-party integration work?"

> "Cleanest option: `pip install trl[diagnostics]`. The callback uses the standard TrainerCallback interface, so it's already compatible. The YAML heuristics could live in trl/diagnostics/ and be community-maintained."

### "What about custom trainers?"

> "Auto-detection falls back gracefully. Unknown trainers get all heuristics run. Custom YAML heuristics can target specific setups."

### "Why YAML instead of Python for heuristics?"

> "Lower barrier to contribution. ML researchers know 'this metric should be X.' They shouldn't need to understand the codebase to share that knowledge. YAML captures the intent, we handle the execution."

---

## Key Messages to Hit

1. **"One line to integrate"** - Zero friction adoption
2. **"Would have caught your GRPO bug"** - Solves real problems
3. **"YAML = anyone can contribute"** - Community scalability
4. **"Trainer-aware, not just thresholds"** - Domain knowledge
5. **"Production-ready today"** - Not a prototype

---

## Files to Have Open

1. `demo/huggingface_demo.py` - Run this for the live demo
2. `post_training_toolkit/heuristics/builtin/grpo/importance_sampling_ratio.yaml` - Show this
3. `post_training_toolkit/integrations/trl.py` - If they ask about implementation

---

## Backup: If Something Breaks

Just run the individual pieces:

```python
# Test YAML loading
from post_training_toolkit.heuristics.loader import HeuristicLoader
loader = HeuristicLoader()
print(loader.load_for_trainer("grpo"))

# Test inline alerts
from post_training_toolkit import DiagnosticsCallback
cb = DiagnosticsCallback(custom_alerts=["grpo: x < 0.5 -> high: Test"])
print(cb._custom_alerts)
```

---

## Post-Demo Follow-up

Offer:
1. Share the repo for them to try
2. Open an issue/discussion about first-party integration
3. Happy to iterate on the API based on their feedback

---

## Recording Notes

Since this will be recorded for async viewing:
- Speak clearly, assume no context
- Pause after key points
- Keep terminal font large
- Consider adding captions/annotations if editing
