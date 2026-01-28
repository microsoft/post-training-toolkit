# Post-Training Toolkit

Diagnostics and training control for long-horizon RL with TRL.

Add one callback to your TRL trainer to get live heuristics, failure-conditioned control, distributed profiling, and reproducible artifacts for post-training runs (DPO, PPO, GRPO, SFT, ORPO, KTO, CPO).

---

## What this is

Post-Training Toolkit (PTT) is a production-style **diagnostics and control layer** for TRL trainers, designed for long-running and expensive RL jobs where training correctness matters.

It makes training dynamics observable and enables deterministic intervention when runs drift into unreliable regimes.

**Key capabilities**
- Auto-detecting integration for TRL trainers
- Live heuristics over training metrics (Python + YAML)
- Failure-conditioned training control on critical or high-severity issues
- Behavior snapshots and diffs to track drift and refusal changes
- Distributed profiling (stragglers, throughput, memory balance)
- Auditable artifacts for real training platforms

This is infrastructure you build after you’ve burned GPU hours and don’t want to debug bad runs in the dark again.

## Why it matters

Post-training failures rarely appear as clean crashes. More often, training silently degrades long before aggregate metrics flag a problem.

PTT is built to:
- Detect NaNs, divergence, margin collapse, KL spikes, reward variance spikes, and truncation
- Make long unattended and multi-GPU runs debuggable via snapshots, diffs, and postmortems
- Encode paper-inspired and practitioner heuristics in extensible YAML + Python
- Produce reproducible artifacts that plug into experiment trackers

With `stop_on_critical=True`, clearly invalid runs can be halted early and annotated with the last known-stable step.

---

## Installation

```bash
pip install post-training-toolkit
````

PTT assumes you are using Hugging Face TRL for post-training.

## Quick start

Add a single callback to any TRL trainer to enable diagnostics and control:

```python
from post_training_toolkit import DiagnosticsCallback
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[
        DiagnosticsCallback(
            run_dir="ptt_run",
            stop_on_critical=True,      # failure-conditioned training control
            enable_live_warnings=True,  # emit live heuristic warnings
        )
    ],
)

trainer.train()
```

**Supported trainers**
DPOTrainer, PPOTrainer (and PPOv2), GRPOTrainer, SFTTrainer, ORPOTrainer, KTOTrainer, CPOTrainer.

At the end of a run, PTT produces:

* `metrics.jsonl` – step-level trainer-aware metrics
* `run_metadata*.json` – immutable provenance
* `snapshots/` and `diffs/` – optional behavior tracking
* `postmortem.json` – crash or invalidation context
* `reports/` – auto-generated diagnostics summaries

---

## Heuristics and training control

PTT maintains a rolling metric history and evaluates Python and YAML-defined heuristics against it.

* Built-in heuristics for common RL failure modes
* Trainer-specific YAML rules
* Custom rules via inline definitions or a heuristics directory

```python
DiagnosticsCallback(
    run_dir="ptt_run",
    stop_on_critical=True,
    custom_heuristics_dir="./my_heuristics",
)
```

High-severity findings emit prominent warnings and can request control actions through TRL’s `TrainerControl` interface.

## Distributed training

The same callback works transparently with `torchrun` or Accelerate:

* Aggregates metrics across ranks
* Detects stragglers and slowdown
* Tracks GPU memory balance and OOM risk
* Writes artifacts only on the main process

## Agent traces and datasets

PTT also supports agent-trace analysis and dataset construction for outcome-driven post-training:

```python
from post_training_toolkit.agents import AgentRunLog, analyze_runs, to_preference_pairs
```

## CLI

```bash
ptt-diagnose --input ./my_run --make-plots
ptt-agent-diagnose --input agent_runs.jsonl --export-dpo pairs.parquet
```

## License

MIT
