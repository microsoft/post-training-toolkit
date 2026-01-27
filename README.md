# Post-Training Toolkit

Frontier-lab style diagnostics and guardrails for TRL training.

Add one callback to your TRL trainer to get live heuristics, optional safe stopping, distributed profiling, and reproducible artifacts for post-training runs (DPO, PPO, GRPO, SFT, ORPO, KTO, CPO).

---

## What this is (systems lens)

A production-style diagnostics and guardrail layer for TRL trainers:

- Auto-detecting integration for TRL trainers (DPO, PPO, GRPO, SFT, ORPO, KTO, CPO)
- Live heuristics over training metrics (Python + YAML)
- Optional safe stopping on critical failures or high-severity issues
- Behavior snapshots and diffs to track model drift and refusal changes
- Distributed profiling (stragglers, throughput, memory balance)
- Provenance and artifacts designed for real training platforms

This is the kind of infrastructure you build after you’ve burned a lot of GPU hours and never want to debug a bad run in the dark again.

## Why it matters operationally

Post-training runs fail in ways that standard logs don’t surface until it’s too late. The toolkit is built to catch and explain those failure modes:

- Prevents silent failures with automatic detection of NaNs, divergence, margin collapse, KL spikes, reward variance spikes, and truncated outputs
- Makes long unattended and multi-GPU jobs debuggable via snapshots, diffs, postmortems, and resume validation
- Encodes both paper-inspired and practitioner heuristics in YAML + Python, so you can extend it with your own lab’s rules
- Produces auditable artifacts (metrics JSONL, metadata, snapshots, diffs, postmortems, reports) that plug cleanly into experiment trackers

With `stop_on_critical=True`, the callback can also stop clearly doomed runs early and record a recommended resume step.

## How it’s engineered (for people who build systems)

The interesting part is under the hood:

- Auto-detecting TRL integration with real CPU-level integration tests (see `tests/test_trl_integration.py`)
- YAML heuristics engine over pandas DataFrames (see `post_training_toolkit/heuristics/executor.py`) plus a suite of built-in heuristics for DPO, PPO, GRPO, etc.
- Distributed straggler and memory detection, plus throughput and slowdown profiling (see `post_training_toolkit/models/distributed` and `post_training_toolkit/models/profiling`)
- Artifact, checkpoint, and postmortem design you can plug into a real training platform (see `post_training_toolkit/models/artifacts.py`, `checkpoints.py`, `postmortem.py`)

If you’re evaluating this as a codebase, those modules are where to start reading.

---

## Installation

```bash
pip install post-training-toolkit
```

The toolkit assumes you are using Hugging Face TRL for post-training (DPO, PPO, GRPO, SFT, ORPO, KTO, CPO).

## Quick start: diagnostics for TRL

Add a single callback to any TRL trainer to get metrics logging, live heuristics, and artifacts:

```python
from post_training_toolkit import DiagnosticsCallback
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[
        DiagnosticsCallback(
            run_dir="diagnostic_run",
            stop_on_critical=True,      # optionally stop on NaNs / high-severity issues
            enable_live_warnings=True,  # print live heuristic warnings
        )
    ],
)

trainer.train()
```

Supported trainers: DPOTrainer, PPOTrainer (and PPOv2), GRPOTrainer, SFTTrainer, ORPOTrainer, KTOTrainer, CPOTrainer.

At the end of a run you get:

- `run_metadata_start.json` / `run_metadata.json` – immutable run metadata and provenance
- `metrics.jsonl` – step-level metrics with trainer-aware fields
- `snapshots/` – behavior snapshots on fixed prompts (if enabled)
- `diffs/` – behavior diffs between snapshots (if enabled)
- `postmortem.json` – crash/interrupt and critical failure context
- `reports/` – auto-generated diagnostics report with insights and plots

## Live heuristics and YAML alerts

Under the hood, the callback maintains a rolling history of metrics and runs a combination of Python and YAML-defined heuristics against it:

- Built-in heuristics for reward margin collapse, KL too high, loss going random, entropy collapse, reward variance spikes, etc.
- Additional heuristics loaded from YAML files per trainer type (see `post_training_toolkit/heuristics/builtin`)
- Custom alerts via inline rules or your own YAML directory

Example: enable built-in heuristics and your own YAML directory, and allow high-severity issues to stop training:

```python
DiagnosticsCallback(
    run_dir="diagnostic_run",
    stop_on_critical=True,
    custom_heuristics_dir="./my_heuristics",  # optional
)
```

High-severity insights trigger prominent warnings; when `stop_on_critical=True`, they can also request training to stop via the `TrainerControl` object.

## Distributed training support

The same callback works transparently in distributed training:

- Aggregates metrics across ranks for logging
- Detects stragglers and reports slowdown factors and likely causes
- Tracks GPU memory balance and can warn about impending OOMs
- Only writes artifacts on the main process

You enable distributed training as usual with `torchrun` or Accelerate; the toolkit auto-detects the environment and adds diagnostics on top.

## Agent trace analysis and datasets

The toolkit also includes an agent-trace → dataset pipeline for outcome-driven post-training:

```python
from post_training_toolkit.agents import AgentRunLog, analyze_runs, to_preference_pairs

runs = AgentRunLog.from_jsonl("agent_runs.jsonl")
report = analyze_runs(runs)

dataset = to_preference_pairs(
    runs,
    positive=lambda e: e.success and e.total_steps < 15,
    negative=lambda e: not e.success or e.has_repeated_tool_pattern(),
)
```

You can also use `AgentTrainingLoop` for a higher-level workflow (diagnose traces, build DPO/KTO/SFT/GRPO datasets, compare before/after training).

## CLI

The CLI wraps the diagnostics engine for existing runs and agent logs:

```bash
ptt-diagnose --input ./my_run --make-plots
ptt-agent-diagnose --input agent_runs.jsonl --export-dpo pairs.parquet
```

These commands generate reports under an outputs directory (plots, insights, and Markdown summaries).

## License

MIT
