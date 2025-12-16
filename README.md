# Post-Training Toolkit

Train models and agents the way frontier labs do.

Get frontier-level RL and post training results without building frontier-level infrastructure.

## Installation

```bash
pip install post-training-toolkit
```

## Training Diagnostics

Add one callback to any TRL trainer. Get live warnings, automatic failure detection, and auditable artifacts.

```python
from post_training_toolkit import DiagnosticsCallback
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    args=training_args,
    callbacks=[DiagnosticsCallback()],
    ...
)
trainer.train()
```

Supports DPO, PPO, SFT, ORPO, KTO, CPO, and GRPO.

## Agent Trace Analysis

Turn agent logs into diagnostics and preference datasets.

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

## CLI

```bash
ptt-diagnose --input ./my_run --make-plots
ptt-agent-diagnose --input agent_runs.jsonl --export-dpo pairs.parquet
```

## License

MIT
