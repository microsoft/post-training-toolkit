# Post-Training Toolkit: A Walkthrough for TRL

**TL;DR**: One callback gives you auto-diagnostics, crash postmortems, and literature-backed failure detection for any TRL trainer. Plus: convert agent traces to DPO pairs with simple predicates.

---

## The Problem We Kept Running Into

When debugging RLHF runs with TRL, we found ourselves repeatedly:

1. **Manually checking for known failure modes** - Is DPO loss stuck at 0.693? Is KL exploding? Did win-rate plateau?
2. **Losing context when runs crashed** - What step did it die on? What were the metrics right before?
3. **No structured way to compare behavior** - Did the model get more/less refusal-prone? Did capabilities regress?

We built this toolkit to operationalize the diagnostic patterns we found most useful.

---

## Quick Start: 3 Lines

```python
from post_training_toolkit import DiagnosticsCallback
from trl import DPOTrainer  # or PPOTrainer, SFTTrainer, ORPOTrainer, KTOTrainer, CPOTrainer, GRPOTrainer

trainer = DPOTrainer(
    model=model,
    args=training_args,
    callbacks=[DiagnosticsCallback()],  # ← This is it
    ...
)
trainer.train()
```

**What happens automatically:**
- Detects trainer type (DPO, PPO, SFT, ORPO, KTO, CPO, GRPO)
- Captures algorithm-specific metrics in structured format
- Runs heuristics at end of training (or on crash)
- Generates diagnostic report with severity-ranked insights

---

## What You Get

### 1. Auto-Detecting Metric Collection

The callback knows which metrics matter for each trainer:

| Trainer | Key Metrics Captured |
|---------|---------------------|
| **DPO** | loss, win_rate, reward_margin, logps_chosen/rejected, rewards_chosen/rejected |
| **PPO** | policy_loss, value_loss, entropy, clip_fraction, advantages, KL |
| **GRPO** | group rewards, advantages, policy loss, KL |
| **SFT** | loss, perplexity, accuracy |
| **ORPO** | sft_loss, odds_ratio_loss, log_odds_ratio |
| **KTO** | kl, logps for desirable/undesirable |

No configuration needed—just add the callback.

### 2. Crash Postmortems

If training crashes or gets interrupted, you get a `postmortem.json`:

```json
{
  "exit_reason": "exception",
  "last_step": 847,
  "timestamp": "2025-12-17T19:26:04Z",
  "traceback": "...",
  "final_metrics": {
    "dpo_loss": 0.693,
    "win_rate": 0.52
  }
}
```

No more "what step did it die on?"

### 3. Literature-Backed Heuristics

We encoded failure modes from the RLHF literature into automatic checks:

| Heuristic | What It Catches | Reference |
|-----------|-----------------|-----------|
| `dpo_loss_random` | Loss stuck at ln(2) ≈ 0.693 = not learning | Rafailov et al. (2023) §4.2 |
| `kl_instability` | KL exceeding safe thresholds | Schulman et al. (2017) |
| `margin_collapse` | Reward margin → 0 (chosen ≈ rejected) | Rafailov et al. (2023) |
| `entropy_collapse` | PPO entropy → 0 (mode collapse) | Zheng et al. (2023) |
| `value_head_divergence` | Value predictions exploding | Common PPO failure |
| `win_rate_plateau` | No improvement in preference accuracy | — |
| `policy_drift` | Cosine similarity to SFT dropping | — |

---

## Real Example: Catching DPO Loss at 0.693

Here's output from an actual run where the model wasn't learning:

```markdown
## RLHF Run Diagnostic Report

**Trainer:** DPO | **Status:** Crashed (exception)

### Key Insights

1. [HIGH] DPO loss stuck at ~0.693 (random chance). Model may not be learning preferences.
   *Ref: Rafailov et al. (2023) 'DPO', Section 4.2 - Loss at ln(2) indicates no preference signal*

2. [MEDIUM] Win rate shows high volatility (std=0.53), indicating inconsistent preference learning.

### Recommended Actions

- DPO loss at random chance: increase learning rate 2-5x, check data quality, or reduce beta.
- Win rate unstable: increase batch size for more stable gradient estimates.
```

The insight here is actionable: **loss at 0.693 means the model sees chosen and rejected as equally likely**. Common fixes: bump learning rate, check that chosen/rejected aren't too similar, or reduce beta.

---

## Bonus: Agent Traces → DPO Pairs

This is something we found surprisingly useful. If you're running agents and logging traces, you can convert them directly to preference pairs:

```python
from post_training_toolkit.agents import AgentRunLog, to_preference_pairs

# Load agent traces (simple JSONL format)
runs = AgentRunLog.from_jsonl("agent_runs.jsonl")

# Define what's "good" and "bad"
dataset = to_preference_pairs(
    runs,
    positive=lambda e: e.success and e.total_steps < 15,
    negative=lambda e: not e.success or e.has_repeated_tool_pattern(),
)

# Use directly with TRL
from trl import DPOTrainer
trainer = DPOTrainer(model, train_dataset=dataset, ...)
```

**The key insight**: you often already know what good vs bad agent behavior looks like (succeeded quickly vs failed/looped). This makes that knowledge into training signal.

### Trace Format

We use a minimal JSONL format that's easy to adapt to:

```jsonl
{"episode_id": "ep_001", "step": 0, "type": "user_message", "content": "Find flights to Paris"}
{"episode_id": "ep_001", "step": 1, "type": "assistant_message", "content": "I'll search..."}
{"episode_id": "ep_001", "step": 2, "type": "tool_call", "tool": "search_flights", "args": {"dest": "Paris"}}
{"episode_id": "ep_001", "step": 3, "type": "tool_result", "tool": "search_flights", "result": "..."}
{"episode_id": "ep_001", "step": 4, "type": "episode_end", "success": true, "reward": 1.0}
```

### Built-in Heuristics for Agent Traces

```python
# Detect common failure patterns
runs.analyze()  # Returns:
# - Episodes with repeated tool calls (stuck in loop)
# - Episodes with tool parse errors
# - Success rate by prompt type
# - Token/cost distribution
```

---

## Artifact Structure

A run with the callback produces:

```
my_run/
├── run_metadata_start.json   # Immutable provenance at start
├── run_metadata_final.json   # Immutable provenance at end  
├── metrics.jsonl             # Step-level metrics
├── postmortem.json           # Crash/interrupt diagnostics (if applicable)
├── snapshots/                # Behavior snapshots (optional)
├── diffs/                    # Behavior diffs between snapshots
└── reports/
    ├── run_report.md         # Auto-generated diagnostics
    └── plots/                # Reward, KL, drift visualizations
```

**Provenance includes**: git commit, config hash, hardware info, package versions, model/tokenizer/dataset identity.

---

## Design Principles (Feedback Welcome)

These are the principles we tried to follow. Would love feedback on where the abstractions could be cleaner:

### 1. Zero Configuration as Default
The callback should "just work" without any arguments. Advanced users can customize, but the default should be useful.

### 2. Algorithm-Aware, Not Algorithm-Specific
One callback for all trainers, but it knows what metrics matter for each. We didn't want `DPODiagnosticsCallback`, `PPODiagnosticsCallback`, etc.

### 3. Heuristics Should Be Citable
Each heuristic references the paper or empirical finding it's based on. Makes it easier to trust and to update.

### 4. Structured Artifacts Over Logs
Metrics in JSONL, metadata in JSON, reports in Markdown. Easy to parse, version, and compare across runs.

### 5. Graceful in Distributed Settings
Only rank 0 writes artifacts. Works with `torchrun` out of the box.

---

## Running the Demo

```bash
# Clone and install
git clone https://github.com/your-org/post-training-toolkit
cd post-training-toolkit
pip install -e .

# Run minimal DPO demo (~2-3 min on CPU)
python demo/scripts/minimal_dpo_demo.py

# Check outputs
cat demo/outputs/dpo_run/reports/run_report.md
```

Or open `demo/notebooks/demo.ipynb` for an interactive walkthrough.

---

## What We'd Love Feedback On

1. **Callback API surface** - Is the current set of options right? Too many? Missing something obvious?

2. **Heuristic thresholds** - We used values from papers where possible, but some are empirical. Would love to calibrate against more runs.

3. **Integration depth** - Currently this is a callback you add. Would deeper integration (e.g., into base Trainer) be valuable?

4. **Agent trace format** - Is the JSONL schema reasonable? Should we support other formats natively?

5. **What's missing?** - What diagnostic info do you wish you had during TRL runs that this doesn't capture?

---

## Links

- **Repo**: [post-training-toolkit](https://github.com/your-org/post-training-toolkit)
- **Demo notebook**: [demo/notebooks/demo.ipynb](demo/notebooks/demo.ipynb)
- **Heuristics source**: [post_training_toolkit/models/heuristics.py](post_training_toolkit/models/heuristics.py)
- **TRL integration**: [post_training_toolkit/integrations/trl.py](post_training_toolkit/integrations/trl.py)

---

*Built from patterns we found useful debugging real RLHF runs. Happy to walk through any part in more depth.*
