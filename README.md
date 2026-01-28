# Post-Training Toolkit

**Training diagnostics and control for long-horizon TRL runs.**

Post-Training Toolkit (PTT) is a diagnostics and training control layer for RLHF and agent post-training with Hugging Face TRL. Add a single callback to your TRL trainer to make long-running training dynamics observable, debuggable, and controllable under scale.

PTT is designed for expensive, unattended training jobs where correctness depends on detecting and intervening in failure modes before aggregate metrics make them obvious.

---

## What this is (systems lens)

PTT is a production-style diagnostics **and control** layer for TRL trainers:

- Auto-detecting integration with TRL trainers (DPO, PPO, GRPO, SFT, ORPO, KTO, CPO)
- Live heuristics over training metrics (Python + YAML)
- Failure-conditioned training control via `TrainerControl`
- Behavior snapshots and diffs to track model drift and refusal changes
- Distributed profiling (stragglers, throughput, memory balance)
- Provenance and artifacts designed for real training platforms

This is the kind of infrastructure you build after you’ve burned a lot of GPU hours and never want to debug a bad run in the dark again.

---

## Why it matters operationally

Post-training runs often fail gradually and silently. Loss, reward, or KL can look stable long after training dynamics have entered an unreliable regime.

PTT is built to surface and act on those failure modes early:

- Detects NaNs, divergence, margin collapse, KL spikes, reward variance spikes, and truncated outputs
- Makes long unattended and multi-GPU jobs debuggable via snapshots, diffs, postmortems, and resume validation
- Encodes both paper-inspired and practitioner heuristics in YAML + Python, extensible with lab-specific rules
- Produces auditable artifacts (metrics, metadata, snapshots, diffs, reports) that plug cleanly into experiment tracking systems

With `stop_on_critical=True`, PTT can also invalidate clearly broken runs early and record a recommended resume step.

---

## From diagnostics to training control

In long-horizon RL, failures rarely appear as sudden crashes. More often, training drifts into regimes where updates become unreliable long before aggregate metrics reflect a problem.

PTT treats post-training as a **controlled system**, not a blind batch process. Diagnostics make failure modes observable; control hooks make them actionable.

This enables deterministic intervention patterns such as pausing, stopping, or invalidating runs based on explicit heuristics, rather than reacting after a run has already consumed significant compute.

---

## How it’s engineered (for people who build systems)

The core design is intentionally simple and inspectable:

- Auto-detecting TRL integration with CPU-level integration tests  
  (see `tests/test_trl_integration.py`)
- YAML-based heuristics engine over pandas DataFrames  
  (see `post_training_toolkit/heuristics/executor.py`)
- Built-in heuristics for DPO, PPO, GRPO, SFT, ORPO, KTO, CPO  
  (see `post_training_toolkit/heuristics/builtin`)
- Distributed straggler detection, throughput profiling, and memory balance analysis  
  (see `post_training_toolkit/models/distributed` and `models/profiling`)
- Artifact, checkpoint, and postmortem design suitable for real training platforms  
  (see `models/artifacts.py`, `checkpoints.py`, `postmortem.py`)

If you’re evaluating this as a codebase, those modules are the right place to start reading.

---

## Installation

```bash
pip install post-training-toolkit
