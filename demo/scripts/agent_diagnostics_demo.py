"""Demo: Agent trace diagnostics and dataset conversion.

This example shows how to:
1. Load agent traces from JSONL
2. Run diagnostics to detect issues
3. Convert to preference pairs for TRL DPO training
4. Convert to KTO/SFT datasets

Run with:
    python demo/scripts/agent_diagnostics_demo.py
"""
from pathlib import Path

# Add project root to path for development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from post_training_toolkit.agents import (
    AgentRunLog,
    analyze_runs,
    to_preference_pairs,
    to_kto_dataset,
)


def main():
    # Path to demo traces
    traces_path = Path(__file__).parent.parent / "logs" / "agent_traces.jsonl"
    
    print("=" * 60)
    print("AGENT DIAGNOSTICS DEMO")
    print("=" * 60)
    print()
    
    # 1. Load traces
    print(f"Loading traces from: {traces_path}")
    runs = AgentRunLog.from_jsonl(traces_path)
    print(f"Loaded {len(runs)} episodes")
    print()
    
    # 2. Quick stats
    print("--- Quick Stats ---")
    print(f"  Success rate: {runs.success_rate:.1%}")
    print(f"  Avg steps: {runs.avg_steps:.1f}")
    print(f"  Avg tokens: {runs.avg_tokens:.0f}")
    print(f"  Total cost: ${runs.total_cost:.4f}")
    print(f"  Tool error rate: {runs.tool_error_rate:.1%}")
    print()
    
    # 3. Run full diagnostics
    print("Running diagnostics...")
    print()
    report = analyze_runs(runs)
    print(report)
    print()
    
    # 4. Convert to preference pairs for DPO
    print("=" * 60)
    print("DATASET CONVERSION")
    print("=" * 60)
    print()
    
    print("Creating preference pairs for DPO...")
    print("  Positive: successful episodes with < 8 steps")
    print("  Negative: failed episodes OR episodes with loops")
    
    try:
        dpo_dataset = to_preference_pairs(
            runs,
            positive=lambda e: e.success is True and e.total_steps < 8,
            negative=lambda e: (
                e.success is False or 
                e.has_repeated_tool_pattern(min_repeats=3)
            ),
        )
        print(f"  → Created {len(dpo_dataset)} preference pairs")
        print()
        print("  Sample pair:")
        print(f"    Prompt: {dpo_dataset[0]['prompt'][:60]}...")
        print(f"    Chosen length: {len(dpo_dataset[0]['chosen'])} chars")
        print(f"    Rejected length: {len(dpo_dataset[0]['rejected'])} chars")
    except ValueError as e:
        print(f"  → Could not create pairs: {e}")
    print()
    
    # 5. Convert to KTO dataset
    print("Creating KTO dataset (binary labels)...")
    print("  Desirable: successful AND cost < $0.003")
    
    kto_dataset = to_kto_dataset(
        runs,
        desirable=lambda e: e.success is True and (e.total_cost or 0) < 0.003,
    )
    
    desirable_count = sum(1 for ex in kto_dataset if ex['label'])
    undesirable_count = len(kto_dataset) - desirable_count
    print(f"  → Created {len(kto_dataset)} examples")
    print(f"     Desirable: {desirable_count}, Undesirable: {undesirable_count}")
    print()
    
    # 6. Show how to use with TRL
    print("=" * 60)
    print("USAGE WITH TRL")
    print("=" * 60)
    print("""
# DPO Training
from trl import DPOTrainer, DPOConfig
from post_training_toolkit import DiagnosticsCallback

trainer = DPOTrainer(
    model=model,
    args=DPOConfig(...),
    train_dataset=dpo_dataset,  # From to_preference_pairs()
    callbacks=[DiagnosticsCallback()],  # Add training diagnostics!
)
trainer.train()

# KTO Training  
from trl import KTOTrainer, KTOConfig

trainer = KTOTrainer(
    model=model,
    args=KTOConfig(...),
    train_dataset=kto_dataset,  # From to_kto_dataset()
    callbacks=[DiagnosticsCallback()],
)
trainer.train()
""")


if __name__ == "__main__":
    main()
