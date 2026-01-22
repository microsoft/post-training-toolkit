#!/usr/bin/env python3
"""
Real Scenario 2: DPO Training with PTT Auto-Stop

This runs REAL TRL DPO training with PTT auto-stop enabled.
Bug injected: Very high LR → loss stuck at 0.693 (random chance) → CRITICAL
PTT will auto-stop training.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
import sys
from pathlib import Path

# Add PTT to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from post_training_toolkit import DiagnosticsCallback

def main():
    print("="*70)
    print("SCENARIO 2: REAL DPO TRAINING WITH PTT AUTO-STOP")
    print("="*70)
    print("\nSetup:")
    print("  • Model: gpt2 (124M)")
    print("  • Trainer: DPO")
    print("  • Dataset: Anthropic/hh-rlhf (25 examples)")
    print("  • Bug: Very high LR (5e-3) → DPO loss explodes (>2.0)")
    print("  • PTT: stop_on_critical=True (AUTO-STOP ENABLED)")
    print("  • Expected: Auto-stop when loss exceeds 2.0")
    print()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model_ref = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    dataset = dataset.select(range(25))  # Just 25 examples

    # Format for DPO
    def format_dataset(example):
        return {
            "prompt": example["chosen"].split("\n\nAssistant:")[0] + "\n\nAssistant:",
            "chosen": example["chosen"].split("\n\nAssistant:")[-1],
            "rejected": example["rejected"].split("\n\nAssistant:")[-1],
        }

    dataset = dataset.map(format_dataset)

    # Training config with bug: VERY HIGH LEARNING RATE
    print("\nConfiguring DPO training...")
    config = DPOConfig(
        output_dir="./real_scenario2_output",
        beta=0.1,  # Normal beta
        learning_rate=5e-3,  # 🐛 BUG: 100x too high! Will cause loss to explode
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        max_length=256,
        max_prompt_length=128,
        logging_steps=1,  # Log every step so PTT can detect issues quickly
        save_steps=100,
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )

    # PTT callback - AUTO-STOP ENABLED
    # Custom alert: DPO loss > 2.0 is abnormally high (should be ~0.693)
    # This WILL trigger because the high LR causes loss to explode
    # Note: "for 1 steps" sets the window small so it triggers early
    callback = DiagnosticsCallback(
        run_dir="./real_scenario2_output/diagnostics",
        stop_on_critical=True,  # 🛑 AUTO-STOP ON CRITICAL ISSUES
        enable_live_warnings=True,
        live_warning_interval=1,  # Check every step for fast detection
        enable_snapshots=False,
        verbose=True,
        custom_alerts=[
            # This triggers on the REAL issue: loss exploding due to high LR
            # "for 1 steps" = small window so min_steps is only 11, not 30
            "dpo: dpo_loss > 2.0 for 1 steps -> high: DPO loss exploded! Normal range is 0.5-1.0. High LR is destabilizing training.",
        ],
    )

    # Create trainer
    print("\nCreating DPO trainer with PTT callback...")
    trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[callback],
    )

    print("\n" + "="*70)
    print("STARTING TRAINING - PTT WILL AUTO-STOP ON CRITICAL ISSUES")
    print("="*70)
    print("\n👀 PTT will check metrics every step (logging_steps=1)")
    print("🚨 Expected: Critical alert + AUTO-STOP around step 11 when loss > 2.0")
    print()

    try:
        trainer.train()
        print("\n✅ Training completed (if you see this, bug didn't trigger)")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("TRAINING ENDED")
    print("="*70)

    # Check if auto-stopped
    if callback._critical_failure_detected:
        print(f"\n🛑 PTT AUTO-STOPPED TRAINING")
        print(f"   Reason: {callback._stop_reason}")
        print(f"   Step when stopped: {callback._metrics_history[-1]['step'] if callback._metrics_history else 'unknown'}")
        print(f"   Last good step: {callback._last_good_step}")

        # Calculate compute saved (rough estimate)
        if callback._metrics_history:
            import pandas as pd
            df = pd.DataFrame(callback._metrics_history)
            stopped_at = df['step'].iloc[-1]
            planned_steps = 25 * 3 // 4  # (examples * epochs) / batch_size
            saved_steps = max(0, planned_steps - stopped_at)
            print(f"   Compute saved: ~{saved_steps} steps")

        print(f"\n💰 In a real GPU run at $2.50/hr with 8 GPUs:")
        print(f"   This would have saved: ${saved_steps * 0.5:.0f}+")

        # Show summary
        if callback._metrics_history:
            df = pd.DataFrame(callback._metrics_history)
            print(f"\n📊 Metrics at Stop:")
            if 'dpo_loss' in df.columns:
                print(f"   Loss: {df['dpo_loss'].iloc[-1]:.4f} (should be < 0.5, stuck at ~0.693 = random)")
            if 'reward_margin' in df.columns:
                print(f"   Margin: {df['reward_margin'].iloc[-1]:.4f}")
            if 'win_rate' in df.columns:
                print(f"   Win Rate: {df['win_rate'].iloc[-1]*100:.1f}%")
    else:
        print("\n✅ Training completed without critical issues")
        if callback._metrics_history:
            import pandas as pd
            df = pd.DataFrame(callback._metrics_history)
            print(f"\n📊 Final Metrics:")
            if 'dpo_loss' in df.columns:
                print(f"   Loss: {df['dpo_loss'].iloc[0]:.4f} → {df['dpo_loss'].iloc[-1]:.4f}")

    print(f"\n📁 Artifacts saved to: {callback.run_dir}")
    print(f"   • Metrics log: {callback.run_dir}/metrics.jsonl")

    # Check for critical failure artifact
    failure_file = Path(callback.run_dir) / "critical_failure.json"
    if failure_file.exists():
        print(f"   • Critical failure report: {failure_file}")
        import json
        with open(failure_file) as f:
            print(f"\n📋 Critical Failure Details:")
            print(json.dumps(json.load(f), indent=2))

    print()

if __name__ == "__main__":
    main()
