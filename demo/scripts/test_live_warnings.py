#!/usr/bin/env python3
"""
Test the live warnings and auto-diagnostics features.

This runs a quick DPO training to demonstrate:
1. Live warnings during training (enable_live_warnings=True)
2. Auto-stopping on critical issues (stop_on_critical=True)
3. Auto-diagnostics at end (auto_diagnostics=True)

Usage:
    python demo/scripts/test_live_warnings.py

The callback will print live warnings like:
    [DiagnosticsCallback] ⚠️ MEDIUM at step 20: DPO loss stuck near 0.693
    [DiagnosticsCallback] 🚨 HIGH at step 50: ...
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig

from post_training_toolkit import DiagnosticsCallback


def create_tiny_preference_dataset(n_samples: int = 60) -> Dataset:
    """Create a minimal preference dataset for demo purposes."""
    prompts = [
        "Explain what machine learning is in one sentence.",
        "What is the capital of France?",
        "Write a haiku about coding.",
        "What does CPU stand for?",
        "Name a programming language.",
        "What is 2 + 2?",
        "Define the word 'algorithm'.",
        "What color is the sky?",
        "Name a planet in our solar system.",
        "What is an API?",
    ]
    
    chosen_responses = [
        "Machine learning is a subset of AI where computers learn patterns from data to make predictions.",
        "The capital of France is Paris.",
        "Bugs in the code / Debug until the dawn breaks / Coffee fuels the night",
        "CPU stands for Central Processing Unit.",
        "Python is a popular programming language.",
        "2 + 2 equals 4.",
        "An algorithm is a step-by-step procedure for solving a problem.",
        "The sky is typically blue during the day.",
        "Earth is a planet in our solar system.",
        "An API is an Application Programming Interface that allows software to communicate.",
    ]
    
    rejected_responses = [
        "idk lol",
        "paris i think maybe london",
        "roses are red violets are blue",
        "computer stuff",
        "coding thing",
        "math is hard",
        "something with computers",
        "depends",
        "the moon",
        "letters",
    ]
    
    data = {
        "prompt": [prompts[i % len(prompts)] for i in range(n_samples)],
        "chosen": [chosen_responses[i % len(chosen_responses)] for i in range(n_samples)],
        "rejected": [rejected_responses[i % len(rejected_responses)] for i in range(n_samples)],
    }
    
    return Dataset.from_dict(data)


def main():
    print("=" * 70)
    print("TESTING LIVE WARNINGS + AUTO-DIAGNOSTICS")
    print("=" * 70)
    print("\nThis demo shows:")
    print("  • Live warnings printed during training (⚠️ ℹ️ 🚨)")
    print("  • Auto-stop on high-severity issues (if any)")
    print("  • Auto-diagnostics report at end")
    print("=" * 70)
    
    model_name = "sshleifer/tiny-gpt2"
    
    print(f"\n[1/4] Loading tiny model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    print(f"   Model parameters: {model.num_parameters():,}")
    
    print("\n[2/4] Creating preference dataset...")
    dataset = create_tiny_preference_dataset(n_samples=60)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(f"   Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    print("\n[3/4] Setting up DPO training with NEW callback options...")
    
    output_dir = Path(__file__).parent.parent / "outputs" / "test_live_warnings"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        logging_steps=1,  # Log every step to see live warnings
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
        max_length=128,
        max_prompt_length=64,
        beta=0.1,
        report_to="none",
        use_cpu=True,
        fp16=False,
        bf16=False,
    )
    
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # NEW: Callback with live warnings enabled (default)
    diagnostics_callback = DiagnosticsCallback(
        run_dir=output_dir,
        verbose=False,  # Keep False so we see live warnings more clearly
        
        # Live warnings (default ON)
        enable_live_warnings=True,
        live_warning_interval=5,  # Check every 5 steps
        
        # Auto-stop on critical issues (OFF for demo so we see full output)
        stop_on_critical=False,
        
        # Auto-diagnostics summary at end (default ON)
        auto_diagnostics=True,
        
        # Snapshots disabled to speed up demo
        enable_snapshots=False,
    )
    
    print(f"   Output dir: {output_dir}")
    print("   Options:")
    print("     • enable_live_warnings=True (check every 5 steps)")
    print("     • stop_on_critical=False (for demo)")
    print("     • auto_diagnostics=True (prints summary at end)")
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[diagnostics_callback],
    )
    
    print("\n[4/4] Training (watch for live warnings!)...")
    print("=" * 70)
    
    trainer.train()
    
    # Diagnostics summary is printed automatically by the callback
    
    print(f"\n✅ Detected trainer type: {diagnostics_callback.trainer_type.upper()}")
    print(f"   Metrics saved to: {output_dir / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()
