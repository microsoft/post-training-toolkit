#!/usr/bin/env python3
"""
Minimal DPO training demo for CPU - demonstrates auto-detecting diagnostics.

This runs a tiny DPO training loop (~2-3 min on CPU) to demonstrate
the diagnostics callback integration with TRL. The callback automatically:
- Detects the trainer type (DPO in this case)
- Captures DPO-specific metrics
- Runs DPO-specific heuristics (loss at 0.693, margin collapse, etc.)

Usage:
    python demo/scripts/minimal_dpo_demo.py

Requirements:
    pip install transformers trl datasets torch accelerate peft
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

from post_training_toolkit import DiagnosticsCallback, run_diagnostics


def create_tiny_preference_dataset(n_samples: int = 50) -> Dataset:
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
    print("=" * 60)
    print("MINIMAL DPO DEMO - Auto-Detecting Diagnostics Callback")
    print("=" * 60)
    
    model_name = "sshleifer/tiny-gpt2"
    
    print(f"\n[1/5] Loading tiny model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    
    print(f"   Model parameters: {model.num_parameters():,}")
    
    print("\n[2/5] Creating preference dataset...")
    dataset = create_tiny_preference_dataset(n_samples=40)
    print(f"   Dataset size: {len(dataset)} samples")
    
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print("\n[3/5] Setting up DPO training with auto-detecting callback...")
    
    output_dir = Path(__file__).parent.parent / "outputs" / "dpo_run"
    log_path = output_dir / "diagnostics_log.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=5,
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
    
    # Zero-config callback - it auto-detects DPOTrainer!
    diagnostics_callback = DiagnosticsCallback(
        log_path=log_path,
        verbose=True,
    )
    
    print(f"   Diagnostics logging to: {log_path}")
    print("   ⚡ Callback will auto-detect trainer type (DPO)")
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[diagnostics_callback],
    )
    
    print("\n[4/5] Training (this may take 2-3 minutes on CPU)...")
    print("-" * 60)
    
    trainer.train()
    
    print("-" * 60)
    
    # Show what trainer type was detected
    print(f"\n   ✅ Detected trainer type: {diagnostics_callback.trainer_type.upper()}")
    
    print("\n[5/5] Running trainer-aware diagnostics...")
    
    reports_dir = output_dir / "reports"
    report_path = run_diagnostics(log_path, reports_dir, make_plots=True)
    
    print(f"\n{'=' * 60}")
    print("DEMO COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\nOutputs:")
    print(f"  - Training logs: {log_path}")
    print(f"  - Diagnostic report: {report_path}")
    print(f"  - Plots: {reports_dir / 'plots'}")
    print(f"\nOpen the report to see DPO-specific diagnostics!")
    
    print(f"\n{'=' * 60}")
    print("REPORT PREVIEW:")
    print(f"{'=' * 60}")
    with open(report_path, "r") as f:
        for i, line in enumerate(f):
            if i >= 40:
                print("...")
                break
            print(line.rstrip())


if __name__ == "__main__":
    main()
