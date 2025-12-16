#!/usr/bin/env python3
"""
Full-featured DPO demo with behavior snapshots and all diagnostics features.

This demonstrates the complete post-training toolkit:
- Auto-detecting diagnostics callback
- Behavior snapshots at intervals
- Crash/interrupt postmortem recording
- Behavior diff analysis
- Checkpoint comparison and recommendation

Usage:
    python examples/full_diagnostics_demo.py

Requirements:
    pip install transformers trl datasets torch accelerate peft
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig

from post_training_toolkit import (
    DiagnosticsCallback, 
    run_diagnostics,
    RefusalDetector,
    is_refusal,
)


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


def demo_refusal_detection():
    """Quick demo of refusal detection."""
    print("\n--- Refusal Detection Demo ---")
    
    test_responses = [
        "The capital of France is Paris.",
        "I cannot help with that request.",
        "I'm sorry, but I can't provide information about that topic.",
        "Here's a simple explanation of machine learning...",
        "I apologize, but I will not assist with this.",
    ]
    
    detector = RefusalDetector()
    
    for response in test_responses:
        result = detector.detect(response)
        status = "🚫 REFUSAL" if result.is_refusal else "✅ OK"
        print(f"  {status}: {response[:50]}...")
        if result.is_refusal:
            print(f"       Type: {result.refusal_type.value}, Confidence: {result.confidence}")


def main():
    print("=" * 70)
    print("FULL DIAGNOSTICS DEMO - All Post-Training Toolkit Features")
    print("=" * 70)
    
    # Show refusal detection
    demo_refusal_detection()
    
    model_name = "sshleifer/tiny-gpt2"
    
    print(f"\n[1/6] Loading tiny model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    
    print(f"   Model parameters: {model.num_parameters():,}")
    
    print("\n[2/6] Creating preference dataset...")
    dataset = create_tiny_preference_dataset(n_samples=40)
    print(f"   Dataset size: {len(dataset)} samples")
    
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print("\n[3/6] Setting up DPO training with FULL diagnostics...")
    
    # Use new run_dir based artifact management
    run_dir = Path(__file__).parent.parent / "demo_outputs" / "full_dpo_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = DPOConfig(
        output_dir=str(run_dir / "checkpoints"),
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
    
    # FULL-FEATURED callback with all new options
    diagnostics_callback = DiagnosticsCallback(
        run_dir=run_dir,  # New: unified artifact directory
        verbose=True,
        enable_snapshots=True,  # New: capture behavior snapshots
        snapshot_interval=10,   # Every 10 steps
        enable_postmortem=True, # New: crash recording
    )
    
    print(f"   Run directory: {run_dir}")
    print(f"   Features enabled:")
    print(f"     - Auto-detecting trainer type")
    print(f"     - Behavior snapshots every 10 steps")
    print(f"     - Postmortem recording on crash/interrupt")
    print(f"     - Unified artifact structure")
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[diagnostics_callback],
    )
    
    print("\n[4/6] Training (this may take 2-3 minutes on CPU)...")
    print("-" * 70)
    
    trainer.train()
    
    print("-" * 70)
    
    # Show what trainer type was detected
    print(f"\n   ✅ Detected trainer type: {diagnostics_callback.trainer_type.upper()}")
    
    print("\n[5/6] Examining run artifacts...")
    
    # List artifacts
    print(f"\n   Run Directory Contents:")
    for item in sorted(run_dir.iterdir()):
        if item.is_dir():
            print(f"     📁 {item.name}/")
            for sub in sorted(item.iterdir())[:5]:
                print(f"        - {sub.name}")
            if len(list(item.iterdir())) > 5:
                print(f"        ... and more")
        else:
            print(f"     📄 {item.name}")
    
    print("\n[6/6] Running full diagnostics with behavior analysis...")
    
    reports_dir = run_dir / "reports"
    report_path = run_diagnostics(run_dir, reports_dir, make_plots=True)
    
    print(f"\n{'=' * 70}")
    print("DEMO COMPLETE!")
    print(f"{'=' * 70}")
    print(f"\nRun Artifacts:")
    print(f"  - Run metadata: {run_dir / 'run_metadata.json'}")
    print(f"  - Metrics log:  {run_dir / 'metrics.jsonl'}")
    print(f"  - Snapshots:    {run_dir / 'snapshots/'}")
    print(f"\nDiagnostic Report:")
    print(f"  - Report:       {report_path}")
    print(f"  - Plots:        {reports_dir / 'plots'}")
    
    print(f"\n{'=' * 70}")
    print("REPORT PREVIEW:")
    print(f"{'=' * 70}")
    with open(report_path, "r") as f:
        for i, line in enumerate(f):
            if i >= 60:
                print("...")
                break
            print(line.rstrip())
    
    print(f"\n\n✨ The report now includes:")
    print(f"   - Trainer-aware metric analysis")
    print(f"   - Behavior drift analysis (if snapshots were captured)")
    print(f"   - Checkpoint recommendations (if applicable)")
    print(f"   - Postmortem data (if run crashed)")


if __name__ == "__main__":
    main()
