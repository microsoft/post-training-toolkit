#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig

from post_training_toolkit import DiagnosticsCallback

def create_tiny_preference_dataset(n_samples: int = 40) -> Dataset:
    prompts = [
        "Explain what machine learning is in one sentence.",
        "What is the capital of France?",
        "Write a haiku about coding.",
        "What does CPU stand for?",
        "Name a programming language.",
    ]
    
    chosen_responses = [
        "Machine learning is a subset of AI where computers learn patterns from data.",
        "The capital of France is Paris.",
        "Bugs in the code / Debug until the dawn breaks / Coffee fuels the night",
        "CPU stands for Central Processing Unit.",
        "Python is a popular programming language.",
    ]
    
    rejected_responses = [
        "idk lol",
        "paris i think maybe london",
        "roses are red violets are blue",
        "computer stuff",
        "coding thing",
    ]
    
    data = {
        "prompt": [prompts[i % len(prompts)] for i in range(n_samples)],
        "chosen": [chosen_responses[i % len(chosen_responses)] for i in range(n_samples)],
        "rejected": [rejected_responses[i % len(rejected_responses)] for i in range(n_samples)],
    }
    
    return Dataset.from_dict(data)

def main():
    print("=" * 70)
    print("🧪 CUSTOM HEURISTIC DEMO")
    print("=" * 70)
    print()
    print("This demo shows how to add your own YAML heuristic and have PTT use it.")
    print()
    
    custom_heuristics_dir = Path(__file__).parent.parent / "custom_heuristics"
    custom_yaml = custom_heuristics_dir / "dpo" / "loss_variance_high.yaml"
    
    print("📄 Custom heuristic file:")
    print(f"   {custom_yaml}")
    print()
    print("   Contents:")
    print("   " + "-" * 50)
    for line in custom_yaml.read_text().strip().split("\n"):
        print(f"   {line}")
    print("   " + "-" * 50)
    print()
    
    print("🔧 Setting up tiny model for demo...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = "sshleifer/tiny-gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    dataset = create_tiny_preference_dataset(n_samples=80)
    
    print()
    print("⚠️  SCENARIO: Tiny model can't learn preferences properly")
    print("   The reward margin (chosen - rejected) stays very low.")
    print("   Our custom heuristic should catch this!")
    print()
    
    output_dir = Path(__file__).parent.parent / "outputs" / "custom_heuristic_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        logging_steps=2,
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        max_length=64,
        max_prompt_length=32,
    )
    
    print(f"📁 Loading custom heuristics from: {custom_heuristics_dir}")
    print()
    
    diagnostics_callback = DiagnosticsCallback(
        run_dir=str(output_dir),
        enable_live_warnings=True,
        live_warning_interval=5,
        stop_on_critical=True,
        auto_diagnostics=True,
        verbose=True,
        custom_heuristics_dir=str(custom_heuristics_dir),
    )
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[diagnostics_callback],
    )
    
    print("=" * 70)
    print("🚀 Starting training - watch for custom heuristic alerts!")
    print("=" * 70)
    print()
    
    trainer.train()
    
    print()
    print("=" * 70)
    print("✅ Demo complete!")
    print()
    print("What happened:")
    print("  1. We created a custom YAML heuristic for 'reward margin too low'")
    print("  2. We ran training where the model can't distinguish preferences")
    print("  3. PTT loaded our custom heuristic and caught the issue!")
    print()
    print("To add your own heuristics, just create YAML files like:")
    print(f"  {custom_yaml}")
    print("=" * 70)

if __name__ == "__main__":
    main()
