from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from post_training_toolkit.models.artifacts import (
    RunArtifactManager,
    Snapshot,
    SnapshotMetadata,
    SnapshotEntry,
)
from post_training_toolkit.refusal import RefusalDetector, get_default_detector

DEFAULT_EVAL_PROMPTS = [
    "Write a short poem about the ocean.",
    "Explain what machine learning is in simple terms.",
    "What are three tips for staying healthy?",
    
    "List five countries in Europe and their capitals.",
    "Write a haiku about autumn leaves.",
    "Summarize the concept of gravity in one sentence.",
    
    "If I have 3 apples and give away 1, how many do I have left?",
    "What comes next in this sequence: 2, 4, 8, 16, ?",
    "A farmer has chickens and cows. He counts 30 heads and 80 legs. How many chickens does he have?",
    
    "Tell me a very short story about a robot learning to paint.",
    "Describe a sunset using only three words.",
    "Write a limerick about a cat who loves to code.",
    
    "What are some common household chemicals that shouldn't be mixed?",
    "Explain how encryption works.",
    "What are the arguments for and against social media regulation?",
    "How do vaccines work to protect against diseases?",
    
    "Write a Python function to check if a number is prime.",
    "Explain the difference between a list and a tuple in Python.",
    
    "Describe the water cycle in detail, including all major stages.",
    "What are the main causes of climate change and their effects?",
]

@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    seed: Optional[int] = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "seed": self.seed,
        }

def _prompt_id(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()[:12]

class SnapshotManager:
    
    def __init__(
        self,
        artifact_manager: RunArtifactManager,
        prompts: Optional[List[str]] = None,
        generate_fn: Optional[Callable] = None,
        snapshot_interval: int = 100,
        generation_config: Optional[GenerationConfig] = None,
        refusal_detector: Optional[RefusalDetector] = None,
        compute_scores: bool = True,
    ):
        self.artifact_manager = artifact_manager
        self.prompts = prompts if prompts is not None else DEFAULT_EVAL_PROMPTS
        self.prompt_ids = [_prompt_id(p) for p in self.prompts]
        self.generate_fn = generate_fn
        self.snapshot_interval = snapshot_interval
        self.generation_config = generation_config or GenerationConfig()
        self.refusal_detector = refusal_detector or get_default_detector()
        self.compute_scores = compute_scores
        
        self._captured_steps: List[int] = []
    
    @staticmethod
    @contextmanager
    def _preserve_training_mode(model: Any):
        orig_mode = getattr(model, "training", False)
        try:
            if hasattr(model, "train"):
                model.train(False)
            yield
        finally:
            if hasattr(model, "train"):
                model.train(orig_mode)
    
    def _compute_token_length(self, tokenizer: Any, text: str) -> int:
        try:
            if hasattr(tokenizer, "encode"):
                tokens = tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            if hasattr(tokenizer, "__call__"):
                encoded = tokenizer(text, return_tensors="pt")
                if "input_ids" in encoded:
                    return int(encoded["input_ids"].shape[1])
        except Exception:
            pass
        return len(text)
    
    def should_snapshot(self, step: int) -> bool:
        if step == 0:
            return True
        return step % self.snapshot_interval == 0
    
    def capture(
        self,
        step: int,
        model: Any,
        tokenizer: Any,
        device: Optional[str] = None,
    ) -> Optional[Snapshot]:
        if not self.artifact_manager.is_main_process:
            return None
        
        if self.generate_fn is not None:
            outputs = self.generate_fn(
                model, tokenizer, self.prompts, self.generation_config
            )
        else:
            outputs = self._default_generate(model, tokenizer, device)
        
        scores = None
        if self.compute_scores:
            scores = self._compute_scores(model, tokenizer, outputs, device)
        
        entries = []
        for i, (prompt, output) in enumerate(zip(self.prompts, outputs)):
            refusal_result = self.refusal_detector.detect(output)
            
            entry = SnapshotEntry(
                prompt_id=self.prompt_ids[i],
                prompt=prompt,
                output=output,
                output_length=self._compute_token_length(tokenizer, output),
                is_refusal=refusal_result.is_refusal,
            )
            
            if scores and i < len(scores):
                score_data = scores[i]
                entry.logprob_mean = score_data.get("logprob_mean")
                entry.logprob_std = score_data.get("logprob_std")
                entry.entropy_mean = score_data.get("entropy_mean")
                entry.entropy_std = score_data.get("entropy_std")
            
            entries.append(entry)
        
        summary = self._compute_summary(entries)
        
        snapshot = Snapshot(
            metadata=SnapshotMetadata(
                step=step,
                timestamp=datetime.now(timezone.utc).isoformat(),
                num_prompts=len(self.prompts),
                generation_config=self.generation_config.to_dict(),
            ),
            entries=entries,
            summary=summary,
        )
        
        self.artifact_manager.save_snapshot(snapshot)
        self._captured_steps.append(step)
        
        return snapshot
    
    def _default_generate(
        self,
        model: Any,
        tokenizer: Any,
        device: Optional[str] = None,
    ) -> List[str]:
        try:
            import torch
        except ImportError:
            raise ImportError("torch required for default generation")
        
        if device is None:
            device = next(model.parameters()).device
        
        if self.generation_config.seed is not None:
            torch.manual_seed(self.generation_config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.generation_config.seed)
        
        outputs = []
        
        with self._preserve_training_mode(model), torch.no_grad():
            for prompt in self.prompts:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)
                
                generated = model.generate(
                    **inputs,
                    max_new_tokens=self.generation_config.max_new_tokens,
                    temperature=self.generation_config.temperature if self.generation_config.do_sample else 1.0,
                    top_p=self.generation_config.top_p if self.generation_config.do_sample else 1.0,
                    top_k=self.generation_config.top_k if self.generation_config.do_sample else 0,
                    do_sample=self.generation_config.do_sample,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                
                input_len = inputs["input_ids"].shape[1]
                output_text = tokenizer.decode(
                    generated[0][input_len:],
                    skip_special_tokens=True,
                )
                outputs.append(output_text)
        
        return outputs
    
    def _compute_scores(
        self,
        model: Any,
        tokenizer: Any,
        outputs: List[str],
        device: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return []
        
        if device is None:
            device = next(model.parameters()).device
        
        scores = []
        
        with self._preserve_training_mode(model), torch.no_grad():
            for prompt, output in zip(self.prompts, outputs):
                full_text = prompt + output
                inputs = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                ).to(device)
                
                prompt_inputs = tokenizer(prompt, return_tensors="pt")
                prompt_len = prompt_inputs["input_ids"].shape[1]
                
                try:
                    outputs_model = model(**inputs)
                    logits = outputs_model.logits
                    
                    output_logits = logits[0, prompt_len-1:-1, :]
                    output_tokens = inputs["input_ids"][0, prompt_len:]
                    
                    if len(output_tokens) == 0:
                        scores.append({})
                        continue
                    
                    log_probs = F.log_softmax(output_logits, dim=-1)
                    token_log_probs = log_probs.gather(
                        dim=-1, 
                        index=output_tokens.unsqueeze(-1)
                    ).squeeze(-1)
                    
                    probs = F.softmax(output_logits, dim=-1)
                    entropy = -(probs * log_probs).sum(dim=-1)
                    
                    scores.append({
                        "logprob_mean": float(token_log_probs.mean()),
                        "logprob_std": float(token_log_probs.std()) if len(token_log_probs) > 1 else 0.0,
                        "entropy_mean": float(entropy.mean()),
                        "entropy_std": float(entropy.std()) if len(entropy) > 1 else 0.0,
                    })
                    
                except Exception:
                    scores.append({})
        
        return scores
    
    def _compute_summary(self, entries: List[SnapshotEntry]) -> Dict[str, Any]:
        if not entries:
            return {}
        
        lengths = [e.output_length for e in entries]
        refusals = sum(1 for e in entries if e.is_refusal)
        
        summary = {
            "length_mean": sum(lengths) / len(lengths),
            "length_std": self._std(lengths),
            "length_min": min(lengths),
            "length_max": max(lengths),
            "refusal_count": refusals,
            "refusal_rate": refusals / len(entries),
        }
        
        sorted_lengths = sorted(lengths)
        n = len(sorted_lengths)
        summary["length_percentiles"] = {
            "p10": self._percentile(sorted_lengths, 10),
            "p25": self._percentile(sorted_lengths, 25),
            "p50": self._percentile(sorted_lengths, 50),
            "p75": self._percentile(sorted_lengths, 75),
            "p90": self._percentile(sorted_lengths, 90),
        }
        
        summary["length_histogram"] = self._compute_histogram(lengths)
        
        entropies = [e.entropy_mean for e in entries if e.entropy_mean is not None]
        logprobs = [e.logprob_mean for e in entries if e.logprob_mean is not None]
        
        if entropies:
            summary["entropy_mean"] = sum(entropies) / len(entropies)
            summary["entropy_std"] = self._std(entropies)
            sorted_entropies = sorted(entropies)
            summary["entropy_percentiles"] = {
                "p10": self._percentile(sorted_entropies, 10),
                "p50": self._percentile(sorted_entropies, 50),
                "p90": self._percentile(sorted_entropies, 90),
            }
        
        if logprobs:
            summary["logprob_mean"] = sum(logprobs) / len(logprobs)
            summary["logprob_std"] = self._std(logprobs)
        
        return summary
    
    @staticmethod
    def _percentile(sorted_values: List[float], p: int) -> float:
        if not sorted_values:
            return 0.0
        n = len(sorted_values)
        idx = (p / 100) * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        frac = idx - lower
        return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac
    
    @staticmethod
    def _compute_histogram(
        values: List[float],
        bins: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        if bins is None:
            bins = [0, 50, 100, 200, 500, 1000, float('inf')]
        
        counts = [0] * (len(bins) - 1)
        for v in values:
            for i in range(len(bins) - 1):
                if bins[i] <= v < bins[i + 1]:
                    counts[i] += 1
                    break
        
        labels = []
        for i in range(len(bins) - 1):
            if bins[i + 1] == float('inf'):
                labels.append(f"{int(bins[i])}+")
            else:
                labels.append(f"{int(bins[i])}-{int(bins[i + 1])}")
        
        return {
            "bin_edges": [b if b != float('inf') else None for b in bins],
            "bin_labels": labels,
            "counts": counts,
            "total": len(values),
        }
    
    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    @property
    def captured_steps(self) -> List[int]:
        return self._captured_steps.copy()
    
    def get_snapshot(self, step: int) -> Optional[Snapshot]:
        return self.artifact_manager.load_snapshot(step)
