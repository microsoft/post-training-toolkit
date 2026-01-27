
import json
import re
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from post_training_toolkit.models.artifacts import Snapshot
from post_training_toolkit.models.heuristics import Insight, TrainerType

def load_snapshots(run_dir: Path) -> List[Snapshot]:
    snapshots_dir = run_dir / "snapshots"
    if not snapshots_dir.exists():
        return []
    
    snapshots = []
    for f in snapshots_dir.glob("*.json"):
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
                snapshots.append(Snapshot.from_dict(data))
        except Exception:
            continue
            
    return sorted(snapshots, key=lambda s: s.metadata.step)

def detect_verbosity_bias(snapshots: List[Snapshot]) -> List[Insight]:
    if len(snapshots) < 2:
        return []
        
    steps = [s.metadata.step for s in snapshots]
    mean_lengths = [
        np.mean([e.output_length for e in s.entries]) 
        for s in snapshots
    ]
    
    if len(steps) > 2:
        slope, intercept = np.polyfit(steps, mean_lengths, 1)
        
        initial_len = mean_lengths[0]
        if initial_len > 0:
            rel_growth = (slope * (steps[-1] - steps[0])) / initial_len
            
            if rel_growth > 0.5:
                return [Insight(
                    type="verbosity_bias",
                    severity="high" if rel_growth > 1.0 else "medium",
                    message=f"Model response length increased by {rel_growth*100:.1f}% over training. This may indicate reward hacking (verbosity bias).",
                    steps=[steps[-1]],
                    data={"slope": slope, "growth": rel_growth},
                    trainer_types={TrainerType.DPO, TrainerType.PPO}
                )]
    return []

def detect_repetition_loops(snapshots: List[Snapshot]) -> List[Insight]:
    insights = []
    
    for s in snapshots:
        repeated_count = 0
        for entry in s.entries:
            text = entry.output
            if len(text) < 50:
                continue
                
            compressed = zlib.compress(text.encode("utf-8"))
            ratio = len(text) / len(compressed)
            
            if ratio > 3.5: 
                repeated_count += 1
                
        if repeated_count > len(s.entries) * 0.2:
             insights.append(Insight(
                type="repetition_collapse",
                severity="high",
                message=f"High repetition detected in {repeated_count}/{len(s.entries)} responses at step {s.metadata.step}.",
                steps=[s.metadata.step],
                trainer_types={TrainerType.DPO, TrainerType.PPO, TrainerType.SFT}
            ))
            
    return insights

def detect_pattern_collapse(snapshots: List[Snapshot]) -> List[Insight]:
    insights = []
    
    for s in snapshots:
        prefixes = [e.output[:15].lower() for e in s.entries if len(e.output) >= 15]
        if not prefixes:
            continue
            
        counts = Counter(prefixes)
        if not counts:
            continue
            
        most_common, count = counts.most_common(1)[0]
        
        if count > len(s.entries) * 0.5 and len(s.entries) > 5:
             insights.append(Insight(
                type="pattern_collapse",
                severity="medium",
                message=f"Pattern collapse: {count}/{len(s.entries)} responses start with '{most_common}...' at step {s.metadata.step}.",
                steps=[s.metadata.step],
                trainer_types={TrainerType.DPO, TrainerType.PPO}
            ))
            
    return insights

def run_text_heuristics(run_dir: Path) -> List[Insight]:
    snapshots = load_snapshots(run_dir)
    if not snapshots:
        return []
        
    insights = []
    insights.extend(detect_verbosity_bias(snapshots))
    insights.extend(detect_repetition_loops(snapshots))
    insights.extend(detect_pattern_collapse(snapshots))
    
    return insights
