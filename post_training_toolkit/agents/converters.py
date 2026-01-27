from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union
import json

from post_training_toolkit.agents.traces import (
    AgentRunLog, 
    Episode, 
    Step,
    StepType,
)

def format_episode_as_conversation(
    episode: Episode,
    include_tool_calls: bool = True,
    include_tool_results: bool = True,
    max_steps: Optional[int] = None,
) -> str:
    lines = []
    steps = episode.steps[:max_steps] if max_steps else episode.steps
    
    for step in steps:
        if step.type == StepType.USER_MESSAGE:
            lines.append(f"User: {step.content}")
        
        elif step.type == StepType.ASSISTANT_MESSAGE:
            lines.append(f"Assistant: {step.content}")
        
        elif step.type == StepType.TOOL_CALL and include_tool_calls:
            args_str = json.dumps(step.args) if step.args else "{}"
            lines.append(f"Assistant: [Tool Call: {step.tool}({args_str})]")
        
        elif step.type == StepType.TOOL_RESULT and include_tool_results:
            if step.error:
                lines.append(f"[Tool Error: {step.error}]")
            else:
                result = step.result or ""
                if len(result) > 500:
                    result = result[:500] + "..."
                lines.append(f"[Tool Result: {result}]")
        
        elif step.type == StepType.EPISODE_END:
            continue
    
    return "\n".join(lines)

def format_episode_as_messages(
    episode: Episode,
    include_tool_calls: bool = True,
    include_tool_results: bool = True,
) -> List[Dict[str, str]]:
    messages = []
    
    for step in episode.steps:
        if step.type == StepType.USER_MESSAGE:
            messages.append({"role": "user", "content": step.content or ""})
        
        elif step.type == StepType.ASSISTANT_MESSAGE:
            messages.append({"role": "assistant", "content": step.content or ""})
        
        elif step.type == StepType.TOOL_CALL and include_tool_calls:
            args_str = json.dumps(step.args) if step.args else "{}"
            content = f"[Tool Call: {step.tool}({args_str})]"
            if messages and messages[-1]["role"] == "assistant":
                messages[-1]["content"] += "\n" + content
            else:
                messages.append({"role": "assistant", "content": content})
        
        elif step.type == StepType.TOOL_RESULT and include_tool_results:
            if step.error:
                content = f"[Tool Error: {step.error}]"
            else:
                result = step.result or ""
                if len(result) > 500:
                    result = result[:500] + "..."
                content = f"[Tool Result: {result}]"
            messages.append({"role": "user", "content": content})
    
    return messages

def to_preference_pairs(
    runs: AgentRunLog,
    positive: Callable[[Episode], bool],
    negative: Callable[[Episode], bool],
    prompt_key: str = "prompt",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
    format_fn: Optional[Callable[[Episode], str]] = None,
    require_same_prompt: bool = False,
) -> "Dataset":
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "datasets package required for to_preference_pairs. "
            "Install with: pip install datasets"
        )
    
    format_fn = format_fn or format_episode_as_conversation
    
    positives = [e for e in runs if positive(e)]
    negatives = [e for e in runs if negative(e)]
    
    if not positives:
        raise ValueError("No episodes matched the positive predicate")
    if not negatives:
        raise ValueError("No episodes matched the negative predicate")
    
    pairs = []
    
    if require_same_prompt:
        pos_by_prompt: Dict[str, List[Episode]] = {}
        neg_by_prompt: Dict[str, List[Episode]] = {}
        
        for e in positives:
            prompt = e.initial_prompt or ""
            if prompt not in pos_by_prompt:
                pos_by_prompt[prompt] = []
            pos_by_prompt[prompt].append(e)
        
        for e in negatives:
            prompt = e.initial_prompt or ""
            if prompt not in neg_by_prompt:
                neg_by_prompt[prompt] = []
            neg_by_prompt[prompt].append(e)
        
        for prompt in pos_by_prompt:
            if prompt in neg_by_prompt:
                for pos_ep in pos_by_prompt[prompt]:
                    for neg_ep in neg_by_prompt[prompt]:
                        pairs.append({
                            prompt_key: prompt,
                            chosen_key: format_fn(pos_ep),
                            rejected_key: format_fn(neg_ep),
                        })
    else:
        for pos_ep in positives:
            prompt = pos_ep.initial_prompt or ""
            for neg_ep in negatives:
                pairs.append({
                    prompt_key: prompt,
                    chosen_key: format_fn(pos_ep),
                    rejected_key: format_fn(neg_ep),
                })
    
    if not pairs:
        raise ValueError(
            "No preference pairs could be created. "
            "If require_same_prompt=True, ensure positive and negative episodes "
            "share some prompts."
        )
    
    return Dataset.from_list(pairs)

def to_kto_dataset(
    runs: AgentRunLog,
    desirable: Callable[[Episode], bool],
    prompt_key: str = "prompt",
    completion_key: str = "completion",
    label_key: str = "label",
    format_fn: Optional[Callable[[Episode], str]] = None,
) -> "Dataset":
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "datasets package required for to_kto_dataset. "
            "Install with: pip install datasets"
        )
    
    format_fn = format_fn or format_episode_as_conversation
    
    examples = []
    for episode in runs:
        prompt = episode.initial_prompt or ""
        completion = format_fn(episode)
        label = desirable(episode)
        
        examples.append({
            prompt_key: prompt,
            completion_key: completion,
            label_key: label,
        })
    
    return Dataset.from_list(examples)

def to_sft_dataset(
    runs: AgentRunLog,
    include: Optional[Callable[[Episode], bool]] = None,
    text_key: str = "text",
    format_fn: Optional[Callable[[Episode], str]] = None,
) -> "Dataset":
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "datasets package required for to_sft_dataset. "
            "Install with: pip install datasets"
        )
    
    format_fn = format_fn or format_episode_as_conversation
    include = include or (lambda e: e.success is True)
    
    examples = []
    for episode in runs:
        if include(episode):
            examples.append({text_key: format_fn(episode)})
    
    return Dataset.from_list(examples)

def to_grpo_dataset(
    runs: AgentRunLog,
    reward_fn: Optional[Callable[[Episode], float]] = None,
    prompt_key: str = "prompt",
    completion_key: str = "completion",
    reward_key: str = "reward",
    format_fn: Optional[Callable[[Episode], str]] = None,
) -> "Dataset":
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "datasets package required for to_grpo_dataset. "
            "Install with: pip install datasets"
        )
    
    format_fn = format_fn or format_episode_as_conversation
    
    def default_reward(e: Episode) -> float:
        if e.reward is not None:
            return e.reward
        if e.success is True:
            return 1.0
        if e.success is False:
            return 0.0
        return 0.5
    
    reward_fn = reward_fn or default_reward
    
    examples = []
    for episode in runs:
        prompt = episode.initial_prompt or ""
        completion = format_fn(episode)
        reward = reward_fn(episode)
        
        examples.append({
            prompt_key: prompt,
            completion_key: completion,
            reward_key: reward,
        })
    
    return Dataset.from_list(examples)
