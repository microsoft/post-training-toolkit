"""Convert agent traces to training datasets for TRL.

Transforms agent episodes into preference pairs (for DPO) or 
labeled examples (for KTO/SFT) that can be directly used with TRL trainers.

Example:
    from post_training_toolkit.agents import AgentRunLog, to_preference_pairs
    
    runs = AgentRunLog.from_jsonl("agent_runs.jsonl")
    
    # Create DPO dataset: successful short runs vs failed/long runs
    dataset = to_preference_pairs(
        runs,
        positive=lambda e: e.success and e.total_steps < 15,
        negative=lambda e: not e.success or e.total_steps > 30,
    )
    
    # Use with TRL
    from trl import DPOTrainer
    trainer = DPOTrainer(model, train_dataset=dataset, ...)
"""
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
    """Format an episode as a conversation string.
    
    This produces a text representation suitable for training.
    
    Args:
        episode: The episode to format
        include_tool_calls: Whether to include tool call steps
        include_tool_results: Whether to include tool result steps
        max_steps: Maximum steps to include (None = all)
        
    Returns:
        Formatted conversation string
    """
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
                # Truncate long results
                if len(result) > 500:
                    result = result[:500] + "..."
                lines.append(f"[Tool Result: {result}]")
        
        elif step.type == StepType.EPISODE_END:
            # Skip episode_end markers in conversation
            continue
    
    return "\n".join(lines)


def format_episode_as_messages(
    episode: Episode,
    include_tool_calls: bool = True,
    include_tool_results: bool = True,
) -> List[Dict[str, str]]:
    """Format an episode as a list of message dicts.
    
    Produces chat-format messages suitable for chat templates.
    
    Returns:
        List of {"role": "user"|"assistant"|"system", "content": "..."}
    """
    messages = []
    
    for step in episode.steps:
        if step.type == StepType.USER_MESSAGE:
            messages.append({"role": "user", "content": step.content or ""})
        
        elif step.type == StepType.ASSISTANT_MESSAGE:
            messages.append({"role": "assistant", "content": step.content or ""})
        
        elif step.type == StepType.TOOL_CALL and include_tool_calls:
            args_str = json.dumps(step.args) if step.args else "{}"
            content = f"[Tool Call: {step.tool}({args_str})]"
            # Append to last assistant message or create new one
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
            # Tool results go as system or user depending on preference
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
    """Convert agent runs to preference pairs for DPO training.
    
    Creates pairs of (chosen, rejected) completions for the same prompt.
    
    Args:
        runs: Agent run log
        positive: Predicate for "chosen" episodes (good behavior)
        negative: Predicate for "rejected" episodes (bad behavior)
        prompt_key: Key for prompt in output dataset
        chosen_key: Key for chosen completion in output dataset
        rejected_key: Key for rejected completion in output dataset
        format_fn: Custom function to format episodes as strings.
                   Defaults to format_episode_as_conversation.
        require_same_prompt: If True, only pair episodes with identical prompts.
                            If False, pair any positive with any negative.
    
    Returns:
        HuggingFace Dataset with columns [prompt, chosen, rejected]
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "datasets package required for to_preference_pairs. "
            "Install with: pip install datasets"
        )
    
    format_fn = format_fn or format_episode_as_conversation
    
    # Split into positive and negative
    positives = [e for e in runs if positive(e)]
    negatives = [e for e in runs if negative(e)]
    
    if not positives:
        raise ValueError("No episodes matched the positive predicate")
    if not negatives:
        raise ValueError("No episodes matched the negative predicate")
    
    pairs = []
    
    if require_same_prompt:
        # Group by prompt
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
        
        # Create pairs for matching prompts
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
        # Pair all positives with all negatives (using prompts from positives)
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
    """Convert agent runs to KTO format (binary desirable/undesirable).
    
    KTO only needs a binary label per example, not paired preferences.
    
    Args:
        runs: Agent run log
        desirable: Predicate for desirable episodes (label=True)
        prompt_key: Key for prompt in output dataset
        completion_key: Key for completion in output dataset
        label_key: Key for binary label in output dataset
        format_fn: Custom function to format episodes as strings.
        
    Returns:
        HuggingFace Dataset with columns [prompt, completion, label]
    """
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
    """Convert agent runs to SFT format (just completions).
    
    Args:
        runs: Agent run log
        include: Optional predicate to filter episodes. Defaults to successful only.
        text_key: Key for text in output dataset
        format_fn: Custom function to format episodes as strings.
        
    Returns:
        HuggingFace Dataset with column [text]
    """
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
    """Convert agent runs to GRPO format (completions with rewards).
    
    GRPO uses reward signals directly rather than preferences.
    
    Args:
        runs: Agent run log
        reward_fn: Function to compute reward for episode. 
                  Defaults to using episode.reward or success-based.
        prompt_key: Key for prompt in output dataset
        completion_key: Key for completion in output dataset
        reward_key: Key for reward in output dataset
        format_fn: Custom function to format episodes as strings.
        
    Returns:
        HuggingFace Dataset with columns [prompt, completion, reward]
    """
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
        return 0.5  # Unknown
    
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
