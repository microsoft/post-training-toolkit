
import re
from typing import List, Optional

from post_training_toolkit.heuristics.schema import YAMLHeuristic

INLINE_PATTERN = re.compile(
    r"^\s*"
    r"(?P<trainer>\w+)"
    r"\s*:\s*"
    r"(?P<metric>[\w_]+)"
    r"\s+"
    r"(?P<condition>[^-]+?)"
    r"(?:\s+for\s+(?P<window>\d+)\s+steps?)?"
    r"\s*->\s*"
    r"(?P<severity>high|medium|low)"
    r"\s*:\s*"
    r"(?P<message>.+)"
    r"\s*$",
    re.IGNORECASE
)

def parse_inline_alert(alert_string: str) -> Optional[YAMLHeuristic]:
    match = INLINE_PATTERN.match(alert_string.strip())
    if not match:
        return None

    trainer = match.group("trainer").lower()
    metric = match.group("metric").strip()
    condition = match.group("condition").strip()
    window_str = match.group("window")
    severity = match.group("severity").lower()
    message = match.group("message").strip()

    window = int(window_str) if window_str else 20

    name = _generate_name(metric, condition)

    if trainer in ("common", "all"):
        trainers = ["all"]
    else:
        trainers = [trainer]

    try:
        return YAMLHeuristic(
            name=name,
            description=f"Inline alert: {message}",
            trainers=trainers,
            metric=metric,
            condition=condition,
            window=window,
            severity=severity,
            message=message,
            reference=None,
            min_steps=window + 10,
            enabled=True,
        )
    except ValueError:
        return None

def parse_inline_alerts(alert_strings: List[str]) -> List[YAMLHeuristic]:
    heuristics = []
    for alert_string in alert_strings:
        h = parse_inline_alert(alert_string)
        if h is not None:
            heuristics.append(h)
    return heuristics

def _generate_name(metric: str, condition: str) -> str:
    cond_clean = condition.lower()

    if "<=" in cond_clean:
        cond_type = "lte"
    elif ">=" in cond_clean:
        cond_type = "gte"
    elif "<" in cond_clean:
        cond_type = "lt"
    elif ">" in cond_clean:
        cond_type = "gt"
    elif "==" in cond_clean:
        cond_type = "eq"
    elif "range" in cond_clean:
        cond_type = "range"
    elif "drop" in cond_clean:
        cond_type = "drop"
    elif "spike" in cond_clean:
        cond_type = "spike"
    else:
        cond_type = "custom"

    return f"inline_{metric}_{cond_type}"

def validate_inline_alert(alert_string: str) -> tuple[bool, Optional[str]]:
    if not alert_string or not alert_string.strip():
        return False, "Empty alert string"

    match = INLINE_PATTERN.match(alert_string.strip())
    if not match:
        return False, (
            "Invalid format. Expected: 'trainer: metric condition -> severity: message'. "
            "Example: 'dpo: margin < 0.1 -> high: Margin collapsed'"
        )

    trainer = match.group("trainer").lower()
    valid_trainers = {"dpo", "ppo", "sft", "orpo", "kto", "cpo", "grpo", "common", "all"}
    if trainer not in valid_trainers:
        return False, f"Invalid trainer '{trainer}'. Must be one of: {valid_trainers}"

    severity = match.group("severity").lower()
    if severity not in {"high", "medium", "low"}:
        return False, f"Invalid severity '{severity}'. Must be: high, medium, or low"

    from post_training_toolkit.heuristics.parser import ConditionParser
    parser = ConditionParser()
    condition = match.group("condition").strip()
    try:
        parser.parse(condition)
    except ValueError as e:
        return False, f"Invalid condition: {e}"

    return True, None
