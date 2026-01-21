"""Parse inline custom_alerts strings into YAMLHeuristic objects.

This module parses the inline alert syntax used in DiagnosticsCallback:

    custom_alerts=[
        "dpo: margin < 0.1 for 20 steps -> high: Margin collapsed",
        "ppo: entropy drop(50%) -> high: Entropy collapsing",
    ]

Syntax:
    trainer: metric condition [for N steps] -> severity: message

Examples:
    "dpo: margin < 0.1 -> high: Margin collapsed"
    "ppo: entropy drop(50%) for 30 steps -> medium: Entropy dropping"
    "common: loss > 10 -> high: Loss exploded"
"""

import re
from typing import List, Optional

from post_training_toolkit.heuristics.schema import YAMLHeuristic


# Regex pattern for parsing inline alerts
# Format: trainer: metric condition [for N steps] -> severity: message
INLINE_PATTERN = re.compile(
    r"^\s*"
    r"(?P<trainer>\w+)"          # Trainer type (e.g., "dpo", "ppo", "common")
    r"\s*:\s*"                    # Colon separator
    r"(?P<metric>[\w_]+)"         # Metric name (e.g., "margin", "entropy")
    r"\s+"                        # Space
    r"(?P<condition>[^-]+?)"      # Condition (everything before -> except trailing spaces)
    r"(?:\s+for\s+(?P<window>\d+)\s+steps?)?"  # Optional: for N steps
    r"\s*->\s*"                   # Arrow separator
    r"(?P<severity>high|medium|low)"  # Severity level
    r"\s*:\s*"                    # Colon separator
    r"(?P<message>.+)"            # Message text
    r"\s*$",
    re.IGNORECASE
)


def parse_inline_alert(alert_string: str) -> Optional[YAMLHeuristic]:
    """Parse a single inline alert string into a YAMLHeuristic.

    Args:
        alert_string: The inline alert string

    Returns:
        YAMLHeuristic if parsing succeeds, None otherwise

    Example:
        >>> h = parse_inline_alert("dpo: margin < 0.1 -> high: Margin collapsed")
        >>> h.name
        'inline_margin_lt'
        >>> h.metric
        'margin'
        >>> h.severity
        'high'
    """
    match = INLINE_PATTERN.match(alert_string.strip())
    if not match:
        return None

    trainer = match.group("trainer").lower()
    metric = match.group("metric").strip()
    condition = match.group("condition").strip()
    window_str = match.group("window")
    severity = match.group("severity").lower()
    message = match.group("message").strip()

    # Parse window (default 20)
    window = int(window_str) if window_str else 20

    # Generate a unique name based on metric and condition
    name = _generate_name(metric, condition)

    # Determine trainer list
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
            min_steps=window + 10,  # Need at least window + buffer steps
            enabled=True,
        )
    except ValueError:
        # Invalid heuristic specification
        return None


def parse_inline_alerts(alert_strings: List[str]) -> List[YAMLHeuristic]:
    """Parse multiple inline alert strings into YAMLHeuristic objects.

    Args:
        alert_strings: List of inline alert strings

    Returns:
        List of successfully parsed YAMLHeuristic objects
        (invalid strings are silently skipped)

    Example:
        >>> alerts = [
        ...     "dpo: margin < 0.1 -> high: Margin collapsed",
        ...     "ppo: entropy drop(50%) -> medium: Entropy dropping",
        ... ]
        >>> heuristics = parse_inline_alerts(alerts)
        >>> len(heuristics)
        2
    """
    heuristics = []
    for alert_string in alert_strings:
        h = parse_inline_alert(alert_string)
        if h is not None:
            heuristics.append(h)
    return heuristics


def _generate_name(metric: str, condition: str) -> str:
    """Generate a unique name for an inline heuristic.

    Args:
        metric: The metric name
        condition: The condition string

    Returns:
        A unique identifier string
    """
    # Clean up the condition for use in a name
    cond_clean = condition.lower()

    # Determine condition type for the name
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
    """Validate an inline alert string without creating a heuristic.

    Args:
        alert_string: The inline alert string to validate

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if the string is valid
        - error_message: Error description if invalid, None otherwise
    """
    if not alert_string or not alert_string.strip():
        return False, "Empty alert string"

    match = INLINE_PATTERN.match(alert_string.strip())
    if not match:
        return False, (
            "Invalid format. Expected: 'trainer: metric condition -> severity: message'. "
            "Example: 'dpo: margin < 0.1 -> high: Margin collapsed'"
        )

    # Validate trainer
    trainer = match.group("trainer").lower()
    valid_trainers = {"dpo", "ppo", "sft", "orpo", "kto", "cpo", "grpo", "common", "all"}
    if trainer not in valid_trainers:
        return False, f"Invalid trainer '{trainer}'. Must be one of: {valid_trainers}"

    # Validate severity
    severity = match.group("severity").lower()
    if severity not in {"high", "medium", "low"}:
        return False, f"Invalid severity '{severity}'. Must be: high, medium, or low"

    # Validate condition (try to parse it)
    from post_training_toolkit.heuristics.parser import ConditionParser
    parser = ConditionParser()
    condition = match.group("condition").strip()
    try:
        parser.parse(condition)
    except ValueError as e:
        return False, f"Invalid condition: {e}"

    return True, None
