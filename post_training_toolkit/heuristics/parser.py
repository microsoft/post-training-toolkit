"""Condition DSL parser for YAML heuristics.

This module parses the condition DSL syntax used in YAML heuristics.

Supported operators:
    | Operator | Syntax | Description |
    |----------|--------|-------------|
    | Less than | `< 0.1` | Value below threshold |
    | Greater than | `> 0.5` | Value above threshold |
    | Less than or equal | `<= 0.1` | Value at or below threshold |
    | Greater than or equal | `>= 0.5` | Value at or above threshold |
    | Equals | `== 0.693` | Value equals (with tolerance) |
    | Range | `range(0.68, 0.71)` | Mean stuck in range |
    | Drop | `drop(20%)` | Dropped 20% from baseline |
    | Spike | `spike(3x)` | 3x above rolling average |
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union


class ConditionType(Enum):
    """Types of conditions supported by the DSL."""
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    EQUALS = "equals"
    RANGE = "range"
    DROP = "drop"
    SPIKE = "spike"


@dataclass
class ParsedCondition:
    """A parsed condition from the DSL.

    Attributes:
        type: The type of condition
        threshold: Primary threshold value (for comparison operators)
        threshold2: Secondary threshold (for range operator)
        percentage: Percentage value (for drop operator)
        multiplier: Multiplier value (for spike operator)
        tolerance: Tolerance for equality comparison (default: 0.02)
    """
    type: ConditionType
    threshold: Optional[float] = None
    threshold2: Optional[float] = None
    percentage: Optional[float] = None
    multiplier: Optional[float] = None
    tolerance: float = 0.02

    def __repr__(self) -> str:
        if self.type == ConditionType.LESS_THAN:
            return f"< {self.threshold}"
        elif self.type == ConditionType.LESS_THAN_OR_EQUAL:
            return f"<= {self.threshold}"
        elif self.type == ConditionType.GREATER_THAN:
            return f"> {self.threshold}"
        elif self.type == ConditionType.GREATER_THAN_OR_EQUAL:
            return f">= {self.threshold}"
        elif self.type == ConditionType.EQUALS:
            return f"== {self.threshold} (±{self.tolerance})"
        elif self.type == ConditionType.RANGE:
            return f"range({self.threshold}, {self.threshold2})"
        elif self.type == ConditionType.DROP:
            return f"drop({self.percentage}%)"
        elif self.type == ConditionType.SPIKE:
            return f"spike({self.multiplier}x)"
        return f"ParsedCondition({self.type})"


class ConditionParser:
    """Parser for the condition DSL.

    Examples:
        >>> parser = ConditionParser()
        >>> parser.parse("< 0.1")
        ParsedCondition(type=LESS_THAN, threshold=0.1)
        >>> parser.parse("drop(20%)")
        ParsedCondition(type=DROP, percentage=20.0)
        >>> parser.parse("spike(3x)")
        ParsedCondition(type=SPIKE, multiplier=3.0)
    """

    # Regex patterns for each operator type
    PATTERNS = {
        # <= and >= must come before < and > to match correctly
        "less_than_or_equal": re.compile(r"^\s*<=\s*(-?\d+\.?\d*)\s*$"),
        "greater_than_or_equal": re.compile(r"^\s*>=\s*(-?\d+\.?\d*)\s*$"),
        "less_than": re.compile(r"^\s*<\s*(-?\d+\.?\d*)\s*$"),
        "greater_than": re.compile(r"^\s*>\s*(-?\d+\.?\d*)\s*$"),
        "equals": re.compile(r"^\s*==\s*(-?\d+\.?\d*)\s*$"),
        "range": re.compile(r"^\s*range\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)\s*$"),
        "drop": re.compile(r"^\s*drop\s*\(\s*(\d+\.?\d*)\s*%?\s*\)\s*$"),
        "spike": re.compile(r"^\s*spike\s*\(\s*(\d+\.?\d*)\s*x?\s*\)\s*$"),
    }

    def parse(self, condition: str) -> ParsedCondition:
        """Parse a condition DSL string.

        Args:
            condition: The condition string (e.g., "< 0.1", "drop(20%)")

        Returns:
            ParsedCondition with the parsed values

        Raises:
            ValueError: If the condition string is invalid
        """
        condition = condition.strip()

        # Try less_than_or_equal first (before less_than)
        match = self.PATTERNS["less_than_or_equal"].match(condition)
        if match:
            return ParsedCondition(
                type=ConditionType.LESS_THAN_OR_EQUAL,
                threshold=float(match.group(1)),
            )

        # Try greater_than_or_equal first (before greater_than)
        match = self.PATTERNS["greater_than_or_equal"].match(condition)
        if match:
            return ParsedCondition(
                type=ConditionType.GREATER_THAN_OR_EQUAL,
                threshold=float(match.group(1)),
            )

        # Try less_than
        match = self.PATTERNS["less_than"].match(condition)
        if match:
            return ParsedCondition(
                type=ConditionType.LESS_THAN,
                threshold=float(match.group(1)),
            )

        # Try greater_than
        match = self.PATTERNS["greater_than"].match(condition)
        if match:
            return ParsedCondition(
                type=ConditionType.GREATER_THAN,
                threshold=float(match.group(1)),
            )

        # Try equals
        match = self.PATTERNS["equals"].match(condition)
        if match:
            return ParsedCondition(
                type=ConditionType.EQUALS,
                threshold=float(match.group(1)),
            )

        # Try range
        match = self.PATTERNS["range"].match(condition)
        if match:
            return ParsedCondition(
                type=ConditionType.RANGE,
                threshold=float(match.group(1)),
                threshold2=float(match.group(2)),
            )

        # Try drop
        match = self.PATTERNS["drop"].match(condition)
        if match:
            return ParsedCondition(
                type=ConditionType.DROP,
                percentage=float(match.group(1)),
            )

        # Try spike
        match = self.PATTERNS["spike"].match(condition)
        if match:
            return ParsedCondition(
                type=ConditionType.SPIKE,
                multiplier=float(match.group(1)),
            )

        raise ValueError(
            f"Invalid condition: '{condition}'. "
            f"Supported formats: '< N', '> N', '<= N', '>= N', '== N', "
            f"'range(A, B)', 'drop(N%)', 'spike(Nx)'"
        )


# Module-level parser instance for convenience
_parser = ConditionParser()


def parse_condition(condition: str) -> ParsedCondition:
    """Parse a condition DSL string.

    This is a convenience function that uses a module-level parser instance.

    Args:
        condition: The condition string (e.g., "< 0.1", "drop(20%)")

    Returns:
        ParsedCondition with the parsed values

    Raises:
        ValueError: If the condition string is invalid
    """
    return _parser.parse(condition)
