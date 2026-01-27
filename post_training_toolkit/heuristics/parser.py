
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

class ConditionType(Enum):
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
    PATTERNS = {
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
        condition = condition.strip()

        match = self.PATTERNS["less_than_or_equal"].match(condition)
        if match:
            return ParsedCondition(
                type=ConditionType.LESS_THAN_OR_EQUAL,
                threshold=float(match.group(1)),
            )

        match = self.PATTERNS["greater_than_or_equal"].match(condition)
        if match:
            return ParsedCondition(
                type=ConditionType.GREATER_THAN_OR_EQUAL,
                threshold=float(match.group(1)),
            )

        match = self.PATTERNS["less_than"].match(condition)
        if match:
            return ParsedCondition(
                type=ConditionType.LESS_THAN,
                threshold=float(match.group(1)),
            )

        match = self.PATTERNS["greater_than"].match(condition)
        if match:
            return ParsedCondition(
                type=ConditionType.GREATER_THAN,
                threshold=float(match.group(1)),
            )

        match = self.PATTERNS["equals"].match(condition)
        if match:
            return ParsedCondition(
                type=ConditionType.EQUALS,
                threshold=float(match.group(1)),
            )

        match = self.PATTERNS["range"].match(condition)
        if match:
            return ParsedCondition(
                type=ConditionType.RANGE,
                threshold=float(match.group(1)),
                threshold2=float(match.group(2)),
            )

        match = self.PATTERNS["drop"].match(condition)
        if match:
            return ParsedCondition(
                type=ConditionType.DROP,
                percentage=float(match.group(1)),
            )

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

_parser = ConditionParser()

def parse_condition(condition: str) -> ParsedCondition:
    return _parser.parse(condition)
