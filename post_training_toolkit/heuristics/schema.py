"""Pydantic models for YAML heuristic validation.

This module defines the schema for YAML-based heuristics that can be
contributed without writing Python code.
"""

from typing import List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class YAMLHeuristic:
    """A heuristic defined in YAML format.

    Attributes:
        name: Unique identifier for the heuristic (e.g., "margin_collapse")
        description: Human-readable description of what this detects
        trainers: List of trainer types this applies to (e.g., ["dpo", "orpo"])
        metric: The metric column name to check (e.g., "reward_margin")
        condition: The condition DSL string (e.g., "< 0.1", "drop(20%)")
        window: Number of steps to consider for the condition (default: 20)
        severity: Severity level - "high", "medium", or "low"
        message: Alert message template (can use {value:.3f} for substitution)
        reference: Optional URL or citation for documentation
        min_steps: Minimum steps before this heuristic activates (default: 30)
        enabled: Whether this heuristic is enabled (default: True)

    Example YAML:
        name: margin_collapse
        description: Detect collapse in chosen/rejected reward margin
        trainers: [dpo, orpo, cpo]
        metric: reward_margin
        condition: "< 0.1"
        window: 20
        severity: high
        message: "Reward margin collapsed to {value:.3f}"
        reference: "https://arxiv.org/abs/2305.18290"
        min_steps: 30
        enabled: true
    """
    name: str
    description: str
    trainers: List[str]
    metric: str
    condition: str
    severity: str
    message: str
    window: int = 20
    reference: Optional[str] = None
    min_steps: int = 30
    enabled: bool = True

    def __post_init__(self):
        """Validate the heuristic after initialization."""
        # Validate severity
        valid_severities = {"high", "medium", "low"}
        if self.severity not in valid_severities:
            raise ValueError(f"Invalid severity '{self.severity}'. Must be one of: {valid_severities}")

        # Validate trainers
        valid_trainers = {"dpo", "ppo", "sft", "orpo", "kto", "cpo", "grpo", "common", "all"}
        for trainer in self.trainers:
            if trainer not in valid_trainers:
                raise ValueError(f"Invalid trainer '{trainer}'. Must be one of: {valid_trainers}")

        # Validate window
        if self.window < 1:
            raise ValueError(f"Window must be >= 1, got {self.window}")

        # Validate min_steps
        if self.min_steps < 0:
            raise ValueError(f"min_steps must be >= 0, got {self.min_steps}")

    @classmethod
    def from_dict(cls, data: dict) -> "YAMLHeuristic":
        """Create a YAMLHeuristic from a dictionary (parsed YAML).

        Args:
            data: Dictionary with heuristic fields

        Returns:
            YAMLHeuristic instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        required_fields = {"name", "description", "trainers", "metric", "condition", "severity", "message"}
        missing = required_fields - set(data.keys())
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        return cls(
            name=data["name"],
            description=data["description"],
            trainers=data["trainers"],
            metric=data["metric"],
            condition=data["condition"],
            severity=data["severity"],
            message=data["message"],
            window=data.get("window", 20),
            reference=data.get("reference"),
            min_steps=data.get("min_steps", 30),
            enabled=data.get("enabled", True),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "description": self.description,
            "trainers": self.trainers,
            "metric": self.metric,
            "condition": self.condition,
            "severity": self.severity,
            "message": self.message,
            "window": self.window,
            "min_steps": self.min_steps,
            "enabled": self.enabled,
        }
        if self.reference:
            result["reference"] = self.reference
        return result

    def applies_to_trainer(self, trainer_type: str) -> bool:
        """Check if this heuristic applies to a given trainer type.

        Args:
            trainer_type: The trainer type (e.g., "dpo", "ppo")

        Returns:
            True if this heuristic should run for the given trainer
        """
        if not self.enabled:
            return False
        if "all" in self.trainers:
            return True
        if "common" in self.trainers:
            return True
        return trainer_type.lower() in [t.lower() for t in self.trainers]
