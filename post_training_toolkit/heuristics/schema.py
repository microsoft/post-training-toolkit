
from typing import List, Optional, Union
from dataclasses import dataclass, field

@dataclass
class YAMLHeuristic:
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
        valid_severities = {"high", "medium", "low"}
        if self.severity not in valid_severities:
            raise ValueError(f"Invalid severity '{self.severity}'. Must be one of: {valid_severities}")

        valid_trainers = {"dpo", "ppo", "sft", "orpo", "kto", "cpo", "grpo", "common", "all"}
        for trainer in self.trainers:
            if trainer not in valid_trainers:
                raise ValueError(f"Invalid trainer '{trainer}'. Must be one of: {valid_trainers}")

        if self.window < 1:
            raise ValueError(f"Window must be >= 1, got {self.window}")

        if self.min_steps < 0:
            raise ValueError(f"min_steps must be >= 0, got {self.min_steps}")

    @classmethod
    def from_dict(cls, data: dict) -> "YAMLHeuristic":
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
        if not self.enabled:
            return False
        if "all" in self.trainers:
            return True
        if "common" in self.trainers:
            return True
        return trainer_type.lower() in [t.lower() for t in self.trainers]
