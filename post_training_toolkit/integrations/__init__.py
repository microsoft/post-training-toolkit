"""TRL trainer integrations for diagnostics logging.

Provides auto-configuring callbacks that work with any TRL trainer:
- DPOTrainer
- PPOTrainer
- SFTTrainer
- ORPOTrainer
- KTOTrainer
- CPOTrainer
"""

from post_training_toolkit.integrations.trl import DiagnosticsCallback, TrainerType

__all__ = ["DiagnosticsCallback", "TrainerType"]
