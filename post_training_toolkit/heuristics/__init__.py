"""YAML-based heuristics system for easy contribution of training alerts.

This package provides a YAML-based way to define training heuristics without
writing Python code.

Quick Start:
    >>> from post_training_toolkit.heuristics import run_yaml_heuristics
    >>> insights = run_yaml_heuristics(df, trainer_type="dpo")

Custom Alerts in DiagnosticsCallback:
    >>> from post_training_toolkit import DiagnosticsCallback
    >>> cb = DiagnosticsCallback(
    ...     custom_alerts=[
    ...         "dpo: margin < 0.1 -> high: Margin collapsed",
    ...         "ppo: entropy drop(50%) for 30 steps -> medium: Entropy dropping",
    ...     ]
    ... )

Creating Custom YAML Heuristics:
    Create a YAML file in your custom directory:

    ```yaml
    # my_heuristics/dpo/custom_margin.yaml
    name: custom_margin_check
    description: Custom margin threshold
    trainers: [dpo]
    metric: reward_margin
    condition: "< 0.2"
    window: 30
    severity: medium
    message: "Margin dropped to {value:.3f}"
    min_steps: 50
    enabled: true
    ```

    Then load with:
    >>> from post_training_toolkit.heuristics import HeuristicLoader
    >>> loader = HeuristicLoader(custom_dirs=[Path("my_heuristics")])
    >>> heuristics = loader.load_for_trainer("dpo")
"""

from post_training_toolkit.heuristics.schema import YAMLHeuristic
from post_training_toolkit.heuristics.parser import (
    ConditionParser,
    ParsedCondition,
    ConditionType,
    parse_condition,
)
from post_training_toolkit.heuristics.loader import (
    HeuristicLoader,
    get_loader,
    load_heuristics_for_trainer,
)
from post_training_toolkit.heuristics.executor import (
    ConditionEvaluator,
    YAMLHeuristicExecutor,
    run_yaml_heuristics,
)
from post_training_toolkit.heuristics.inline import (
    parse_inline_alert,
    parse_inline_alerts,
    validate_inline_alert,
)

__all__ = [
    # Schema
    "YAMLHeuristic",
    # Parser
    "ConditionParser",
    "ParsedCondition",
    "ConditionType",
    "parse_condition",
    # Loader
    "HeuristicLoader",
    "get_loader",
    "load_heuristics_for_trainer",
    # Executor
    "ConditionEvaluator",
    "YAMLHeuristicExecutor",
    "run_yaml_heuristics",
    # Inline
    "parse_inline_alert",
    "parse_inline_alerts",
    "validate_inline_alert",
]
