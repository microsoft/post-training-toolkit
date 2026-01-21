"""Execute YAML heuristics against DataFrames.

This module evaluates YAML-defined conditions against training metrics
DataFrames and generates Insight objects when conditions are triggered.
"""

from typing import List, Optional, Set
from pathlib import Path

import numpy as np
import pandas as pd

from post_training_toolkit.heuristics.schema import YAMLHeuristic
from post_training_toolkit.heuristics.parser import (
    ConditionParser,
    ConditionType,
    ParsedCondition,
)
from post_training_toolkit.models.heuristics import Insight, TrainerType


class ConditionEvaluator:
    """Evaluates parsed conditions against metric data.

    This class takes a parsed condition and a pandas Series of metric values,
    returning True if the condition is triggered.

    Examples:
        >>> evaluator = ConditionEvaluator()
        >>> series = pd.Series([0.5, 0.3, 0.1, 0.05])
        >>> cond = ParsedCondition(type=ConditionType.LESS_THAN, threshold=0.1)
        >>> evaluator.evaluate(cond, series, window=2)
        True
    """

    def evaluate(
        self,
        condition: ParsedCondition,
        series: pd.Series,
        window: int = 20,
    ) -> tuple[bool, Optional[float]]:
        """Evaluate a condition against a metric series.

        Args:
            condition: The parsed condition to evaluate
            series: Pandas Series of metric values
            window: Number of recent values to consider

        Returns:
            Tuple of (triggered: bool, value: Optional[float])
            - triggered: Whether the condition was met
            - value: The relevant metric value for message substitution
        """
        if len(series) == 0:
            return False, None

        # Get recent window of data
        recent = series.tail(window)
        if len(recent) == 0:
            return False, None

        # Handle based on condition type
        if condition.type == ConditionType.LESS_THAN:
            mean_val = float(recent.mean())
            return mean_val < condition.threshold, mean_val

        elif condition.type == ConditionType.LESS_THAN_OR_EQUAL:
            mean_val = float(recent.mean())
            return mean_val <= condition.threshold, mean_val

        elif condition.type == ConditionType.GREATER_THAN:
            mean_val = float(recent.mean())
            return mean_val > condition.threshold, mean_val

        elif condition.type == ConditionType.GREATER_THAN_OR_EQUAL:
            mean_val = float(recent.mean())
            return mean_val >= condition.threshold, mean_val

        elif condition.type == ConditionType.EQUALS:
            mean_val = float(recent.mean())
            tolerance = condition.tolerance
            return abs(mean_val - condition.threshold) < tolerance, mean_val

        elif condition.type == ConditionType.RANGE:
            mean_val = float(recent.mean())
            low, high = condition.threshold, condition.threshold2
            return low <= mean_val <= high, mean_val

        elif condition.type == ConditionType.DROP:
            return self._evaluate_drop(series, window, condition.percentage)

        elif condition.type == ConditionType.SPIKE:
            return self._evaluate_spike(series, window, condition.multiplier)

        return False, None

    def _evaluate_drop(
        self,
        series: pd.Series,
        window: int,
        percentage: float,
    ) -> tuple[bool, Optional[float]]:
        """Evaluate a drop condition (value dropped X% from baseline).

        Args:
            series: Full metric series
            window: Window size for recent and baseline
            percentage: Drop percentage threshold (e.g., 20.0 for 20%)

        Returns:
            Tuple of (triggered, current_value)
        """
        if len(series) < window * 2:
            return False, None

        # Baseline is the early part of the series
        baseline = series.head(window)
        recent = series.tail(window)

        baseline_mean = float(baseline.mean())
        recent_mean = float(recent.mean())

        if baseline_mean == 0:
            return False, recent_mean

        # Calculate percentage drop
        drop_pct = ((baseline_mean - recent_mean) / abs(baseline_mean)) * 100

        return drop_pct >= percentage, recent_mean

    def _evaluate_spike(
        self,
        series: pd.Series,
        window: int,
        multiplier: float,
    ) -> tuple[bool, Optional[float]]:
        """Evaluate a spike condition (value X times above average).

        Args:
            series: Full metric series
            window: Window size for baseline average
            multiplier: Spike multiplier threshold (e.g., 3.0 for 3x)

        Returns:
            Tuple of (triggered, current_value)
        """
        if len(series) < window:
            return False, None

        # Rolling average as baseline (excluding last few points)
        if len(series) <= window + 5:
            baseline_mean = float(series.head(len(series) - 5).mean())
        else:
            baseline_mean = float(series.iloc[:-5].rolling(window, min_periods=window // 2).mean().iloc[-1])

        if pd.isna(baseline_mean) or baseline_mean == 0:
            return False, None

        recent = series.tail(5)
        current_max = float(recent.max())

        return current_max >= baseline_mean * multiplier, current_max


class YAMLHeuristicExecutor:
    """Executes YAML heuristics against training metrics DataFrames.

    This is the main entry point for evaluating YAML-defined heuristics.

    Usage:
        >>> from post_training_toolkit.heuristics.loader import HeuristicLoader
        >>> executor = YAMLHeuristicExecutor()
        >>> loader = HeuristicLoader()
        >>> heuristics = loader.load_for_trainer("dpo")
        >>> insights = executor.run(df, heuristics)
    """

    def __init__(self):
        """Initialize the executor."""
        self._parser = ConditionParser()
        self._evaluator = ConditionEvaluator()

    def run(
        self,
        df: pd.DataFrame,
        heuristics: List[YAMLHeuristic],
        trainer_type: str = "unknown",
    ) -> List[Insight]:
        """Run YAML heuristics against a DataFrame.

        Args:
            df: DataFrame with training metrics (must have a 'step' column)
            heuristics: List of YAMLHeuristic objects to evaluate
            trainer_type: The current trainer type

        Returns:
            List of Insight objects for triggered heuristics
        """
        insights: List[Insight] = []

        if df.empty or "step" not in df.columns:
            return insights

        num_steps = len(df)

        for heuristic in heuristics:
            # Skip if not applicable to this trainer
            if not heuristic.applies_to_trainer(trainer_type):
                continue

            # Skip if not enough steps
            if num_steps < heuristic.min_steps:
                continue

            # Skip if metric not present
            if heuristic.metric not in df.columns:
                continue

            # Parse the condition
            try:
                condition = self._parser.parse(heuristic.condition)
            except ValueError:
                # Invalid condition - skip this heuristic
                continue

            # Get the metric series
            series = df[heuristic.metric].astype(float)

            # Evaluate the condition
            triggered, value = self._evaluator.evaluate(
                condition,
                series,
                window=heuristic.window,
            )

            if triggered:
                # Format the message with the value
                message = self._format_message(heuristic.message, value)

                # Create the insight
                insight = Insight(
                    type=heuristic.name,
                    severity=heuristic.severity,
                    message=message,
                    steps=None,  # YAML heuristics don't track specific steps
                    data={
                        "metric": heuristic.metric,
                        "condition": heuristic.condition,
                        "value": value,
                        "window": heuristic.window,
                        "source": "yaml",
                    },
                    trainer_types=set(heuristic.trainers),
                    reference=heuristic.reference,
                )
                insights.append(insight)

        return insights

    def run_single(
        self,
        df: pd.DataFrame,
        heuristic: YAMLHeuristic,
        trainer_type: str = "unknown",
    ) -> Optional[Insight]:
        """Run a single YAML heuristic against a DataFrame.

        Args:
            df: DataFrame with training metrics
            heuristic: YAMLHeuristic to evaluate
            trainer_type: The current trainer type

        Returns:
            Insight if triggered, None otherwise
        """
        insights = self.run(df, [heuristic], trainer_type)
        return insights[0] if insights else None

    def _format_message(self, template: str, value: Optional[float]) -> str:
        """Format a message template with the metric value.

        Args:
            template: Message template (e.g., "Value dropped to {value:.3f}")
            value: The metric value to substitute

        Returns:
            Formatted message string
        """
        if value is None:
            # Remove any format specifiers
            import re
            return re.sub(r"\{value[^}]*\}", "N/A", template)

        try:
            return template.format(value=value)
        except (KeyError, ValueError):
            # Fallback if formatting fails
            return template.replace("{value}", str(value))


def run_yaml_heuristics(
    df: pd.DataFrame,
    trainer_type: str = "unknown",
    custom_dirs: Optional[List[Path]] = None,
    custom_alerts: Optional[List[str]] = None,
    disable_builtin: bool = False,
) -> List[Insight]:
    """Convenience function to run all applicable YAML heuristics.

    This is the main entry point for running YAML heuristics. It:
    1. Loads applicable heuristics from builtin and custom directories
    2. Parses any inline custom_alerts strings
    3. Executes all heuristics against the DataFrame
    4. Returns a list of triggered insights

    Args:
        df: DataFrame with training metrics
        trainer_type: The trainer type (e.g., "dpo", "ppo")
        custom_dirs: Optional list of custom heuristic directories
        custom_alerts: Optional list of inline alert strings
        disable_builtin: Whether to disable builtin heuristics

    Returns:
        List of Insight objects for triggered heuristics
    """
    from post_training_toolkit.heuristics.loader import HeuristicLoader
    from post_training_toolkit.heuristics.inline import parse_inline_alerts

    # Load YAML heuristics
    loader = HeuristicLoader(
        custom_dirs=custom_dirs,
        include_builtin=not disable_builtin,
    )
    yaml_heuristics = loader.load_for_trainer(trainer_type)

    # Parse inline custom alerts
    inline_heuristics = []
    if custom_alerts:
        inline_heuristics = parse_inline_alerts(custom_alerts)

    # Combine all heuristics
    all_heuristics = yaml_heuristics + inline_heuristics

    # Execute
    executor = YAMLHeuristicExecutor()
    return executor.run(df, all_heuristics, trainer_type)
