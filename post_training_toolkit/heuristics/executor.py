
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
    def evaluate(
        self,
        condition: ParsedCondition,
        series: pd.Series,
        window: int = 20,
    ) -> tuple[bool, Optional[float]]:
        if len(series) == 0:
            return False, None

        recent = series.tail(window)
        if len(recent) == 0:
            return False, None

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
        if len(series) < window * 2:
            return False, None

        baseline = series.head(window)
        recent = series.tail(window)

        baseline_mean = float(baseline.mean())
        recent_mean = float(recent.mean())

        if baseline_mean == 0:
            return False, recent_mean

        drop_pct = ((baseline_mean - recent_mean) / abs(baseline_mean)) * 100

        return drop_pct >= percentage, recent_mean

    def _evaluate_spike(
        self,
        series: pd.Series,
        window: int,
        multiplier: float,
    ) -> tuple[bool, Optional[float]]:
        if len(series) < window:
            return False, None

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
    def __init__(self):
        self._parser = ConditionParser()
        self._evaluator = ConditionEvaluator()

    def run(
        self,
        df: pd.DataFrame,
        heuristics: List[YAMLHeuristic],
        trainer_type: str = "unknown",
    ) -> List[Insight]:
        insights: List[Insight] = []

        if df.empty or "step" not in df.columns:
            return insights

        num_steps = len(df)

        for heuristic in heuristics:
            if not heuristic.applies_to_trainer(trainer_type):
                continue

            if num_steps < heuristic.min_steps:
                continue

            if heuristic.metric not in df.columns:
                continue

            try:
                condition = self._parser.parse(heuristic.condition)
            except ValueError:
                continue

            series = df[heuristic.metric].astype(float)

            triggered, value = self._evaluator.evaluate(
                condition,
                series,
                window=heuristic.window,
            )

            if triggered:
                message = self._format_message(heuristic.message, value)

                insight = Insight(
                    type=heuristic.name,
                    severity=heuristic.severity,
                    message=message,
                    steps=None,
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
        insights = self.run(df, [heuristic], trainer_type)
        return insights[0] if insights else None

    def _format_message(self, template: str, value: Optional[float]) -> str:
        if value is None:
            import re
            return re.sub(r"\{value[^}]*\}", "N/A", template)

        try:
            return template.format(value=value)
        except (KeyError, ValueError):
            return template.replace("{value}", str(value))

def run_yaml_heuristics(
    df: pd.DataFrame,
    trainer_type: str = "unknown",
    custom_dirs: Optional[List[Path]] = None,
    custom_alerts: Optional[List[str]] = None,
    disable_builtin: bool = False,
) -> List[Insight]:
    from post_training_toolkit.heuristics.loader import HeuristicLoader
    from post_training_toolkit.heuristics.inline import parse_inline_alerts

    loader = HeuristicLoader(
        custom_dirs=custom_dirs,
        include_builtin=not disable_builtin,
    )
    yaml_heuristics = loader.load_for_trainer(trainer_type)

    inline_heuristics = []
    if custom_alerts:
        inline_heuristics = parse_inline_alerts(custom_alerts)

    all_heuristics = yaml_heuristics + inline_heuristics

    executor = YAMLHeuristicExecutor()
    return executor.run(df, all_heuristics, trainer_type)
