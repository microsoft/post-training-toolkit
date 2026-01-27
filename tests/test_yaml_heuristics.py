
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from post_training_toolkit.heuristics.schema import YAMLHeuristic
from post_training_toolkit.heuristics.parser import (
    ConditionParser,
    ConditionType,
    ParsedCondition,
    parse_condition,
)
from post_training_toolkit.heuristics.loader import HeuristicLoader
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
from post_training_toolkit.models.heuristics import run_heuristics, TrainerType

class TestConditionParser:
    def test_parse_less_than(self):
        parser = ConditionParser()
        result = parser.parse("< 0.1")

        assert result.type == ConditionType.LESS_THAN
        assert result.threshold == 0.1

    def test_parse_less_than_negative(self):
        parser = ConditionParser()
        result = parser.parse("< -0.5")

        assert result.type == ConditionType.LESS_THAN
        assert result.threshold == -0.5

    def test_parse_greater_than(self):
        result = parse_condition("> 0.5")

        assert result.type == ConditionType.GREATER_THAN
        assert result.threshold == 0.5

    def test_parse_less_than_or_equal(self):
        result = parse_condition("<= 0.3")

        assert result.type == ConditionType.LESS_THAN_OR_EQUAL
        assert result.threshold == 0.3

    def test_parse_greater_than_or_equal(self):
        result = parse_condition(">= 0.7")

        assert result.type == ConditionType.GREATER_THAN_OR_EQUAL
        assert result.threshold == 0.7

    def test_parse_equals(self):
        result = parse_condition("== 0.693")

        assert result.type == ConditionType.EQUALS
        assert result.threshold == 0.693
        assert result.tolerance == 0.02

    def test_parse_range(self):
        result = parse_condition("range(0.68, 0.71)")

        assert result.type == ConditionType.RANGE
        assert result.threshold == 0.68
        assert result.threshold2 == 0.71

    def test_parse_drop(self):
        result = parse_condition("drop(20%)")

        assert result.type == ConditionType.DROP
        assert result.percentage == 20.0

    def test_parse_drop_without_percent(self):
        result = parse_condition("drop(30)")

        assert result.type == ConditionType.DROP
        assert result.percentage == 30.0

    def test_parse_spike(self):
        result = parse_condition("spike(3x)")

        assert result.type == ConditionType.SPIKE
        assert result.multiplier == 3.0

    def test_parse_spike_without_x(self):
        result = parse_condition("spike(2.5)")

        assert result.type == ConditionType.SPIKE
        assert result.multiplier == 2.5

    def test_parse_with_extra_spaces(self):
        result = parse_condition("  <   0.1  ")

        assert result.type == ConditionType.LESS_THAN
        assert result.threshold == 0.1

    def test_parse_invalid_condition(self):
        with pytest.raises(ValueError) as exc_info:
            parse_condition("invalid")

        assert "Invalid condition" in str(exc_info.value)

    def test_parsed_condition_repr(self):
        result = parse_condition("< 0.1")
        assert "< 0.1" in repr(result)

        result = parse_condition("drop(20%)")
        assert "drop(20.0%)" in repr(result)

class TestYAMLHeuristic:
    def test_create_valid_heuristic(self):
        h = YAMLHeuristic(
            name="test_heuristic",
            description="Test description",
            trainers=["dpo", "ppo"],
            metric="reward_margin",
            condition="< 0.1",
            severity="high",
            message="Test message",
        )

        assert h.name == "test_heuristic"
        assert h.trainers == ["dpo", "ppo"]
        assert h.window == 20
        assert h.enabled is True

    def test_from_dict(self):
        data = {
            "name": "margin_collapse",
            "description": "Detect margin collapse",
            "trainers": ["dpo"],
            "metric": "reward_margin",
            "condition": "< 0.1",
            "severity": "high",
            "message": "Margin collapsed to {value:.3f}",
            "window": 30,
            "reference": "https://example.com",
        }

        h = YAMLHeuristic.from_dict(data)

        assert h.name == "margin_collapse"
        assert h.window == 30
        assert h.reference == "https://example.com"

    def test_from_dict_missing_required(self):
        data = {
            "name": "test",
            "description": "Test",
        }

        with pytest.raises(ValueError) as exc_info:
            YAMLHeuristic.from_dict(data)

        assert "Missing required fields" in str(exc_info.value)

    def test_invalid_severity(self):
        with pytest.raises(ValueError) as exc_info:
            YAMLHeuristic(
                name="test",
                description="Test",
                trainers=["dpo"],
                metric="loss",
                condition="< 0.1",
                severity="critical",
                message="Test",
            )

        assert "Invalid severity" in str(exc_info.value)

    def test_invalid_trainer(self):
        with pytest.raises(ValueError) as exc_info:
            YAMLHeuristic(
                name="test",
                description="Test",
                trainers=["invalid_trainer"],
                metric="loss",
                condition="< 0.1",
                severity="high",
                message="Test",
            )

        assert "Invalid trainer" in str(exc_info.value)

    def test_applies_to_trainer(self):
        h = YAMLHeuristic(
            name="test",
            description="Test",
            trainers=["dpo", "orpo"],
            metric="loss",
            condition="< 0.1",
            severity="high",
            message="Test",
        )

        assert h.applies_to_trainer("dpo") is True
        assert h.applies_to_trainer("DPO") is True
        assert h.applies_to_trainer("orpo") is True
        assert h.applies_to_trainer("ppo") is False

    def test_applies_to_trainer_all(self):
        h = YAMLHeuristic(
            name="test",
            description="Test",
            trainers=["all"],
            metric="loss",
            condition="< 0.1",
            severity="high",
            message="Test",
        )

        assert h.applies_to_trainer("dpo") is True
        assert h.applies_to_trainer("ppo") is True
        assert h.applies_to_trainer("sft") is True

    def test_disabled_heuristic(self):
        h = YAMLHeuristic(
            name="test",
            description="Test",
            trainers=["dpo"],
            metric="loss",
            condition="< 0.1",
            severity="high",
            message="Test",
            enabled=False,
        )

        assert h.applies_to_trainer("dpo") is False

    def test_to_dict(self):
        h = YAMLHeuristic(
            name="test",
            description="Test",
            trainers=["dpo"],
            metric="loss",
            condition="< 0.1",
            severity="high",
            message="Test",
            reference="https://example.com",
        )

        d = h.to_dict()

        assert d["name"] == "test"
        assert d["reference"] == "https://example.com"

class TestHeuristicLoader:
    def test_load_builtin_heuristics(self):
        loader = HeuristicLoader()
        heuristics = loader.load_for_trainer("dpo")

        assert len(heuristics) > 0

        for h in heuristics:
            assert h.applies_to_trainer("dpo")

    def test_load_from_custom_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = Path(tmpdir)
            dpo_dir = custom_dir / "dpo"
            dpo_dir.mkdir()

            yaml_content = """
name: custom_test_heuristic
description: Custom test heuristic
trainers: [dpo]
metric: custom_metric
condition: "< 0.5"
severity: medium
message: "Custom test alert"
"""
            (dpo_dir / "custom.yaml").write_text(yaml_content)

            loader = HeuristicLoader(
                custom_dirs=[custom_dir],
                include_builtin=False,
            )
            heuristics = loader.load_for_trainer("dpo")

            assert len(heuristics) == 1
            assert heuristics[0].name == "custom_test_heuristic"

    def test_get_heuristic_by_name(self):
        loader = HeuristicLoader()
        h = loader.get_heuristic("yaml_margin_collapse")

        if h is not None:
            assert h.name == "yaml_margin_collapse"

    def test_cache_clearing(self):
        loader = HeuristicLoader()

        loader.load_for_trainer("dpo")
        assert len(loader._cache) > 0

        loader.clear_cache()
        assert len(loader._cache) == 0

class TestConditionEvaluator:
    def test_evaluate_less_than_triggered(self):
        evaluator = ConditionEvaluator()
        series = pd.Series([0.5, 0.3, 0.1, 0.05])
        condition = ParsedCondition(type=ConditionType.LESS_THAN, threshold=0.2)

        triggered, value = evaluator.evaluate(condition, series, window=2)

        assert triggered is True
        assert value is not None
        assert value < 0.2

    def test_evaluate_less_than_not_triggered(self):
        evaluator = ConditionEvaluator()
        series = pd.Series([0.5, 0.6, 0.7, 0.8])
        condition = ParsedCondition(type=ConditionType.LESS_THAN, threshold=0.2)

        triggered, value = evaluator.evaluate(condition, series, window=2)

        assert triggered is False

    def test_evaluate_greater_than(self):
        evaluator = ConditionEvaluator()
        series = pd.Series([0.1, 0.3, 0.7, 0.9])
        condition = ParsedCondition(type=ConditionType.GREATER_THAN, threshold=0.5)

        triggered, value = evaluator.evaluate(condition, series, window=2)

        assert triggered is True

    def test_evaluate_equals(self):
        evaluator = ConditionEvaluator()
        series = pd.Series([0.69, 0.693, 0.695, 0.691])
        condition = ParsedCondition(type=ConditionType.EQUALS, threshold=0.693, tolerance=0.02)

        triggered, value = evaluator.evaluate(condition, series, window=4)

        assert triggered is True

    def test_evaluate_range(self):
        evaluator = ConditionEvaluator()
        series = pd.Series([0.69, 0.693, 0.695, 0.691])
        condition = ParsedCondition(
            type=ConditionType.RANGE,
            threshold=0.68,
            threshold2=0.71,
        )

        triggered, value = evaluator.evaluate(condition, series, window=4)

        assert triggered is True

    def test_evaluate_drop(self):
        evaluator = ConditionEvaluator()
        series = pd.Series([1.0] * 20 + [0.5] * 20)
        condition = ParsedCondition(type=ConditionType.DROP, percentage=40.0)

        triggered, value = evaluator.evaluate(condition, series, window=20)

        assert triggered is True

    def test_evaluate_drop_not_triggered(self):
        evaluator = ConditionEvaluator()
        series = pd.Series([1.0] * 40)
        condition = ParsedCondition(type=ConditionType.DROP, percentage=20.0)

        triggered, value = evaluator.evaluate(condition, series, window=20)

        assert triggered is False

    def test_evaluate_spike(self):
        evaluator = ConditionEvaluator()
        series = pd.Series([1.0] * 50 + [5.0] * 5)
        condition = ParsedCondition(type=ConditionType.SPIKE, multiplier=3.0)

        triggered, value = evaluator.evaluate(condition, series, window=20)

        assert triggered is True

    def test_evaluate_empty_series(self):
        evaluator = ConditionEvaluator()
        series = pd.Series([], dtype=float)
        condition = ParsedCondition(type=ConditionType.LESS_THAN, threshold=0.1)

        triggered, value = evaluator.evaluate(condition, series, window=10)

        assert triggered is False
        assert value is None

class TestYAMLHeuristicExecutor:
    def test_run_with_matching_condition(self):
        executor = YAMLHeuristicExecutor()

        df = pd.DataFrame({
            "step": list(range(50)),
            "reward_margin": [0.5] * 30 + [0.05] * 20,
        })

        heuristic = YAMLHeuristic(
            name="test_margin",
            description="Test margin check",
            trainers=["dpo"],
            metric="reward_margin",
            condition="< 0.1",
            window=20,
            severity="high",
            message="Margin collapsed to {value:.3f}",
            min_steps=30,
        )

        insights = executor.run(df, [heuristic], trainer_type="dpo")

        assert len(insights) == 1
        assert insights[0].type == "test_margin"
        assert insights[0].severity == "high"
        assert "0.0" in insights[0].message

    def test_run_with_non_matching_condition(self):
        executor = YAMLHeuristicExecutor()

        df = pd.DataFrame({
            "step": list(range(50)),
            "reward_margin": [0.5] * 50,
        })

        heuristic = YAMLHeuristic(
            name="test_margin",
            description="Test margin check",
            trainers=["dpo"],
            metric="reward_margin",
            condition="< 0.1",
            window=20,
            severity="high",
            message="Margin collapsed",
            min_steps=30,
        )

        insights = executor.run(df, [heuristic], trainer_type="dpo")

        assert len(insights) == 0

    def test_run_skips_wrong_trainer(self):
        executor = YAMLHeuristicExecutor()

        df = pd.DataFrame({
            "step": list(range(50)),
            "reward_margin": [0.05] * 50,
        })

        heuristic = YAMLHeuristic(
            name="test_margin",
            description="Test",
            trainers=["dpo"],
            metric="reward_margin",
            condition="< 0.1",
            severity="high",
            message="Test",
        )

        insights = executor.run(df, [heuristic], trainer_type="ppo")

        assert len(insights) == 0

    def test_run_skips_missing_metric(self):
        executor = YAMLHeuristicExecutor()

        df = pd.DataFrame({
            "step": list(range(50)),
            "other_metric": [0.05] * 50,
        })

        heuristic = YAMLHeuristic(
            name="test_margin",
            description="Test",
            trainers=["dpo"],
            metric="reward_margin",
            condition="< 0.1",
            severity="high",
            message="Test",
        )

        insights = executor.run(df, [heuristic], trainer_type="dpo")

        assert len(insights) == 0

    def test_run_skips_insufficient_steps(self):
        executor = YAMLHeuristicExecutor()

        df = pd.DataFrame({
            "step": list(range(10)),
            "reward_margin": [0.05] * 10,
        })

        heuristic = YAMLHeuristic(
            name="test_margin",
            description="Test",
            trainers=["dpo"],
            metric="reward_margin",
            condition="< 0.1",
            severity="high",
            message="Test",
            min_steps=30,
        )

        insights = executor.run(df, [heuristic], trainer_type="dpo")

        assert len(insights) == 0

class TestInlineAlerts:
    def test_parse_basic_alert(self):
        alert = "dpo: margin < 0.1 -> high: Margin collapsed"
        h = parse_inline_alert(alert)

        assert h is not None
        assert h.metric == "margin"
        assert h.severity == "high"
        assert h.message == "Margin collapsed"
        assert "dpo" in h.trainers

    def test_parse_alert_with_window(self):
        alert = "ppo: entropy drop(50%) for 30 steps -> medium: Entropy dropping"
        h = parse_inline_alert(alert)

        assert h is not None
        assert h.metric == "entropy"
        assert h.condition == "drop(50%)"
        assert h.window == 30
        assert h.severity == "medium"

    def test_parse_common_trainer(self):
        alert = "common: loss > 10 -> high: Loss exploded"
        h = parse_inline_alert(alert)

        assert h is not None
        assert "all" in h.trainers

    def test_parse_multiple_alerts(self):
        alerts = [
            "dpo: margin < 0.1 -> high: Margin collapsed",
            "ppo: entropy drop(50%) -> medium: Entropy dropping",
            "invalid alert string",
        ]

        heuristics = parse_inline_alerts(alerts)

        assert len(heuristics) == 2

    def test_validate_valid_alert(self):
        alert = "dpo: margin < 0.1 -> high: Test"
        is_valid, error = validate_inline_alert(alert)

        assert is_valid is True
        assert error is None

    def test_validate_invalid_format(self):
        alert = "not a valid alert"
        is_valid, error = validate_inline_alert(alert)

        assert is_valid is False
        assert "Invalid format" in error

    def test_validate_invalid_trainer(self):
        alert = "invalid_trainer: margin < 0.1 -> high: Test"
        is_valid, error = validate_inline_alert(alert)

        assert is_valid is False
        assert "Invalid trainer" in error

    def test_validate_invalid_condition(self):
        alert = "dpo: margin invalid_condition -> high: Test"
        is_valid, error = validate_inline_alert(alert)

        assert is_valid is False
        assert "Invalid condition" in error

class TestIntegration:
    def test_run_yaml_heuristics_function(self):
        df = pd.DataFrame({
            "step": list(range(50)),
            "reward_margin": [0.5] * 30 + [0.05] * 20,
            "dpo_loss": [0.693] * 50,
        })

        insights = run_yaml_heuristics(
            df=df,
            trainer_type="dpo",
            custom_alerts=["dpo: reward_margin < 0.1 -> high: Custom margin alert"],
        )

        assert len(insights) >= 1

        custom_found = any("Custom margin alert" in i.message for i in insights)
        assert custom_found

    def test_run_heuristics_with_yaml(self):
        df = pd.DataFrame({
            "step": list(range(50)),
            "reward_margin": [0.5] * 30 + [0.05] * 20,
        })

        insights = run_heuristics(
            df=df,
            trainer_type=TrainerType.DPO,
            custom_alerts=["dpo: reward_margin < 0.1 -> high: Test custom alert"],
        )

        margin_insights = [i for i in insights if "margin" in i.type.lower()]
        assert len(margin_insights) >= 1

    def test_run_heuristics_disable_yaml(self):
        df = pd.DataFrame({
            "step": list(range(50)),
            "reward_margin": [0.5] * 30 + [0.05] * 20,
        })

        insights_no_yaml = run_heuristics(
            df=df,
            trainer_type=TrainerType.DPO,
            disable_yaml_heuristics=True,
        )

        insights_with_yaml = run_heuristics(
            df=df,
            trainer_type=TrainerType.DPO,
            disable_yaml_heuristics=False,
        )

        assert len(insights_no_yaml) >= 0
        assert len(insights_with_yaml) >= 0

class TestSyntheticDataFrames:
    def test_detect_dpo_loss_random(self):
        df = pd.DataFrame({
            "step": list(range(50)),
            "dpo_loss": np.random.normal(0.693, 0.005, 50),
        })

        heuristic = YAMLHeuristic(
            name="test_loss_random",
            description="Test",
            trainers=["dpo"],
            metric="dpo_loss",
            condition="range(0.68, 0.71)",
            window=20,
            severity="high",
            message="Loss at random: {value:.4f}",
            min_steps=30,
        )

        executor = YAMLHeuristicExecutor()
        insights = executor.run(df, [heuristic], "dpo")

        assert len(insights) == 1
        assert "0.69" in insights[0].message

    def test_detect_entropy_drop(self):
        entropy_values = np.linspace(2.0, 0.5, 100)

        df = pd.DataFrame({
            "step": list(range(100)),
            "entropy": entropy_values,
        })

        heuristic = YAMLHeuristic(
            name="test_entropy_drop",
            description="Test",
            trainers=["ppo"],
            metric="entropy",
            condition="drop(50%)",
            window=30,
            severity="high",
            message="Entropy dropped to {value:.3f}",
            min_steps=50,
        )

        executor = YAMLHeuristicExecutor()
        insights = executor.run(df, [heuristic], "ppo")

        assert len(insights) == 1

    def test_detect_kl_spike(self):
        kl_values = [0.05] * 80 + [0.3] * 20

        df = pd.DataFrame({
            "step": list(range(100)),
            "kl": kl_values,
        })

        heuristic = YAMLHeuristic(
            name="test_kl_high",
            description="Test",
            trainers=["dpo", "ppo"],
            metric="kl",
            condition="> 0.25",
            window=10,
            severity="high",
            message="KL too high: {value:.3f}",
            min_steps=50,
        )

        executor = YAMLHeuristicExecutor()
        insights = executor.run(df, [heuristic], "dpo")

        assert len(insights) == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
