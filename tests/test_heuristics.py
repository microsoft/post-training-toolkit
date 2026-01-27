import pytest
import pandas as pd
import numpy as np

from post_training_toolkit.models.heuristics import (
    TrainerType,
    Insight,
    detect_reward_variance_spikes,
    detect_kl_instability,
    detect_policy_drift,
    detect_output_length_collapse,
    detect_refusal_regressions,
    detect_dpo_loss_random,
    detect_dpo_margin_collapse,
    detect_ppo_entropy_collapse,
    detect_ppo_value_head_divergence,
    detect_sft_loss_plateau,
    run_heuristics,
)

class TestInsight:
    
    def test_insight_creation(self):
        insight = Insight(
            type="test_insight",
            severity="high",
            message="Test message",
            steps=[1, 2, 3],
            data={"key": "value"},
            trainer_types={TrainerType.DPO},
            reference="Test reference",
        )
        assert insight.type == "test_insight"
        assert insight.severity == "high"
        assert insight.message == "Test message"
        assert insight.steps == [1, 2, 3]
        assert insight.data == {"key": "value"}
        assert TrainerType.DPO in insight.trainer_types
        assert insight.reference == "Test reference"

class TestCommonHeuristics:
    
    @pytest.fixture
    def stable_df(self):
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "step": range(n),
            "reward_mean": np.random.normal(0.5, 0.05, n),
            "reward_std": np.random.normal(0.1, 0.01, n),
            "kl": np.random.uniform(0.05, 0.10, n),
            "embedding_cosine_to_sft": np.random.uniform(0.94, 0.98, n),
            "output_length_mean": np.random.normal(100, 5, n),
            "refusal_rate": np.random.uniform(0.01, 0.05, n),
        })
    
    @pytest.fixture
    def unstable_df(self):
        n = 100
        df = pd.DataFrame({
            "step": range(n),
            "reward_mean": [0.5] * 50 + [2.0] * 50,
            "reward_std": [0.1] * 50 + [1.0] * 50,
            "kl": [0.1] * 50 + [0.5] * 50,
            "embedding_cosine_to_sft": [0.95] * 50 + [0.80] * 50,
            "output_length_mean": [100] * 50 + [30] * 50,
            "refusal_rate": [0.05] * 50 + [0.35] * 50,
        })
        return df
    
    def test_reward_variance_spikes_stable(self, stable_df):
        insights = detect_reward_variance_spikes(stable_df)
        assert len(insights) == 0
    
    def test_reward_variance_spikes_unstable(self, unstable_df):
        insights = detect_reward_variance_spikes(unstable_df)
        assert isinstance(insights, list)
    
    def test_kl_instability_stable(self, stable_df):
        insights = detect_kl_instability(stable_df)
        assert len(insights) == 0
    
    def test_kl_instability_unstable(self, unstable_df):
        insights = detect_kl_instability(unstable_df)
        assert len(insights) > 0
        assert any(i.type in ("kl_instability", "kl_above_target") for i in insights)
    
    def test_policy_drift_stable(self, stable_df):
        insights = detect_policy_drift(stable_df)
        assert len(insights) == 0
    
    def test_policy_drift_unstable(self, unstable_df):
        insights = detect_policy_drift(unstable_df)
        assert len(insights) > 0
        assert insights[0].type in ("policy_drift_alert", "policy_drift_warn")
    
    def test_output_length_collapse_stable(self, stable_df):
        insights = detect_output_length_collapse(stable_df)
        assert len(insights) == 0
    
    def test_output_length_collapse_unstable(self, unstable_df):
        insights = detect_output_length_collapse(unstable_df)
        assert len(insights) > 0
        assert insights[0].type == "length_collapse"
    
    def test_refusal_regressions_stable(self, stable_df):
        insights = detect_refusal_regressions(stable_df)
        assert len(insights) == 0
    
    def test_refusal_regressions_unstable(self, unstable_df):
        insights = detect_refusal_regressions(unstable_df)
        assert len(insights) > 0
        assert any(i.type in ("refusal_alert", "refusal_warn", "refusal_uptick") for i in insights)
    
    def test_missing_columns_graceful(self):
        empty_df = pd.DataFrame({"step": range(10)})
        assert detect_reward_variance_spikes(empty_df) == []
        assert detect_kl_instability(empty_df) == []
        assert detect_policy_drift(empty_df) == []
        assert detect_output_length_collapse(empty_df) == []
        assert detect_refusal_regressions(empty_df) == []

class TestDPOHeuristics:
    
    @pytest.fixture
    def healthy_dpo_df(self):
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "step": range(n),
            "dpo_loss": np.linspace(0.69, 0.45, n) + np.random.normal(0, 0.02, n),
            "reward_margin": np.linspace(0.1, 0.5, n) + np.random.normal(0, 0.03, n),
            "win_rate": np.linspace(0.5, 0.7, n) + np.random.normal(0, 0.02, n),
        })
    
    @pytest.fixture
    def stuck_dpo_df(self):
        n = 100
        return pd.DataFrame({
            "step": range(n),
            "dpo_loss": np.random.normal(0.693, 0.01, n),
            "reward_margin": np.random.normal(0.01, 0.005, n),
            "win_rate": np.random.normal(0.5, 0.02, n),
        })
    
    def test_dpo_loss_random_healthy(self, healthy_dpo_df):
        insights = detect_dpo_loss_random(healthy_dpo_df)
        assert len(insights) == 0
    
    def test_dpo_loss_random_stuck(self, stuck_dpo_df):
        insights = detect_dpo_loss_random(stuck_dpo_df)
        assert len(insights) > 0
        assert insights[0].type == "dpo_loss_random"
        assert insights[0].severity == "high"
    
    def test_dpo_margin_collapse_healthy(self, healthy_dpo_df):
        insights = detect_dpo_margin_collapse(healthy_dpo_df)
        assert len(insights) == 0
    
    def test_dpo_margin_collapse_stuck(self, stuck_dpo_df):
        n = 100
        df_with_collapse = pd.DataFrame({
            "step": range(n),
            "dpo_loss": np.random.normal(0.693, 0.01, n),
            "reward_margin": np.concatenate([
                np.random.normal(0.5, 0.05, 50),
                np.random.normal(0.01, 0.005, 50)
            ]),
            "win_rate": np.random.normal(0.5, 0.02, n),
        })
        insights = detect_dpo_margin_collapse(df_with_collapse)
        assert len(insights) > 0
        assert insights[0].type == "margin_collapse"

class TestPPOHeuristics:
    
    @pytest.fixture
    def healthy_ppo_df(self):
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "step": range(n),
            "entropy": np.linspace(1.5, 1.2, n) + np.random.normal(0, 0.05, n),
            "value_loss": np.random.uniform(0.1, 0.3, n),
            "ppo_loss": np.random.uniform(0.01, 0.05, n),
        })
    
    @pytest.fixture
    def collapsed_ppo_df(self):
        n = 100
        return pd.DataFrame({
            "step": range(n),
            "entropy": np.linspace(1.5, 0.01, n),
            "value_loss": np.linspace(0.1, 10.0, n),
            "ppo_loss": np.random.uniform(0.01, 0.05, n),
        })
    
    def test_ppo_entropy_collapse_healthy(self, healthy_ppo_df):
        insights = detect_ppo_entropy_collapse(healthy_ppo_df)
        assert len(insights) == 0
    
    def test_ppo_entropy_collapse_collapsing(self, collapsed_ppo_df):
        insights = detect_ppo_entropy_collapse(collapsed_ppo_df)
        assert len(insights) > 0
        assert insights[0].type == "entropy_collapse"
    
    def test_ppo_value_divergence_healthy(self, healthy_ppo_df):
        insights = detect_ppo_value_head_divergence(healthy_ppo_df)
        assert len(insights) == 0
    
    def test_ppo_value_divergence_exploding(self, collapsed_ppo_df):
        insights = detect_ppo_value_head_divergence(collapsed_ppo_df)
        assert len(insights) > 0
        assert insights[0].type == "value_head_divergence"

class TestSFTHeuristics:
    
    @pytest.fixture
    def healthy_sft_df(self):
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "step": range(n),
            "sft_loss": np.linspace(3.0, 1.5, n) + np.random.normal(0, 0.05, n),
        })
    
    @pytest.fixture
    def plateau_sft_df(self):
        n = 100
        return pd.DataFrame({
            "step": range(n),
            "sft_loss": np.concatenate([
                np.linspace(3.0, 2.0, 30),
                np.random.normal(2.0, 0.01, 70)
            ]),
        })
    
    def test_sft_loss_plateau_healthy(self, healthy_sft_df):
        insights = detect_sft_loss_plateau(healthy_sft_df)
        assert len(insights) == 0
    
    def test_sft_loss_plateau_stuck(self, plateau_sft_df):
        insights = detect_sft_loss_plateau(plateau_sft_df)
        assert len(insights) > 0
        assert insights[0].type == "sft_loss_plateau"

class TestRunAllHeuristics:
    
    def test_run_all_heuristics_dpo(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "step": range(100),
            "dpo_loss": np.random.normal(0.693, 0.01, 100),
            "reward_mean": np.random.normal(0.5, 0.05, 100),
            "kl": np.random.uniform(0.05, 0.10, 100),
        })
        insights = run_heuristics(df, trainer_type=TrainerType.DPO)
        assert isinstance(insights, list)
        assert any(i.type == "dpo_loss_random" for i in insights)
    
    def test_run_all_heuristics_ppo(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "step": range(100),
            "entropy": np.linspace(1.5, 0.01, 100),
            "reward_mean": np.random.normal(0.5, 0.05, 100),
            "kl": np.random.uniform(0.05, 0.10, 100),
        })
        insights = run_heuristics(df, trainer_type=TrainerType.PPO)
        assert isinstance(insights, list)
        assert any(i.type == "entropy_collapse" for i in insights)
    
    def test_run_all_heuristics_filters_by_trainer(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "step": range(100),
            "dpo_loss": np.random.normal(0.693, 0.01, 100),
            "entropy": np.linspace(1.5, 0.01, 100),
        })
        
        dpo_insights = run_heuristics(df, trainer_type=TrainerType.DPO)
        assert not any(i.type == "entropy_collapse" for i in dpo_insights)
        
        ppo_insights = run_heuristics(df, trainer_type=TrainerType.PPO)
        assert not any(i.type == "dpo_loss_random" for i in ppo_insights)
