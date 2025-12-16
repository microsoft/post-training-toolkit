## RLHF Run Diagnostic Report

Generated: {{ generated_at }}

**Trainer:** {{ trainer_type }} | **Status:** {{ status }}

### Run Summary
- Steps: {{ summary.num_steps }}
{%- if summary.final_reward_mean is defined and summary.final_reward_mean == summary.final_reward_mean %}
- Final Reward Mean: {{ "%.3f"|format(summary.final_reward_mean) }}
{%- endif %}
{%- if summary.final_kl is defined and summary.final_kl == summary.final_kl %}
- Final KL: {{ "%.3f"|format(summary.final_kl) }}
{%- endif %}
{%- if summary.final_refusal_rate is defined and summary.final_refusal_rate == summary.final_refusal_rate %}
- Final Refusal Rate: {{ "%.3f"|format(summary.final_refusal_rate) }}
{%- endif %}
{%- if summary.final_cosine is defined and summary.final_cosine == summary.final_cosine %}
- Policy Cosine to SFT (final): {{ "%.3f"|format(summary.final_cosine) }}
{%- endif %}
{# DPO-specific metrics #}
{%- if summary.final_dpo_loss is defined and summary.final_dpo_loss == summary.final_dpo_loss %}
- Final DPO Loss: {{ "%.4f"|format(summary.final_dpo_loss) }}
{%- endif %}
{%- if summary.mean_win_rate is defined and summary.mean_win_rate == summary.mean_win_rate %}
- Mean Win Rate: {{ "%.1f"|format(summary.mean_win_rate * 100) }}%
{%- endif %}
{%- if summary.final_win_rate is defined and summary.final_win_rate == summary.final_win_rate %}
- Final Win Rate: {{ "%.1f"|format(summary.final_win_rate * 100) }}%
{%- endif %}
{# PPO-specific metrics #}
{%- if summary.final_ppo_loss is defined and summary.final_ppo_loss == summary.final_ppo_loss %}
- Final PPO Loss: {{ "%.4f"|format(summary.final_ppo_loss) }}
{%- endif %}
{%- if summary.final_entropy is defined and summary.final_entropy == summary.final_entropy %}
- Final Entropy: {{ "%.3f"|format(summary.final_entropy) }}
{%- endif %}
{%- if summary.final_value_loss is defined and summary.final_value_loss == summary.final_value_loss %}
- Final Value Loss: {{ "%.4f"|format(summary.final_value_loss) }}
{%- endif %}
{%- if summary.mean_clip_fraction is defined and summary.mean_clip_fraction == summary.mean_clip_fraction %}
- Mean Clip Fraction: {{ "%.3f"|format(summary.mean_clip_fraction) }}
{%- endif %}
{# SFT-specific metrics #}
{%- if summary.final_sft_loss is defined and summary.final_sft_loss == summary.final_sft_loss %}
- Final SFT Loss: {{ "%.4f"|format(summary.final_sft_loss) }}
{%- endif %}
{%- if summary.final_perplexity is defined and summary.final_perplexity == summary.final_perplexity %}
- Final Perplexity: {{ "%.2f"|format(summary.final_perplexity) }}
{%- endif %}
{# ORPO-specific metrics #}
{%- if summary.final_orpo_loss is defined and summary.final_orpo_loss == summary.final_orpo_loss %}
- Final ORPO Loss: {{ "%.4f"|format(summary.final_orpo_loss) }}
{%- endif %}
{%- if summary.final_log_odds_ratio is defined and summary.final_log_odds_ratio == summary.final_log_odds_ratio %}
- Final Log Odds Ratio: {{ "%.3f"|format(summary.final_log_odds_ratio) }}
{%- endif %}
{# KTO-specific metrics #}
{%- if summary.final_kto_loss is defined and summary.final_kto_loss == summary.final_kto_loss %}
- Final KTO Loss: {{ "%.4f"|format(summary.final_kto_loss) }}
{%- endif %}

### Key Insights
{% if insights %}
{% for ins in insights %}
{{ loop.index }}. [{{ ins.severity|upper }}] {{ ins.message }}{% if ins.steps %} (steps: {{ ins.steps[:6] }}{% if ins.steps|length > 6 %}...{% endif %}){% endif %}
{%- if ins.reference %}
   *Ref: {{ ins.reference }}*
{%- endif %}
{% endfor %}
{% else %}
No significant issues detected. Training appears stable.
{% endif %}

{% if behavior_drift %}
### Behavior Drift Analysis
**Overall Drift:** {{ behavior_drift.severity|upper }}

{% if behavior_drift.refusal_trend %}
#### Refusal Rate Trend
- Initial: {{ "%.1f"|format(behavior_drift.refusal_trend.initial * 100) }}%
- Final: {{ "%.1f"|format(behavior_drift.refusal_trend.final * 100) }}%
- Change: {{ "%+.1f"|format(behavior_drift.refusal_trend.delta * 100) }}%
{% if behavior_drift.refusal_trend.delta > 0.1 %}
⚠️ Significant increase in refusals detected
{% endif %}
{% endif %}

{% if behavior_drift.length_trend %}
#### Output Length Trend
- Mean Change: {{ "%+.1f"|format(behavior_drift.length_trend.mean_delta) }} chars
- Increased: {{ behavior_drift.length_trend.increased }} prompts
- Decreased: {{ behavior_drift.length_trend.decreased }} prompts
{% endif %}

{% if behavior_drift.flagged_prompts %}
#### Flagged Prompts ({{ behavior_drift.flagged_prompts|length }})
{% for p in behavior_drift.flagged_prompts[:5] %}
- {{ p.id }}: {{ p.reason }}
{% endfor %}
{% if behavior_drift.flagged_prompts|length > 5 %}
... and {{ behavior_drift.flagged_prompts|length - 5 }} more
{% endif %}
{% endif %}
{% endif %}

{% if checkpoint_recommendation %}
### Checkpoint Recommendation
**Recommended:** Step {{ checkpoint_recommendation.step }}

{{ checkpoint_recommendation.justification }}

| Checkpoint | Stability Score | Drift Score | Refusal Rate |
|------------|-----------------|-------------|--------------|
{% for cp in checkpoint_recommendation.candidates[:5] %}
| {{ cp.step }} | {{ "%.2f"|format(cp.stability) }} | {{ "%.2f"|format(cp.drift) }} | {{ "%.1f"|format(cp.refusal_rate * 100) }}% |
{% endfor %}
{% endif %}

{% if postmortem %}
### Postmortem
**Exit Reason:** {{ postmortem.exit_reason }}
- Last Step: {{ postmortem.last_step }}
- Timestamp: {{ postmortem.timestamp }}
{% if postmortem.traceback %}

Traceback (truncated):
```
{{ postmortem.traceback }}
```
{% endif %}
{% endif %}

### Recommended Actions
{% if actions %}
{% for action in actions %}
- {{ action }}
{% endfor %}
{% else %}
- No critical actions recommended based on current signals.
{% endif %}
