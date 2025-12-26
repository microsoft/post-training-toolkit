## RLHF Run Diagnostic Report

Generated: 2025-12-10T05:18:55.105612Z

Status: Unstable

### Run Summary
- Steps: 24
- Final DPO Loss: 0.6931
- Mean Win Rate: 64.6%

### Key Insights


1. [HIGH] KL exceeded hard cap (0.30) at steps like [2, 3, 4, 5, 6]... (steps: [2, 3, 4, 5, 6, 7]...)

2. [HIGH] Policy cosine to SFT dropped below 0.88. (steps: [2, 3, 4, 5, 6, 7]...)

3. [HIGH] Instability hotspot detected around step ~15. (steps: [15])

4. [HIGH] DPO loss stuck at ~0.693 (random chance). Model may not be learning preferences.

5. [MEDIUM] KL shows high short-term volatility. (steps: [3, 4, 5, 6, 7, 8]...)

6. [MEDIUM] Win rate shows high volatility (std=0.53), indicating inconsistent preference learning.



### Recommended Actions


- Adjust KL schedule or reduce learning rate.

- Increase KL strength or add anchor tasks to reduce drift.

- Increase batch size or use gradient clipping; smooth rewards.

- DPO loss stuck at random: increase learning rate, check data quality, or try larger beta.

- Win rate unstable: increase batch size for more stable gradient estimates.



### Plots

![Reward](plots/reward.png)


![KL](plots/kl.png)


![Drift](plots/drift.png)


![Slices](plots/slices.png)


