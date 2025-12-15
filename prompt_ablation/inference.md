# Prompt Ablation Run (Dec 15 2025)

- Run log: `logs/prompt_ablation.log` (seed 42, k_shot=6, max_new_tokens=4, repeats=2, zeroing=True).
- Baseline accuracy with the stricter prompt: **0.4086** (down from ~0.642 in the original harness).
- Because the baseline dropped sharply, the head-perturbation sweep was halted partway (progress logged through head index 3 when stopped).
- No new ranking conclusions drawn; this prompt variant underperformed and is not recommended as-is.

Recommendation: revert to the original prompt harness or redesign the instruction/few-shot set before rerunning perturbations.
