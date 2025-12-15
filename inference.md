# Quick Inference from head_perturbation_outproj_results_v2.csv

## Context
- Baseline accuracy (inferred): ~0.642 (per stored contributions and accuracies).
- Noise contribution = baseline − noise_mean_acc; Zeroing contribution = baseline − zero_acc. Positive values mean the head helps performance; negative means the perturbation slightly improves or leaves performance unchanged.

## Top heads by zeroing contribution (most impactful)
1) Head 10: zero_acc=0.6000, zero_contrib=+0.0421
2) Head 2: zero_acc=0.6105, zero_contrib=+0.0316
3) Heads 1/3/7: zero_acc≈0.6316, zero_contrib=+0.0105 (tie)

## Top heads by noise contribution (drop under noise)
1) Head 0: noise_contrib=+0.0126, noise_mean_acc=0.6295
2) Head 7/9/5: noise_contrib=+0.0105, noise_mean_acc≈0.6316
3) Head 8: noise_contrib=+0.0084, noise_mean_acc=0.6337
4) Head 10/3/1/11: noise_contrib=+0.0063

## Heads showing slight negative contribution (possible redundancy)
- Head 4: zero_contrib=-0.0105 (zeroing improves accuracy); noise_contrib=+0.0042 (mildly helpful under noise)
- Head 5: zero_contrib=-0.0105 (zeroing improves accuracy); noise_contrib=+0.0105
- Head 6: zero_contrib=-0.0211 (zeroing improves accuracy most strongly); noise_contrib=+0.0021

## Takeaways
- Head 10 is the most operationally critical in the last layer (largest accuracy drop when zeroed).
- Head 2 is also important but with smaller sensitivity to noise than to zeroing.
- Heads 4, 5, and especially 6 may be redundant or even slightly harmful under this setup (zeroing helps).
- Noise sensitivity highlights different heads (0, 7/9/5) than zeroing, suggesting some heads degrade gradually under noise but fail less catastrophically when removed.

## Next steps
- Validate with a larger eval set to confirm head rankings stability.
- Consider targeted regularization or pruning trials on low/negative-contribution heads (e.g., head 6) to see if performance holds.
- Re-run with multiple seeds to check robustness of the ordering.
