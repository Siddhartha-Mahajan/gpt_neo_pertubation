# Ranking Attention Heads by Operational Contribution

This repo reproduces the assignment workflow without notebooks. It loads GPT-Neo-125M, runs few-shot ICL sentiment classification on `IMDB_Dataset_100.csv`, perturbs each attention head in the last layer, and ranks heads by their impact on accuracy. Logs and plots are written to dedicated folders for inspection.

## Project layout

- `main.py` — orchestrates the full pipeline (data prep → baseline → perturbations → reporting).
- `config.py` — central experiment settings (model, seeds, perturbation knobs, paths).
- `data_utils.py` — dataset loading, cleaning, shuffling, and few-shot split.
- `model_utils.py` — seeding, model, and tokenizer loading.
- `prompt_utils.py` — prompt construction with safe truncation to fit the context window.
- `evaluation.py` — greedy generation and accuracy measurement with a fixed few-shot set.
- `perturbation.py` — c_proj discovery in the last block plus noise/zeroing sweeps per head.
- `plotting_utils.py` — saves heatmaps for contributions and robustness stats to `plots/`.
- `logs/` — run logs (auto-created).
- `plots/` — generated figures (auto-created).
- `head_perturbation_outproj_results.csv` — experiment output (overwritten on each run).

## Quick start

1) Install dependencies (prefer a fresh venv):

```bash
pip install -r requirements.txt
```

2) Run the full assignment pipeline:

```bash
python main.py
```

This will:

- Load and clean `IMDB_Dataset_100.csv`.
- Fix the few-shot set (`k_shot` in `config.py`) and evaluate the baseline accuracy (`result_original`).
- Locate the last-layer attention output projection (`attn.attention.out_proj`).
- For each head, inject Gaussian noise multiple times and optionally zero-out the head, measuring accuracy drops (`result_atth_i`).
- Rank heads by accuracy degradation and write a CSV plus plots.

## Outputs

- Logs: `logs/run_YYYYMMDD_HHMMSS.log` (console+file) with configuration, baseline, and per-head summaries.
- Metrics: `head_perturbation_outproj_results.csv` with noise/zeroing accuracies and contributions.
- Plots in `plots/`:
	- `head_contribution_heatmap.png`
	- `head_accuracy_heatmap.png`
	- `noise_stats_heatmap.png`
	- `head_noise_contrib_ranking.png` (sorted descending by noise contribution; negative values stay negative)

### Regenerate plots from existing CSV (no rerun)

If you already have perturbation results, regenerate plots without re-running the model:

```bash
python generate_plots_from_csv.py --csv head_perturbation_outproj_results_v2.csv
```

Use `--baseline <value>` to override the inferred baseline if needed.

## Prompt ablation (stronger prompt, separate harness)

A separate, non-intrusive harness lives in `prompt_ablation/` and does not modify the original pipeline. It uses a more explicit instruction prompt, balanced k-shot, and shorter label decoding.

Run it end-to-end:

```bash
python -m prompt_ablation.main
```

Outputs:

- Logs: `logs/prompt_ablation.log`
- Results CSV: `prompt_ablation/head_perturbation_outproj_results_ablation.csv`

Key prompt format:

```
You are a strict sentiment classifier. Given a review, respond with exactly one word: positive or negative.
### Example 1
Review: ...
Sentiment: positive
... (few-shot examples)
### Test Example
Review: <test review>
Answer:
```

The ablation pipeline recomputes the baseline with this prompt before running perturbations.

Status note (Dec 15 2025): The prompt-ablation variant yielded a lower baseline (≈0.41) and was halted mid-perturbation sweep (see `prompt_ablation/inference.md` and `logs/prompt_ablation.log`). The original harness remains recommended.

## Configuration knobs (edit `config.py`)

- `model_name`: Hugging Face identifier (default GPT-Neo 125M).
- `k_shot`: few-shot count.
- `max_test_examples`: limit eval size (None uses all remaining rows).
- `noise_eps`, `repeats`, `zeroing`: perturbation strength, Monte Carlo repeats, and whether to include zeroing ablation.
- `max_new_tokens`: generation length for the sentiment label.
- `target_cproj_name`: override if the last-layer projection name differs (printed in logs).

## Notes

- The pipeline uses greedy decoding for a binary sentiment decision; heuristic keyword matching maps text to `positive`/`negative`.
- If you change the dataset, ensure it has `review` and `sentiment` columns formatted for few-shot prompting.
- GPU is preferred; CPU works but will be slow for repeated perturbations.
