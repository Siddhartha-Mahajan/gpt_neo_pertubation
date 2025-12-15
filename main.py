from __future__ import annotations

from pathlib import Path

from config import ExperimentConfig
from data_utils import load_and_prepare_dataset
from evaluation import evaluate_model_on_evalset_fixed_fs
from logging_utils import setup_logging
from model_utils import load_model_and_tokenizer, set_global_seed
from perturbation import run_head_perturbation_experiment
from plotting_utils import plot_accuracy_heatmap, plot_contributions_heatmap, plot_noise_stats


def main() -> None:
    config = ExperimentConfig()
    log_file = config.ensure_directories()
    logger = setup_logging(log_file)

    logger.info("Device: %s", config.device)
    logger.info("Configuration: %s", config)

    set_global_seed(config.seed)

    logger.info("Loading dataset from %s", config.dataset_path)
    few_shot, eval_set, _ = load_and_prepare_dataset(
        config.dataset_path,
        k_shot=config.k_shot,
        seed=config.seed,
        max_test_examples=config.max_test_examples,
    )
    logger.info("Few-shot examples: %d | Eval examples: %d", len(few_shot), len(eval_set))

    model, tokenizer, n_layers, n_heads, embed_dim, head_dim = load_model_and_tokenizer(
        config.model_name,
        device=config.device,
    )
    logger.info(
        "Model %s | layers=%d heads=%d embed_dim=%d head_dim=%d",
        config.model_name,
        n_layers,
        n_heads,
        embed_dim,
        head_dim,
    )

    logger.info("Running baseline evaluation (few-shot ICL)")
    baseline_acc, _ = evaluate_model_on_evalset_fixed_fs(
        model,
        tokenizer,
        few_shot,
        eval_set,
        device=config.device,
        max_new_tokens=config.max_new_tokens,
        max_examples=config.max_test_examples,
    )
    logger.info("Baseline accuracy: %.4f (%.1f%%)", baseline_acc, baseline_acc * 100)

    logger.info("Starting head perturbation experiment")
    df_results = run_head_perturbation_experiment(
        model=model,
        tokenizer=tokenizer,
        few_shot_examples=few_shot,
        eval_set=eval_set,
        device=config.device,
        baseline_acc=baseline_acc,
        head_dim=head_dim,
        n_heads=n_heads,
        noise_eps=config.noise_eps,
        repeats=config.repeats,
        zeroing=config.zeroing,
        max_new_tokens=config.max_new_tokens,
        max_test_examples=config.max_test_examples,
        target_cproj_name=config.target_cproj_name,
        logger=logger,
    )

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(config.output_csv, index=False)
    logger.info("Saved head contribution results to %s", config.output_csv)

    logger.info("Creating plots")
    plot_paths = [
        plot_contributions_heatmap(df_results, config.plots_dir),
        plot_accuracy_heatmap(df_results, baseline_acc, config.plots_dir),
        plot_noise_stats(df_results, config.plots_dir),
    ]
    for p in plot_paths:
        logger.info("Saved plot: %s", p)

    logger.info("Done. Top 5 heads by zeroing contribution:\n%s", df_results.head(5))


if __name__ == "__main__":
    main()
