from __future__ import annotations

import logging
from pathlib import Path

from data_utils import load_and_prepare_dataset
from logging_utils import setup_logging
from model_utils import load_model_and_tokenizer, set_global_seed

from prompt_ablation.config import AblationConfig
from prompt_ablation.evaluation import evaluate
from prompt_ablation.perturbation import run


def main():
    cfg = AblationConfig()
    cfg.ensure_dirs()

    log_file = cfg.logs_dir / "prompt_ablation.log"
    logger = setup_logging(log_file, name="prompt_ablation")

    logger.info("Prompt ablation run with config: %s", cfg)
    set_global_seed(cfg.seed)

    few_shot, eval_set, _ = load_and_prepare_dataset(cfg.dataset_path, cfg.k_shot, cfg.seed, cfg.max_test_examples)
    logger.info("Few-shot=%d Eval=%d", len(few_shot), len(eval_set))

    model, tokenizer, n_layers, n_heads, embed_dim, head_dim = load_model_and_tokenizer(cfg.model_name, cfg.device)
    logger.info(
        "Model=%s layers=%d heads=%d embed_dim=%d head_dim=%d", cfg.model_name, n_layers, n_heads, embed_dim, head_dim
    )

    baseline_acc, _ = evaluate(
        model,
        tokenizer,
        few_shot,
        eval_set,
        device=cfg.device,
        instruction=cfg.instruction,
        delimiter=cfg.delimiter,
        answer_format=cfg.answer_format,
        max_new_tokens=cfg.max_new_tokens,
        max_examples=cfg.max_test_examples,
    )
    logger.info("Baseline (prompt ablation) accuracy: %.4f (%.1f%%)", baseline_acc, baseline_acc * 100)

    df_results = run(
        model,
        tokenizer,
        few_shot,
        eval_set,
        device=cfg.device,
        baseline_acc=baseline_acc,
        head_dim=head_dim,
        n_heads=n_heads,
        noise_eps=cfg.noise_eps,
        repeats=cfg.repeats,
        zeroing=cfg.zeroing,
        max_new_tokens=cfg.max_new_tokens,
        max_test_examples=cfg.max_test_examples,
        target_cproj_name=cfg.target_cproj_name,
        instruction=cfg.instruction,
        delimiter=cfg.delimiter,
        answer_format=cfg.answer_format,
        logger=logger,
    )

    df_results.to_csv(cfg.output_csv, index=False)
    logger.info("Saved ablation results to %s", cfg.output_csv)


if __name__ == "__main__":
    main()
