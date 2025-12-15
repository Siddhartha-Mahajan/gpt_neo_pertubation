from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm

from evaluation import evaluate_model_on_evalset_fixed_fs


def find_c_proj_module(last_block: nn.Module, embed_dim: int, target_name: str | None = None) -> tuple[nn.Linear, str]:
    """Locate the attention output projection (c_proj) in the last block."""
    c_proj: Optional[nn.Linear] = None
    c_proj_name: Optional[str] = None

    if target_name:
        for name, mod in last_block.named_modules():
            if name == target_name and isinstance(mod, nn.Linear):
                c_proj, c_proj_name = mod, name
                break

    if c_proj is None:
        for name, mod in last_block.named_modules():
            if isinstance(mod, nn.Linear) and any(k in name.lower() for k in ["out", "o_proj", "out_proj", "c_proj"]):
                if tuple(mod.weight.shape) == (embed_dim, embed_dim):
                    c_proj, c_proj_name = mod, name
                    break

    if c_proj is None:
        for name, mod in last_block.named_modules():
            if isinstance(mod, nn.Linear) and tuple(mod.weight.shape) == (embed_dim, embed_dim):
                c_proj, c_proj_name = mod, name
                break

    if c_proj is None:
        raise RuntimeError("Could not locate c_proj in the last block.")

    return c_proj, c_proj_name or "unknown"


def _perturb_once(c_proj_layer: nn.Linear, head_idx: int, head_dim: int, noise_eps: float) -> callable:
    W = c_proj_layer.weight.data
    b = c_proj_layer.bias.data if c_proj_layer.bias is not None else None
    start = head_idx * head_dim
    end = (head_idx + 1) * head_dim

    W_backup = W.clone()
    b_backup = b.clone() if b is not None else None
    noise = noise_eps * torch.randn_like(W[:, start:end], device=W.device)
    W[:, start:end] += noise

    def restore():
        c_proj_layer.weight.data.copy_(W_backup)
        if b is not None:
            c_proj_layer.bias.data.copy_(b_backup)

    return restore


def _zero_once(c_proj_layer: nn.Linear, head_idx: int, head_dim: int) -> callable:
    W = c_proj_layer.weight.data
    b = c_proj_layer.bias.data if c_proj_layer.bias is not None else None
    start = head_idx * head_dim
    end = (head_idx + 1) * head_dim

    W_backup = W.clone()
    b_backup = b.clone() if b is not None else None
    W[:, start:end] = 0.0

    def restore():
        c_proj_layer.weight.data.copy_(W_backup)
        if b is not None:
            c_proj_layer.bias.data.copy_(b_backup)

    return restore


def run_head_perturbation_experiment(
    model,
    tokenizer,
    few_shot_examples,
    eval_set,
    device: str,
    baseline_acc: float,
    head_dim: int,
    n_heads: int,
    noise_eps: float,
    repeats: int,
    zeroing: bool,
    max_new_tokens: int,
    max_test_examples: int | None,
    target_cproj_name: str,
    logger,
) -> pd.DataFrame:
    last_block = model.transformer.h[-1]
    c_proj, c_proj_name = find_c_proj_module(last_block, embed_dim=head_dim * n_heads, target_name=target_cproj_name)
    logger.info("Using c_proj='%s' with shape=%s", c_proj_name, tuple(c_proj.weight.shape))

    head_results: List[dict] = []
    logger.info("Running perturbation across %d heads (noise_eps=%.4f, repeats=%d, zeroing=%s)", n_heads, noise_eps, repeats, zeroing)

    for head_idx in range(n_heads):
        logger.info("Head %d/%d", head_idx, n_heads - 1)
        noise_accs = []
        for r in range(repeats):
            restore = _perturb_once(c_proj, head_idx, head_dim, noise_eps)
            acc_r, _ = evaluate_model_on_evalset_fixed_fs(
                model,
                tokenizer,
                few_shot_examples,
                eval_set,
                device=device,
                max_new_tokens=max_new_tokens,
                max_examples=max_test_examples,
            )
            restore()
            noise_accs.append(acc_r)
            logger.debug(" head=%d repeat=%d acc=%.4f", head_idx, r + 1, acc_r)

        noise_mean = float(torch.tensor(noise_accs).mean().item())
        noise_std = float(torch.tensor(noise_accs).std(unbiased=True).item()) if repeats > 1 else 0.0
        noise_contrib = baseline_acc - noise_mean

        zero_acc = None
        zero_contrib = None
        if zeroing:
            restore = _zero_once(c_proj, head_idx, head_dim)
            zero_acc, _ = evaluate_model_on_evalset_fixed_fs(
                model,
                tokenizer,
                few_shot_examples,
                eval_set,
                device=device,
                max_new_tokens=max_new_tokens,
                max_examples=max_test_examples,
            )
            restore()
            zero_contrib = baseline_acc - zero_acc
            logger.debug(" head=%d zero_acc=%.4f", head_idx, zero_acc)

        head_results.append(
            {
                "head": head_idx,
                "noise_mean_acc": noise_mean,
                "noise_std_acc": noise_std,
                "noise_contrib": noise_contrib,
                "zero_acc": zero_acc,
                "zero_contrib": zero_contrib,
            }
        )

    df_results = pd.DataFrame(head_results)
    df_results = df_results.sort_values(by="zero_contrib", ascending=False).reset_index(drop=True)
    df_results.index = df_results.index + 1
    logger.info("Finished perturbations. Example top rows:\n%s", df_results.head())
    return df_results
