from __future__ import annotations

from typing import List, Tuple

import torch
from tqdm.auto import tqdm

from prompt_utils import safe_build_prompt_and_encode


def predict_sentiment_for_input_ids(input_ids, model, tokenizer, device: str, max_new_tokens: int):
    """Generate a continuation and map to a binary sentiment label."""
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=None,
        )
    gen_tokens = outputs[0, input_ids.shape[-1] :]
    generated = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().lower()

    if "positive" in generated:
        pred = "positive"
    elif "negative" in generated:
        pred = "negative"
    elif generated.startswith("pos") or "good" in generated or "great" in generated:
        pred = "positive"
    elif generated.startswith("neg") or "bad" in generated or "terrible" in generated:
        pred = "negative"
    else:
        pred = "positive"
    return pred, generated


def evaluate_model_on_evalset_fixed_fs(
    model,
    tokenizer,
    few_shot_examples_df,
    eval_set_df,
    device: str,
    max_new_tokens: int,
    max_examples: int | None = None,
    verbose: bool = False,
):
    """Evaluate using one fixed few-shot set for all examples."""
    correct = 0
    total = 0
    details: List[dict] = []

    fs_pairs: List[Tuple[str, str]] = list(
        zip(few_shot_examples_df["review"].tolist(), few_shot_examples_df["sentiment"].tolist())
    )

    idxs = list(range(len(eval_set_df))) if max_examples is None else list(range(min(max_examples, len(eval_set_df))))

    for i in tqdm(idxs, desc="Evaluating", leave=False):
        test_row = eval_set_df.iloc[i]
        test_review = test_row["review"]
        true_label = test_row["sentiment"]

        prompt_text, input_ids = safe_build_prompt_and_encode(
            test_review,
            fs_pairs,
            tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            verbose=False,
        )

        try:
            pred, gen_text = predict_sentiment_for_input_ids(
                input_ids,
                model,
                tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
            )
        except IndexError:
            prompt_text, input_ids = safe_build_prompt_and_encode(
                test_review,
                [],
                tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                verbose=True,
            )
            pred, gen_text = predict_sentiment_for_input_ids(
                input_ids,
                model,
                tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
            )

        is_correct = pred == true_label
        correct += int(is_correct)
        total += 1
        details.append(
            {
                "index": i,
                "true": true_label,
                "pred": pred,
                "gen": gen_text,
                "prompt_len": input_ids.shape[-1],
            }
        )

    acc = correct / total if total > 0 else 0.0
    return acc, details
