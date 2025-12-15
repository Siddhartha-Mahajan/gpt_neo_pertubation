from __future__ import annotations

from typing import List, Tuple

import torch
from tqdm.auto import tqdm

from prompt_ablation.prompt_utils import safe_build_prompt_and_encode


def predict_label(input_ids, model, tokenizer, device: str, max_new_tokens: int):
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

    if generated.startswith("positive"):
        pred = "positive"
    elif generated.startswith("negative"):
        pred = "negative"
    elif "positive" in generated:
        pred = "positive"
    elif "negative" in generated:
        pred = "negative"
    else:
        pred = "positive"
    return pred, generated


def evaluate(
    model,
    tokenizer,
    few_shot_df,
    eval_df,
    device: str,
    instruction: str,
    delimiter: str,
    answer_format: str,
    max_new_tokens: int,
    max_examples: int | None = None,
):
    correct = 0
    total = 0
    details: List[dict] = []

    fs_pairs: List[Tuple[str, str]] = list(zip(few_shot_df["review"].tolist(), few_shot_df["sentiment"].tolist()))
    idxs = list(range(len(eval_df))) if max_examples is None else list(range(min(max_examples, len(eval_df))))

    for i in tqdm(idxs, desc="Evaluating", leave=False):
        row = eval_df.iloc[i]
        test_review = row["review"]
        true_label = row["sentiment"]

        prompt_text, input_ids = safe_build_prompt_and_encode(
            instruction=instruction,
            delimiter=delimiter,
            answer_format=answer_format,
            test_review=test_review,
            few_shot_examples=fs_pairs,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            verbose=False,
        )

        pred, gen_text = predict_label(input_ids, model, tokenizer, device=device, max_new_tokens=max_new_tokens)

        correct += int(pred == true_label)
        total += 1
        details.append({"index": i, "true": true_label, "pred": pred, "gen": gen_text, "prompt": prompt_text})

    acc = correct / total if total else 0.0
    return acc, details
