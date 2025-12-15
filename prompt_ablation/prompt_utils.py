from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

FewShotPairs = Sequence[Tuple[str, str]]


def build_prompt(
    instruction: str,
    delimiter: str,
    answer_format: str,
    few_shot_examples: FewShotPairs,
    test_review: str,
) -> str:
    parts: List[str] = []
    parts.append(instruction)
    for idx, (review, sentiment) in enumerate(few_shot_examples, start=1):
        parts.append(f"{delimiter} Example {idx}")
        parts.append(f"Review: {review}")
        parts.append(f"Sentiment: {sentiment}")
    parts.append(f"{delimiter} Test Example")
    parts.append(f"Review: {test_review}")
    parts.append(f"{answer_format}")
    return "\n".join(parts)


def safe_build_prompt_and_encode(
    instruction: str,
    delimiter: str,
    answer_format: str,
    test_review: str,
    few_shot_examples: FewShotPairs,
    tokenizer,
    device: str,
    max_new_tokens: int,
    verbose: bool = False,
):
    model_limit = tokenizer.model_max_length
    max_input_tokens = max(16, model_limit - max_new_tokens - 4)

    encoded_fs = []
    for review, sentiment in few_shot_examples:
        rt = tokenizer.encode(review, add_special_tokens=False)
        lt = tokenizer.encode(sentiment, add_special_tokens=False)
        encoded_fs.append((rt, lt))

    test_tokens = tokenizer.encode(test_review, add_special_tokens=False)
    overhead = 12  # instruction + labels overhead estimate per block
    total_tokens = sum(len(rt) + len(lt) + overhead for rt, lt in encoded_fs)
    total_tokens += len(test_tokens) + overhead

    if total_tokens <= max_input_tokens:
        prompt_text = build_prompt(instruction, delimiter, answer_format, few_shot_examples, test_review)
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        return prompt_text, input_ids

    num_items = max(1, len(encoded_fs) + 1)
    per_item_budget = max(16, max_input_tokens // num_items)

    truncated_fs = []
    for (rt, lt), (review, sentiment) in zip(encoded_fs, few_shot_examples):
        rt_trunc = rt[:per_item_budget]
        truncated_review = tokenizer.decode(rt_trunc, clean_up_tokenization_spaces=True)
        truncated_fs.append((truncated_review, sentiment))

    test_trunc = test_tokens[:per_item_budget]
    test_trunc_text = tokenizer.decode(test_trunc, clean_up_tokenization_spaces=True)

    prompt_text = build_prompt(instruction, delimiter, answer_format, truncated_fs, test_trunc_text)
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    if input_ids.shape[-1] <= max_input_tokens:
        if verbose:
            print(f"Truncated prompt length {input_ids.shape[-1]} tokens (budget {max_input_tokens})")
        return prompt_text, input_ids

    fs_copy = list(few_shot_examples)
    while len(fs_copy) > 0:
        fs_copy = fs_copy[:-1]
        prompt_text = build_prompt(instruction, delimiter, answer_format, fs_copy, test_trunc_text)
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        if input_ids.shape[-1] <= max_input_tokens:
            if verbose:
                print(f"Reduced k_shot to {len(fs_copy)} to fit prompt (len {input_ids.shape[-1]})")
            return prompt_text, input_ids

    test_trunc = test_tokens[:max_input_tokens]
    test_trunc_text = tokenizer.decode(test_trunc, clean_up_tokenization_spaces=True)
    prompt_text = f"{instruction}\n{delimiter} Test Example\nReview: {test_trunc_text}\n{answer_format}"
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    if verbose:
        print(f"Zero-shot truncated prompt length {input_ids.shape[-1]} tokens (budget {max_input_tokens})")
    return prompt_text, input_ids
