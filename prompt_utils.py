from __future__ import annotations

from typing import List, Sequence, Tuple
import torch


FewShotPairs = Sequence[Tuple[str, str]]


def build_prompt_from_pairs(few_shot_examples_list: FewShotPairs, test_review: str) -> str:
    parts: List[str] = []
    for review, sentiment in few_shot_examples_list:
        parts.append(f"Review: {review}\nSentiment: {sentiment}\n")
    parts.append(f"Review: {test_review}\nSentiment:")
    return "\n".join(parts)


def safe_build_prompt_and_encode(
    test_review: str,
    few_shot_examples_list: FewShotPairs,
    tokenizer,
    device: str,
    max_new_tokens: int,
    verbose: bool = False,
):
    """Ensure prompt fits model length; truncate fairly when needed."""
    model_limit = tokenizer.model_max_length
    max_input_tokens = max(16, model_limit - max_new_tokens - 4)

    encoded_fs = []
    for review, sentiment in few_shot_examples_list:
        rt = tokenizer.encode(review, add_special_tokens=False)
        lt = tokenizer.encode(sentiment, add_special_tokens=False)
        encoded_fs.append((rt, lt))

    test_tokens = tokenizer.encode(test_review, add_special_tokens=False)

    overhead_tokens_per_example = 6
    total_tokens = sum(len(rt) + len(lt) + overhead_tokens_per_example for rt, lt in encoded_fs)
    total_tokens += len(test_tokens) + overhead_tokens_per_example

    if total_tokens <= max_input_tokens:
        prompt_text = build_prompt_from_pairs(few_shot_examples_list, test_review)
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        return prompt_text, input_ids

    num_items = max(1, len(encoded_fs) + 1)
    per_item_budget = max(16, max_input_tokens // num_items)

    truncated_fs = []
    for (rt, lt), (review, sentiment) in zip(encoded_fs, few_shot_examples_list):
        rt_trunc = rt[:per_item_budget]
        truncated_review = tokenizer.decode(rt_trunc, clean_up_tokenization_spaces=True)
        truncated_fs.append((truncated_review, sentiment))

    test_trunc = test_tokens[:per_item_budget]
    test_trunc_text = tokenizer.decode(test_trunc, clean_up_tokenization_spaces=True)

    prompt_text = build_prompt_from_pairs(truncated_fs, test_trunc_text)
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    if input_ids.shape[-1] <= max_input_tokens:
        if verbose:
            print(f"Built truncated prompt with length {input_ids.shape[-1]} tokens (budget {max_input_tokens})")
        return prompt_text, input_ids

    fs_copy = list(few_shot_examples_list)
    while len(fs_copy) > 0:
        fs_copy = fs_copy[:-1]
        prompt_text = build_prompt_from_pairs(fs_copy, test_trunc_text)
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        if input_ids.shape[-1] <= max_input_tokens:
            if verbose:
                print(f"Reduced K_shot to {len(fs_copy)} to fit prompt (len {input_ids.shape[-1]} tokens)")
            return prompt_text, input_ids

    test_trunc = test_tokens[:max_input_tokens]
    test_trunc_text = tokenizer.decode(test_trunc, clean_up_tokenization_spaces=True)
    prompt_text = f"Review: {test_trunc_text}\nSentiment:"
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    if verbose:
        print(f"Fell back to zero-shot/truncated prompt length {input_ids.shape[-1]} (budget {max_input_tokens})")
    return prompt_text, input_ids
