from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def clean_review_text(text: str) -> str:
    """Basic cleanup to remove HTML breaks and excessive whitespace."""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ")
    text = " ".join(text.split())
    return text


def load_and_prepare_dataset(
    csv_path: Path,
    k_shot: int,
    seed: int,
    max_test_examples: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load dataset, clean text, shuffle, and split into few-shot/eval sets."""
    df = pd.read_csv(csv_path)
    df["review"] = df["review"].apply(clean_review_text)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if k_shot < 1 or k_shot >= len(df):
        raise ValueError("k_shot must be at least 1 and less than dataset size")

    few_shot = df.iloc[:k_shot].reset_index(drop=True)
    eval_set = df.iloc[k_shot:].reset_index(drop=True)
    if max_test_examples is not None:
        eval_set = eval_set.head(max_test_examples)

    return few_shot, eval_set, df
