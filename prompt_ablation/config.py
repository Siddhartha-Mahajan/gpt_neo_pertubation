from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class AblationConfig:
    dataset_path: Path = Path("IMDB_Dataset_100.csv")
    output_csv: Path = Path("prompt_ablation/head_perturbation_outproj_results_ablation.csv")
    logs_dir: Path = Path("logs")
    plots_dir: Path = Path("plots")

    model_name: str = "EleutherAI/gpt-neo-125M"
    seed: int = 42

    k_shot: int = 6
    max_test_examples: int | None = None
    max_new_tokens: int = 4

    noise_eps: float = 0.05
    repeats: int = 2
    zeroing: bool = True
    target_cproj_name: str = "attn.attention.out_proj"

    verbose: bool = True

    instruction: str = (
        "You are a strict sentiment classifier. Given a review, respond with exactly one word: positive or negative."
    )
    delimiter: str = "###"
    answer_format: str = "Answer:"

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_dirs(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
