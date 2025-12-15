from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class ExperimentConfig:
    """Centralized configuration for the perturbation assignment."""

    # data / paths
    dataset_path: Path = Path("IMDB_Dataset_100.csv")
    output_csv: Path = Path("head_perturbation_outproj_results_v2.csv")
    logs_dir: Path = Path("logs")
    plots_dir: Path = Path("plots")

    # model / reproducibility
    model_name: str = "EleutherAI/gpt-neo-125M"
    seed: int = 42

    # few-shot + generation
    k_shot: int = 4
    max_test_examples: int | None = None
    max_new_tokens: int = 8

    # perturbation experiment
    noise_eps: float = 0.05
    repeats: int = 5
    zeroing: bool = True
    target_cproj_name: str = "attn.attention.out_proj"

    verbose: bool = True

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_directories(self) -> Path:
        """Create required folders and return a unique log file path."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.logs_dir / f"run_{timestamp}.log"
