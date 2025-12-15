from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def set_global_seed(seed: int) -> None:
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    config = model.config
    n_layers = config.num_layers
    n_heads = config.num_heads
    embed_dim = config.hidden_size
    head_dim = embed_dim // n_heads

    return model, tokenizer, n_layers, n_heads, embed_dim, head_dim
