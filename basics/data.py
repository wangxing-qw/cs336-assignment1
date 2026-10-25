"""Language-modelling batch sampling."""
from __future__ import annotations

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_starts = len(dataset) - context_length
    if num_starts <= 0:
        raise ValueError("dataset is shorter than context_length")
    starts = np.random.randint(0, num_starts, size=batch_size)
    offsets = np.arange(context_length + 1)
    indices = starts[:, None] + offsets[None, :]
    windows = np.asarray(dataset[indices], dtype=np.int64)
    x = torch.from_numpy(windows[:, :-1].copy()).to(device)
    y = torch.from_numpy(windows[:, 1:].copy()).to(device)
    return x, y
