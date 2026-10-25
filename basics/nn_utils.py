"""Loss, softmax and gradient clipping utilities."""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from .model import softmax

__all__ = ["softmax", "cross_entropy", "gradient_clipping"]


def cross_entropy_per_token(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Numerically stable per-example cross entropy (no reduction)."""
    max_value = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - max_value
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1)) + max_value.squeeze(-1)
    target_logits = inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return log_sum_exp - target_logits


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross entropy over the leading dimensions (numerically stable)."""
    return cross_entropy_per_token(inputs, targets).mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip the total L2 norm of all present gradients in place."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum((g.detach().float() ** 2).sum() for g in grads))
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.mul_(clip_coef.to(g.dtype))
