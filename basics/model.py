"""Transformer language model components (CS336 assignment 1).

All modules are implemented from scratch with explicit parameters so that the
exported state dict matches the course contract key names exactly.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """y = x W^T, without bias."""

    def __init__(self, d_in: int, d_out: int, device=None, dtype=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (d_in + d_out))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-2.0, b=2.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms * self.weight.float()).to(in_dtype)


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class SwiGLU(nn.Module):
    """w2(SiLU(w1 x) * w3 x)."""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SiLUFeedForward(nn.Module):
    """w2(SiLU(w1 x))."""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)))


class RoPE(nn.Module):
    """Rotary position embedding operating on adjacent (even, odd) pairs."""

    def __init__(self, d_k: int, theta: float, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: (..., seq, d_k)
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        pos = token_positions
        # Add leading singleton dimensions until it broadcasts against x[..., seq, :].
        while pos.dim() < x.dim() - 1:
            pos = pos.unsqueeze(0)
        cos = self.cos_cached[pos].to(x.dtype)
        sin = self.sin_cached[pos].to(x.dtype)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    shifted = x - x_max
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return torch.matmul(attn, V)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.use_rope = use_rope
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RoPE(self.d_head, rope_theta, max_seq_len, device=device) if use_rope else None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (..., seq, d_model) -> (..., heads, seq, d_head) -> (prod(leading), seq, d_head)
        x = x.unflatten(-1, (self.num_heads, self.d_head)).transpose(-3, -2)
        return x.reshape(-1, x.shape[-2], self.d_head)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        *leading, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        out = scaled_dot_product_attention(q, k, v, mask=causal)
        out = out.reshape(*leading, self.num_heads, seq_len, self.d_head).transpose(-3, -2)
        out = out.flatten(-2)
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        norm: str = "pre",
        position: str = "rope",
        ffn: str = "swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert norm in {"pre", "post", "none"}
        assert position in {"rope", "none"}
        assert ffn in {"swiglu", "silu"}
        self.norm = norm
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len,
            rope_theta,
            use_rope=(position == "rope"),
            device=device,
            dtype=dtype,
        )
        if norm != "none":
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        if ffn == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        else:
            self.ffn = SiLUFeedForward(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm == "pre":
            x = x + self.attn(self.ln1(x), token_positions)
            x = x + self.ffn(self.ln2(x))
        elif self.norm == "post":
            x = self.ln1(x + self.attn(x, token_positions))
            x = self.ln2(x + self.ffn(x))
        else:
            x = x + self.attn(x, token_positions)
            x = x + self.ffn(x)
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        norm: str = "pre",
        position: str = "rope",
        ffn: str = "swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.norm = norm
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    norm=norm,
                    position=position,
                    ffn=ffn,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        if norm != "none":
            self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        if self.norm != "none":
            x = self.ln_final(x)
        return self.lm_head(x)


def build_model(model_config: dict, device=None, dtype=None) -> TransformerLM:
    cfg = dict(model_config)
    return TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=cfg.get("rope_theta", 10000.0),
        norm=cfg.get("norm", "pre"),
        position=cfg.get("position", "rope"),
        ffn=cfg.get("ffn", "swiglu"),
        device=device,
        dtype=dtype,
    )
