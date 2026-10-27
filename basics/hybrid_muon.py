"""Hybrid Muon + AdamW optimizer used by the training-only "kdd" recipe.

Muon (orthogonalized-momentum SGD) handles the hidden 2-D matrices inside the
transformer layers; AdamW handles embeddings, norms and the LM head, matching
the approach in ``kdd`` (another student's reproduction materials).

Training only: the model forward and the exported state dict are unchanged.
"""
from __future__ import annotations

import math

import torch

from basics.optimizer import AdamW


class HybridMuonAdamW(torch.optim.Optimizer):
    def __init__(self, model, config: dict):
        multiplier = float(config.get("muon_lr_multiplier", 1.75))
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("muon_lr_multiplier must be finite and positive")
        weight_decay = float(config["weight_decay"])
        base_lr = float(config["learning_rate"])

        hidden, other = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            is_hidden = name.startswith("layers.") and parameter.ndim == 2
            (hidden if is_hidden else other).append(parameter)
        if not hidden or not other:
            raise ValueError("hybrid Muon requires both hidden matrices and other parameters")

        super().__init__(hidden + other, dict(lr=base_lr))
        self.model = model
        self.config = config
        self.muon = torch.optim.Muon(
            hidden,
            lr=base_lr * multiplier,
            weight_decay=weight_decay / multiplier,
            momentum=float(config.get("muon_momentum", 0.95)),
            ns_steps=5,
            adjust_lr_fn="match_rms_adamw",
        )
        self.muon.param_groups[0]["lr_multiplier"] = multiplier

        # AdamW groups: embedding (no decay), final norm (extra LR), internal
        # norms, and everything else.
        embedding = model.token_embeddings.weight
        final_norm = model.ln_final.weight if getattr(model, "ln_final", None) is not None else None
        internal_norms = [
            p
            for name, p in model.named_parameters()
            if name.startswith("layers.") and p.ndim == 1
        ]
        rest = [
            p
            for p in other
            if id(p) not in {id(embedding), id(final_norm)}
            and all(p is not n for n in internal_norms)
        ]
        groups = []
        groups.append(
            dict(
                params=[embedding],
                lr_multiplier=1.0,
                weight_decay=0.0 if config.get("tied_embedding_weight_decay_zero", True) else weight_decay,
            )
        )
        if final_norm is not None:
            groups.append(
                dict(
                    params=[final_norm],
                    lr_multiplier=float(config.get("final_norm_lr_multiplier", 1.0)),
                    weight_decay=0.0,
                )
            )
        if internal_norms:
            groups.append(
                dict(
                    params=internal_norms,
                    lr_multiplier=float(config.get("internal_norm_lr_multiplier", 1.0)),
                    weight_decay=weight_decay,
                )
            )
        if rest:
            groups.append(dict(params=rest, lr_multiplier=1.0, weight_decay=weight_decay))
        self.adamw = AdamW(
            groups,
            lr=base_lr,
            betas=tuple(config["betas"]),
            eps=float(config["eps"]),
            weight_decay=weight_decay,
        )
        self.param_groups = self.muon.param_groups + self.adamw.param_groups

    # -- optimizer interface -------------------------------------------------
    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.adamw.step()
        return loss

    def state_dict(self):
        return {
            "format": "hybrid_muon_v1",
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if state_dict.get("format") != "hybrid_muon_v1":
            raise ValueError("Incompatible hybrid Muon checkpoint")
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = self.muon.param_groups + self.adamw.param_groups

    # -- vocabulary untying --------------------------------------------------
    def untie(self, model) -> None:
        """Split the tied LM head off the embedding with its own AdamW group."""
        shared = model.token_embeddings.weight
        if model.lm_head.weight is not shared:
            return
        head = torch.nn.Parameter(shared.detach().clone())
        model.lm_head.weight = head
        multiplier = float(self.config.get("untied_head_lr_multiplier", 0.25))
        self.adamw.add_param_group(
            dict(
                params=[head],
                lr_multiplier=multiplier,
                weight_decay=float(self.config["weight_decay"]),
            )
        )
        if shared in self.adamw.state:
            source = self.adamw.state[shared]
            self.adamw.state[head] = {
                key: (value.clone() if torch.is_tensor(value) else value)
                for key, value in source.items()
            }
        self.param_groups = self.muon.param_groups + self.adamw.param_groups
