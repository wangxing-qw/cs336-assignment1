"""Student implementations of the three course interfaces.

``train``  trains from scratch or resumes, storing the real configuration, a
per-update CSV, full validation NLL, best weights and milestone checkpoints.
``load_model`` accepts either exported scoring weights or a full checkpoint.
``export`` writes a compact CPU FP32 ``state_dict``.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from basics.model import build_model
from basics.nn_utils import cross_entropy, cross_entropy_per_token, gradient_clipping
from basics.optimizer import AdamW, get_lr_cosine_schedule

CSV_FIELDS = [
    "event",
    "step",
    "train_tokens",
    "train_loss",
    "valid_nll",
    "valid_ppl",
    "lr",
    "elapsed_seconds",
]


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def _resolve_model_config(config: dict) -> dict:
    if "model" in config:
        model_config = dict(config["model"])
    elif "kind" in config:  # scoring-model specification
        model_config = dict(config["model"])
    else:
        model_config = dict(config)
    model_config.setdefault("rope_theta", 10000.0)
    model_config.setdefault("norm", "pre")
    model_config.setdefault("position", "rope")
    model_config.setdefault("ffn", "swiglu")
    return model_config


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent / p


def _validate_config(config: dict, mode: str) -> None:
    from course_tools.contracts import normalize_model, validate_v2_run

    normalize_model(_resolve_model_config(config), course=True)
    if mode == "ranked":
        validate_v2_run(config)
    else:
        from course_tools.contracts import checked_budget

        checked_budget(config)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _prepare_validation(tokens: np.ndarray, context_length: int):
    n_targets = len(tokens) - 1
    num_windows = math.ceil(n_targets / context_length)
    padded = num_windows * context_length
    x = np.zeros(padded, dtype=np.int64)
    y = np.zeros(padded, dtype=np.int64)
    x[:n_targets] = tokens[:-1]
    y[:n_targets] = tokens[1:]
    mask = np.zeros(padded, dtype=bool)
    mask[:n_targets] = True
    return (
        torch.from_numpy(x.reshape(num_windows, context_length)),
        torch.from_numpy(y.reshape(num_windows, context_length)),
        torch.from_numpy(mask.reshape(num_windows, context_length)),
        n_targets,
    )


@torch.inference_mode()
def evaluate_nll(model, valid_tokens, context_length, batch_size, device) -> float:
    x, y, mask, n_targets = _prepare_validation(valid_tokens, context_length)
    total_loss = 0.0
    total_count = 0
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        was_training = model.training
        model.eval()
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].to(device)
            yb = y[start : start + batch_size].to(device)
            mb = mask[start : start + batch_size].to(device)
            with torch.autocast(device_type=device.type, enabled=False):
                logits = model(xb)
                losses = cross_entropy_per_token(
                    logits.reshape(-1, logits.shape[-1]).float(), yb.reshape(-1)
                ).reshape(yb.shape)
            total_loss += float((losses * mb).sum().item())
            total_count += int(mb.sum().item())
        if was_training:
            model.train()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return total_loss / n_targets


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _checkpoint_payload(
    model, optimizer, scaler, step, train_tokens, best_nll, config, run_dir, elapsed
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "iteration": step,
        "train_tokens": train_tokens,
        "best_valid_nll": best_nll,
        "config": copy.deepcopy(config),
        "run_dir": str(run_dir),
        "elapsed_seconds": elapsed,
        "rng": _get_rng_state(),
    }


def _save_checkpoint(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(config: dict, run_dir: str, resume: str | None = None):
    config = copy.deepcopy(config)
    mode = config.get("mode", "ranked")
    _validate_config(config, mode)

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    seed = int(config["seed"])
    _set_seed(seed)
    if config.get("cpu_threads"):
        torch.set_num_threads(int(config["cpu_threads"]))

    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model_config = _resolve_model_config(config)
    context_length = int(model_config["context_length"])
    vocab_size = int(model_config["vocab_size"])

    train_tokens_data = np.load(_resolve_path(config["train_data"]))
    valid_tokens_data = np.load(_resolve_path(config["valid_data"]))

    micro_batch = int(config["micro_batch_size"])
    grad_accum = int(config["gradient_accumulation"])
    batch_tokens = micro_batch * grad_accum * context_length
    total_steps = int(config["token_budget"]) // batch_tokens
    validate_steps = int(config["validate_every"]) // batch_tokens
    milestones = {int(m) for m in config.get("milestones", [])}
    token_budget = int(config["token_budget"])

    model = build_model(model_config).to(device)
    model.train()
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=tuple(config["betas"]),
        eps=float(config["eps"]),
        weight_decay=float(config["weight_decay"]),
    )

    precision = config.get("precision", "bf16")
    use_amp = precision in {"bf16", "fp16"} and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda") if (use_amp and precision == "fp16") else None

    step = 0
    train_tokens = 0
    best_nll = float("inf")
    elapsed_offset = 0.0
    resumed_metrics: list[dict] = []

    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint["iteration"])
        train_tokens = int(checkpoint["train_tokens"])
        best_nll = float(checkpoint.get("best_valid_nll", best_nll))
        elapsed_offset = float(checkpoint.get("elapsed_seconds", 0.0))
        _set_rng_state(checkpoint.get("rng"))
        source_metrics = Path(checkpoint.get("run_dir", "")) / "metrics.csv"
        if source_metrics.is_file():
            with source_metrics.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    if int(row["step"]) <= step:
                        resumed_metrics.append(row)
        source_best = Path(checkpoint.get("run_dir", "")) / "best.pt"
        if source_best.is_file() and not (run_path / "best.pt").exists():
            shutil.copyfile(source_best, run_path / "best.pt")
        print(f"resumed from {resume} at step {step} ({train_tokens} tokens)")

    metrics_path = run_path / "metrics.csv"
    start_time = time.time()

    def elapsed() -> float:
        return elapsed_offset + (time.time() - start_time)

    def write_row(writer, **values):
        row = {field: "" for field in CSV_FIELDS}
        row.update(values)
        writer.writerow(row)
        stream.flush()

    milestones_saved = set()
    stream = metrics_path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    if resumed_metrics:
        for row in resumed_metrics:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        stream.flush()
        # Milestones already passed must not be overwritten.
        for m in milestones:
            if m <= train_tokens:
                milestones_saved.add(m)

    def save_full_checkpoint(path: Path):
        payload = _checkpoint_payload(
            model,
            optimizer,
            scaler,
            step,
            train_tokens,
            best_nll,
            config,
            run_path,
            elapsed(),
        )
        _save_checkpoint(path, payload)

    if resume is None:
        save_full_checkpoint(run_path / "checkpoint_000.pt")
        if 0 in milestones:
            milestones_saved.add(0)

    accumulation = grad_accum
    loss_since_log = 0.0
    last_log_time = time.time()

    try:
        while step < total_steps:
            lr = get_lr_cosine_schedule(
                step,
                float(config["learning_rate"]),
                float(config["min_learning_rate"]),
                int(config["warmup_steps"]),
                total_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(accumulation):
                x, y = _get_training_batch(
                    train_tokens_data, micro_batch, context_length, device
                )
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    logits = model(x)
                    loss = cross_entropy(
                        logits.reshape(-1, vocab_size).float(), y.reshape(-1)
                    )
                accumulated_loss += float(loss.detach())
                scaled = loss / accumulation
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

            if scaler is not None:
                scaler.unscale_(optimizer)
            gradient_clipping(model.parameters(), float(config["grad_clip"]))
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            step += 1
            train_tokens += batch_tokens
            train_loss = accumulated_loss / accumulation
            write_row(
                writer,
                event="train",
                step=step,
                train_tokens=train_tokens,
                train_loss=train_loss,
                lr=lr,
                elapsed_seconds=elapsed(),
            )

            if step % validate_steps == 0:
                nll = evaluate_nll(
                    model,
                    valid_tokens_data,
                    context_length,
                    int(config.get("eval_batch_size", 8)),
                    device,
                )
                model.train()
                write_row(
                    writer,
                    event="valid",
                    step=step,
                    train_tokens=train_tokens,
                    valid_nll=nll,
                    valid_ppl=math.exp(nll),
                    lr=lr,
                    elapsed_seconds=elapsed(),
                )
                if nll < best_nll:
                    best_nll = nll
                    save_full_checkpoint(run_path / "best.pt")
                print(
                    f"step {step}/{total_steps} tokens {train_tokens} "
                    f"train_loss {train_loss:.4f} valid_nll {nll:.6f} "
                    f"lr {lr:.3e} elapsed {elapsed():.1f}s"
                )

            for milestone in milestones:
                if train_tokens >= milestone and milestone not in milestones_saved:
                    milestones_saved.add(milestone)
                    percent = round(100 * milestone / token_budget)
                    save_full_checkpoint(run_path / f"checkpoint_{percent:03d}.pt")

            now = time.time()
            if now - last_log_time > 300:
                last_log_time = now
                print(
                    f"step {step}/{total_steps} train_loss {train_loss:.4f} "
                    f"elapsed {elapsed():.1f}s"
                )
    finally:
        stream.close()

    # Guarantee the final model is represented even for smoke budgets without a
    # validation boundary.
    if not (run_path / "best.pt").exists():
        save_full_checkpoint(run_path / "best.pt")
    save_full_checkpoint(run_path / "checkpoint_100.pt")
    print(f"training finished: step {step}, tokens {train_tokens}, best_nll {best_nll:.6f}")
    return {"step": step, "train_tokens": train_tokens, "best_valid_nll": best_nll}


def _get_training_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_starts = len(dataset) - context_length
    starts = np.random.randint(0, num_starts, size=batch_size)
    offsets = np.arange(context_length + 1)
    indices = starts[:, None] + offsets[None, :]
    windows = dataset[indices].astype(np.int64)
    x = torch.from_numpy(windows[:, :-1].copy()).to(device, non_blocking=True)
    y = torch.from_numpy(windows[:, 1:].copy()).to(device, non_blocking=True)
    return x, y


# ---------------------------------------------------------------------------
# Loading / exporting
# ---------------------------------------------------------------------------
def _extract_state_dict(obj) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = obj.get(key)
            if isinstance(value, dict) and value and all(
                isinstance(v, torch.Tensor) for v in value.values()
            ):
                return value
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("checkpoint does not contain a Tensor state dict")


def load_model(checkpoint: str, config: dict):
    model_config = _resolve_model_config(config)
    model = build_model(model_config)
    try:
        obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        # Full checkpoints contain NumPy/Python RNG state, which requires the
        # unrestricted unpickler. Only our own checkpoint files take this path.
        obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = _extract_state_dict(obj)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def export(checkpoint: str, output: str):
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = _extract_state_dict(obj)
    out_state = {}
    for key in state:
        value = state[key]
        out_state[key] = value.detach().cpu().float().contiguous().clone()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_state, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    training = sub.add_parser("train")
    training.add_argument("--config", required=True)
    training.add_argument("--run-dir", required=True)
    training.add_argument("--resume")
    exporting = sub.add_parser("export")
    exporting.add_argument("--checkpoint", required=True)
    exporting.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "train":
        train(json.loads(Path(args.config).read_text()), args.run_dir, args.resume)
    else:
        export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
