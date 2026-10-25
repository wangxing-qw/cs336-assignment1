"""Draw training and public-validation NLL from metrics.csv only."""
import argparse
import csv
import math
import os
from pathlib import Path
import warnings

REQUIRED_COLUMNS = (
    "event", "step", "train_tokens", "train_loss", "valid_nll",
    "valid_ppl", "lr", "elapsed_seconds",
)


def read_metrics(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError("CSV缺少必需列：" + ", ".join(sorted(missing)))
        for line, raw in enumerate(reader, start=2):
            if not any(value for value in raw.values()):
                continue
            if None in raw:
                raise ValueError(f"CSV第{line}行的字段数超过表头，请检查分隔符")
            event = (raw.get("event") or "").strip()
            if event not in {"train", "valid"}:
                raise ValueError(f"CSV第{line}行event须为train或valid")
            row = {"event": event}
            needed = {"step", "train_tokens", "lr", "elapsed_seconds"}
            needed.update({"train_loss"} if event == "train" else {"valid_nll", "valid_ppl"})
            for name in REQUIRED_COLUMNS[1:]:
                value = (raw.get(name) or "").strip()
                if not value:
                    if name in needed:
                        raise ValueError(f"CSV第{line}行{name}不能为空")
                    row[name] = None
                    continue
                try:
                    number = int(value) if name in {"step", "train_tokens"} else float(value)
                except ValueError as error:
                    raise ValueError(f"CSV第{line}行{name}必须是数值") from error
                if name in {"step", "train_tokens", "lr", "elapsed_seconds"}:
                    if not math.isfinite(number) or number < 0:
                        raise ValueError(f"CSV第{line}行{name}须为有限非负数")
                elif not math.isfinite(number):
                    # Failed runs may contain NaN/Inf; keep a visible gap, never invent a loss.
                    number = math.nan
                row[name] = number
            rows.append(row)
    return rows


def plot_run(run_dir):
    root = Path(run_dir)
    rows = read_metrics(root / "metrics.csv")
    train = [r for r in rows if r["event"] == "train"]
    valid = [r for r in rows if r["event"] == "valid"]
    if not train:
        raise ValueError("CSV没有train记录，无法绘制训练曲线")
    plotted = [r["train_loss"] for r in train] + [r["valid_nll"] for r in valid]
    if not all(math.isfinite(value) for value in plotted):
        warnings.warn("日志中的NaN/Inf将显示为曲线缺口，原始CSV不会修改", stacklevel=2)

    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".build/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot([r["train_tokens"] for r in train], [r["train_loss"] for r in train],
            color="#3d788b", linewidth=0.8, alpha=.65, label="Training loss (raw)")
    if valid:
        ax.plot([r["train_tokens"] for r in valid], [r["valid_nll"] for r in valid],
                color="#c0583d", marker="o", linewidth=1.6, label="Full public validation NLL")
    ax.set(xlabel="Training target tokens", ylabel="NLL", title=root.name)
    ax.grid(alpha=.2)
    ax.legend()
    output = root / "curve.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description="读取运行目录的metrics.csv，生成curve.png；无需其他日志文件")
    parser.add_argument("run_dir", help="包含metrics.csv的运行目录")
    args = parser.parse_args()
    try:
        print(plot_run(args.run_dir))
    except (ValueError, OSError, csv.Error) as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
