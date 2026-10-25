"""Generate TinyStories samples with greedy and temperature/top-p sampling."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import course_entry  # noqa: E402
from basics.tokenizer import load_course_tokenizer  # noqa: E402


@torch.inference_mode()
def sample(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    context_length: int,
    device: torch.device,
    *,
    greedy: bool,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int = 0,
) -> list[int]:
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(max_new_tokens):
        window = ids[-context_length:].unsqueeze(0)
        logits = model(window)[0, -1].float()
        if greedy:
            next_id = int(torch.argmax(logits))
        else:
            probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumulative > top_p
                cutoff[1:] = cutoff[:-1].clone()
                cutoff[0] = False
                sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum()
                choice = torch.multinomial(sorted_probs, 1, generator=generator)
                next_id = int(sorted_idx[choice])
            else:
                next_id = int(torch.multinomial(probs, 1, generator=generator))
        ids = torch.cat(
            [ids, torch.tensor([next_id], dtype=torch.long, device=device)]
        )
    return ids.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/final/best.pt")
    parser.add_argument("--config", default="runs/final/config.json")
    parser.add_argument("--tokenizer-dir", default="data/public/tokenizer")
    parser.add_argument("--prompts", default="configs/prompts.json")
    parser.add_argument("--output", default="runs/generation/samples.json")
    args = parser.parse_args()

    config = json.loads((ROOT / args.config).read_text())
    prompts_config = json.loads((ROOT / args.prompts).read_text())
    tokenizer = load_course_tokenizer(ROOT / args.tokenizer_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = course_entry.load_model(str(ROOT / args.checkpoint), config).to(device).eval()
    context_length = config["model"]["context_length"]

    results = []
    for prompt in prompts_config["prompts"]:
        prompt_ids = tokenizer.encode(prompt)
        for mode, kwargs in (
            ("greedy", dict(greedy=True)),
            (
                "sample",
                dict(
                    greedy=False,
                    temperature=prompts_config["sampling"]["temperature"],
                    top_p=prompts_config["sampling"]["top_p"],
                ),
            ),
        ):
            output_ids = sample(
                model,
                prompt_ids,
                prompts_config["max_new_tokens"],
                context_length,
                device,
                seed=prompts_config["seed"],
                **kwargs,
            )
            results.append(
                {
                    "prompt": prompt,
                    "mode": mode,
                    "text": tokenizer.decode(output_ids),
                }
            )
            print(f"[{mode}] {prompt}\n{tokenizer.decode(output_ids)}\n")

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
