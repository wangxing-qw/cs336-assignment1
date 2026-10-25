"""Train a BPE tokenizer on the public corpus, then encode/decode a sample."""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basics.tokenizer import Tokenizer, train_bpe  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/public/train.txt")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--special", default="<|endoftext|>")
    parser.add_argument("--max-bytes", type=int, default=0, help="0 trains on the full file")
    parser.add_argument("--output", default="runs/bpe_demo")
    args = parser.parse_args()

    source = ROOT / args.input
    if args.max_bytes:
        text = source.read_text(encoding="utf-8", errors="replace")[: args.max_bytes]
        temp = ROOT / args.output / "corpus_subset.txt"
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_text(text, encoding="utf-8")
        input_path = temp
    else:
        input_path = source

    started = time.time()
    vocab, merges = train_bpe(input_path, args.vocab_size, [args.special])
    elapsed = time.time() - started
    print(
        f"trained {len(vocab)} tokens / {len(merges)} merges in {elapsed:.2f}s"
    )

    tokenizer = Tokenizer(vocab, merges, [args.special])
    sample = (
        "Once upon a time, there was a little rabbit who lived in a small house. "
        "<|endoftext|> She liked to play in the garden."
    )
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)
    assert decoded == sample, "round trip failed"
    print(f"sample encoded to {len(ids)} tokens and decoded identically")

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "vocab.json").write_text(
        json.dumps(
            {str(i): base64.b64encode(token).decode() for i, token in vocab.items()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "merges.json").write_text(
        json.dumps(
            [
                [base64.b64encode(a).decode(), base64.b64encode(b).decode()]
                for a, b in merges
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "tokenizer.json").write_text(
        json.dumps(
            {
                "format": "cs336-course-bpe-base64-v1",
                "pattern": r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
                "special_tokens": [args.special],
                "vocab_size": len(vocab),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "input": str(input_path.relative_to(ROOT)),
                "vocab_size": len(vocab),
                "merges": len(merges),
                "seconds": elapsed,
                "sample_tokens": len(ids),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
