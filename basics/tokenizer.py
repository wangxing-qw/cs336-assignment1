"""Byte-level BPE training and tokenizer (CS336 assignment 1)."""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import regex

# GPT-2 style pre-tokenization pattern required by the course.
PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PRETOKEN_RE = regex.compile(PRETOKEN_PATTERN)


def _special_split_regex(special_tokens: list[str]):
    if not special_tokens:
        return None
    ordered = sorted(set(special_tokens), key=len, reverse=True)
    return regex.compile("(" + "|".join(regex.escape(t) for t in ordered) + ")")


def _merge_pair(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out: list[int] = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def _count_pairs(word: tuple[int, ...]) -> Counter:
    counts = Counter()
    for i in range(len(word) - 1):
        counts[(word[i], word[i + 1])] += 1
    return counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer on ``input_path``."""
    special_tokens = list(special_tokens or [])
    text = Path(input_path).read_text(encoding="utf-8", errors="replace")

    split_re = _special_split_regex(special_tokens)
    segments = split_re.split(text) if split_re is not None else [text]
    special_set = set(special_tokens)

    word_counts: Counter[bytes] = Counter()
    for segment in segments:
        if not segment or segment in special_set:
            continue
        for match in _PRETOKEN_RE.finditer(segment):
            word_counts[match.group().encode("utf-8")] += 1

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    merges: list[tuple[bytes, bytes]] = []

    # Represent words as tuples of token ids (starting as raw byte ids).
    words: dict[tuple[int, ...], int] = {
        tuple(word): count for word, count in word_counts.items()
    }
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    for word, count in words.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] += count
            pair_words[pair].add(word)

    while len(vocab) < vocab_size and pair_counts:
        # Ties are broken by the pair of merged byte strings, matching the
        # course reference tokenizer.
        best_pair = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], vocab[pair[0]], vocab[pair[1]]),
        )
        best_count = pair_counts[best_pair]
        if best_count <= 0:
            del pair_counts[best_pair]
            continue
        new_id = len(vocab)
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        affected = list(pair_words.pop(best_pair, ()))
        for word in affected:
            count = words.pop(word, 0)
            if count == 0:
                continue
            old_pairs = _count_pairs(word)
            new_word = _merge_pair(word, best_pair, new_id)
            new_pairs = _count_pairs(new_word)
            for pair, n in old_pairs.items():
                pair_counts[pair] -= n * count
                pair_words[pair].discard(word)
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair, None)
                    pair_words.pop(pair, None)
            for pair, n in new_pairs.items():
                pair_counts[pair] += n * count
                pair_words[pair].add(new_word)
            words[new_word] = words.get(new_word, 0) + count

    return vocab, merges


class Tokenizer:
    """Byte-level BPE tokenizer with optional special tokens."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.special_to_id = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.token_to_id[token_bytes] = new_id
            self.special_to_id[token] = self.token_to_id[token_bytes]
        # BPE merge ranks keyed by token ids, so encoding only touches ints.
        self.merge_ranks = {}
        for rank, (left, right) in enumerate(self.merges):
            left_id = self.token_to_id.get(left)
            right_id = self.token_to_id.get(right)
            if left_id is not None and right_id is not None:
                self.merge_ranks[(left_id, right_id)] = rank
        self._split_re = _special_split_regex(self.special_tokens)

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        parts = [self.token_to_id[bytes([byte])] for byte in pretoken.encode("utf-8")]
        if not parts:
            return []
        while len(parts) > 1:
            best_rank = None
            best_index = -1
            for i in range(len(parts) - 1):
                rank = self.merge_ranks.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = i
            if best_index < 0:
                break
            pair = (parts[best_index], parts[best_index + 1])
            new_id = self.token_to_id[
                self.vocab[pair[0]] + self.vocab[pair[1]]
            ]
            merged: list[int] = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == pair:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        return parts

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        ids: list[int] = []
        segments = self._split_re.split(text) if self._split_re is not None else [text]
        for segment in segments:
            if not segment:
                continue
            special_id = self.special_to_id.get(segment)
            if special_id is not None:
                ids.append(special_id)
                continue
            for match in _PRETOKEN_RE.finditer(segment):
                ids.extend(self._encode_pretoken(match.group()))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


def load_course_tokenizer(tokenizer_dir: str | os.PathLike) -> Tokenizer:
    """Load the tokenizer distributed with the course data (base64 JSON)."""
    import base64
    import json

    directory = Path(tokenizer_dir)
    vocab_raw = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    merges_raw = json.loads((directory / "merges.json").read_text(encoding="utf-8"))
    meta = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = {int(k): base64.b64decode(v) for k, v in vocab_raw.items()}
    merges = [
        (base64.b64decode(pair[0]), base64.b64decode(pair[1])) for pair in merges_raw
    ]
    return Tokenizer(vocab, merges, meta.get("special_tokens"))
