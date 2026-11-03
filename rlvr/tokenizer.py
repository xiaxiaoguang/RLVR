"""Character-level tokenizer (frozen after week 1).

Vocab: <pad> <bos> <eos>, then the alphabet (default a-z), F, 0-9, ( ) = ; :, Q, A, newline
(48 tokens for a-z).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SPECIALS = ["<pad>", "<bos>", "<eos>"]
IGNORE_INDEX = -100


class CharTokenizer:
    def __init__(self, alphabet: str = "abcdefghijklmnopqrstuvwxyz"):
        chars = list(alphabet) + ["F"] + list("0123456789") + list("()=;:") + ["Q", "A", "\n"]
        assert len(set(chars)) == len(chars), "alphabet clashes with syntax chars"
        self.alphabet = alphabet
        self.vocab: list[str] = SPECIALS + chars
        self.tok2id = {t: i for i, t in enumerate(self.vocab)}
        self.pad_id, self.bos_id, self.eos_id = 0, 1, 2
        self._split = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        ids = [self.bos_id] if add_bos else []
        for part in self._split.split(text):
            if part in self.tok2id and part in SPECIALS:
                ids.append(self.tok2id[part])
            else:
                ids.extend(self.tok2id[c] for c in part)
        return ids

    def decode(self, ids, skip_pad: bool = True) -> str:
        return "".join(self.vocab[i] for i in ids if not (skip_pad and i == self.pad_id))

    def encode_example(self, prompt: str, target: str) -> tuple[list[int], list[int]]:
        """input_ids = <bos> prompt target; labels mask <bos> and prompt with -100."""
        p = self.encode(prompt, add_bos=True)
        t = self.encode(target)
        return p + t, [IGNORE_INDEX] * len(p) + t

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"alphabet": self.alphabet, "vocab": self.vocab}, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        d = json.loads(Path(path).read_text())
        tok = cls(d["alphabet"])
        assert tok.vocab == d["vocab"], "saved vocab does not match current tokenizer"
        return tok
