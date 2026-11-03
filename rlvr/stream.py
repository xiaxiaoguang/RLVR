"""Streamed pretraining corpus: examples are generated on the fly and never repeated.

    mix = Mixture(task, splits, cfg["pretrain"], seed)        # one example at a time
    ds = PretrainStream(cfg, rank=0, exclude_pids=eval_pids)  # length-bucketed micro-batches
    loader = DataLoader(ds, batch_size=None, num_workers=4)

Composition (doc: 预训练数据生成). `pretrain.comp_frac` is the share of supervised
(target) tokens that come from composition examples; atoms (depth 1, all
functions uniform) fill the rest. Because compositions have longer targets,
the per-example probability of a composition is set from Monte-Carlo estimates
of the mean target length of each component, so the *token* share matches.

Compositions: depth from `comp_depth_mix`; depths in `h_only_depths` use only H
functions; the others are H-only with prob `h_ratio`, else contain >= 1 L.

Leak control: every example is rejection-sampled into the pretrain pool (pid
hash) without held-out function pairs, and dropped if its pid is in
`exclude_pids` (the eval sets; cannot happen by construction, counted anyway).

Determinism: worker w of rank r draws from random.Random(f"{seed}-{r}-{w}"), so a
config with the same world size and num_workers yields the same stream.

Batching: `bucket_examples` examples are sorted by length and cut into
micro-batches holding >= `micro_target_tokens` supervised tokens each, so all
levels see the same supervised tokens per step and padding stays small. Rows
are right-padded; padding sits after every real token, so a plain causal mask
suffices (labels are -100 on <bos>, prompt and padding). Each row gets a random
position offset in [0, pos_offset_max] (P2b).
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Iterator

import torch

from .splits import Splits
from .task import Problem, Task
from .tokenizer import IGNORE_INDEX, CharTokenizer

KINDS = ("atom", "comp")
GROUP_IDS = {"H": 0, "L": 1, "mixed": 2}
REJECT_KINDS = ("pool", "heldout", "eval_dup")


def corrupt(task: Task, p: Problem, rng: random.Random) -> list[str]:
    """Replace one step's output with a random different string and recompute
    the downstream steps from it (an internally consistent wrong chain)."""
    i = rng.randrange(p.depth)
    steps = list(p.steps)
    while True:
        bad = task.sample_x(rng, len(p.x))
        if bad != steps[i]:
            break
    steps[i:] = [bad] + task.run(p.funcs[i + 1:], bad)
    return steps


class Mixture:
    """Samples single pretraining examples according to the `pretrain` config."""

    def __init__(self, task: Task, splits: Splits, pc: dict, seed: int,
                 exclude_pids: set | frozenset = frozenset(), n_estimate: int = 4000):
        self.task, self.splits, self.pc = task, splits, pc
        self.exclude = exclude_pids
        mix = {int(d): float(w) for d, w in pc["comp_depth_mix"].items()}
        assert all(2 <= d <= task.max_depth for d in mix), f"comp depths must be in [2, max_depth]: {mix}"
        self.comp_depths, self.comp_weights = list(mix), list(mix.values())
        self.h_only = {int(d) for d in pc["h_only_depths"]}
        self.rejects = Counter()
        f = pc["comp_frac"]
        assert 0.0 <= f <= 1.0
        # Mean target tokens per component -> per-example composition probability.
        tok, est = CharTokenizer(task.alphabet), random.Random(f"{seed}-estimate")
        self.mean_target = {k: sum(len(tok.encode(self.target(*self._sample(k, est))))
                                   for _ in range(n_estimate)) / n_estimate for k in KINDS}
        if f in (0.0, 1.0):
            self.p_comp = f
        else:
            a, c = (1 - f) / self.mean_target["atom"], f / self.mean_target["comp"]
            self.p_comp = c / (a + c)

    def _draw(self, depth: int, rng: random.Random, **kw) -> Problem:
        task, splits = self.task, self.splits
        while True:
            p = task.sample_problem(depth, rng, **kw)
            if splits.pool_of(p.pid) != "pretrain":
                self.rejects["pool"] += 1
            elif depth > 1 and splits.has_heldout_pair(p.funcs):
                self.rejects["heldout"] += 1
            elif p.pid in self.exclude:
                self.rejects["eval_dup"] += 1
            else:
                return p

    def _sample(self, kind: str, rng: random.Random) -> tuple[Problem, list[str] | None]:
        if kind == "atom":
            return self._draw(1, rng, func_pool=[rng.randrange(self.task.n_funcs)]), None
        depth = rng.choices(self.comp_depths, self.comp_weights)[0]
        if depth in self.h_only:
            group = "H"
        else:
            group = "H" if rng.random() < self.pc["h_ratio"] else "anyL"
        p = self._draw(depth, rng, group=group)
        steps = corrupt(self.task, p, rng) if rng.random() < self.pc["noise_frac"] else None
        return p, steps

    def target(self, p: Problem, steps: list[str] | None) -> str:
        if steps is None:
            return self.task.format_target(p)
        return self.task.format_target(p, steps, steps[-1])

    def sample(self, rng: random.Random) -> tuple[str, Problem, list[str] | None]:
        kind = "comp" if rng.random() < self.p_comp else "atom"
        return (kind, *self._sample(kind, rng))


class PretrainStream(torch.utils.data.IterableDataset):
    """Infinite stream of collated micro-batches for one rank."""

    def __init__(self, cfg: dict, rank: int = 0, exclude_pids: set | frozenset = frozenset()):
        super().__init__()
        self.cfg, self.rank, self.exclude = cfg, rank, exclude_pids
        self.pc = cfg["pretrain"]
        self.max_len = cfg["model"]["max_len"]

    def _setup(self, worker: int):
        from .config import task_and_splits
        task, splits = task_and_splits(self.cfg)
        self.task, self.tok = task, CharTokenizer(task.alphabet)
        self.mix = Mixture(task, splits, self.pc, self.cfg["seed"], self.exclude)
        self.rng = random.Random(f"{self.cfg['seed']}-{self.rank}-{worker}")

    def encode(self, kind: str, p: Problem, steps) -> dict:
        ids, labels = self.tok.encode_example(self.task.format_prompt(p), self.mix.target(p, steps))
        assert len(ids) <= self.max_len, f"example of {len(ids)} tokens > model.max_len"
        return {"ids": ids, "labels": labels, "n_target": sum(t != IGNORE_INDEX for t in labels),
                "kind": KINDS.index(kind), "depth": p.depth, "group": GROUP_IDS[p.group],
                "funcs": p.funcs, "noisy": steps is not None}

    def collate(self, rows: list[dict]) -> dict:
        T = max(len(r["ids"]) for r in rows)
        T = -(-T // 8) * 8
        n = len(rows)
        ids = torch.full((n, T), self.tok.pad_id, dtype=torch.long)
        labels = torch.full((n, T), IGNORE_INDEX, dtype=torch.long)
        pos = torch.arange(T).repeat(n, 1)
        omax = self.pc["pos_offset_max"]
        for i, r in enumerate(rows):
            L = len(r["ids"])
            ids[i, :L] = torch.tensor(r["ids"])
            labels[i, :L] = torch.tensor(r["labels"])
            if omax:
                pos[i] += self.rng.randint(0, max(0, min(omax, self.max_len - T)))
        func_counts = torch.zeros(self.task.n_funcs, dtype=torch.long)
        for r in rows:
            if r["kind"] == 1:
                for f in r["funcs"]:
                    func_counts[f] += 1
        rejects = torch.tensor([self.mix.rejects.pop(k, 0) for k in REJECT_KINDS])
        meta = {k: torch.tensor([r[k] for r in rows]) for k in ("kind", "depth", "group", "n_target", "noisy")}
        return {"input_ids": ids, "labels": labels, "position_ids": pos, **meta,
                "func_counts": func_counts, "rejects": rejects}

    def batches(self) -> Iterator[dict]:
        budget, nbuf = self.pc["micro_target_tokens"], self.pc["bucket_examples"]
        while True:
            buf = sorted((self.encode(*self.mix.sample(self.rng)) for _ in range(nbuf)),
                         key=lambda r: len(r["ids"]))
            groups, cur, ntgt = [], [], 0
            for r in buf:
                cur.append(r)
                ntgt += r["n_target"]
                if ntgt >= budget:
                    groups.append(cur)
                    cur, ntgt = [], 0
            # The leftover (< budget) is the longest tail, where rare long compositions
            # sit; dropping or carrying it would starve them, so it joins the last group.
            if cur:
                if groups:
                    groups[-1] += cur
                else:
                    groups.append(cur)
            self.rng.shuffle(groups)
            for g in groups:
                yield self.collate(g)

    def __iter__(self) -> Iterator[dict]:
        info = torch.utils.data.get_worker_info()
        self._setup(info.id if info is not None else 0)
        return self.batches()
