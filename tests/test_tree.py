"""Semantic, structural and integration checks for multi-input string problems."""

import copy
import json
import random

import pytest

from rlvr.config import load_config, task_and_splits
from rlvr.task import Task
from rlvr.tokenizer import CharTokenizer, IGNORE_INDEX
from rlvr.tree_sampling import count, sample_shape


@pytest.fixture(scope="module")
def setup():
    cfg = load_config("configs/tasks/tree_v1.yaml")
    task, splits = task_and_splits(cfg)
    return cfg, task, splits


def call(task, name, *args):
    return {"op": task.fname(task.name2id[name]), "args": [a if isinstance(a, dict) else {"value": a} for a in args]}


@pytest.mark.parametrize(
    "name,args,want",
    [
        ("zip_add", ("abcdef", "bcdefg"), "bdfhjl"),
        ("zip_sub", ("abcdef", "bcdefg"), "zzzzzz"),
        ("zip_mul", ("abcdef", "bbbbbb"), "abcdef"),
        ("gather", ("abcdef", "fedcba"), "fedcba"),
        ("gather_add", ("abcdef", "fedcba"), "kigeca"),
        ("scatter_add", ("abcdef", "aaaaaa"), "paaaaa"),
        ("rotate_by_sum", ("abcdef", "baaaaa"), "bcdefa"),
        ("prefix_add", ("aaaaaa", "bbbbbb"), "bcdefg"),
        ("zip_add3", ("zzzzzz", "bbbbbb", "bbbbbb"), "bbbbbb"),
        ("zip_mul_add", ("bbbbbb", "cccccc", "dddddd"), "ffffff"),
        ("gather_mix", ("abcdef", "fedcba", "bbbbbb"), "gfedcb"),
        ("select_parity", ("aaaaaa", "zzzzzz", "abcdef"), "azazaz"),
    ],
)
def test_known_multi_operator_results(setup, name, args, want):
    _, task, _ = setup
    assert task.apply(task.name2id[name], *args) == want


def test_operator_replacement_and_groups(setup):
    _, t, _ = setup
    old = Task()
    removed = set(old.id2name) - set(t.id2name)
    assert removed == {
        "reverse",
        "rotl1",
        "rotr1",
        "rotl2",
        "rotl3",
        "swap_ends",
        "shift_p1",
        "shift_m1",
        "shift_p3",
        "shift_p5",
        "complement",
        "sort_desc",
    }
    assert t.n_funcs == 24
    for group in (t.H, t.L):
        assert [sum(t.arities[f] == a for f in group) for a in (1, 2, 3)] == [6, 4, 2]
    rng = random.Random(4)
    for fid, arity in enumerate(t.arities):
        for n in range(1, 13):
            args = [t.sample_x(rng, n) for _ in range(arity)]
            result = t.apply(fid, *args)
            assert len(result) == n and set(result) <= set(t.alphabet)
        with pytest.raises(ValueError, match="arguments"):
            t.apply(fid, *(["abcdef"] * (arity + 1)))
        if arity > 1:
            with pytest.raises(ValueError, match="equal-length"):
                t.apply(fid, "abc", *(["abcdef"] * (arity - 1)))


def test_branching_postorder_roundtrip_and_reward(setup):
    _, t, _ = setup
    expr = call(t, "zip_add", call(t, "swap_adjacent", "abcdef"), call(t, "zip_sub", "bcdefg", "aaaaaa"))
    p = t.make_problem(expr)
    assert (p.depth, p.n_ops, p.n_binary, p.n_ternary, p.n_leaves) == (2, 3, 2, 0, 3)
    assert p.n_leaves == 1 + p.n_binary + 2 * p.n_ternary
    assert p.max_live_values == 3
    assert p.steps == ("badcfe", "bcdefg", "ccggkk")
    prompt, target = t.format_prompt(p), t.format_target(p)
    assert "," in prompt and t.parse_steps(target) == list(p.steps)
    assert t.reward(target, p) == t.reward(target, p, "partial") == 1
    assert t.reward(target.replace("<eos>", ""), p) == 0
    assert t.reward(target.replace("=badcfe;", "=aaaaaa;"), p, "partial") == pytest.approx(2 / 3)
    assert t.from_dict(json.loads(json.dumps(p.to_dict()))) == p
    # Parameter order is part of identity even for a commutative operator.
    swapped = copy.deepcopy(expr)
    swapped["args"].reverse()
    assert t.make_problem(swapped).pid != p.pid


def test_invalid_and_stale_data_rejected(setup):
    cfg, t, _ = setup
    p = t.make_problem(call(t, "zip_add", "abcdef", "bcdefg"))
    for key, value in [
        ("answer", "aaaaaa"),
        ("depth", 9),
        ("funcs", [0]),
        ("task_fingerprint", "stale"),
        ("n_ops", 10),
        ("prompt", "Q:bad=\n"),
    ]:
        row = p.to_dict()
        row[key] = value
        with pytest.raises(ValueError):
            t.from_dict(row)
    with pytest.raises(ValueError):
        Task().from_dict(p.to_dict())
    with pytest.raises(ValueError):
        t.from_dict(Task().sample_problem(2, random.Random(0)).to_dict())
    with pytest.raises(ValueError, match="arity"):
        t.make_problem(call(t, "zip_add", "abcdef"))
    with pytest.raises(ValueError, match="equal-length"):
        t.make_problem(call(t, "zip_add", "abcdef", "abcdefg"))
    cfg2 = copy.deepcopy(cfg)
    cfg2["task"]["h_group"] = list(reversed(cfg2["task"]["h_group"]))
    # Mapping/semantics changes must invalidate old rows.
    cfg2["task"]["name_seed"] += 1
    t2, _ = task_and_splits(cfg2)
    with pytest.raises(ValueError, match="fingerprint"):
        t2.from_dict(p.to_dict())


def test_sampling_sizes_groups_and_independent_interpreter(setup):
    _, t, _ = setup
    rng = random.Random(15)

    def interpret(node):
        if "value" in node:
            return node["value"]
        return t.apply(int(node["op"][1:]), *(interpret(a) for a in node["args"]))

    for depth in range(1, 9):
        for group in ("H", "L", None, "anyL", *(["mixed"] if depth > 1 else [])):
            p = t.sample_problem(depth, rng, group=group)
            assert p.depth == depth and depth <= p.n_ops <= min(10, 2 * depth)
            assert p.answer == interpret(p.expr.to_dict())
            assert p.n_leaves == 1 + p.n_binary + 2 * p.n_ternary
            if group in ("H", "L", "mixed"):
                assert p.group == group
            elif group == "anyL":
                assert p.group != "H"
    p = t.sample_problem(2, rng, n_ops=4)
    assert p.n_ops == 4
    for kw in ({"depth": 2, "n_ops": 1}, {"depth": 1, "group": "mixed"}, {"depth": 9}, {"depth": 2, "func_pool": []}):
        with pytest.raises(ValueError):
            t.sample_problem(rng=rng, **kw)
    assert count(1, 1, (1, 2, 3)) == 3
    assert count(2, 2, (1, 2, 3)) == 18
    with pytest.raises(ValueError, match="no tree"):
        sample_shape(4, 2, (1,), rng)


def test_actual_edges_not_siblings_and_pool_isolation(setup):
    _, t, splits = setup
    assert len(splits.heldout_edges) == 36
    p = t.make_problem(call(t, "zip_add", call(t, "swap_adjacent", "abcdef"), call(t, "swap_halves", "bcdefg")))
    root, left, right = [t.name2id[n] for n in ("zip_add", "swap_adjacent", "swap_halves")]
    assert set(t.edges(p)) == {(root, 0, left), (root, 1, right)}
    rng = random.Random(1)
    for pool in ("pretrain", "rl", "eval"):
        for depth in (1, 2, 4, 8):
            q = splits.sample(pool, depth, rng)
            assert splits.pool_of(q.pid) == pool and not splits.has_heldout(q)
    for group in (None, "H", "L", "mixed"):
        q = splits.sample("eval", 3, rng, group=group, novel=True)
        assert splits.admissible(q, "eval", novel=True)
    with pytest.raises(ValueError, match="expression"):
        splits.has_heldout_pair(p.funcs)


def test_tokenizer_and_global_length_bounds(setup, tmp_path):
    cfg, t, _ = setup
    tok = t.tokenizer()
    assert tok.vocab_size == 49 and tok.vocab[:48] == CharTokenizer().vocab
    for tokenizer in (tok, CharTokenizer()):
        tokenizer.save(tmp_path / "tokenizer.json")
        assert CharTokenizer.load(tmp_path / "tokenizer.json").vocab == tokenizer.vocab
    total, target = t.length_bounds()
    rng = random.Random(3)
    for depth in range(1, 9):
        for _ in range(20):
            p = t.sample_problem(depth, rng, str_len=10)
            prompt, response = t.format_prompt(p), t.format_target(p)
            ids, labels = tok.encode_example(prompt, response)
            assert len(ids) <= total and len(tok.encode(response)) <= target
            n = len(tok.encode(prompt, add_bos=True))
            assert labels[:n] == [IGNORE_INDEX] * n and labels[n:] == ids[n:]
    bad = copy.deepcopy(cfg)
    bad["model"]["max_len"] = 512
    with pytest.raises(ValueError, match="length bounds"):
        task_and_splits(bad)


def test_noise_recomputes_only_ancestors(setup):
    _, t, _ = setup
    p = t.make_problem(call(t, "zip_add", call(t, "swap_adjacent", "abcdef"), call(t, "swap_halves", "bcdefg")))

    class FirstNode(random.Random):
        def randrange(self, *args):
            return 0

    steps = t.corrupt_steps(p, FirstNode(0))
    assert steps[0] != p.steps[0] and steps[1] == p.steps[1]
    assert steps[2] == t.apply(t.name2id["zip_add"], steps[0], steps[1])
    assert f"({steps[0]},{steps[1]})={steps[2]};" in t.format_target(p, steps)


def test_stream_supervised_share_and_structure_metadata(setup):
    from rlvr.stream import Mixture, PretrainStream

    cfg, task, splits = setup
    cfg = copy.deepcopy(cfg)
    cfg["pretrain"].update(comp_frac=0.35, noise_frac=0.2, bucket_examples=60, micro_target_tokens=400, num_workers=0)
    mix = Mixture(task, splits, cfg["pretrain"], 0, n_estimate=200)
    rng, tokens = random.Random(15), {"atom": 0, "comp": 0}
    seen_arities = set()
    for _ in range(3000):
        kind, p, steps = mix.sample(rng)
        tokens[kind] += len(task.tokenizer().encode(mix.target(p, steps)))
        seen_arities.update(task.arities[f] for f in p.funcs)
        assert not splits.has_heldout(p) and splits.pool_of(p.pid) == "pretrain"
    assert seen_arities == {1, 2, 3}
    assert tokens["comp"] / sum(tokens.values()) == pytest.approx(0.35, abs=0.055)
    ds = PretrainStream(cfg)
    ds._setup(0)
    b = next(ds.batches())
    assert (b["n_ops"] >= b["depth"]).all()
    assert (b["position_ids"] < cfg["model"]["max_len"]).all()
    for ids, labels in zip(b["input_ids"], b["labels"]):
        mask = labels != IGNORE_INDEX
        assert (ids[mask] == labels[mask]).all()
    from scripts.pretrain import Stats

    stats = Stats(task.n_funcs)
    stats.add(b)
    summary = stats.merge([stats])
    assert sum(summary["by_structure_depth_ops_binary_ternary"].values()) == len(b["input_ids"])


def test_tree_model_loss_and_checkpoint_compatibility(setup, tmp_path):
    import torch
    from rlvr.model import build_model, save_model, load_model, generate
    from scripts.pretrain import token_losses
    from rlvr.stream import PretrainStream

    cfg, task, _ = setup
    cfg = copy.deepcopy(cfg)
    cfg["model"].update(n_layer=1, d_model=32, d_ff=64, n_head=2)
    cfg["pretrain"].update(comp_frac=1.0, bucket_examples=12, micro_target_tokens=100, num_workers=0)
    ds = PretrainStream(cfg)
    ds._setup(0)
    batch = next(ds.batches())
    model = build_model(cfg["model"], task.tokenizer())
    ce, mask = token_losses(model, batch)
    loss = ce.sum() / mask.sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(model.lm_head.weight.grad).all()
    save_model(model, task.tokenizer(), cfg, tmp_path)
    loaded, tok = load_model(tmp_path, "cpu")
    assert tok.vocab_size == 49 and loaded.config.task_fingerprint == task.fingerprint
    p = task.sample_problem(2, random.Random(6))
    prompt = [task.format_prompt(p)]
    assert generate(model, tok, prompt, max_new_tokens=4) == generate(loaded, tok, prompt, max_new_tokens=4)
    config_path = tmp_path / "run_config.yaml"
    import yaml

    cfg["task"]["name_seed"] += 1
    config_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="fingerprint"):
        load_model(tmp_path, "cpu")


def test_dataset_structure_selection(setup):
    import pandas as pd
    from scripts.build_dataset import build, summarize

    _, task, splits = setup
    rng = random.Random(8)
    rows = [splits.sample("rl", 2, rng, n_ops=n).to_dict() | {"origin": "base"} for n in (2, 3, 4) for _ in range(30)]
    spec = {"name": "matched_ops", "n": 12, "seed": 0, "cells": [{"n_ops": 3, "frac": 1.0}]}
    result = build(spec, pd.DataFrame(rows), None)
    assert len(result) == 12 and (result.n_ops == 3).all()
    assert summarize(result)["n"] == 12


def test_fingerprint_tracks_display_order_and_data_namespace(setup, tmp_path):
    import yaml
    from rlvr.io import check_task_directory

    cfg, task, _ = setup
    changed = copy.deepcopy(cfg)
    changed["task"]["functions"] = dict(reversed(list(changed["task"]["functions"].items())))
    other, _ = task_and_splits(changed)
    assert task.id2name != other.id2name and task.fingerprint != other.fingerprint
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "task_config.yaml").write_text(yaml.safe_dump({"task": cfg["task"]}, sort_keys=False))
    (meta / "task_fingerprint.json").write_text(json.dumps({"task_fingerprint": task.fingerprint}))
    check_task_directory(task, tmp_path)
    with pytest.raises(ValueError, match="fingerprint"):
        check_task_directory(other, tmp_path)
