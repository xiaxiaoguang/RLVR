"""Step checker (rlvr/steps.py), pass@k estimator and greedy_eval step / pass@k columns (CPU)."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
import torch  # noqa: E402

import rlvr.evaluate as E  # noqa: E402
from rlvr.config import load_config, task_and_splits  # noqa: E402
from rlvr.model import build_model  # noqa: E402
from rlvr.steps import first_error  # noqa: E402
from rlvr.stream import corrupt  # noqa: E402
from rlvr.task import EOS  # noqa: E402

TINY = ["model.n_layer=2", "model.d_model=32", "model.d_ff=64", "model.n_head=2"]
CHAIN = task_and_splits(load_config(None, TINY))[0]
TREE_CFG = load_config("configs/tasks/tree_v1.yaml", TINY)
TREE = task_and_splits(TREE_CFG)[0]


def call(task, name, *args):
    return {"op": task.fname(task.name2id[name]), "args": [a if isinstance(a, dict) else {"value": a} for a in args]}


def random_problems(task, n=300, seed=0):
    rng = random.Random(seed)
    return [task.sample_problem(rng.randint(1, 5 if task is TREE else task.max_depth), rng) for _ in range(n)]


@pytest.mark.parametrize("task", [CHAIN, TREE], ids=["chain", "tree"])
def test_true_and_corrupted_targets(task):
    rng = random.Random(1)
    for p in random_problems(task):
        assert first_error(task, p, task.format_target(p)) == ("ok", None, len(p.funcs))
        steps = corrupt(task, p, rng)  # one step wrong, everything downstream recomputed from it
        i = next(j for j, (a, b) in enumerate(zip(steps, p.steps)) if a != b)
        text = task.format_target(p, steps, steps[-1])  # as in the pretraining stream's noisy targets
        assert first_error(task, p, text) == ("apply", i, i)


def test_chain_error_types():
    t = CHAIN
    f0, f1 = t.name2id["reverse"], t.name2id["shift_p1"]
    p = t.make_problem([f0, f1], "abcdefgh")
    s0, s1 = p.steps
    assert (s0, s1) == ("hgfedcba", "ihgfedcb")
    good = t.format_target(p)
    assert good == f"F{f0:02d}(abcdefgh)=hgfedcba;F{f1:02d}(hgfedcba)=ihgfedcb;\nA:ihgfedcb{EOS}"
    other = (f1 + 1) % t.n_funcs
    line1 = f"F{f0:02d}(abcdefgh)={s0};"
    cases = {
        f"{line1}F{other:02d}({s0})={s1};\nA:{s1}{EOS}": ("wrong_func", 1, 1),
        f"{line1}F99({s0})={s1};\nA:{s1}{EOS}": ("wrong_func", 1, 1),
        f"{line1}F{f1:02d}(abcdefgh)={s1};\nA:{s1}{EOS}": ("copy", 1, 1),
        f"{line1}F{f1:02d}({s0})=aaaaaaaa;\nA:aaaaaaaa{EOS}": ("apply", 1, 1),
        f"{line1}\nA:{s1}{EOS}": ("missing_step", 1, 1),
        f"{line1}F{f1:02d}({s0})={s1};F{f1:02d}({s1})=jihgfedc;\nA:{s1}{EOS}": ("extra_step", 2, 2),
        good.replace(EOS, ""): ("format", 2, 2),
        f"{line1}F{f1:02d}({s0})={s1};\nA:{s1}A:{s1}{EOS}": ("format", 2, 2),
        f"{line1}F{f1:02d}({s0})={s1};\nA:{s0}{EOS}": ("answer_copy", 2, 2),
        f"{line1}F{f1:02d}(hgf": ("missing_step", 1, 1),  # truncated, no <eos>
        "zq;;F(=" + EOS: ("missing_step", 0, 0),
        "": ("missing_step", 0, 0),
        good + "F00(x)=y;": ("ok", None, 2),  # text after <eos> is ignored
    }
    for text, want in cases.items():
        assert first_error(t, p, text) == want, text


def test_tree_error_types():
    t = TREE
    p = t.make_problem(call(t, "zip_add", call(t, "swap_adjacent", "abcdef"), call(t, "swap_halves", "bcdefg")))
    s0, s1, s2 = p.steps
    assert (s0, s1) == ("badcfe", "efgbcd")
    good = t.format_target(p)
    fa, fb, fz = (t.fname(t.name2id[n]) for n in ("swap_adjacent", "swap_halves", "zip_add"))
    head = f"{fa}(abcdef)={s0};{fb}(bcdefg)={s1};"
    assert good == f"{head}{fz}({s0},{s1})={s2};\nA:{s2}{EOS}"
    cases = {
        f"{head}{fz}({s1},{s0})={s2};\nA:{s2}{EOS}": ("copy", 2, 2),  # operands swapped
        f"{head}{fz}({s0})={s2};\nA:{s2}{EOS}": ("copy", 2, 2),  # an operand dropped
        f"{fa}(abcdef)={s0};{fz}({s0},{s1})={s2};\nA:{s2}{EOS}": ("wrong_func", 1, 1),  # node skipped
        f"{fb}(bcdefg)={s1};{fa}(abcdef)={s0};{fz}({s0},{s1})={s2};\nA:{s2}{EOS}": ("wrong_func", 0, 0),
        f"{head}\nA:{s2}{EOS}": ("missing_step", 2, 2),
        f"{head}{fz}({s0},{s1})=aaaaaa;\nA:aaaaaa{EOS}": ("apply", 2, 2),
        f"{head}{fz}({s0},{s1})={s2};\nA:{s1}{EOS}": ("answer_copy", 3, 3),
        good.replace(EOS, ""): ("format", 3, 3),
    }
    for text, want in cases.items():
        assert first_error(t, p, text) == want, text


@pytest.mark.parametrize("task", [CHAIN, TREE], ids=["chain", "tree"])
def test_ok_implies_reward_one(task):
    """Random character edits of the true target: ok => reward 1 and all steps counted."""
    rng = random.Random(2)
    chars = task.alphabet[:3] + ",;=()\nA:F0"
    n_ok = 0
    for p in random_problems(task, 200, seed=3):
        good = task.format_target(p)
        for _ in range(10):
            s = list(good)
            for _ in range(rng.randint(1, 3)):
                i = rng.randrange(len(s) + 1)
                op = rng.choice("dir")
                if op == "d" and i < len(s):
                    del s[i]
                elif op == "i":
                    s.insert(i, rng.choice(chars))
                elif i < len(s):
                    s[i] = rng.choice(chars)
            text = "".join(s)
            err, err_step, n_correct = first_error(task, p, text)
            assert 0 <= n_correct <= len(p.funcs)
            if err == "ok":
                n_ok += 1
                assert task.reward(text, p) == 1.0 and n_correct == len(p.funcs) and err_step is None
            else:
                assert n_correct == err_step  # the lines before the first error are all correct
    assert n_ok > 0  # some edits (e.g. after <eos>, between lines) leave the completion correct


def test_pass_at_k():
    for n in (1, 4, 16):
        for c in range(n + 1):
            assert E.pass_at_k(n, c, 1) == pytest.approx(c / n)
            assert E.pass_at_k(n, c, n) == (1.0 if c >= 1 else 0.0)
    assert all(E.pass_at_k(16, 0, k) == 0.0 for k in (1, 2, 4, 8, 16))
    assert E.pass_at_k(4, 2, 2) == pytest.approx(5 / 6)  # 1 - C(2,2)/C(4,2) = 1 - 1/6
    assert E.pass_at_k(5, 1, 2) == pytest.approx(0.4)  # 1 - C(4,2)/C(5,2) = 1 - 6/10


def test_greedy_eval_tiny_model():
    torch.manual_seed(0)
    tok = CHAIN.tokenizer()
    model = build_model(load_config(None, TINY)["model"], tok)
    rng = random.Random(4)
    sets = {"E1": [CHAIN.sample_problem(1, rng) for _ in range(6)],
            "E2": [CHAIN.sample_problem(d, rng) for d in (2, 3, 3, 4)]}
    for n_s in (0, 4):
        records = []
        rows = E.greedy_eval(model, tok, CHAIN, sets, 7, max_new_tokens=12, batch_size=8,
                             pass_k_samples=n_s, pass_k=(1, 2, 4, 8), records=records)
        assert len(records) == 10 and {r["eval_set"] for r in records} == {"E1", "E2"}
        for r in rows:
            assert 0.0 <= r["step_frac"] <= 1.0 and r["step_score"] >= 0
            assert {c for c in r if c.startswith("pass@")} == ({"pass@1", "pass@2", "pass@4"} if n_s else set())
        by_depth = {(r["eval_set"], r["depth"]): r["n"] for r in rows if r.get("aggregation") == "depth"}
        assert by_depth == {("E1", 1): 6, ("E2", 2): 1, ("E2", 3): 2, ("E2", 4): 1}
        timed = [r for r in rows if "sec" in r]
        assert [r["eval_set"] for r in timed] == ["E1", "E2"] and all(r["sec"] >= 0 for r in timed)
        assert all(r["depth"] == r["group"] == "all" for r in timed)


def test_greedy_eval_step_columns(monkeypatch):
    """Fake generation with known step scores: correct targets, cut-off targets and junk."""
    t = TREE
    tok = t.tokenizer()
    rng = random.Random(5)
    probs = [t.sample_problem(d, rng) for d in (2, 3, 3, 5, 5, 5)]
    by_prompt = {t.format_prompt(p): p for p in probs}
    keep = {p.pid: i % 3 for i, p in enumerate(probs)}  # problem i: 0 junk, 1 first line only, 2 all

    def fake_generate(model, tok, prompts, max_new_tokens, temperature=0.0, num_return_sequences=1, batch_size=1,
                      progress=""):
        out = []
        for pr in prompts:
            p = by_prompt[pr]
            text = {0: "zz", 1: t.format_target(p).split(";")[0] + ";", 2: t.format_target(p)}[keep[p.pid]]
            out += [text] * num_return_sequences
        return out

    monkeypatch.setattr(E, "generate", fake_generate)
    records = []
    rows = E.greedy_eval(None, tok, t, {"E2": probs}, pass_k_samples=2, pass_k=(1, 2, 4), records=records)
    want = {0: 0, 1: 1}
    n_correct = [want.get(keep[p.pid], p.n_ops) for p in probs]
    assert [r["n_correct"] for r in records] == n_correct
    assert [r["err_type"] for r in records] == [("missing_step", "missing_step", "ok")[keep[p.pid]] for p in probs]
    assert records[0]["n_ops"] == probs[0].n_ops and "n_binary" in records[0]
    top = next(r for r in rows if r["depth"] == "all")
    assert top["step_score"] == pytest.approx(sum(n_correct) / 6)
    assert top["step_frac"] == pytest.approx(sum(c / p.n_ops for c, p in zip(n_correct, probs)) / 6)
    assert top["greedy_acc"] == pytest.approx(2 / 6) and top["pass@1"] == top["pass@2"] == pytest.approx(2 / 6)
    assert "pass@4" not in top
    d5 = next(r for r in rows if r.get("aggregation") == "depth" and r["depth"] == 5)
    assert d5["n"] == 3 and d5["greedy_acc"] == pytest.approx(1 / 3)
    assert all("step_frac" in r for r in rows if r.get("aggregation") == "structure")
    assert top["sec"] >= 0 and sum("sec" in r for r in rows) == 1
