"""v2 tests: binary operators, tree problems, skippable check, splits, eval v3, pilot logic.
Run: python tests/test_tree.py (or pytest)."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from rlvr import pilot as PL  # noqa: E402
from rlvr.atoms import BINARY  # noqa: E402
from rlvr.config import load_config, task_and_splits  # noqa: E402
from rlvr.gate import _in_cells, cell_name  # noqa: E402
from rlvr.io import read_jsonl  # noqa: E402
from rlvr.sets import cell_candidates, stratified_set  # noqa: E402
from rlvr.task import Task  # noqa: E402
from rlvr.tokenizer import CharTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "v2" / "base.yaml")
TASK, SPLITS = task_and_splits(CFG)
V1 = task_and_splits(load_config())[0]
F = {n: TASK.name2id[n] for n in TASK.id2name}


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    raise AssertionError(f"expected {exc.__name__}")


def test_binary_operators():
    a, b = "abcdefg", "zyxwvut"
    want = {"add": "zzzzzzz", "alternate": "aycweug", "splice": "abcwvut", "max": "zyxwvut"}
    for name, out in want.items():
        assert TASK.apply(F[name], a, b) == out, name
    assert TASK.apply(F["max"], "azbycx", "zazbzc") == "zzzyzx"
    A = 26
    assert BINARY.build("sub")([0, 5], [1, 2], A) == [25, 3]
    assert BINARY.build("minimum")([0, 5], [1, 2], A) == [0, 2]
    # ids: unary keep v1 ids, binary F24.. in config order
    assert TASK.id2name[:24] == V1.id2name and TASK.B == (24, 25, 26, 27)
    assert [TASK.id2name[i] for i in TASK.B] == ["add", "alternate", "splice", "max"]
    # an operator that ignores an input is rejected when the task is built
    BINARY.add("first_only", lambda: (lambda a, b, A: list(a)), override=True)
    cfg = {**CFG["task"], "binary_functions": {"first": "first_only"}}
    assert "both inputs" in raises(AssertionError, Task, cfg)


def test_tree_format_and_structure():
    # design.md example shape: F25(F11(F03(x)),F07(y)) with illustrative ids
    u1, u2, u3 = F["rotl1"], F["shift_p1"], F["reverse"]
    funcs, args = (u1, u2, u3, F["add"]), ((-1,), (0,), (-2,), (1, 2))
    p = TASK.make_tree(funcs, args, ("abcdefgh", "ijklmnop"))
    assert (p.n_ops, p.tree_depth, p.n_bin, p.dep_dist) == (4, 3, 1, 2)
    n = TASK.fname
    assert TASK.format_prompt(p) == f"Q:{n(F['add'])}({n(u2)}({n(u1)}(abcdefgh)),{n(u3)}(ijklmnop))=\n"
    s = p.steps
    assert s[0] == "bcdefgha" and s[1] == "cdefghib" and s[2] == "ponmlkji"
    target = (f"{n(u1)}(abcdefgh)={s[0]};{n(u2)}({s[0]})={s[1]};{n(u3)}(ijklmnop)={s[2]};"
              f"{n(F['add'])}({s[1]},{s[2]})={s[3]};\nA:{s[3]}<eos>")
    assert TASK.format_target(p) == target
    assert TASK.parse_steps(target) == list(s) and TASK.parse_answer(target) == p.answer
    assert TASK.reward(target, p) == 1.0 and TASK.reward(target, p, "partial") == 1.0
    bad = target.replace(f"={s[2]};", "=aaaaaaaa;", 1)
    assert abs(TASK.reward(bad, p, "partial") - 3 / 4) < 1e-9
    # round trip, pid stable, group from unary functions only
    q = TASK.from_dict(p.to_dict())
    assert q == p and q.pid == p.pid and q.steps == p.steps
    assert p.group == TASK.group_of((u1, u2, u3))
    assert TASK.make_tree((F["add"],), ((-1, -2),), ("abc", "def")).group == "bin"
    # a tree without binary operators is a v1 chain (same pid and representation)
    c = TASK.make_tree((u1, u2), ((-1,), (0,)), ("abcdefgh",))
    assert not c.is_tree and c.pid == V1.make_problem((u1, u2), "abcdefgh").pid
    assert (c.tree_depth, c.n_bin, c.dep_dist) == (2, 0, 1)
    assert TASK.format_prompt(c) == V1.format_prompt(V1.make_problem((u1, u2), "abcdefgh"))
    # malformed trees
    raises(AssertionError, TASK.make_tree, (F["add"],), ((-1,),), ("abc",))
    raises(AssertionError, TASK.make_tree, (u1, F["add"]), ((-1,), (-2, -1)), ("abc", "def"))
    raises(AssertionError, TASK.make_tree, (F["add"],), ((-1, -2),), ("abc", "defg"))


def test_sample_tree_cells():
    rng = random.Random(0)
    for n_ops in range(1, 9):
        for n_bin in range(4):
            for g in ("H", "L", "mixed", "bin", None):
                if not TASK.valid_cell(n_ops, n_bin, g):
                    continue
                for _ in range(20):
                    p = TASK.sample_tree(n_ops, n_bin, rng, g)
                    assert p.n_ops == n_ops and p.n_bin == n_bin and len(p.inputs) == n_bin + 1
                    assert g is None or p.group == g
                    assert len({len(x) for x in p.inputs}) == 1 and len(p.x) in TASK.str_lens
                    assert TASK.from_dict(p.to_dict()).pid == p.pid
                    assert 1 <= p.tree_depth <= n_ops
    assert not TASK.valid_cell(2, 2, "H") and TASK.valid_cell(2, 2, "bin") and not TASK.valid_cell(2, 0, "bin")
    assert not TASK.valid_cell(3, 2, "mixed") and TASK.valid_cell(4, 2, "mixed")
    assert not V1.valid_cell(2, 1, "H"), "v1 task has no binary operators"


def test_skippable():
    srt = F["sort_asc"]
    assert TASK.is_skippable(TASK.make_problem((srt,), "abcdef"))        # already sorted
    assert not TASK.is_skippable(TASK.make_problem((srt,), "fedcba"))
    # max(x, y) with y >= x everywhere: replacing the node by y gives the same answer
    p = TASK.make_tree((F["max"],), ((-1, -2),), ("aaaaaa", "bbbbbb"))
    assert TASK.is_skippable(p)
    p = TASK.make_tree((F["add"],), ((-1, -2),), ("bcdefg", "bbbbbb"))
    assert not TASK.is_skippable(p)
    # a later sort erases an earlier reverse
    assert TASK.is_skippable(TASK.make_problem((F["reverse"], srt), "dbecaf"))


def test_heldout_edges_in_trees():
    a, b = min(SPLITS.heldout_pairs)
    x, y = "abcdefgh", "ijklmnop"
    p = TASK.make_tree((a, b, F["add"]), ((-1,), (0,), (1, -2)), (x, y))
    assert SPLITS.has_heldout(p)
    q = TASK.make_tree((a, F["add"], b), ((-1,), (0, -2), (1,)), (x, y))   # binary node in between
    assert not SPLITS.has_heldout(q)
    assert SPLITS.has_heldout(TASK.make_problem((a, b), x)) == SPLITS.has_heldout_pair((a, b))


def test_tokenizer_versions():
    v1 = CharTokenizer.load(ROOT / "data" / "meta" / "tokenizer.json")
    v3 = CharTokenizer.load(ROOT / "data" / "eval" / "v3" / "meta" / "tokenizer.json")
    assert v1.vocab_size == 48 and v3.vocab_size == 49 and v3.vocab[:48] == v1.vocab and v3.vocab[48] == ","
    assert CharTokenizer.for_task(V1).vocab == v1.vocab and CharTokenizer.for_task(TASK).vocab == v3.vocab


def test_v1_eval_sets_regenerate():
    """New code + v1 config reproduces the tracked v2 eval sets (first two sets, same rng)."""
    cfg = load_config()
    splits = task_and_splits(cfg)[1]
    ec = cfg["eval"]
    rng, seen = random.Random(ec["seed"]), set()
    for name in ("E1", "E2"):
        s = ec["sets"][name]
        probs = stratified_set(splits, "eval", s["depths"], s["groups"], s["per_cell"], rng, seen,
                               novel=s.get("novel", False), by_function=s.get("by_function", False))
        assert [p.pid for p in probs] == [r["pid"] for r in read_jsonl(ROOT / "data/eval/v2" / f"{name}.jsonl")]


def test_eval_v3_sets():
    rows = read_jsonl(ROOT / "data/eval/v3/E1.jsonl")
    binary = [r for r in rows if r.get("n_bin")]
    assert len(binary) == 50 * len(TASK.B) and {r["funcs"][0] for r in binary} == set(TASK.B)
    calib = read_jsonl(ROOT / "data/eval/v3/calib.jsonl")
    cells = {(r["depth"], r.get("n_bin", 0), r["group"]) for r in calib}
    assert len(cells) == 77 and len(calib) == 77 * 200
    for r in calib[::500]:
        p = TASK.from_dict(r)
        assert p.pid == r["pid"] and SPLITS.pool_of(p.pid) == "eval" and not SPLITS.has_heldout(p)


def test_cell_candidates_prefix():
    c = (3, 1, "H")
    a = cell_candidates(SPLITS, "rl", c, 30, 7)
    b = cell_candidates(SPLITS, "rl", c, 10, 7)
    assert [p.pid for p in a[:10]] == [p.pid for p in b] and len({p.pid for p in a}) == 30
    assert all(SPLITS.pool_of(p.pid) == "rl" and p.n_bin == 1 and p.depth == 3 and p.group == "H" for p in a)
    assert [len(p.x) for p in a[:5]] == TASK.str_lens


def test_gate_cells_with_n_bin():
    df = pd.DataFrame({"depth": [4, 4, 4], "n_bin": [0, 1, 1], "group": ["H", "H", "L"]})
    assert _in_cells(df, [{"depth": 4, "group": "H"}]).tolist() == [True, False, False]
    assert _in_cells(df, [{"depth": 4, "n_bin": 1, "group": "H"}]).tolist() == [False, True, False]
    assert _in_cells(df.drop(columns="n_bin"), [{"depth": 4, "group": "H"}]).tolist() == [True, True, False]
    assert cell_name(4, "H") == "d4-H" and cell_name(4, "H", 1) == "d4b1-H"


def test_pilot_logic():
    rows = []
    for (d, nb, g, ps) in [(2, 0, "L", [0.5] * 10 + [1.0] * 10), (4, 1, "H", [0.45] * 5 + [0.0] * 15),
                           (3, 0, "H", [0.5] * 20), (5, 2, "mixed", [0.0] * 20)]:
        rows += [{"depth": d, "n_bin": nb, "group": g, "p_hat": p, "skippable": False} for p in ps]
    rates = PL.cell_rates(pd.DataFrame(rows), [1, 2, 3, 4, 5], [0.4, 0.6])
    r = {(x.depth, x.n_bin, x.group): x.rate for x in rates.itertuples()}
    assert r[(2, 0, "L")] == 0.5 and r[(4, 1, "H")] == 0.25 and r[(5, 2, "mixed")] == 0.0
    cells, pairs = PL.choose_cells(rates, 2, 4)
    assert cells == [(2, 0, "L"), (4, 1, "H")] and pairs[0]["score"] == 0.25
    assert PL.shared_mix_supply({6: 10, 7: 10}, {6: 100, 7: 5}) == 10   # mix 50/50: limited by 5 -> 10
    assert PL.shared_mix_supply({6: 600, 7: 0}, {6: 700, 7: 900}) == 600
    pool = pd.DataFrame({"depth": [2] * 4 + [4] * 4, "n_bin": [0] * 4 + [1] * 4, "group": ["L"] * 4 + ["H"] * 4,
                         "str_len": [6, 7, 6, 7] * 2, "skippable": [False, False, True, False] * 2,
                         "p_hat": [0.5] * 8, "p_hat_relabel": [0.5, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.5],
                         "tree_depth": [2] * 4 + [3] * 4, "dep_dist": [1] * 8})
    rep = PL.report(pool, cells, pairs, [0.4, 0.6], 2)
    assert rep["cells"]["d2-L"]["n_in_range"] == 2 and rep["cells"]["d4b1-H"]["n_in_range"] == 3
    assert rep["pairs"][0]["shared_mix_supply"] == 2 and rep["go"]


def test_diag_first_error_on_trees():
    sys.path.insert(0, str(ROOT / "scripts"))
    from diag_steps import first_error, parse_cell
    assert parse_cell("4-H") == (4, 0, "H") and parse_cell("4b1-mixed") == (4, 1, "mixed")
    u1, u2, u3 = F["rotl1"], F["shift_p1"], F["reverse"]
    p = TASK.make_tree((u1, u2, u3, F["add"]), ((-1,), (0,), (-2,), (1, 2)), ("abcdefgh", "ijklmnop"))
    t, s, n = TASK.format_target(p), p.steps, TASK.fname
    assert first_error(TASK, p, t) == ("ok", None, {})
    # second operand of the binary step copied wrong (step 3 reads step 2's output)
    bad = t.replace(f"{n(F['add'])}({s[1]},{s[2]})", f"{n(F['add'])}({s[1]},{s[2][::-1]})")
    err = first_error(TASK, p, bad)
    assert err[:2] == ("copy", 3) and err[2]["operand"] == 1 and not err[2]["from_leaf"]
    # leaf operand copied wrong in step 2
    err = first_error(TASK, p, t.replace(f"{n(u3)}(ijklmnop)", f"{n(u3)}(ijklmnoo)"))
    assert err[:2] == ("copy", 2) and err[2]["from_leaf"]
    assert first_error(TASK, p, t.replace(f"={s[2]};", "=aaaaaaaa;", 1))[:2] == ("apply", 2)
    assert first_error(TASK, p, t.replace(f"{n(F['add'])}({s[1]},", f"{n(F['add'])}("))[:2] == ("copy", 3)
    assert first_error(TASK, p, t.replace(f"A:{p.answer}", "A:aaaaaaaa"))[:2] == ("answer_copy", 4)
    # chains as in v1
    c = TASK.make_problem((u1, u2), "abcdefgh")
    assert first_error(TASK, c, TASK.format_target(c)) == ("ok", None, {})
    assert first_error(TASK, c, TASK.format_target(c).replace(f"{n(u2)}({c.steps[0]})", f"{n(u2)}(abcdefgh)"))[:2] \
        == ("copy", 1)


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
