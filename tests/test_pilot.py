"""v2 pilot stage B helpers (design.md "v2 pilot", decisions 6 and 11): skippable check, per-cell
candidates, D/E cell choice and the go report. Run: python tests/test_pilot.py (or pytest)."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from rlvr import pilot as PL  # noqa: E402
from rlvr.config import load_config, task_and_splits  # noqa: E402
from rlvr.sets import cell_candidates  # noqa: E402
from rlvr.tree_task import Call, Leaf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TREE, TREE_SPLITS = task_and_splits(load_config(ROOT / "configs" / "tasks" / "tree_v1.yaml"))
CHAIN, _ = task_and_splits(load_config())


def test_skippable_trees():
    T, op = TREE, TREE.name2id
    x, y = "abcdef", "aaaaaa"  # y is all zeros: zip_add(x, y) = x, so the node can be skipped
    p = T.make_problem(Call(op["zip_add"], (Leaf(x), Leaf(y))))
    assert p.answer == x and T.is_skippable(p)
    q = T.make_problem(Call(op["zip_add"], (Leaf(x), Leaf("bbbbbb"))))
    assert not T.is_skippable(q)
    # an inner unary node whose removal does not change the root
    inner = Call(op["sort_asc"], (Leaf("fedcba"),))
    r = T.make_problem(Call(op["sort_asc"], (inner,)))
    assert T.is_skippable(r)


def test_skippable_chains():
    C = CHAIN
    f = C.name2id
    p = C.make_problem((f["sort_asc"], f["sort_asc"]), "dcbaef")
    assert C.is_skippable(p)
    q = C.make_problem((f["shift_p1"], f["shift_p1"]), "abcdef")
    assert not C.is_skippable(q)


def test_cell_candidates_prefix_and_lengths():
    a = cell_candidates(TREE_SPLITS, "rl", 3, "H", 25, 7)
    b = cell_candidates(TREE_SPLITS, "rl", 3, "H", 10, 7)
    assert [p.pid for p in a[:10]] == [p.pid for p in b]
    assert len({p.pid for p in a}) == 25 and {len(p.x) for p in a} == set(TREE.str_lens)
    assert all(p.depth == 3 and p.group == "H" and TREE_SPLITS.pool_of(p.pid) == "rl" for p in a)
    assert not any(TREE_SPLITS.has_heldout(p) for p in a)


def test_choose_cells_and_report():
    calib = pd.DataFrame({"depth": [1] * 4 + [2] * 4 + [4] * 4, "group": ["H"] * 12,
                          "p_hat": [0.5, 0.5, 1, 1, 0.5, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.0],
                          "skippable": [False] * 11 + [True]})
    rates = PL.cell_rates(calib, [1, 2, 3, 4, 5], [0.4, 0.6])
    assert dict(zip(rates.depth, rates.rate)) == {1: 0.5, 2: 0.25, 4: 0.75}
    cells, pairs = PL.choose_cells(rates, 2, 4)
    assert (pairs[0]["d"], pairs[0]["e"], pairs[0]["score"]) == ((1, "H"), (4, "H"), 0.5)  # min(.5, .75)
    assert cells == [(1, "H"), (4, "H"), (2, "H")] and len(pairs) == 2  # d1-d2 is not 2 apart
    pool = pd.DataFrame({"depth": [2] * 4 + [4] * 4, "group": ["H"] * 8, "str_len": [6, 7, 6, 7] * 2,
                         "skippable": [False, False, True, False] * 2, "p_hat": [0.5] * 8,
                         "p_hat_relabel": [0.5, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.5], "n_ops": [3] * 8})
    rep = PL.report(pool, [(2, "H"), (4, "H")], [{"d": (2, "H"), "e": (4, "H"), "score": 0.25}], [0.4, 0.6], 2)
    assert rep["cells"]["d2-H"]["n_in_range"] == 2 and rep["cells"]["d4-H"]["n_in_range"] == 3
    assert rep["pairs"][0]["shared_mix_supply"] == 2 and rep["go"]
    assert PL.shared_mix_supply({6: 600, 7: 0}, {6: 700, 7: 900}) == 600


if __name__ == "__main__":
    for k, fn in dict(globals()).items():
        if k.startswith("test_"):
            fn()
            print("ok", k)
