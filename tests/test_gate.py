"""Gate checks (rlvr/gate.py) and p_label=relabel in build_dataset.py, on synthetic tables.
No model or GPU needed. Run: python tests/test_gate.py (or pytest)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rlvr import gate as G  # noqa: E402
from rlvr.config import load_config  # noqa: E402
from rlvr.difficulty import in_p_range  # noqa: E402

_spec = importlib.util.spec_from_file_location("build_dataset", ROOT / "scripts" / "build_dataset.py")
BD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BD)

GC = load_config()["gate"]
LC = load_config()["label"]


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    raise AssertionError(f"expected {exc.__name__}")


def e1_rows(correct_per_func: list[int], n: int = 50) -> list[dict]:
    rows = [{"depth": "all", "group": "all", "n": n * len(correct_per_func),
             "greedy_acc": sum(correct_per_func) / (n * len(correct_per_func))}]
    rows += [{"depth": 1, "group": "H", "func": f"F{i:02d}", "n": n, "greedy_acc": c / n}
             for i, c in enumerate(correct_per_func)]
    return rows


def test_depth1_wilson():
    ok = G.depth1_greedy(e1_rows([50] * 22 + [48, 48]), GC)  # two functions at 96%: noise at n=50
    assert ok["pass"] and not ok["funcs_below_min"], ok
    bad = G.depth1_greedy(e1_rows([50] * 23 + [47]), GC)  # 94%: upper bound < 0.98
    assert not bad["pass"] and list(bad["funcs_below_min"]) == ["F23"], bad
    low = G.depth1_greedy(e1_rows([48] * 24), GC)  # every function at 96%: overall 0.96 < 0.98
    assert not low["pass"] and not low["funcs_below_min"]


def pool_frame(n_per_len: dict, depth=2, group="L", p=0.5, start=0) -> pd.DataFrame:
    rows, i = [], start
    for ln, n in n_per_len.items():
        for _ in range(n):
            rows.append({"pid": f"p{i}", "depth": depth, "group": group, "str_len": ln, "p_hat": p,
                         "funcs": [0, 1][:depth], "origin": "base"})
            i += 1
    return pd.DataFrame(rows)


def test_matched_cells_per_length():
    lens = {6: 150, 7: 150, 8: 150, 9: 150, 10: 50}  # 650 in range, but length 10 has only 50
    pool = pd.concat([pool_frame(lens), pool_frame({L: 150 for L in range(6, 11)}, 4, "H", start=10_000)])
    r = G.matched_cells(pool, GC)
    assert r["cells"]["d4-H"]["pass"] and not r["cells"]["d2-L"]["pass"] and not r["pass"]
    assert r["cells"]["d2-L"]["n_in_range"] == 650 and r["cells"]["d2-L"]["n_in_range_by_len"][10] == 50
    # relabelled cells are judged on p_hat_relabel, not on the first pass
    pool = pd.concat([pool_frame({L: 150 for L in range(6, 11)}),
                      pool_frame({L: 150 for L in range(6, 11)}, 4, "H", start=10_000)])
    pool["p_hat_relabel"] = np.where(pool.depth == 2, 0.8, 0.5)
    r = G.matched_cells(pool, GC, LC["relabel_cells"])
    assert r["cells"]["d2-L"]["p_col"] == "p_hat_relabel" and r["cells"]["d2-L"]["n_in_range"] == 0
    assert r["cells"]["d4-H"]["pass"]


def test_relabel_mask_and_attach():
    pool = pd.DataFrame({"pid": ["a", "b", "c", "d"], "depth": [2, 2, 4, 3], "group": ["L", "L", "H", "H"],
                         "p_hat": [0.5, 0.9, 0.31, 0.5]})
    m = G.relabel_mask(pool, LC)
    assert m.tolist() == [True, False, True, False]  # b: outside the screen; d: d3-H not relabelled
    again = pd.DataFrame({"pid": ["a", "c"], "n_correct": [30, 20], "k": [64, 64], "p_hat": [30 / 64, 20 / 64],
                          "ci_lo": [0.3, 0.2], "ci_hi": [0.6, 0.4]})
    out = G.attach_relabel(pool, m, again)
    assert out.p_hat_relabel.isna().tolist() == [False, True, False, True] and out.k_relabel[0] == 64
    raises(ValueError, G.attach_relabel, pool, m, again.iloc[::-1])
    assert G.attach_relabel(pool, np.zeros(4, bool), None).p_hat_relabel.isna().all()
    assert not G.relabel_mask(pool, {**LC, "relabel_k": 0}).any()


def test_final_loss_and_verdict():
    metrics = [{"step": 9000, "loss": 3e-5, "loss_d1": 1e-5, "loss_d3": 5e-3},
               {"step": 9500, "loss": 2e-5, "loss_d1": 1e-5, "loss_d2": 1e-4}]  # no d3 in the last row
    fl = G.final_loss(metrics, GC)
    assert fl["value"] == 2e-5 and fl["by_depth"] == {"d3": 5e-3} and fl["pass"]
    assert G.final_loss(metrics, GC, ckpt_step=100) is None
    checks = {"depth1_greedy": {"pass": True}, "spread": {"pass": False}, "final_loss": {"pass": False}}
    v = G.verdict(checks, ["depth1_greedy", "matched_cells"])
    assert v == {"pass": True, "warnings": ["spread", "final_loss"], "not_run": ["matched_cells"]}
    assert not G.verdict(checks, GC["hard_checks"])["pass"]


def test_level_configs():
    for lv in ("weak", "strong"):
        g = load_config(f"configs/pretrain/{lv}.yaml")["gate"]
        assert g["hard_checks"] == ["depth1_greedy"] and g["target"]["depth"] in (3, 4)
    for lv in ("mid", "midbal"):
        cfg = load_config(f"configs/pretrain/{lv}.yaml")
        assert "matched_cells" in cfg["gate"]["hard_checks"]
        G.check_config(cfg["gate"], cfg["label"])
    assert "unknown" in raises(ValueError, G.check_config, {**GC, "hard_checks": ["spred"]}, LC)
    cells = GC["matched_cells"] + [{"depth": 3, "group": "L"}]
    assert "relabel_cells" in raises(ValueError, G.check_config, {**GC, "matched_cells": cells}, LC)
    calib = pd.DataFrame({"depth": [3, 3, 4], "p_hat": [0.05, 0.15, 0.9]})
    assert G.target(calib, {"target": {"depth": 3, "p_range": [0.05, 0.2]}})["pass"]
    assert G.target(calib, {"target": None}) is None


def test_in_p_range_nan():
    p = [0.5, float("nan"), 0.1]
    assert in_p_range(p, "0.4-0.6").tolist() == [True, False, False]
    assert in_p_range(p, [0.0, 0.2]).tolist() == [False, False, True]


def synthetic_pool_and_labels(seed=0):
    """d2-L candidates, 200 per length; first-pass p_hat and an independent relabel."""
    rng = np.random.default_rng(seed)
    pool = pool_frame({L: 200 for L in range(6, 11)}).drop(columns="p_hat")  # candidates carry no p
    labels = pool[["pid"]].copy()
    labels["p_hat"] = rng.choice([0.2, 0.5, 0.65], size=len(pool))
    labels["n_correct"], labels["k"], labels["ci_lo"], labels["ci_hi"] = 0, 32, 0.0, 1.0
    screened = (labels.p_hat >= 0.3) & (labels.p_hat <= 0.7)
    labels["p_hat_relabel"] = np.where(screened, np.where(rng.random(len(pool)) < 0.5, 0.5, 0.7), np.nan)
    for c in ("n_correct_relabel", "k_relabel", "ci_lo_relabel", "ci_hi_relabel"):
        labels[c] = np.where(screened, 1.0, np.nan)
    return pool, labels


def test_build_dataset_relabel():
    pool, labels = synthetic_pool_and_labels()
    spec = {"name": "t", "n": 200, "seed": 0, "cells": [{"depth": 2, "group": "L", "p_bin": "0.4-0.6", "frac": 1.0}],
            "match_covariates": ["str_len"], "pool_origin": ["base"]}
    first = BD.build(spec, pool, labels)
    assert len(first) == 200 and "p_hat_relabel" not in first and (first.p_hat == 0.5).all()
    out = BD.build({**spec, "p_label": "relabel"}, pool, labels)
    assert len(out) == 200 and (out.p_hat_relabel == 0.5).all()
    assert out.str_len.value_counts().to_dict() == {L: 40 for L in range(6, 11)}  # whole-pool proportions
    s = BD.summarize(out)
    assert s["mean_p_relabel"] == 0.5 and "mean_p" in s
    # a short stratum names the covariate value
    short = labels.copy()
    short.loc[pool.str_len.to_numpy() == 10, "p_hat_relabel"] = np.nan
    msg = raises(BD.SpecError, BD.build, {**spec, "p_label": "relabel"}, pool, short)
    assert "'str_len': 10" in msg and "need 40" in msg, msg
    no_relabel = labels.drop(columns=list(G.RELABEL_COLS.values()))
    assert "relabel_k" in raises(BD.SpecError, BD.build, {**spec, "p_label": "relabel"}, pool, no_relabel)
    raises(BD.SpecError, BD.validate_spec, {**spec, "p_label": "second"})


def test_matched_specs_use_relabel():
    from rlvr.config import load_yaml
    for name in ("matched_p_shallow", "matched_p_deep", "matched_p_deeper"):
        spec = load_yaml(ROOT / "configs" / "datasets" / f"{name}.yaml")
        BD.validate_spec(spec)
        assert spec["p_label"] == "relabel" and spec["match_covariates"] == ["str_len"]


def test_relabel_report():
    spec = importlib.util.spec_from_file_location("relabel_report", ROOT / "scripts" / "relabel_report.py")
    RR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RR)
    pool, labels = synthetic_pool_and_labels()
    df = pool.merge(labels, on="pid")
    rep = RR.report(df, [{"depth": 2, "group": "L"}, {"depth": 4, "group": "H"}], [0.4, 0.6])
    r, first = rep["d2-L"], df[df.p_hat == 0.5]
    assert r["n_cell"] == 1000 and r["n_relabelled"] == int(df.p_hat_relabel.notna().sum())
    assert r["first_selected"]["n"] == len(first) and r["first_selected"]["mean_p_hat"] == 0.5
    assert abs(r["first_selected"]["mean_p_hat_relabel"] - first.p_hat_relabel.mean()) < 1e-4
    assert r["first_selected"]["frac_relabel_in_range"] == round(float((first.p_hat_relabel == 0.5).mean()), 4)
    assert r["relabel_selected"]["n"] == int((df.p_hat_relabel == 0.5).sum())
    assert sum(r["relabel_selected"]["n_by_len"].values()) == r["relabel_selected"]["n"]
    assert rep["d4-H"]["n_cell"] == 0 and rep["d4-H"]["first_selected"]["mean_p_hat"] is None


def test_calibrate_shards():
    """Multi-GPU labelling: strided shards gathered back give the single-GPU row order."""
    spec = importlib.util.spec_from_file_location("calibrate", ROOT / "scripts" / "calibrate.py")
    CA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(CA)
    full = pd.DataFrame({"pid": [f"p{i}" for i in range(10)], "p_hat": np.linspace(0, 1, 10)})
    for world in (1, 3, 4, 12):
        idx = [CA.shard(len(full), r, world) for r in range(world)]
        assert sorted(j for i in idx for j in i) == list(range(len(full)))
        parts = [full.iloc[i].assign(_i=i) if i else pd.DataFrame() for i in idx]  # ranks may hold nothing
        assert CA.merge_shards(parts[::-1]).equals(full)


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
