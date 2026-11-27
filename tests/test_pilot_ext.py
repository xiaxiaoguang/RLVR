"""Pilot extension (rl_pool.pilot_cells, rl_pool.pilot_skip) and its helper scripts: the stream check,
the bridge test's selection and the label merge. CPU only, no model weights.
Run: python tests/test_pilot_ext.py (or pytest)."""
import importlib.util
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from rlvr import pilot as PL  # noqa: E402
from rlvr.config import load_config, task_and_splits  # noqa: E402
from rlvr.io import problems_to_rows  # noqa: E402
from rlvr.sets import cell_candidates  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TREE_CFG = load_config(ROOT / "configs" / "tasks" / "tree_v1.yaml")
TREE, SPLITS = task_and_splits(TREE_CFG)


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    raise AssertionError(f"expected {exc.__name__}")


def _rc(**kw):
    return {**TREE_CFG["rl_pool"], **kw}


def test_defaults_unchanged():
    rc = _rc(pilot_per_cell=6)
    assert rc["pilot_cells"] == "" and rc["pilot_skip"] == 0
    cells = [(2, "L"), (4, "H")]
    old = [p for d, g in cells for p in cell_candidates(SPLITS, "rl", d, g, rc["pilot_per_cell"], rc["pilot_seed"])]
    assert [p.pid for p in PL.candidates(SPLITS, cells, rc)] == [p.pid for p in old]
    calib = pd.DataFrame({"depth": [2] * 4 + [4] * 4, "group": ["L"] * 4 + ["H"] * 4,
                          "p_hat": [0.5, 0.5, 0.1, 1.0, 0.5, 0.0, 0.0, 0.0], "skippable": [False] * 8})
    rates = PL.cell_rates(calib, rc["pilot_depths"], [0.4, 0.6])
    assert PL.select_cells(rc, rates) == PL.choose_cells(rates, rc["pilot_min_gap"], rc["pilot_max_cells"])


def test_skip_continues_stream():
    rc = _rc(pilot_cells="d3-H,d2-L", pilot_skip=7, pilot_per_cell=5)
    kept = PL.candidates(SPLITS, PL.parse_cells(rc["pilot_cells"]), rc)
    for i, (d, g) in enumerate([(3, "H"), (2, "L")]):
        longer = cell_candidates(SPLITS, "rl", d, g, 12, rc["pilot_seed"])
        part = kept[5 * i:5 * (i + 1)]
        assert [p.pid for p in part] == [p.pid for p in longer[7:]]  # the tail of a longer run
        assert not {p.pid for p in part} & {p.pid for p in longer[:7]}  # no repeat of the head
        assert all(p.depth == d and p.group == g for p in part)


def test_cells_key():
    rates = pd.DataFrame({"depth": [2, 4], "group": ["L", "H"], "n": [4, 4], "mean_p": [0.5, 0.5],
                          "skippable_frac": [0, 0], "rate": [0.5, 0.5]})
    assert PL.select_cells(_rc(pilot_cells="d4-H"), rates) == ([(4, "H")], [])
    assert PL.parse_cells(" d2-L , d4-mixed") == [(2, "L"), (4, "mixed")] and PL.parse_cells("") == []
    for bad in ("d4-X", "4-H", "d4H", "d0-H", "d1-mixed", "d4-H,d4-H", "d4-H,", "d4-h"):
        raises(ValueError, PL.parse_cells, bad)


def test_calibrate_rejects_bad_cells_before_model():
    CA = _script("calibrate")
    CA.load_model = lambda *a, **k: (_ for _ in ()).throw(AssertionError("model loaded"))
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "pt-x-s0"
        ckpt.mkdir()
        cfg = {**TREE_CFG, "paths": {**TREE_CFG["paths"], "data": f"{d}/data", "runs": f"{d}/runs"},
               "rl_pool": {**TREE_CFG["rl_pool"], "mode": "pilot"}}
        (ckpt / "run_config.yaml").write_text(yaml.safe_dump(cfg))
        args = [str(ckpt), "--device", "cpu", "--allow-unknown-commit"]
        assert "pilot_cells" in raises(SystemExit, CA.main, [str(ckpt), "rl_pool.pilot_cells=d4-X", *args[1:]])
        assert "pilot_skip" in raises(SystemExit, CA.main, [str(ckpt), "rl_pool.pilot_skip=-1", *args[1:]])
        assert "pilot_cells" in raises(ValueError, CA.main, [str(ckpt), "rl_pool.pilot_cells=[d4-H]", *args[1:]])
        assert sorted(p.name for p in Path(d).iterdir()) == ["pt-x-s0"]  # nothing written


def test_stream_check():
    SC = _script("check_pilot_stream")
    rc = _rc(pilot_cells="d4-H", pilot_skip=6, pilot_per_cell=4)
    head = PL.candidates(SPLITS, [(2, "L"), (4, "H")], _rc(pilot_per_cell=6))
    old = [dict(r) for r in problems_to_rows(TREE, head, origin="pilot")]
    res = SC.check(TREE, SPLITS, rc, old)
    assert res["ok"] and res["d4-H"]["n_old_rows"] == 6 and res["d4-H"]["n_run"] == 4
    tampered = [dict(r) for r in old]
    tampered[8] = {**tampered[8], "prompt": "x"}  # d4-H row 2: same pid, other content
    r = SC.check(TREE, SPLITS, rc, tampered)["d4-H"]
    assert not r["ok"] and r["head_pids_ok"] and r["first_row_mismatch"] == 2
    r = SC.check(TREE, SPLITS, _rc(pilot_cells="d4-H", pilot_skip=5, pilot_per_cell=4), old)["d4-H"]
    assert not r["ok"] and not r["head_count_ok"] and r["run_pids_in_old_file"] == 1  # overlaps the old head


def _labels(n=40, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.choice([0.1, 0.35, 0.5, 0.65, 0.9], n)
    return pd.DataFrame({"pid": [f"p{i}" for i in range(n)], "depth": rng.choice([2, 4], n),
                         "group": rng.choice(["H", "L"], n), "skippable": rng.random(n) < 0.2, "p_hat": p,
                         "p_hat_relabel": np.where(rng.random(n) < 0.8, 0.5, np.nan), "str_len": 6,
                         "model_id": "m", "task_fingerprint": "f", "n_ops": 4})


def test_bridge_selection():
    BT = _script("bridge_test")
    df = _labels(200)
    ok = BT.eligible(df, (4, "H"), [0.3, 0.7])
    a = BT.select(df, (4, "H"), [0.3, 0.7], 5, 0)
    assert list(a.pid) == list(BT.select(df, (4, "H"), [0.3, 0.7], 5, 0).pid)  # deterministic
    assert list(a.pid) != list(BT.select(df, (4, "H"), [0.3, 0.7], 5, 1).pid)
    assert set(a.pid) <= set(ok.pid) and len(set(a.pid)) == 5 and list(a.index) == sorted(a.index)
    assert ((ok.depth == 4) & (ok.group == "H") & ~ok.skippable & ok.p_hat.between(0.3, 0.7)
            & ok.p_hat_relabel.notna()).all()
    raises(ValueError, BT.select, df, (4, "H"), [0.3, 0.7], len(ok) + 1, 0)
    s = BT.summary(pd.DataFrame({"p_hat_relabel_old": [0.5, 0.4], "p_hat_new": [0.52, 0.7],
                                 "diff": [0.02, 0.3]}), [0.4, 0.6])
    assert not s["pass"] and s["frac_in_range_old"] == 1.0 and s["frac_in_range_new"] == 0.5


def test_merge():
    ML = _script("merge_labels")
    old, new = _labels(30, 1), _labels(20, 2)
    new["pid"] = [f"q{i}" for i in range(20)]
    m = ML.merge(old, new[list(new.columns)[::-1]])
    assert list(m.source) == ["pilot"] * 30 + ["ext"] * 20 and list(m.columns) == list(old.columns) + ["source"]
    assert "in both" in raises(ValueError, ML.merge, old, new.assign(pid=old.pid[:20].to_numpy()))
    assert "columns" in raises(ValueError, ML.merge, old, new.drop(columns="n_ops"))
    assert "model_id" in raises(ValueError, ML.merge, old, new.assign(model_id="m2"))
    assert "task_fingerprint" in raises(ValueError, ML.merge, old, new.assign(task_fingerprint="g"))
    rep = ML.report(m, ("pilot", "ext"), [0.4, 0.6])
    for c, r in rep["total"].items():
        assert r["n_candidates"] == rep["pilot"][c]["n_candidates"] + rep["ext"][c]["n_candidates"]
        assert r["n_in_range"] == rep["pilot"][c]["n_in_range"] + rep["ext"][c]["n_in_range"]
        assert rep["ext"][c]["yield"] == round(rep["ext"][c]["n_in_range"] / rep["ext"][c]["n_candidates"], 4)
    with tempfile.TemporaryDirectory() as d:
        a, b, out = Path(d) / "a.parquet", Path(d) / "b.parquet", Path(d) / "m" / "x.parquet"
        old.to_parquet(a)
        new.to_parquet(b)
        assert ML.main([str(a), str(b), str(out)]) == 0
        assert len(pd.read_parquet(out)) == 50 and (Path(d) / "m" / "x.merge_meta.json").exists()
        assert "not replaced" in raises(SystemExit, ML.main, [str(a), str(b), str(out)])
        assert ML.main([str(a), str(a), str(Path(d) / "y.parquet")]) == 1  # shared pids: refused
        assert not (Path(d) / "y.parquet").exists()


if __name__ == "__main__":
    for k, fn in dict(globals()).items():
        if k.startswith("test_"):
            fn()
            print("ok", k)
