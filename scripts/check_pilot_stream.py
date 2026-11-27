"""Check that a pilot extension continues the candidate streams of an earlier pilot (CPU, no model).

    python scripts/check_pilot_stream.py checkpoints/tree-v1/pt-tree-mix-cf0.0035-s0 \
        runs/tree-v1/pt-tree-mix-cf0.0035-s0/calib/pilot_candidates.jsonl \
        rl_pool.pilot_cells=d4-H rl_pool.pilot_skip=10000 rl_pool.pilot_per_cell=5000 [--out check.json]

The config is the checkpoint's run_config.yaml plus the overrides, read as scripts/calibrate.py reads it
(give the same overrides as the extension run). For each cell of rl_pool.pilot_cells the script builds the
cell's first pilot_skip + pilot_per_cell candidates with rlvr.pilot.candidates, the function calibrate.py
uses, and checks:
  head  the first pilot_skip equal the cell's rows of the earlier pilot_candidates.jsonl, in order, by pid
        and by full row content (as calibrate.py writes the rows); the file must hold exactly pilot_skip
        rows of the cell
  run   the extension's candidates (rl_pool.pilot_skip as given) are the rest of that stream
  new   no pid of the extension is in the earlier file (any cell), and none appears twice
Exit code 1 on any failed check. --out also writes the result as JSON (an existing file is not replaced).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlvr import pilot as PL  # noqa: E402
from rlvr.config import load_config, resolve, task_and_splits  # noqa: E402
from rlvr.io import problems_to_rows, read_jsonl  # noqa: E402


def _norm(row: dict) -> dict:
    return json.loads(json.dumps(row))  # tuples -> lists, as in the written file


def check(task, splits, rc: dict, old_rows: list[dict]) -> dict:
    """{cell name: {check: ok, ...}, "ok": all passed}; pure, so tests can run it on small streams."""
    skip, n = rc["pilot_skip"], rc["pilot_per_cell"]
    old_pids = {r["pid"] for r in old_rows}
    res: dict = {"ok": True}
    for cell in PL.parse_cells(rc["pilot_cells"]):
        d, g = cell
        t = time.time()
        full = PL.candidates(splits, [cell], {**rc, "pilot_skip": 0, "pilot_per_cell": skip + n})
        run = PL.candidates(splits, [cell], rc)
        secs = time.time() - t
        old = [r for r in old_rows if r["depth"] == d and r["group"] == g]
        head = [_norm(r) for r in problems_to_rows(task, full[:skip], origin="pilot")]
        first_bad = next((i for i, (a, b) in enumerate(zip(head, old)) if a != b), None)
        pid_bad = next((i for i, (a, b) in enumerate(zip(head, old)) if a["pid"] != b["pid"]), None)
        run_pids = [p.pid for p in run]
        r = {"n_old_rows": len(old), "n_head": skip, "n_run": len(run),
             "head_count_ok": len(old) == skip,
             "head_pids_ok": pid_bad is None, "first_pid_mismatch": pid_bad,
             "head_rows_ok": first_bad is None, "first_row_mismatch": first_bad,
             "run_is_stream_tail": run_pids == [p.pid for p in full[skip:]] and len(run) == n,
             "run_pids_in_old_file": len(set(run_pids) & old_pids),
             "run_pids_repeated": len(run_pids) - len(set(run_pids)),
             "generation_seconds": round(secs, 1)}
        r["ok"] = (r["head_count_ok"] and r["head_pids_ok"] and r["head_rows_ok"] and r["run_is_stream_tail"]
                   and r["run_pids_in_old_file"] == 0 and r["run_pids_repeated"] == 0)
        res[PL.cell_name(d, g)] = r
        res["ok"] = res["ok"] and r["ok"]
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="checkpoint dir with run_config.yaml")
    ap.add_argument("old_candidates", help="the earlier pilot's calib/pilot_candidates.jsonl")
    ap.add_argument("--out", default=None, help="also write the result here (JSON)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else None
    if out and out.exists():
        raise SystemExit(f"{out} exists; not replaced")
    cfg = load_config(resolve(args.ckpt) / "run_config.yaml", args.overrides)
    rc = cfg["rl_pool"]
    if not PL.parse_cells(rc["pilot_cells"]) or rc["pilot_skip"] <= 0:
        raise SystemExit("give rl_pool.pilot_cells and rl_pool.pilot_skip > 0, as for the extension run")
    task, splits = task_and_splits(cfg)
    old_path = resolve(args.old_candidates)
    res = check(task, splits, rc, read_jsonl(old_path))
    res = {"old_candidates": str(old_path), "overrides": list(args.overrides),
           "rl_pool": {k: rc[k] for k in ("pilot_cells", "pilot_skip", "pilot_per_cell", "pilot_seed")}, **res}
    print(json.dumps(res, indent=1))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
    if not res["ok"]:
        bad = [c for c, v in res.items() if isinstance(v, dict) and "ok" in v and not v["ok"]]
        print(f"MISMATCH in {', '.join(bad)}: the extension does not continue the earlier stream", file=sys.stderr)
        return 1
    print("stream check PASS: the extension continues the earlier streams with new, unique candidates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
