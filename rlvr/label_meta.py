"""Compare two label_meta.json files (written by scripts/calibrate.py beside the labels).

    python -m rlvr.label_meta data/labels/pt-x-s0.meta.json other/pt-x-s0.meta.json

Prints every key that differs. Keys in IGNORE are skipped; keys in NOTE (or under a NOTE prefix)
describe the run or the machine and are only reported; a difference in any other key means the two
label files were not made the same way (exit code 1), so their p_hat should not be pooled or compared.
"""
from __future__ import annotations

import json
import sys

IGNORE = ("time", "ckpt", "ckpt_path")
NOTE = ("git_commit", "label.batch_size", "seed", "world", "transformers", "skip_pool", "pool_first",
        "calib_per_cell", "reuse_calib", "overrides", "env.", "reused_calib")
# What must be equal for calib labels of an earlier run to be reused (calibrate.py --reuse-calib).
CALIB_KEYS = ("model_id", "weights_sha256", "label.k", "label.temperature", "max_new_tokens", "eval_version")


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and prefix + k != "reused_calib":
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[prefix + k] = v
    return out


def is_note(key: str) -> bool:
    return any(key == n or (n.endswith(".") and key.startswith(n)) for n in NOTE)


def diff(a: dict, b: dict) -> dict[str, tuple]:
    """{dotted key: (value in a, value in b)} for every key that differs, IGNORE keys left out."""
    fa, fb = _flat(a), _flat(b)
    return {k: (fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb))
            if k not in IGNORE and fa.get(k) != fb.get(k)}


def main(argv: list[str] | None = None) -> int:
    paths = sys.argv[1:] if argv is None else argv
    if len(paths) != 2:
        raise SystemExit(__doc__)
    d = diff(*(json.loads(open(p).read()) for p in paths))
    must = [k for k in d if not is_note(k)]
    for k, (va, vb) in d.items():
        print(f"{'note  ' if is_note(k) else 'DIFFER'} {k}: {va!r} | {vb!r}")
    print(f"{len(must)} difference(s) in how the labels were made, {len(d) - len(must)} note(s)")
    return 1 if must else 0


if __name__ == "__main__":
    sys.exit(main())
