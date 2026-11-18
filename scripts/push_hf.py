"""Upload final base-model checkpoints to a Hugging Face model repo, one folder per run.

Checkpoints (~0.4 GB each) are too large for git. Run on a Perlmutter login node from the repo
root, where checkpoints/, runs/ and data/labels/ are the usual symlinks into scratch:

    source scripts/slurm/env.sh       # HF_HOME on scratch, so the token is stored there
    hf auth login                     # once (or export HF_TOKEN=hf_...)
    python scripts/push_hf.py --repo <user-or-org>/rlvr-pretrain-v1 --labels \
        pt-weak-cf0.000125-s0 pt-mid-cf0.00032-s0 pt-strong-cf0.00083-s0 pt-midbal-cf0.00032-s0
    python scripts/push_hf.py --repo <user-or-org>/rlvr-pretrain-v2 --task-dir tree-v1 --labels \
        --note-file <markdown file> pt-tree-mix-cf0.0035-s0 pt-tree-mix-cf0.006-s0

Everything goes in one commit. Per run, the repo gets <run>/ with the checkpoint as save_model()
wrote it (config.json, generation_config.json, model.safetensors, tokenizer.json, run_config.yaml),
the run's stats.json and calib/{gate.json,relabel_report.json,heatmap.png,calib_labels.parquet,pilot.json,
pilot_candidates.jsonl} when present, and with --labels the RL-pool labels data/labels/<run>.parquet as
<run>/labels.parquet (runs without one are listed and skipped). --task-dir names the task's subfolder of
checkpoints/, runs/ and data/ (tree-v1 keeps its outputs in checkpoints/tree-v1/ etc.). A README.md model
card lists the runs of this call, so pass the full list every time; files whose content did not
change are not sent again. The repo is created private unless --public. Use --dry-run to list the
files without uploading.

Loading on the other side (with this repo installed):

    from huggingface_hub import snapshot_download
    from rlvr.model import load_model
    root = snapshot_download("<repo>", allow_patterns=["pt-mid-cf0.00032-s0/*"])
    model, tok = load_model(f"{root}/pt-mid-cf0.00032-s0")
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

CKPT_REQUIRED = ["config.json", "model.safetensors", "tokenizer.json", "run_config.yaml"]
CKPT_OPTIONAL = ["generation_config.json"]
RUN_OPTIONAL = ["stats.json", "calib/gate.json", "calib/relabel_report.json", "calib/heatmap.png",
                "calib/calib_labels.parquet", "calib/pilot.json", "calib/pilot_candidates.jsonl"]
TASKS = {
    "chain": ("string function-composition task (chain-v1)",
              "tags: [rlvr, string-function-composition, from-scratch]"),
    "tree": ("tree-structured string-operator task (tree-v1: 12 unary, 8 binary and 4 ternary operators)",
             "tags: [rlvr, string-operator-trees, from-scratch]"),
}


def dirs(root: Path, task_dir: str) -> tuple[Path, Path, Path]:
    """checkpoints, runs and data folders of the task namespace."""
    return tuple(root / d / task_dir if task_dir else root / d for d in ("checkpoints", "runs", "data"))


def collect(run: str, root: Path, labels: bool, task_dir: str = "") -> list[tuple[str, Path]]:
    """(path_in_repo, local path) for one run; exits if the checkpoint is incomplete."""
    ckpts, runs, data = dirs(root, task_dir)
    ckpt = ckpts / run
    missing = [f for f in CKPT_REQUIRED if not (ckpt / f).is_file()]
    if missing:
        sys.exit(f"{ckpt}: missing {', '.join(missing)}")
    files = [(f"{run}/{f}", ckpt / f) for f in CKPT_REQUIRED + CKPT_OPTIONAL if (ckpt / f).is_file()]
    files += [(f"{run}/{f}", runs / run / f) for f in RUN_OPTIONAL if (runs / run / f).is_file()]
    if labels:
        lab = data / "labels" / f"{run}.parquet"
        if lab.is_file():
            files.append((f"{run}/labels.parquet", lab))
        else:
            print(f"--labels: no {lab}, skipped")
    return files


def model_card(repo: str, runs: list[str], root: Path, task_dir: str = "", note: str = "") -> str:
    rows, kinds = [], set()
    for run in runs:
        cfg = yaml.safe_load((dirs(root, task_dir)[0] / run / "run_config.yaml").read_text())
        kinds.add(cfg.get("task", {}).get("kind", "chain"))
        pt = cfg.get("pretrain", {})
        rows.append(f"| `{run}` | {pt.get('comp_frac')} | {pt.get('h_ratio')} | {pt.get('steps')} "
                    f"| {cfg.get('seed')} | `{str(cfg.get('git_commit', ''))[:7]}` |")
    example = next((r for r in runs if "-mid-" in r), runs[0])
    task, tags = TASKS[kinds.pop()] if len(kinds) == 1 else ("string tasks of the project", TASKS["chain"][1])
    return "\n".join([
        "---",
        "library_name: transformers",
        tags,
        "---",
        "",
        "# RLVR base models",
        "",
        f"Base models pretrained from scratch on the {task} of the RLVR",
        "data-difficulty project: LlamaForCausalLM, 12 layers, d_model 768, ~85M non-embedding",
        "parameters, character-level tokenizer. Settings, results and decisions are in the project",
        "repo's `docs/design.md`; each folder's `run_config.yaml` holds the full resolved config and",
        "the git commit it was trained at.",
        "",
        "| run | comp_frac | h_ratio | steps | seed | commit |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        *([note.strip(), ""] if note.strip() else []),
        "Per folder: the checkpoint (`config.json`, `model.safetensors`, `tokenizer.json`,",
        "`run_config.yaml`), `stats.json` (pretraining corpus actually consumed), `calib/` (gate",
        "check, relabel report, p heatmap, calib-set labels and, for a pilot run, `pilot.json` and the",
        "pilot candidates) and, if uploaded, `labels.parquet` (RL-pool pass-rate labels).",
        "",
        "The tokenizer is the project's own `CharTokenizer`, so load with the project code:",
        "",
        "```python",
        "from huggingface_hub import snapshot_download",
        "from rlvr.model import load_model",
        f'root = snapshot_download("{repo}", allow_patterns=["{example}/*"])',
        f'model, tok = load_model(f"{{root}}/{example}")',
        "```",
        "",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run names under checkpoints/, e.g. pt-mid-cf0.00032-s0")
    ap.add_argument("--repo", required=True, help="Hugging Face repo id, <user-or-org>/<name>")
    ap.add_argument("--public", action="store_true", help="create the repo public (default private)")
    ap.add_argument("--labels", action="store_true", help="also upload data/labels/<run>.parquet")
    ap.add_argument("--root", default=".", help="repo root holding checkpoints/, runs/, data/ (default .)")
    ap.add_argument("--task-dir", default="", help="task subfolder of checkpoints/, runs/, data/, e.g. tree-v1")
    ap.add_argument("--note-file", default=None, help="markdown paragraph(s) for the model card, after the run table")
    ap.add_argument("--message", default=None, help="commit message")
    ap.add_argument("--dry-run", action="store_true", help="list the files, upload nothing")
    args = ap.parse_args()

    root = Path(args.root)
    files = [f for run in args.runs for f in collect(run, root, args.labels, args.task_dir)]
    note = Path(args.note_file).read_text() if args.note_file else ""
    card = model_card(args.repo, args.runs, root, args.task_dir, note).encode()
    total = sum(p.stat().st_size for _, p in files) + len(card)
    for dst, src in files:
        print(f"{src.stat().st_size / 1e6:9.1f} MB  {dst}  <-  {src}")
    print(f"{len(card) / 1e6:9.3f} MB  README.md  (generated)")
    print(f"{total / 1e9:.2f} GB in {len(files) + 1} files -> {args.repo} "
          f"({'public' if args.public else 'private'})")
    if args.dry_run:
        return

    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    api.create_repo(args.repo, private=not args.public, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src)) for dst, src in files]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card))
    info = api.create_commit(args.repo, ops,
                             commit_message=args.message or f"Upload {', '.join(args.runs)}")
    print(info.commit_url)


if __name__ == "__main__":
    main()
