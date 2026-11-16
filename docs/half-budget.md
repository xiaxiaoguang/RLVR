# Half-budget pretraining experiment

Keep the string task unchanged while checking whether less pretraining leaves
useful composition headroom. The stream has no epochs: halve optimizer steps
from 9,500 to 4,750, retaining the supervised-token budget per step.

## First: calibrate existing halfway checkpoints

No retraining is needed for this diagnostic. Run in the `rlvr` conda environment
on a machine with a working CUDA driver:

```bash
python scripts/calibrate.py runs/pt-weak-s0/ckpt/step-4750 --skip-pool
python scripts/calibrate.py runs/pt-strong-s0/ckpt/step-4750 --skip-pool
```

Outputs go to `runs/pt-{weak,strong}-s0-step-4750/calib/`, preserving final-model
calibration results. Compare per-function E1 accuracy, the per-problem success
distribution by depth/group, and the d2-L versus d4-H cells. A depth's mean near
0.5 alone does not establish that individual problems have intermediate p.
If promising, rerun without `--skip-pool` to measure matched-cell counts.
Calibration failure exits with status 1; this is a reported gate result.

## Fresh half-budget training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 scripts/pretrain.py --config configs/pretrain/half/weak.yaml
```

Replace `weak.yaml` with `mid.yaml`, `strong.yaml`, or `midbal.yaml` as needed.
These configurations inherit the corresponding original data recipe and use
distinct `*-half` run names, 4,750 steps and a 30-minute benchmark target.
The pilot and full-budget configurations are unchanged.

The existing step-4750 checkpoints use the original 9,500-step cosine schedule.
Fresh half-budget training compresses that schedule into 4,750 steps, so these
are related but different experiments. Compare models at a common budget for
P4/P6 comparisons; treat checkpoint age as P9 when varying it.
