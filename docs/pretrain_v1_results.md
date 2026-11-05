# v1 预训练结果记录

按轮记录 v1 基座模型的训练结果和 comp_frac 的决定过程。完整的作业日志在
`/pscratch/sd/y/yupengsu/rlvr/runs/NERSC_LOG.md`（scratch，不在 git 里）。

## 设置

- 模型 85.0M 参数；9500 步，每步约 192k 监督 token（共 1.84B），4×A100 约 70 分钟（含启动和评估）。
- seed 0，pos_offset_max 128，h_ratio 0.9（midbal 为 0.5）。其余参数取 `rlvr/params.py` 的默认值。
- 校准：calib 集每格 200 题，T=1.0，k=32。表中数字是各深度的 mean p̂。

## 试点扫描（2026-11-04）

`scripts/sweep.py`：400 步，约为完整训练 token 量的 4%。calib 每格 50 题，k=16，pos_offset_max=128。

| comp_frac | d2 | d3 | d4 | d5 |
|---|---|---|---|---|
| 0.0003 | 0 | 0 | 0 | 0 |
| 0.001 | 0 | 0 | 0 | 0 |
| 0.003 | .007 | 0 | 0 | 0 |
| 0.01 | .112 | .003 | 0 | 0 |
| 0.03 | .461 | .096 | .002 | .001 |
| 0.1 | .813 | .516 | .103 | .010 |

当时按「试点值 ÷ 25（token 量相差 25 倍）」外推，定了 weak = 0.001、strong = 0.01。

## 第 1 轮完整训练（2026-11-05，commit 456e41d）

| 运行 | comp_frac | 组合样本数 | d2 | d3 | d4 | d5 | d2-L | d4-H | E1 | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| pt-weak-s0 | 0.001 | 26,723 | .951 | **.871** | .681 | .017 | .914 | .846 | 1.000 | PASS，警告 target、spread |
| pt-strong-s0 | 0.01 | 269,092 | .999 | .987 | **.973** | .021 | .998 | .991 | 1.000 | PASS，警告 target、spread、final_loss |

两个模型都远超设计目标：weak 目标 d3 为 0.05–0.2，strong 目标 d4 为 0.4–0.6。
gate 判为 PASS，是因为 weak/strong 的硬检查只有 depth1_greedy。
两个模型都只做了 calib 热力图，RL 池标注中途停掉了，所以没有 `data/labels/` 文件。

训练中的 greedy 准确率（weak，E2 集）显示，组合能力是在训练后半段才涨起来的：

| step | 2000 | 4000 | 6000 | 8000 | 9500 |
|---|---|---|---|---|---|
| d3-H | .04 | .21 | .77 | .93 | .99 |
| d3-L | .00 | .02 | .42 | .75 | .83 |
| d4-H | .00 | .01 | .37 | .83 | .92 |
| d4-L | .00 | .01 | .14 | .39 | .69 |

### 结论

- 试点的外推把需要的 comp_frac 高估了约 10 倍。完整训练里，原子函数学透之后，组合只需要很少的样本：
  weak 只见过 149 个 d3-L 样本，d3-L 就到了 .81；只见过 1,349 个 d4-H 样本，d4-H 就到了 .85。
  而 d4-L 一个样本都没见过，也到了 .59。
- 训练过的深度（d2–d4）和没训练过的深度（d5 及以上）之间是断崖：两个模型的 d5 都只有约 0.02。
- 对 comp_frac 的粗略估计，一是对 0.001 和 0.01 两点做 logit 线性拟合，二是读 weak 的训练轨迹。两种方法给出一个范围：
  strong 约 5e-4–7e-4，mid 约 1.3e-4–5e-4，weak 约 2e-5–4e-4。
- 下一步的方案（直接训练最终模型，还是先做一轮完整训练长度的扫描）待定。

### 产出位置（scratch）

- `runs/pt-{weak,strong}-s0/`：metrics.jsonl、eval.jsonl、stats.json、calib/（calib_labels.parquet、heatmap.png、中间版 gate.json）。
- `checkpoints/pt-{weak,strong}-s0/`：最终模型。`runs/pt-*/ckpt/step-{2375,4750}/`：训练中途的 checkpoint。

## 基础设施

- Perlmutter 的计算节点经 DVS 访问 CFS 和 HOME。2026-11-05，放在 CFS 上的 conda 环境卡在 DVS 上，
  两个作业因此被判 NODE_FAIL。现在环境改为在 scratch 上从锁文件重建（见 `scripts/slurm/env.sh`）。
- interactive 节点：用 `scripts/slurm/model_pipeline.sh` 在一个节点上依次做 4 卡训练和 4 卡校准。
  4 卡标注速度约 730 seq/s；完整校准（calib、RL 池和第二次标注）预计每个模型约 35–40 分钟。
