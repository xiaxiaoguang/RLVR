# 两个pretrained base的配对RL已准备并通过真实训练smoke

当前任务 `20261124-231216-start-paired-pretrained-rl-experiments`；RESULT: done仅指执行包/预检/smoke交付。主实验尚未启动，idea未关闭，未宣称PI批准结题。

## 已完成与授权冲突处理

PI最新2026-11-24 23:07:49将总预算升至100 GPU小时（360000s），允许即使最终未完成也先开始两模型原组。旧HANDOFF的48h、第二模型缺失和整组必须保守fit/no-half-group停止条件被替代；原件保存在handoff-before.md，继承ledger保存在inherited-accounting.json。旧comparison-prep/ready、task15/16失败/blocked证据及历史checker都未改。没有更换研究模型、数据、reward、评估定义或D/E门槛；没有API/下载/新预训练/commit/ref变更/push。

两个340128712B权重都通过PI指定SHA256、CPU safetensors逐tensor有限性/111个state key形状和值精确检查、原load_model与native preflight。每模型85029120参数；12层/768hidden/12heads/2048ff/49vocab/1024context/fp32，task fingerprint一致。cf.006的4文件也与既有缓存官方tree的LFS哈希/git blob id一致；此为本地交付+历史元数据相符，**没有重新独立验证远端revision来源，也没有第二次delivery下载receipt**。证据：inputs.json、prepare.log、environment.json。没有读取凭据或联网。

一次CPU准备失败是旧prep YAML引用P/configs/rl/1080ti.yaml不存在；新prepare.py改为读取权威C/configs/rl/1080ti.yaml后第二次成功。保留prepare-attempt1.log；没有GPU失败/重试，也没有改生产源码。

## 两次实际训练smoke（不是完整科学baseline）

同一scripts/grpo.py:main/worker.py/supervise.py路径；每base seed0、B64/G8、1optimizer step、512rollouts；同一固定第一64 PIDs，未挑选有reward批次。tiny eval仅每组前2行，共10题，step0/1用于连线检查。原采样T1/768、loss/reward/AdamW和KL诊断不变。

| base | mean_reward | signal_frac | grad_norm | rollout sec | update sec |
|---|---|---|---|---|---|
| pt-tree-mix-cf0.0035-s0 | 0.513671875 | 0.453125 | 0.05404936149716377 | 107.624025 | 15.310348 |
| pt-tree-mix-cf0.006-s0 | 0.775390625 | 0.28125 | 0.05428840592503548 | 98.359355 | 15.165057 |

来源：smoke/output/*/metrics.jsonl及/data中的rollouts.jsonl；验收重算reward/signal，比较实际PID/order。两者都有finite gradients/AdamW state/parameters、policy改变、reference不变，保存模型CPU重新加载参数digest与训练后policy精确一致。各base的固定前2个E2题，在**新保存policy**上B1/B2原始token/text/EOS/reward/step_score/step_frac精确相同；tiny全部组B2原evaluator的逐题输出由CPU独立重算all行score。证据：smoke/<model>/optimizer.jsonl、training-evidence.json、saved-policy-batch1/2.json、eval-step0/1.json、imported-source.json。这不证明full-set数值等价/全长容量/稳定加速；非新pass@k研究。

两个GPU child含清理墙时分别 165.751867s / 156.492895s，各小于420s监督限时+有界清理、也小于15min。整个worker新增GPU收费323s（完整supervisor startup→cleanup向上取整，包含间隔/监测；上限900s）。历史13257s + 新323s = **13580s**；剩 **346420s = 96.227778h**。budget/accounting每秒持久化elapsed并在结束前保留最坏reservation，最终实际收费在accounting.json与accounting-events.jsonl；两次pre-spawn/child内idle检查、UUID、owned process group、GPU查询及清理在smoke-status.json/smoke-watchdog.jsonl。

## 主组：已可启动，worker未启动

唯一host job：

`JOB: rlvr-paired-rl | script=runs/idea-20261121-01-paired-rl/job.sh | max_hours=96.227`

cf.0035→cf.006顺序，各从原pretrain新开始、seed0/500step/B64/G8（各256000rollouts），fp32 Pascal/T1/768；同40,000候选四cell、同500x64固定PID顺序、static shuffle、不初始化任何p标签。两full resolved configs的差仅rl.base/pretrain.level/pretrain.comp_frac。原AdamW lr1e-5/10warmup/token mean/group std归一化/1update/clip1，保留全部reward与诊断设置。gen_batch_size1/micro_batch_tokens1024/compile=false是既有Pascal执行设置。

主eval使用全部E1–E4/probe=1200/1800/900/800/575行；E1–E4每50step、probe每25step，step0/final都包括；eval_batch2，不改greedy/EOS/768和任何group/depth/step/structure指标。原pass_k_samples=0和pass_k列表保持，因此本组未生成sampled pass@k。405<500/go=false的D/E不解锁。本组是p-independent固定数据比较，不等同均匀depth1–5的A组，更不是A–G/135runs。

host只用物理index1→完整UUID，保留兼容scheduler mask并持有协作锁；不空闲/外部冲突则退出，不动卡0/外部进程。永久claim拒绝重复启动和覆盖；内部monotonic **全序列累计346420s**截止，预留30s关停，独立GNU timeout约束每模型，所有startup/eval/save/failure/cleanup计入。max_hours=96.227仅lab日历notice，取min(1.5×历史示意直接估计89.6541943886414h,剩余授权墙时)后向下取三位小数；不是装得下保证，也不代替内部100h限额。不再等待保守fit许可。超过期限/PI停止时清理owned group、记账并保留未开始/未完成模型；TERM/INT/HUP测试通过。SIGKILL/断电无收尾能力，最坏reservation和claim仍保留，后续需manager核对，不自行恢复预算。

/data保存每50step及final的模型，以及rollouts；项目内symlink保留路径。checkpoint只有model/tokenizer/config，**不含optimizer/scheduler/RNG/sampler，不能精确resume**。eval后保存遵从原trainer顺序，中断于某次eval可能只有此前完成的checkpoint。中断pair必须报告缺失阶段，不拿一个模型的final对另一个中间step作完整比较。

## 对齐PI问题：pretrain差异、RL后差异、结论

两base的主要配方差是composition supervised-token share .0035→.006，均seed0/9500step、depth mix .30/.35/.35、h_ratio .5。历史统计：cf.0035 compositions 38410 examples/6427346 supervised tokens；cf.006 65675 / 10989184。因此并非训练总步数变长，而是组合曝光比例不同；总tokens略有batch overfill差异，见pretrain-inventory.json。

历史greedy结果（旧平台/原日志，仅背景，**不是本次fresh fp32/B2 step0**）：

| model | E1 | E2 | E3 | E4 | probe |
|---|---|---|---|---|---|
| pt-tree-mix-cf0.0035-s0 | 1.0 | 0.3438888888888889 | 0.0022222222222222222 | 0.2725 | 0.27652173913043476 |
| pt-tree-mix-cf0.006-s0 | 1.0 | 0.5461111111111111 | 0.006666666666666667 | 0.46125 | 0.4 |

来源：comparison-prep/pretrain-inventory.json及results/pretrain_v2/models/tree-v1/*/eval.jsonl；历史cf.006在组合/更深/新组合上的起点更强，但不能从历史跨平台值或10题smoke推断新RL收益。主job将在相同冻结test上重新得到两个fresh step0，再以各自final−step0、同step的base差、两模型RL增量差报告greedy/step_score/step_frac，以及E2 depth5/E3 depth6–8和structure分组。inspect_results.py已支持只读提取共同评估step，缺失结果不填充为完成。

**RL后的科学差异仍未知，不能给因果/显著性结论。** 此次完成的是实际训练路径与运行控制验证，不是科学组完成。只有一seed且固定cf.0035先运行，不能把顺序/共享pretrain数据运气当独立复现，也不能检验原H1/H2/H3或拟合可信ceiling。若预算/PI中断，后续分析必须注明不完整pair。最终分析/closure review由后续worker和manager组织，只有PI批准idea关闭。

## 验证、文件与接手

CPU acceptance已通过：check.log、verification.json；control-tests.json包含真实CPU child group timeout/外部冲突注入/信号清理、重复claim与mask测试。主JOB未领取（无main.claimed/main-status/主模型输出）。代码/输入/lock/源码git身份见inputs.json与delivery.json，集成源码备份integrated-source.tar.gz。所有新脚本/配置/小证据/中文指南在本目录；没有改旧tracked文件。git-before.json、git-final.json记录原checkout与集成checkout状态及refs；本目录通常被gitignore，交付时需要保留整个目录及/data symlink目标。

环境复用RLVR-rl1080，lock `/work/users/lab_admin/envs/RLVR-rl1080/uv.lock` SHA256 da9cbed8ab7ca6484c55505bdb4b4d7700efcc328330f823246870bc87f292b9。精确命令/inputs sizes/hashes/seed/GPU/outputs在[REPRO.md](REPRO.md)，日常用法在[USER_GUIDE_ZH.md](USER_GUIDE_ZH.md)，短接手记录在[HANDOFF.md](HANDOFF.md)，共享HANDOFF同样更新。磁盘前后与新增/data路径大小见storage-before-smoke.txt、storage-after.txt、storage-sizes.txt；仅新建本项目/data/runs/idea-20261121-01-paired-rl，既有assets只读。

```bash
cd /home/lab_admin/workspace/RLVR
bash runs/idea-20261121-01-paired-rl/check.sh
cat runs/idea-20261121-01-paired-rl/launch-plan.json
# 以下由lab在worker退出后执行一次：
bash runs/idea-20261121-01-paired-rl/job.sh
# 随时只读查看已有科学结果：
source runs/idea-20261121-01-paired-rl/env.sh
CUDA_VISIBLE_DEVICES='' uv run --project /work/users/lab_admin/envs/RLVR-rl1080 --frozen python "$PAIRED_RUN/inspect_results.py"
```
