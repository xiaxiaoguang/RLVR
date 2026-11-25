# 两个pretrained base的配对RL：操作指南

本目录是新执行包，不修改旧comparison-prep的ready=false或失败证据。PI于2026-11-24 23:07:49批准100 GPU小时且接受未完成的一组；取代旧48h、第二base缺失、保守估计必须装下整组的停止条件。原科研定义、D/E门槛未变。

从项目根目录先读本目录的REPORT_ZH.md、REPRO.md、HANDOFF.md；CPU验收：

```bash
cd /home/lab_admin/workspace/RLVR
bash runs/idea-20261121-01-paired-rl/check.sh
cat runs/idea-20261121-01-paired-rl/launch-plan.json
cat runs/idea-20261121-01-paired-rl/accounting.json
```

验收不初始化CUDA、不训练、不启动JOB、不写结果；读取模型hash、CPU预检证据、两次真实训练smoke、实际固定PID序列和原始eval输出。原历史checker不改。正式JOB尚未领取时应通过；领取后的再次launch会被永久claim拒绝，不要删除claim或覆盖已有结果。

由lab在worker退出后启动唯一host job（不要同时执行smoke.sh；smoke已经执行过）：

```bash
cd /home/lab_admin/workspace/RLVR
bash runs/idea-20261121-01-paired-rl/job.sh
```

job.sh自设uv/缓存/offline/绝对代码路径，从集成worktree运行。先只读验收，接着永久claim与GPU锁；保留传入scheduler mask，仅允许物理index1的完整UUID。发现外部进程/不兼容mask/卡不空闲会退出，不会用卡0、抢卡或扩展mask。新host启动时第一模型cf.0035、然后cf.006，每个均从原pretrain base开始，**不从smoke更新后的权重开始**。

两模型各seed0、500step、B64/G8、fp32、T1、768tokens；同一个40,000四cell候选集、固定无放回shuffle PID顺序；不使用任何初始p标签。这不是均匀depth1–5的A组，更不是A–G/135runs。E1/E2/E3/E4分别1200/1800/900/800行，每50step；probe575行，每25step；均包含step0和final。eval batch2只改调度，不改每序列greedy/EOS/768限制；保存depth/structure/step指标和逐题错误。原pass_k_samples=0不变，因此本组没有新sampled pass@k结论。

观察状态与比较已有共同评估点（CPU、只读，允许在job运行中使用）：

```bash
cd /home/lab_admin/workspace/RLVR
cat runs/idea-20261121-01-paired-rl/main-status.json
cat runs/idea-20261121-01-paired-rl/main.claimed
cat runs/idea-20261121-01-paired-rl/accounting.json
tail -n 30 runs/idea-20261121-01-paired-rl/rlvr-paired-rl.job.log
source runs/idea-20261121-01-paired-rl/env.sh
CUDA_VISIBLE_DEVICES='' uv run --project /work/users/lab_admin/envs/RLVR-rl1080 --frozen python "$PAIRED_RUN/inspect_results.py"
```

main-status.json在首次launch前不存在；缺失不是科学完成。main/<model>/paths.json指向metrics/eval/summary目录。inspect_results.py分别展示旧平台pretrain历史值、fresh step0、最新eval，以及**共同step**的cf.006-cf.0035与各自RL增量/增量差；缺失阶段保留缺失，运行中JSONL半行不作为结果。完整500step仍需summary.json、最终checkpoint和后续独立审查确认；只见metrics step500不等于最终eval完成。

如PI要提前停止，先cat main.claimed，核对其中pid仍是本目录supervise.py --mode main的进程，再 `kill -TERM <pid>`。TERM/INT/HUP触发整个模型进程组清理和持久化账本；独立GNU timeout与内部截止共同约束剩余额度。SIGKILL/断电无法执行Python收尾，此时保留最坏额度reservation与claim，**不要自行重启或降低账**，交给manager核对GPU进程及最终用时。外部进程只触发退出，不会被清理器杀掉。

每50step及final保存模型/tokenizer/config到/data，项目内ckpt和rollouts为symlink。trainer未保存optimizer/scheduler/RNG/sampler状态，**不能精确resume**；中断保留的模型可以供后续评估，新的训练方向/预算由PI决定。若预算到达或PI停止导致cf.006未开始/只完成部分，按main-status明确报告，不用单边final代替配对结论。

一次seed、固定顺序以及共享预训练seed不足以识别一般规律或显著性；后续worker需要按原要求报告两base的fresh起点、RL后值、delta、共同step差、E2深度5/E3 step_score/step_frac和全部depth/structure分组。D/E405<500、go=false仍原样保留，idea只能由PI批准关闭。
