# 新数据形式代码修改提案：字符串计算树（tree-v1）

状态：提案，尚未实现。现有字符串链任务称为 chain-v1。

## 1. 要解决的问题

现有模型在深度 2–4 上学会了组合，但在深度 5 以上出现断崖。
strong 对低曝光函数组合和留出函数对也接近满分，说明当前曝光干预不足以
形成原设计需要的成功率匹配组。weak 也见过约 2.7 万道组合题，不能据此
认定模型仅凭原子题就学会了组合。

本次修改检验一个具体假设：让模型分别计算多个输入分支，再合并中间结果，
是否能提供更丰富、可控制的结构难度。二元运算不保证任务更难，也不保证
出现 p≈0.5；这两点由试点判断。完整函数式语言、递归、类型系统、三元运算
暂不纳入首版。

## 2. 核心决定

| 项目 | 首版决定 |
| --- | --- |
| 输入与输出 | a–z 字符串；同一道题的所有叶子长度相同，初始仍为 6–10 |
| 程序结构 | 有序树，支持一元和二元算子；无变量、共享节点或循环 |
| 叶子 | 每个叶子独立采样字符串；相同字面值允许偶然出现 |
| 计算顺序 | 子节点从左到右、后序执行；一个内部节点对应一步 |
| 输出长度 | 所有算子保持输入长度，二元输入必须等长 |
| 函数命名 | 无语义 F00… 编号；arity 写入元数据，括号语法体现参数数量 |
| 基础奖励 | 保留最终答案 exact match + 必须有 <eos> |
| 运行隔离 | 新 tokenizer、任务指纹、数据目录及模型名称；保留 chain-v1 |
| 模型初始化 | tree-v1 正式实验从随机权重训练，所有对照使用同一模型结构 |

三元算子在接口中允许表示，但生成器首版只接受 arity=1/2；不启用三元训练。

## 3. 算子定义

令 a=0,…,z=25，字符运算模 26；位置从 0 开始，字符串长度为 n。
首版先用 4 个一元、4 个二元算子验证流水线，而非立即扩展到大量函数。

| 算子标签 | 参数数 | 精确定义 | 定位 |
| --- | --- | --- | --- |
| reverse | 1 | 反转字符串 | 复用已有实现 |
| rotl1 | 1 | 循环左移 1 位 | 复用已有实现 |
| shift_p1 | 1 | 每个字符值加 1，模 26 | 复用已有实现 |
| shift_by_pos | 1 | 第 i 位加 i，模 26 | 复用已有实现 |
| zip_add | 2 | z[i]=(x[i]+y[i]) mod 26 | 简单合并基线；可能代数简化 |
| zip_sub | 2 | z[i]=(x[i]-y[i]) mod 26 | 有序、非交换，但仍可线性化 |
| gather | 2 | z[i]=x[y[i] mod n] | y 决定如何读取 x |
| gather_add | 2 | z[i]=(x[y[i] mod n]+y[i]) mod 26 | 同时依赖索引和值 |

例如 zip_add(abcdef,bcdefg)=bdfhjl；gather(abcdef,fedcba)=fedcba。
运行时 Fxx 映射由固定种子产生，上表标签仅用于代码和审计。

初始 H={reverse, shift_by_pos, zip_add, gather}，其余为 L；两组各含
2 个一元、2 个二元。H/L 代表组合曝光，不代表算子本身难度。
这个划分只是试点起点：不能从 H/L 内部差异直接推出曝光的因果效应，
必须保留相同算子表的 h_ratio=0.5 对照，必要时交换 H/L 标签重复试点。

## 4. 一条题长什么样

以下仅为格式示例，临时设 F00=reverse、F01=shift_p1、F04=zip_add：

```text
Q:F04(F00(abcdef),F01(bcdefg))=
F00(abcdef)=fedcba;F01(bcdefg)=cdefgh;F04(fedcba,cdefgh)=hhhhhh;
A:hhhhhh<eos>
```

CoT 每一步只写已算出的字符串参数，不展开整棵子树；避免反复打印子表达式。
首版不引入寄存器名，减少同时改变数据语法带来的影响。
相同表达式的多个出现位置仍分别求值、分别计步，不做公共子表达式消除。

JSONL 保存可重算的树和审计信息，不能只保存扁平 funcs：

```json
{
  "schema_version": "tree-v1",
  "task_fingerprint": "<hash-of-semantics-and-display-mapping>",
  "pid": "<hash-of-task-fingerprint-and-ordered-tree>",
  "expr": {
    "op": "F04",
    "args": [
      {"op": "F00", "args": [{"value": "abcdef"}]},
      {"op": "F01", "args": [{"value": "bcdefg"}]}
    ]
  },
  "depth": 2,
  "n_ops": 3,
  "n_binary": 1,
  "n_leaves": 2,
  "max_live_values": 2,
  "shape_id": "B(U(L),U(L))",
  "funcs": ["F00", "F01", "F04"],
  "group": "mixed",
  "str_len": 6,
  "steps": [
    {"path": "0", "op": "F00", "args": ["abcdef"], "value": "fedcba"},
    {"path": "1", "op": "F01", "args": ["bcdefg"], "value": "cdefgh"},
    {"path": "", "op": "F04", "args": ["fedcba", "cdefgh"], "value": "hhhhhh"}
  ],
  "answer": "hhhhhh"
}
```

示例 group 随实际 Fxx 映射和 H/L 配置重算；以上临时映射下为 mixed。
`from_dict` 必须根据 expr 和冻结任务定义重算答案、步骤、结构统计，并校验
输入元数据；不把文件里缓存的 answer 当作独立真值。

## 5. 分开记录难度，避免把“更多步骤”误认为“树更难”

| 字段 | 定义 |
| --- | --- |
| depth | 叶子深度 0；内部节点为 1+max(孩子深度) |
| n_ops | 内部节点总数，即标准答案的计算步骤数 |
| n_binary | 二元节点数；binary_frac=n_binary/n_ops |
| n_leaves | 叶子出现次数；一元/二元树应满足 n_leaves=n_binary+1 |
| max_live_values | 固定左到右后序执行时，栈上待使用结果的最大数量 |
| shape_id | 只保留 arity 与括号结构，不含函数名或输入值 |
| prompt/target_tokens | 用实际 tokenizer 计算，作为长度协变量 |

max_live_values 的递推为 live(leaf)=1；一个有 k 个孩子的节点取
max((j-1)+live(child_j))，j 从 1 开始。它衡量约定执行顺序下的栈需求，
不声称等同于模型真实内部记忆。

生成器显式指定可行的 (n_ops, depth, n_binary)，从满足限制的形状中抽样，
再填算子和叶子；首版 n_ops<=8，可枚举有序形状并缓存。
不先递归随机长树再截断，以免形状分布无法解释。拒绝不可行组合并说明原因，
每个采样单元有最大尝试次数，不能无限循环。无需追求覆盖完整笛卡尔积。

比较分三类报告：

1. 同 n_ops、同 n_binary、同串长，比较不同深度/形状：检查依赖结构的影响。
2. 同 n_ops、同串长，比较不同 n_binary：检查分支增加的整体影响；叶子数、
   输入长度会一起变化，需要报告，不能声称单独识别了 arity 的作用。
3. 同一分布内固定预算比较不同 RL 数据配方：同时报告 rollout 数、生成 token
   数、优化步数和耗时。等 rollout 不代表等计算量。

跨 chain-v1 与 tree-v1 的成绩只能作为不同基准的描述性比较：算子表、输入数
和 tokenizer 都变化，不能把性能差全部归因于二元运算。

## 6. 生成与质量审计

每次生成先选原子/组合成分，再选结构单元、算子曝光组、串长与叶子。
原子题定义为 n_ops=1，包括一元题和直接接收两个叶子的二元题。
组合题定义为 n_ops>=2，不能继续用 depth>1 判断“有几个操作”。
P4 继续按监督 token 控制组合占比；P6 仍定义为组合题 H-only 概率，
其他题至少含一个 L。depth=1 原子集必须逐个算子测试，二元算子不能缺席。

质量报告包括：

- 按算子/arity、结构单元、H/L、输入长度统计样本和监督 token 数。
- 识别明确的局部简化：reverse(reverse(x))、线性位移合并等。
  只剔除已证明的静态规则，原子题不剔除；不能声称筛出了所有可简化程序。
- 在独立审计样本上逐个扰动二元参数，测量输出发生变化的比例，发现
  长期被忽略的参数。首轮只记录，不根据单样本扰动结果筛题，避免人为改变分布。
- 记录根答案等于叶子/子节点值的比例、输出字符多样性、重复 pid、拒绝原因。
  偶然相等不自动等于捷径，也不一律删除。
- 保存完整原始回答并报告实际输出步骤数、首个错误节点、提前输出 A 的比例。
  这些是诊断指标，最终奖励仍按答案算。

## 7. 数据隔离和组合留出

新数据放在 data/tree-v1/，其下保留 meta、eval/v1、rl_pool、datasets、labels。
新模型使用 pt-tree-* 名称。旧数据和 checkpoint 继续可加载。

pid 使用规范 JSON 序列化的有序树加任务指纹计算；不排序孩子，即使某个
算子可交换也不交换其参数。指纹包含算子语义版本、参数、Fxx 映射、字母表、
H/L 定义、格式与 tokenizer 版本。随机种子、采样配比、训练量不进入题目身份。
语义等价的不同树可能有不同 pid；首版保证语法实例隔离，不保证语义等价类隔离。

保留 pid 哈希的 90%/5%/5% 划分。组合留出改为真实树边的
`(parent_op, child_slot, child_op)`：

- 只检查孩子本身是算子的边；叶子不形成算子对。
- 每一条边都检查，不能检查后序 funcs 的相邻项，后者可能来自兄弟节点。
- held-out 边不进入预训练、RL 池和 E1–E3/probe/calib；E4 至少含一条。
- 按父/子 arity 与 H/L 分层选择留出边；检查剩余连接可生成目标形状。
- 先构造候选树并保留指纹、留出列表和排除规则的 manifest；检查所有产物。

树形状留出可作为后续 E5，首版不同时启用，以便解释 E4。

## 8. 代码修改清单

| 文件 | 拟修改内容 |
| --- | --- |
| 新 rlvr/tree_ops.py | OperatorSpec(arity, semantic_version, factory) 和多元注册表；一元适配现有 ATOMS |
| 新 rlvr/tree_task.py | Leaf/Call、TreeProblem、类型与长度检查、解释器、后序轨迹、格式化和解析 |
| 新 rlvr/tree_sampling.py | 可行树形状枚举、受控采样、静态简化检查和独立质量审计 |
| rlvr/config.py | task.kind 工厂选择 chain/tree Task 与对应 Splits；消除其他脚本直接 Task(cfg) 的调用 |
| rlvr/task.py | 保留旧 Task/Problem；抽取共享协议，不用可选 x/expr 拼成一个混杂对象 |
| rlvr/splits.py | 保留链规则；新增 TreeSplits 或分离 tree_splits.py，实现带参数位置的树边留出 |
| rlvr/sets.py | 支持结构单元及按算子原子评测；不能继续把 depth 当作操作数 |
| rlvr/stream.py | 调用任务采样接口；P4 按 token；统计 n_ops/arity/形状；树噪声按节点重算所有祖先 |
| rlvr/tokenizer.py | 通过任务格式选择词表；tree-v1 增加逗号，48→49 tokens；chain-v1 词表不变 |
| rlvr/rewards.py | binary 保持通用；partial 依节点轨迹比较、分母 n_ops，不复用 p.depth |
| rlvr/io.py | 版本化反序列化和元数据验证；保存 expr 与结构特征 |
| rlvr/evaluate.py | 按 depth/n_ops/arity/shape 聚合；E1 按根算子；可选保存原始回答和错误轨迹 |
| rlvr/model.py | 模型/tokenizer 同步保存、加载并校验任务指纹；禁止误配旧词表 |
| rlvr/params.py | 注册新增配置、验证结构单元、预算与版本组合 |
| scripts/make_eval_sets.py、make_rl_pool.py | 调用任务工厂；使用新元数据和结构分层，不假设一元 funcs/x |
| scripts/preview_stream.py、report_data.py | 增加树示例、结构分布、长度和依赖审计 |
| scripts/pretrain.py | 日志按 n_ops/arity 扩展；数据与模型指纹校验；检查真实序列长度 |
| scripts/calibrate.py | 不直接实例化旧 Task；标注保留结构列，门槛按可配置结构单元检查 |
| scripts/build_dataset.py | SPEC/CELL_KEYS 加结构字段；匹配 n_ops/str_len 等；保留 p-based 采样 |
| scripts/sweep.py | 试点扫描预算与配方，保留统一选择规则及所有候选结果 |
| 新 configs/tasks/tree_v1.yaml 等 | 冻结新任务、结构网格、评测与试点配方；与旧配置分离 |

共享任务协议至少提供 sample_problem、from_dict、format_prompt、format_target、
parse_answer、reward、describe，以及原子枚举和结构统计接口。
树生成所需的结构约束使用单独的 SamplingSpec，旧 depth API 由链适配器接收。
steps 是节点轨迹；funcs 只作曝光统计，不能再充当完整程序或树边列表。

## 9. 配置与原设计变量的对应

以下为拟新增/变更键，并非当前代码已接受的配置。

| 键 | 作用／与 design 的关系 |
| --- | --- |
| task.kind: tree | 选择新任务，旧默认仍为 chain |
| task.version: tree-v1 | 格式与语义版本 |
| task.operators | 算子规格、参数、arity、语义版本；与旧 task.functions 区分 |
| task.h_group | P6；必须在每种 arity 内保持两组覆盖 |
| pretrain.structure_cells | P5 扩展：n_ops/depth/n_binary/frac 的可行单元 |
| pretrain.h_only_depths | tree 首版设 []，避免额外绑定深度和 H；作为 P5b 显式改动 |
| pretrain.comp_frac | P4，仍是组合监督 token 占比 |
| pretrain.h_ratio | P6，仍是非强制单元的 H-only 题概率 |
| pretrain.steps、save_at | P9；沿用已有 4750 步半预算起点并保留中间 checkpoint |
| task.str_len、alphabet、cot | P7/P8，沿用默认但重新冻结 task 指纹 |
| data 的结构 cells | D2 扩展；D6 增加输入/输出 token 与操作数协变量 |
| splits.heldout_edges | 新的有向、带 child_slot 的留出边列表 |
| gate.spread_cells、matched_cells | 替代仅按深度判定；指定真正要用于实验的结构格 |

建议首版组合 n_ops=2–4；RL 候选 n_ops=2–6；更大操作数评测为 7–8。
每个范围内选可行结构格，不能把“操作数外推”和“深度外推”混为一类。
E3 分别报告训练未覆盖的 depth、n_ops，以及两者都未覆盖的子集。
精确结构配比先用预览确定，不硬填无法生成的 depth/branch 组合。

P1/P2/P2b、P10/P11、D1/D3/D4/D7–D10、R1–R13 的研究含义保留。
P3 的原子覆盖扩展到所有 arity。D5 仍不引入评测泄漏干预；M/Y 指标增加结构切片。
GRPO 尚未实现，本提案不把“新增树任务”与“完成 GRPO”合并为一个改动。

## 10. 预算与评测门槛

先生成预览，实际测量最长 prompt/target 和 padding 后长度，再确定
model.max_len、eval/rl.max_new_tokens 与 position offset；不能照搬旧 512/256。
对声明的全部可行结构格验证长度上界，禁止静默截断和按长度丢掉最难样本。
模型架构试点尽量沿用现有 85M；如必须改上下文，所有 tree 对照一起改。

固定监督 token 预算能比较优化投入，但多输入题 prompt 更长，计算成本仍不同；
因此同时记录总非 padding token、监督 token、padding、吞吐和 GPU 时间。

门槛分两类：

- 功能门槛：每个一元/二元原子算子贪心准确率≥98%；无跨池 pid 泄漏、
  无训练留出边泄漏、无截断、无版本混用。
- 研究可行性：预先指定浅/深结构格，两个格中 p∈[0.4,0.6] 各至少 300 题，
  同时匹配可行的协变量；报告匹配前后分布。匹配失败不隐去、不按 E3 分数选格。

不再把总体训练 loss<1e-4 作为独立否决条件；它容易被大量已学会原子题主导。
改为报告分算子、分结构的训练 loss、独立集准确率与 p 分布，是否饱和用校准判断。
K=32 的分箱有噪声；对最终匹配候选用独立新增采样复核，例如额外 96 次，
保留初始与复核结果，报告区间及 p 分布差异，不把 p_hat 当成真实 p。

## 11. 实施顺序和验收

1. **实现语义和文件格式。** 完成算子、AST、解释器、tokenizer、指纹。
   验证已知答案、字母回绕、参数顺序、非法 arity/异长输入、JSON 往返和旧任务回归。
2. **实现生成与隔离。** 小样本遍历可行结构格；独立递归解释器与后序解释器交叉校验；
   验证结构统计、父子边留出、跨池隔离、采样失败终止和长度上界。
3. **接通流式训练、评测与标注。** 在 CPU 验证格式与标签屏蔽；可用 GPU 上做短训练，
   确认模型保存/加载、生成、逐算子 E1、原始回答诊断和校准全链路可用。
4. **做试点后决定正式配方。** 同一 tree 算子表，比较 comp_frac=0 原子对照，
   以及 0.001/0.01/0.1；至少在中间配方对比 h_ratio=0.5/0.9。
   先用短预算排错，再用共同 4750 步预算验证候选，按统一 checkpoint 规则选模型。
   不以某个候选更低的 E3 分数作为“更难、更好”的证据。

试点通过的证据是：原子技能可学、存在足够的单题中间成功率、结构格可匹配，
而且原始回答能解释主要错误来源。若仍是全会/全不会，先区分算子学不会、
执行进度错误、长度限制和结构曝光不足，再决定增加三元运算或更完整语言。

最终交付为新任务实现、冻结元数据、可复现实验配置、数据质量报告、测试、
小规模试点结果及与 chain-v1 的描述性对照；本文件仅定义实施方案。
