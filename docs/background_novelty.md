# Background research for the project

Written 2026-11-25, revised 2026-11-27. An AI-written report. It describes the published work that is closest to the project "What Makes RL Training Data Good?". It also says how the project differs from that work. The related-work table of the project is in `docs/design.md`, section "Related work".

## How to read this report

- Opus agents read 17 papers in full and 16 more in targeted parts. A second agent then compared each notes file with its paper and corrected it. Each statement about a paper below is the statement of its authors, with the location in the paper.
- A statement marked "our reading" is an inference by an agent. The paper does not say it. Most such readings come from a figure.
- A statement marked "abstract only" comes from the arXiv abstract. Nobody read that paper.
- The search ran on 2026-11-25. It was not exhaustive. The section "Limits of this research" says what it missed.
- The notes files, with quotes and locations for each paper, are not in this repository. A team member keeps them in a private coursework repository, in the task folder `2026-11-25T0651Z`.

## In short

1. No paper that the agents read compares RL training sets that have the same measured pass rate and a different structural difficulty. The pass rate of a problem is the share of sampled answers that are correct.
2. One paper matches training pools on initial pass rate. "Where Groups Split" (arXiv 2610.25325) splits each matched pool by a score from a short probe training run. The two halves learn differently (§5.3, Tab. 3). Pass rate alone therefore did not decide the result in that study.
3. Three papers train small transformers from scratch and vary the depth of the RL data. None of the three measures the pass rate of a training problem. Their depth results therefore mix depth with pass rate.
4. The signal formula 1 − p^G − (1−p)^G is in print in at least three papers of 2026.
5. A registered follow-up with three seeds did not reproduce, for GRPO, the result of "Learning from Less" that 100 mixed problems match 500 easy ones.

## Pass rate and structural difficulty are not yet separated

Earlier work selects RL training problems by pass rate, or varies their structural difficulty, but not the two separately. Studies that vary depth do not measure the pass rate of the training problems (Ni et al. 2025, Zhang et al. 2025, Zhu et al. 2026). Studies that control pass rate define difficulty by pass rate alone. The closest design matches pools on initial pass rate and splits them by a score from a probe training run (arXiv 2610.25325). No paper that the agents read trains on sets that are matched on measured pass rate and differ in depth by construction.

Two facts limit this finding:

- "Where Groups Split" holds initial pass rate fixed and varies a different factor. A statement that no published work holds pass rate fixed is therefore false.
- Thirty-three papers is a small share of the field. The arXiv searches returned more than 100 papers of 2025 and 2026 on difficulty, curricula, and data selection for RLVR.

## The closest work

The key question for each paper was: does it vary structural difficulty at a fixed pass rate, or the reverse?

| Paper | Setting | How it defines difficulty | Key question | What it does not cover |
| --- | --- | --- | --- | --- |
| Where Groups Split, arXiv 2610.25325 | Pretrained Qwen3 1.7B and 4B, GRPO, math and logic | Pass rate only, from 4 rollouts (eq. 3) | Partly. Pools matched on initial pass rate, split by a probe score, trained apart (§5.3) | The split factor comes from the model. The authors did not test a factor from the content of a prompt (App. D) |
| Zhang, Neubig, Yue, arXiv 2512.07783 | 100M model from scratch, synthetic arithmetic graphs, GRPO | Operation count (§2.1). Depth bands are named by pass@128 of the base model (§3) | Partly. Two appendices fix the RL data and vary pretraining (App. A.7, App. A.8) | No pass rate per problem. No seeds found |
| Zhu et al., arXiv 2605.26934 | 107M model from scratch, synthetic knowledge-graph stories, GRPO with group size 6 | Depth and distractor complexity, by construction (§2.1) | No | Pass rate of training data never measured (§2.1, §2.4). One seed (§10) |
| Ni et al., arXiv 2510.15990 | About 45M model from scratch, chains of string operators with written steps, GRPO with 8 samples | Number of steps (§4.2) | Partly, in the weak direction. The GRPO data keeps its depth, and the pretraining mix changes the base accuracy (Fig. 3c) | Pass rate of the GRPO prompts not reported, only base accuracy on a test set of the same depth. No seeds found |
| Yuan et al., arXiv 2509.25123 | Pretrained Llama-3.1-8B-Instruct, 25 string functions, nesting levels | Nesting level | No | Pass rate of the RL data not reported |
| Bauer et al., "Learning from Less", arXiv 2604.18381 | Pretrained Qwen3-4B with LoRA, three procedural tasks | Share of a panel of 10 models that answered correctly (§3.2) | No | The authors say that several properties change with the tier (§3.2). One seed (§4.1.3) |
| Chaudhry, arXiv 2610.25812 | Same counting task and model as Bauer et al., 3 seeds, registered hypotheses | Pass rate of the base model from 8 samples (§2.1) | Partly, in the weak direction only | Structure not matched or analyzed |
| Cheng et al., arXiv 2605.28388 | Pretrained Qwen2.5-Math 1.5B and 7B | Success in 8 samples of the base model (§2.2) | No | No structural measure |
| Chen, He, Zhao, arXiv 2605.16787 | Pretrained models, math, GRPO | Initial success rate below 10% is "hard" (Tab. 1) | No | Learnable and unlearnable examples are not matched on pass rate inside that bin |
| Huang et al., arXiv 2602.14872 | Theory, plus synthetic and Qwen2.5-1.5B runs | Composition length | No | Success rate is only an outcome |
| Ran-Milo et al., arXiv 2601.15158 | Theory, plus Qwen and Llama runs | Number of reasoning steps | No | Pass rate never measured |
| Re-Schedule, arXiv 2510.24832 (ICLR 2026 by its header) | Pretrained models, math | A score from a "reasoning tree". The tree is built from the sampled answers of the base model | No | The score measures the behavior of the model on a problem, not the structure of the problem. No comparison at equal accuracy |
| ScaleLogic, arXiv 2605.06638 | Pretrained Qwen3-4B, DAPO, synthetic logic | Proof depth and the expressiveness of the logic, by construction | No | Pass rate per problem never reported. One seed per setting |
| A Step Harder, Not a Step Too Far, arXiv 2610.13669 | Pretrained model, GSM8K, 3 seeds | Pass rate under the current policy (§3.1) | No | "Harder" means a lower pass-rate band only |

Eight more papers all answer no. Four are on the signal mathematics (arXiv 2602.05548, 2609.38018, 2605.05112, 2611.03625). Three are on the selection of prompts (arXiv 2602.01970, 2609.09075, 2502.11886). One is on the limits of RLVR (arXiv 2504.13837).

## What is already known

### GRPO and its filters

- DeepSeekMath proposes GRPO. It replaces the value function of PPO with a group baseline (arXiv 2402.03300 §4.1.2).
- DAPO filters out prompts whose group accuracy is 0 or 1, at 16 samples per prompt, because their advantage is zero (arXiv 2503.14476 §3.2). If p is the accuracy inside the current group, the baseline B of the project matches DAPO. With a p from an earlier measurement, it does not.

### The signal formula and "medium difficulty"

- The chance that a group carries signal, 1 − p^G − (1−p)^G, and its peak at p = 0.5, are in print (arXiv 2609.38018 §3. arXiv 2605.05112 App. E.2. arXiv 2611.03625 Eq. 1). "Where Groups Split" gives the same terms as Lemma 1.
- Yu et al. derive that the summed advantage of a group is 2G√(p(1−p)). They state that GRPO "implicitly prioritizes medium-difficulty samples" (arXiv 2602.05548, Theorem 2 and abstract).
- "Medium is best" depends on the measure. One paper finds the expected update flat in a transformed pass-rate coordinate (arXiv 2609.38018, Prop. 1). In the same paper a fixed target of 0.29 did not beat 0.5 (Tab. 3).
- Cheng et al. do not show that medium difficulty is best. Their abstract says that "easy and medium-difficulty problems yield the strongest and most stable reasoning improvements" (arXiv 2605.28388). Their "medium" is a coarse bin, 1 to 7 correct answers of 8, not p ≈ 0.5.
- The idea of training on tasks that the learner sometimes solves is older than language-model RL. Florensa et al. train on goals with a success rate from 0.1 to 0.9 (arXiv 1705.06366 §4.1). Foster et al. apply the method of SFL to language models (arXiv 2502.12272, abstract only).
- SFL scores levels by p(1−p) (arXiv 2408.15099 §4.1). It keeps the top K levels by that score and samples them uniformly (Alg. 2). It does not sample in proportion to the score. The baseline C of the project samples in proportion to the score.

### Small models from scratch as a controlled testbed

Ni et al., Zhang et al., and Zhu et al. each pretrain a small transformer on a synthetic task, then run GRPO, and vary the depth of the RL data. Their findings on depth:

- Ni et al.: "GRPO with 2-step data improves 3-step accuracy, whereas 1-step data does not" (§5.1).
- Zhang et al.: if the data is at the "edge of competence" of the model, RL gives gains at pass@128, and not in other cases (Takeaway 1).
- Zhu et al.: coverage of both structural axes beats coverage of one (abstract).

Our reading of Zhang et al., Fig. 11: the training reward of their four depth bands starts near 0.75, 0.5, 0.2, and 0. Their depth bands therefore differ in pass rate. An independent checker read the same start values from the figure. The checker adds a fact about the band with the gain on the hardest test (op=11–14). Its reward is about 0.45 to 0.50 for most of training.

Our reading of Ni et al., Fig. 3a to 3c, by the checker: base accuracy is about 87 to 97% at 1 and 2 steps and 0 to 35% at 3 steps. Depth and pass rate move together there too.

The task of the project is closer to that of Ni et al. than to that of Yuan et al. Yuan et al. use 25 functions and a pretrained model that writes free-form reasoning. The project uses 24 operators, a model from scratch, and a fixed step format.

### Pass rate does not decide everything

- At a matched initial pass rate, the half of a pool with the higher probe score reaches higher accuracy. In the low third on math the values are 0.328 against 0.401 (arXiv 2610.25325, Tab. 3). The pools were built once, from the probe of seed 0, and the pass rate comes from 4 rollouts (App. B.5, eq. 3).
- Zhang et al. show a second case. The RL data is fixed, and three models with different pretraining start at nearly the same training reward. The value is about 0.35 to 0.38 (our reading of Fig. 17, by the checker). One model stays flat and two rise to about 0.65 and 0.70. The authors state that models with minimal exposure "exhibit no reward improvement during post-training" (Fig. 17 caption). The reward is a mean over two contexts, so equal means do not prove equal pass rates per problem.
- Ni et al. list two datasets with near-equal accuracy before GRPO (58.20 and 57.93) and gains of 23.70 and 5.17 (App. A, Tab. 2). The datasets differ in more than one property.
- Chen, He, and Zhao report that about half of the hard examples that receive positive reward stay unlearned (arXiv 2605.16787 §3.3). Our reading of their Fig. 2, by the checker: the two groups start at about the same mean training reward, near 0.02, and then separate. A later workshop paper finds that those examples do improve, at a lower rate, and that the labels are not reproducible (arXiv 2609.40115 §3.2, §3.3). It also finds that the gradient evidence weakens once the number of correct rollouts is matched: the ratio falls from 2.327 to 1.539 (§3.5, App. C.2).
- Ran-Milo et al. show a case where accuracy stalls near 1/2. Training on hard examples only gives 52.9% on the hard test, and a mixed set gives 96.2% (arXiv 2601.15158, Tab. 1). Our reading of their Fig. 4, by the checker: the hard-only runs rise from about 1–10% to about 50% in the first 40 to 65 steps and then stop. A finished answer has two possible values. A pass rate near 1/2 is therefore the level of a guess between two values. The paper does not say "chance".

### Selection by a predicted pass rate

Two papers already select prompts by a pass rate that a predictor supplies, with a score that prefers intermediate values (arXiv 2602.01970. arXiv 2609.09075). Baseline F of the project uses the same method. Neither paper predicts the pass rate from the structure of a problem.

### The laws and the curve fits

- ScaleRL fits pass rate against compute with a ceiling A and an efficiency B (arXiv 2510.13786, Eq. 1). It never varies the difficulty of the problems. Its two data choices are filters on pass rate (§3.2). H3 for data difficulty is therefore open.
- The ScaleRL abstract says that curriculum changes efficiency. Its §3.2 reports a gain of 0.015 in the ceiling A, inside the ±0.02 margin of the paper (Fig. 8a). The checker reads the effect on B as no larger than the spread between seeds.
- ScaleLogic reports a power law between RL steps and proof depth (arXiv 2605.06638, abstract). T is the number of RL steps until validation Pass@1 reaches 90% (§4.1). Each fit uses 6 to 9 depths and one seed, and no fit is tested on a held-out depth. The exponent is 1.33 with a curriculum, 1.70 with a uniform mix, and 2.36 with hard problems only (§4.4). The paper does not report the pass rate behind these three numbers.
- RegMix and Data Mixing Laws are about pretraining only. RegMix uses 512 small runs. Data Mixing Laws fits about 20 mixtures.

### "Learning from Less" and its follow-up

- The source paper is at MLSys 2026. It uses one model and one seed, and it calls for empirical laws (arXiv 2604.18381 §3.2, §4.1.3, §5, p. 1 footer).
- In the source paper, 100 mixed problems match 500 easy ones on the counting task (Tab. 2) and on the spatial task. They do not match on the graph task.
- Chaudhry repeated the comparison of mixed against easy with three seeds and registered hypotheses. The gain for GRPO was +4.4 points against a noise bar of 5.6, and the sign reversed out of distribution (arXiv 2610.25812, Tab. 3). The criterion of that paper is a difference above two times the seed noise, with the same sign out of distribution.

## How the project differs from this work

The list starts with the difference that the evidence above supports best.

1. Matched pass rate with different depth. Sets D and E have the same measured pass rate and differ in depth by construction. No paper read does this.
2. A content factor behind "same pass rate, different learning". The authors of "Where Groups Split" name a prior built from prompt features as untested (App. D). Depth is such a feature.
3. Pass rate per problem inside a from-scratch study. The three from-scratch papers do not report it. The project logs it for each problem over training.
4. Seeds and a noise bar. Ni et al. and Zhang et al. report no seeds. Zhu et al. and Bauer et al. use one. The project sets its noise bar before the comparison.
5. A step-level account of what each set taught. Zhang et al. already score each step of a solution (App. A.4), so the metric is not new. Its use to compare two sets at a matched pass rate is new.
6. A structural predictor of pass rate. Baseline F is new only in this narrow form: it predicts p from depth and operators with no rollouts.

Two parts of the project are not new: the from-scratch control by itself, and the signal formula.

## Findings that bear on the experiments

- Noise in pass-rate labels. At a threshold of 0.1, labels from 32 samples agree at about 0.80 between repeats, and 0.90 needs a projected 242 samples (arXiv 2609.40115 §3.4). That paper pools counts and does not intersect labels. The project selects on 32 samples and measures again on 64 new ones.
- Drift after step 0. The correlation with the initial score falls to 0.610 by the end of training in one study (arXiv 2610.25325, Tab. 19). In another, the share of groups with no signal rises from 26–29% to 61–70% (arXiv 2610.25812 §3.2).
- Guesses. The stall of Ran-Milo et al. near 1/2 is on a task with two possible answers. A string answer has a chance level near 0, so p ≈ 0.5 in the project cannot come from guesses.
- Loss scale and length. Huang et al. divide the update by the length of the sequence (arXiv 2602.14872, Eq. 11 and Eq. 14). Two readers derived from the formulas of the paper that this cuts the push from deep problems at an equal pass rate. The paper does not state that result. The GRPO code of the team has both forms, `loss_agg=token` and `loss_agg=sequence` (`rlvr/grpo.py:17`).
- The online filter. It added nothing to GRPO in one study: −1.0 points, within noise (arXiv 2610.25812 §3.2). Baseline B of the project is that filter.
- Metrics in these papers. "Where Groups Split" reports the share of live groups and the steps needed to match the best baseline. Two papers split the gain into pass@1 and pass@k (arXiv 2512.07783. arXiv 2605.26934 App. A.4). Chaudhry plots the no-signal share per tier (arXiv 2610.25812, Fig. 2).

## Limits of this research

- Venues. A venue is given above only where the PDF states it. For Zhu et al., OpenAlex lists EMNLP 2026, and the PDF read states none. For Yuan et al., the PDF read (v3) states no venue.
- A second agent compared all 21 notes files with the papers and corrected them. Every comparison found something to fix, mostly small. Almost all quotes matched the papers. The errors were of three kinds. Some notes worded a reading of a figure as a fact of the paper. Some assigned a number to the wrong quantity. Some left out findings. One comparison changed an answer to the key question, for Ni et al., from no to partly. A checker is also a language model, so a number here can still be wrong. For 9 of the papers, a third pass searched the paper text for the main numbers and quotes that this report uses.
- Papers that look relevant and that nobody read. All are abstract only.
  - Composition-RL, arXiv 2602.12036. It composes prompts to escape a pass rate of 1 and adds a curriculum by composition depth.
  - Foster et al., arXiv 2502.12272, and VCRL, arXiv 2509.19803. Both select by the variance of success.
- FADE was read in targeted parts only. It separates the ability of the policy from a difficulty value per prompt, fitted from outcomes.
- The Semantic Scholar citation search did not run, because no API key was available. OpenAlex listed few citing papers for the newest hubs.
