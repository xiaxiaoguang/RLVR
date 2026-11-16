"""GRPO sanity tests (CPU, tiny model). Run: python tests/test_grpo.py (or pytest)."""
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import grpo as grpo_script  # noqa: E402
from rlvr.config import load_config, task_and_splits  # noqa: E402
from rlvr.grpo import group_advantages, grpo_update, micro_batches, pack, signal_groups, token_logprobs  # noqa: E402
from rlvr.io import problems_to_rows, read_jsonl, write_jsonl  # noqa: E402
from rlvr.model import build_model, generate_ids, left_pad, load_model, save_model  # noqa: E402
from rlvr.sampler import ProblemSampler  # noqa: E402
from rlvr.sets import stratified_set  # noqa: E402

TINY = ["model.n_layer=2", "model.d_model=32", "model.d_ff=64", "model.n_head=2"]
CFG = load_config(None, TINY)
TASK, SPLITS = task_and_splits(CFG)
TOK = TASK.tokenizer()
RC = {**CFG["rl"], "micro_batch_tokens": 10_000}


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return build_model(CFG["model"], TOK)


def some_rollouts(model, n_prompts=5, g=4, max_new=12, seed=0):
    rng = random.Random(seed)
    probs = [TASK.sample_problem(rng.choice([1, 2, 3]), rng) for _ in range(n_prompts)]
    prompts = [TASK.format_prompt(p) for p in probs]
    torch.manual_seed(seed)
    comp = generate_ids(model, TOK, prompts, max_new, temperature=1.0, num_return_sequences=g)
    prompt_ids = [ids for p in prompts for ids in [TOK.encode(p, add_bos=True)] * g]
    return prompts, prompt_ids, comp


# ---- advantages ------------------------------------------------------------

def test_group_advantages():
    r = torch.tensor([[1., 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0]])
    adv = group_advantages(r, std_norm=False)
    assert torch.allclose(adv[0], torch.tensor([0.75, -0.25, -0.25, -0.25]))
    assert (adv[1] == 0).all() and (adv[2] == 0).all()
    norm = group_advantages(r, std_norm=True)
    assert (norm[1] == 0).all() and (norm[2] == 0).all(), "no-signal groups get exactly zero"
    assert torch.allclose(norm[3], adv[3] / (r[3].std() + 1e-6))
    assert torch.allclose(norm.sum(1), torch.zeros(4), atol=1e-6)
    assert signal_groups(r).tolist() == [True, False, False, True]
    assert (group_advantages(torch.tensor([[1.], [0.]]), std_norm=True) == 0).all(), "G=1 has no baseline"


def test_micro_batches_respect_budget_and_cover_everything():
    lengths = [random.Random(0).randint(5, 60) for _ in range(40)]
    chunks = micro_batches(lengths, 200)
    assert sorted(i for ch in chunks for i in ch) == list(range(40))
    for ch in chunks:
        assert len(ch) == 1 or len(ch) * max(lengths[i] for i in ch) <= 200
    assert micro_batches([500, 3], 100) == [[0], [1]], "an over-long sequence gets its own chunk"


# ---- log-probs ---------------------------------------------------------------

def test_pack_masks_only_the_completion():
    ids, mask = pack([[1, 5, 6], [1, 7]], [[8, 2], [9, 9, 2]], TOK.pad_id)
    assert ids.tolist() == [[1, 5, 6, 8, 2], [1, 7, 9, 9, 2]]
    assert mask.tolist() == [[False, False, False, True, True], [False, False, True, True, True]]


def test_teacher_forced_logprobs_match_generation():
    """The right-padded training forward scores sampled tokens like the left-padded generate()."""
    model = tiny_model().eval()
    prompts = [TASK.format_prompt(TASK.sample_problem(d, random.Random(d))) for d in (1, 3, 2)]
    input_ids, attn = left_pad(TOK, prompts, "cpu")
    torch.manual_seed(1)
    out = model.generate(input_ids=input_ids, attention_mask=attn, max_new_tokens=10, do_sample=True,
                         temperature=1.0, output_scores=True, return_dict_in_generate=True)
    new = out.sequences[:, input_ids.shape[1]:]
    step_logp = torch.stack([torch.log_softmax(s.float(), -1) for s in out.scores], 1)  # [N, T, V]
    gen_logp = step_logp.gather(-1, new[..., None]).squeeze(-1)
    comp = []
    for row in new.tolist():
        n = row.index(TOK.eos_id) + 1 if TOK.eos_id in row else len(row)
        comp.append(row[:n])
    prompt_ids = [TOK.encode(p, add_bos=True) for p in prompts]
    ids, mask = pack(prompt_ids, comp, TOK.pad_id)
    with torch.no_grad():
        logp, m = token_logprobs(model, ids, mask)
    for i, (p, c) in enumerate(zip(prompt_ids, comp)):
        got = logp[i, len(p) - 1:len(p) - 1 + len(c)]
        assert m[i].sum() == len(c)
        assert torch.allclose(got, gen_logp[i, :len(c)], atol=1e-4), (got, gen_logp[i, :len(c)])


# ---- update ------------------------------------------------------------------

def _grads(model, prompt_ids, comp, adv, rc):
    """Gradient that grpo_update applies (SGD, lr 1, no clipping) = old params - new params."""
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    stats = grpo_update(model, opt, prompt_ids, comp, adv, {**rc, "grad_clip": 1e9}, TOK.pad_id)
    g = [b - p.detach() for b, p in zip(before, model.parameters())]
    with torch.no_grad():
        for p, b in zip(model.parameters(), before):
            p.copy_(b)
    return g, stats


def test_update_matches_manual_policy_gradient_for_both_aggregations():
    model = tiny_model()
    _, prompt_ids, comp = some_rollouts(model)
    adv = torch.randn(len(comp))
    ids, mask = pack(prompt_ids, comp, TOK.pad_id)
    for agg in ("token", "sequence"):
        logp, m = token_logprobs(model, ids, mask)
        if agg == "token":
            loss = -(adv[:, None] * logp).sum() / m.sum()
        else:
            loss = -(adv * logp.sum(1) / m.sum(1)).mean()
        want = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
        got, stats = _grads(model, prompt_ids, comp, adv, {**RC, "loss_agg": agg})
        assert abs(stats["loss"] - loss.item()) < 1e-5 and stats["n_tokens"] == int(m.sum())
        for w, g in zip(want, got):
            if w is not None:
                assert torch.allclose(w, g, atol=1e-5), agg


def test_micro_batching_does_not_change_the_update():
    model = tiny_model()
    _, prompt_ids, comp = some_rollouts(model, n_prompts=6)
    adv = torch.randn(len(comp))
    for extra in ({}, {"loss_agg": "sequence"}, {"entropy_coef": 0.01}):
        one, s1 = _grads(model, prompt_ids, comp, adv, {**RC, **extra, "micro_batch_tokens": 100_000})
        many, s2 = _grads(model, prompt_ids, comp, adv, {**RC, **extra, "micro_batch_tokens": 120})
        assert s1["n_micro"] == 1 and s2["n_micro"] > 3
        assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(one, many)), extra


def test_zero_advantage_gives_zero_gradient_and_positive_raises_logprob():
    model = tiny_model()
    _, prompt_ids, comp = some_rollouts(model, n_prompts=2, g=2)
    g, _ = _grads(model, prompt_ids, comp, torch.zeros(len(comp)), RC)
    assert all(float(x.abs().max()) == 0 for x in g)
    adv = torch.zeros(len(comp))
    adv[0] = 1.0
    ids, mask = pack(prompt_ids, comp, TOK.pad_id)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    with torch.no_grad():
        before = token_logprobs(model, ids, mask)[0][0].sum()
    for _ in range(5):
        grpo_update(model, opt, prompt_ids, comp, adv, RC, TOK.pad_id)
    with torch.no_grad():
        after = token_logprobs(model, ids, mask)[0][0].sum()
    assert after > before


def test_ppo_epochs_and_kl_to_base():
    model, ref = tiny_model(), tiny_model()
    _, prompt_ids, comp = some_rollouts(model)
    adv = torch.randn(len(comp))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    s = grpo_update(model, opt, prompt_ids, comp, adv, RC, TOK.pad_id, ref, with_kl=True)
    assert abs(s["kl_to_base"]) < 1e-6, "identical to the base before the first update"
    s = grpo_update(model, opt, prompt_ids, comp, adv, {**RC, "ppo_epochs": 3, "kl_coef": 0.1}, TOK.pad_id, ref)
    assert s["kl_to_base"] > 0 and np.isfinite(s["loss"])
    # first PPO epoch: the importance ratio is 1, so one clipped epoch is the plain policy gradient
    plain, _ = _grads(model, prompt_ids, comp, adv, RC)
    old = [token_logprobs(model, *pack(prompt_ids, comp, TOK.pad_id))[0].detach()]
    logp, m = token_logprobs(model, *pack(prompt_ids, comp, TOK.pad_id))
    ratio = torch.exp(logp - old[0])
    a = adv[:, None]
    loss = -(torch.minimum(ratio * a, ratio.clamp(0.8, 1.2) * a) * m).sum() / m.sum()
    want = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    assert all(w is None or torch.allclose(w, g, atol=1e-5) for w, g in zip(want, plain))


# ---- samplers ----------------------------------------------------------------

DEPTHS = [1] * 10 + [2] * 10 + [3] * 10


def test_static_sampler_epochs_and_curriculum():
    s = ProblemSampler(DEPTHS, "static", seed=0)
    draws = np.concatenate([s.sample(7) for _ in range(30)])  # 210 = 7 epochs
    assert (np.bincount(draws, minlength=30) == 7).all(), "every problem equally often"
    assert sorted(draws[:30].tolist()) == list(range(30)), "an epoch is a permutation"
    c = ProblemSampler(DEPTHS, "static", order="shallow_to_deep", seed=0, total=210)
    cur = np.concatenate([c.sample(7) for _ in range(30)])
    assert (np.bincount(cur, minlength=30) == 7).all(), "same draws as shuffle, only reordered"
    d = np.asarray(DEPTHS)[cur]
    assert (np.diff(d) >= 0).all() and d[0] == 1 and d[-1] == 3
    try:
        c.sample(1)
        raise AssertionError("schedule should be exhausted")
    except RuntimeError:
        pass


def test_online_p_ema():
    s = ProblemSampler(DEPTHS, "static", p_init=[0.5] + [np.nan] * 29, p_ema=0.9)
    s.update([0, 1, 1], [1.0, 0.25, 1.0])
    assert abs(s.p[0] - 0.55) < 1e-9
    assert abs(s.p[1] - (0.9 * 0.25 + 0.1)) < 1e-9, "first observation sets p, later ones are an EMA"
    assert np.isnan(s.p[2]) and s.stats()["n_seen"] == 2


def test_online_filter_keeps_signal_problems_and_fills_up():
    p = np.array([0.0] * 10 + [0.5] * 10 + [1.0] * 5 + [np.nan] * 5)
    s = ProblemSampler(DEPTHS, "online_filter", seed=0, p_init=p)
    for _ in range(20):
        idx = s.sample(8)
        assert len(set(idx.tolist())) == 8 and ((idx >= 10) & (idx < 20) | (idx >= 25)).all()
    assert s.last == {"n_eligible": 15, "n_fill": 0}
    idx = s.sample(20)  # only 15 eligible: 5 filled from the rest
    assert len(set(idx.tolist())) == 20 and s.last["n_fill"] == 5
    s.update(np.arange(10, 20), np.ones(10))  # solved problems drift out: 0.5 -> ... -> > 31/32
    for _ in range(40):
        s.update(np.arange(10, 20), np.ones(10))
    assert s.weights()[10:20].sum() == 0
    e = ProblemSampler(DEPTHS, "online_filter", seed=0, p_init=p, explore_frac=0.5)
    seen = np.concatenate([e.sample(8) for _ in range(200)])
    assert (seen < 10).mean() > 0.1, "exploration revisits filtered problems"


def test_learnability_sampler_weights():
    p = np.array([0.0] * 10 + [0.5] * 10 + [0.9] * 10)
    s = ProblemSampler(DEPTHS, "learnability", seed=0, p_init=p)
    draws = np.concatenate([s.sample(4) for _ in range(500)])
    counts = np.bincount(draws, minlength=30)
    assert counts[:10].sum() == 0
    ratio = counts[10:20].sum() / counts[20:].sum()
    assert 2.0 < ratio < 3.6, ratio  # 0.25 / 0.09 = 2.8, a little flatter without replacement
    assert len(ProblemSampler([1, 2], "learnability", seed=0).sample(5)) == 5, "tiny dataset repeats problems"


# ---- end to end ----------------------------------------------------------------

def _workspace(tmp: Path):
    """Tiny random checkpoint, eval sets and an RL dataset under tmp."""
    over = TINY + [f"paths.data={tmp}/data", f"paths.runs={tmp}/runs", f"paths.checkpoints={tmp}/checkpoints",
                   "eval.version=vtest", "pretrain.level=tiny"]
    cfg = load_config(None, over)
    save_model(tiny_model(), TOK, cfg, tmp / "checkpoints" / "pt-tiny-s0")
    rng, seen = random.Random(0), set()
    for name, depths in (("E1", [1]), ("E2", [2, 3]), ("probe", [1, 2])):
        probs = stratified_set(SPLITS, "eval", depths, ["H", "L"], 3, rng, seen)
        write_jsonl(tmp / "data" / "eval" / "vtest" / f"{name}.jsonl", problems_to_rows(TASK, probs, eval_set=name))
    data = [SPLITS.sample("rl", d, rng, exclude=seen) for d in (1, 2) for _ in range(6)]
    write_jsonl(tmp / "data" / "datasets" / "toy.jsonl", problems_to_rows(TASK, data, origin="base"))
    (tmp / "data" / "datasets" / "toy.meta.json").write_text(json.dumps({"spec": {"sampler": "learnability"}}))
    return str(tmp / "checkpoints" / "pt-tiny-s0")


def test_grpo_script_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        base = _workspace(tmp)
        args = ["--base", base, "--dataset", "toy", "--device", "cpu", "rl.steps=3", "rl.prompts_per_step=4",
                "rl.group_size=3", "rl.max_new_tokens=24", "eval.max_new_tokens=24", "rl.eval_every=2",
                "rl.probe_every=1", "rl.eval_sets=[E1,E2]", "rl.kl_every=1", "rl.save_at=[0.34]"]
        summary = grpo_script.main(args)
        run = tmp / "runs" / "rl-pt-tiny-s0-toy-G3-s0"
        assert summary["run_name"] == run.name and summary["sampler"] == "learnability"
        metrics = read_jsonl(run / "metrics.jsonl")
        assert [m["step"] for m in metrics] == [1, 2, 3] and metrics[-1]["rollouts"] == 36
        assert all(k in metrics[0] for k in ("mean_reward", "signal_frac", "entropy", "grad_norm", "kl_to_base"))
        rollouts = read_jsonl(run / "rollouts.jsonl")
        assert len(rollouts) == 12 and len(rollouts[0]["rewards"]) == len(rollouts[0]["completions"]) == 3
        ev = read_jsonl(run / "eval.jsonl")
        steps_by_set = {n: sorted({r["step"] for r in ev if r["eval_set"] == n}) for n in ("E1", "E2", "probe")}
        assert steps_by_set == {"E1": [0, 2, 3], "E2": [0, 2, 3], "probe": [0, 1, 2, 3]}
        assert set(summary["eval_final"]) == {"E1", "E2", "probe"}
        for step in (1, 3):  # RL checkpoints load like base checkpoints (and can seed another RL run)
            model, _ = load_model(run / "ckpt" / f"step-{step}", "cpu")
            assert model.config.hidden_size == 32
        try:
            grpo_script.main(args)
            raise AssertionError("existing run must not be overwritten silently")
        except SystemExit:
            pass
        chained = grpo_script.main(["--base", str(run / "ckpt" / "step-3"), "--dataset", "toy", "--device", "cpu",
                                    "rl.steps=1", "rl.prompts_per_step=2", "rl.group_size=2", "rl.max_new_tokens=8",
                                    "eval.max_new_tokens=8", "rl.save_final=false", "rl.tag=x",
                                    "rl.eval_sets=[E1]"])
        assert chained["run_name"] == "rl-rl-pt-tiny-s0-toy-G3-s0-step-3-toy-G2-s0-x" and chained["ckpt"] is None


def test_rl_config_inherits_and_protects_the_checkpoint_task():
    with tempfile.TemporaryDirectory() as d:
        base = _workspace(Path(d))
        cfg = grpo_script.rl_config(None, [f"rl.base={base}", "data.dataset=toy", "rl.lr=3e-6"])
        assert cfg["model"]["d_model"] == 32 and cfg["eval"]["version"] == "vtest" and cfg["rl"]["lr"] == 3e-6
        assert cfg["paths"]["runs"] == f"{d}/runs" and cfg["pretrain"]["level"] == "tiny"
        for bad in ("model.n_layer=4", "task.str_len=[4]"):
            try:
                grpo_script.rl_config(None, [f"rl.base={base}", bad])
                raise AssertionError(bad)
            except SystemExit:
                pass


def test_sweep_grid_points():
    from rl_sweep import grid_points
    assert grid_points([]) == [{}]
    assert grid_points(["rl.group_size=4,8", "rl.lr=3e-6"]) == [
        {"rl.group_size": "4", "rl.lr": "3e-6"}, {"rl.group_size": "8", "rl.lr": "3e-6"}]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
