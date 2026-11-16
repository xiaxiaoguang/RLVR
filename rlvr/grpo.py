"""GRPO core: group advantages, teacher-forced log-probs of sampled completions, one update.

    comp_ids = generate_ids(model, tok, prompts, max_new, temperature, num_return_sequences=G)
    r = rewards(...).view(B, G)
    adv = group_advantages(r, std_norm=True)
    stats = grpo_update(model, opt, prompt_ids, comp_ids, adv.flatten(), rl_cfg, tok.pad_id)

The policy-gradient forward uses the pretraining layout: <bos> prompt completion, right-padded,
positions from 0 (generation left-pads instead; RoPE only sees relative positions and the pad
is masked there, so both give the same log-probs - tests/test_grpo.py checks this). Only
completion tokens (including <eos>) enter the loss. The rollout batch is split into
length-sorted micro-batches of at most rl.micro_batch_tokens padded tokens; every term is
normalised by totals of the whole batch and gradients are accumulated, so micro-batching does
not change the update.

Loss per rollout batch (N = B*G sequences, T = all completion tokens):
  ppo_epochs = 1   L = -sum(adv_i * logp_it) / T                       (loss_agg=token)
                   L = -mean_i(adv_i * mean_t logp_it)                 (loss_agg=sequence)
  ppo_epochs > 1   logp is replaced by the clipped importance ratio to the rollout policy
  + kl_coef * mean_t KL(pi || pi_base) - entropy_coef * mean_t H(pi)
"""
from __future__ import annotations

import torch


def group_advantages(r: torch.Tensor, std_norm: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """r: [B, G] rewards -> [B, G] advantages. Groups with identical rewards get exactly 0."""
    adv = r - r.mean(1, keepdim=True)
    if std_norm and r.shape[1] > 1:
        adv = adv / (r.std(1, keepdim=True) + eps)
    return adv


def signal_groups(r: torch.Tensor) -> torch.Tensor:
    """[B] bool: the group has mixed rewards, i.e. it produces gradient."""
    return r.max(1).values > r.min(1).values


def pack(prompt_ids: list[list[int]], comp_ids: list[list[int]], pad_id: int,
         device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded input_ids [N, T] (prompt then completion) and the completion mask [N, T]."""
    n = max(len(p) + len(c) for p, c in zip(prompt_ids, comp_ids))
    ids = torch.full((len(prompt_ids), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompt_ids), n), dtype=torch.bool)
    for i, (p, c) in enumerate(zip(prompt_ids, comp_ids)):
        ids[i, :len(p) + len(c)] = torch.tensor(p + c)
        mask[i, len(p):len(p) + len(c)] = True
    return ids.to(device), mask.to(device)


def micro_batches(lengths: list[int], budget: int) -> list[list[int]]:
    """Index chunks, longest sequences first, each with (rows x longest row) <= budget tokens
    (a single sequence longer than the budget gets its own chunk)."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    chunks, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lengths[cur[0]] > budget:
            chunks.append(cur)
            cur = []
        cur.append(i)
    if cur:
        chunks.append(cur)
    return chunks


def token_logprobs(model, input_ids: torch.Tensor, comp_mask: torch.Tensor,
                   full: bool = False):
    """Teacher-forced log-prob of every completion token: ([N, T-1] zeros elsewhere, mask [N, T-1]).
    full=True also returns the whole log-softmax [N, T-1, V] (entropy / KL)."""
    with torch.autocast(input_ids.device.type, dtype=torch.bfloat16, enabled=input_ids.device.type == "cuda"):
        logits = model(input_ids=input_ids).logits
    logp_all = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    m = comp_mask[:, 1:]
    logp = logp_all.gather(-1, input_ids[:, 1:, None]).squeeze(-1) * m
    return (logp, m, logp_all) if full else (logp, m)


def grpo_update(model, opt, prompt_ids: list[list[int]], comp_ids: list[list[int]],
                adv: torch.Tensor, rc: dict, pad_id: int, ref_model=None,
                with_kl: bool = False) -> dict:
    """One GRPO update (rc: the `rl` config section) on N sequences with advantages adv [N].
    Returns loss, entropy, grad_norm, n_tokens and, with a ref_model and (with_kl or kl_coef),
    kl_to_base = mean over completion tokens of KL(pi || pi_ref) before the update."""
    device = next(model.parameters()).device
    n_seq = len(comp_ids)
    n_tok = sum(map(len, comp_ids))
    chunks = micro_batches([len(p) + len(c) for p, c in zip(prompt_ids, comp_ids)], rc["micro_batch_tokens"])
    adv = adv.to(device).float()
    use_kl = ref_model is not None and (with_kl or rc["kl_coef"] > 0)
    epochs = rc["ppo_epochs"]
    packed = [pack([prompt_ids[i] for i in ch], [comp_ids[i] for i in ch], pad_id, device) for ch in chunks]
    old = None
    if epochs > 1:  # log-probs under the rollout policy, for the importance ratio
        with torch.no_grad():
            old = [token_logprobs(model, ids, cm)[0] for ids, cm in packed]
    params = [p for p in model.parameters() if p.requires_grad]
    out: dict = {}
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        tot = {"loss": 0.0, "entropy": 0.0, "kl": 0.0}
        for k, (ch, (ids, cm)) in enumerate(zip(chunks, packed)):
            logp, m, logp_all = token_logprobs(model, ids, cm, full=True)
            a = adv[ch][:, None]
            if old is None:
                obj = a * logp
            else:
                ratio = torch.exp(logp - old[k])
                obj = torch.minimum(ratio * a, ratio.clamp(1 - rc["clip_eps"], 1 + rc["clip_eps"]) * a) * m
            if rc["loss_agg"] == "token":
                loss = -obj.sum() / n_tok
            else:
                loss = -(obj.sum(1) / m.sum(1).clamp(min=1)).sum() / n_seq
            ent = -((logp_all.exp() * logp_all).sum(-1) * m).sum() / n_tok
            if rc["entropy_coef"]:
                loss = loss - rc["entropy_coef"] * ent
            if use_kl:
                with torch.no_grad():
                    ref_all = token_logprobs(ref_model, ids, cm, full=True)[2]
                kl = ((logp_all.exp() * (logp_all - ref_all)).sum(-1) * m).sum() / n_tok
                if rc["kl_coef"] > 0:
                    loss = loss + rc["kl_coef"] * kl
                tot["kl"] += float(kl.detach())
            loss.backward()
            tot["loss"] += float(loss.detach())
            tot["entropy"] += float(ent.detach())
        gn = torch.nn.utils.clip_grad_norm_(params, rc["grad_clip"])
        opt.step()
        if epoch == 0:  # statistics of the rollout policy
            out = {"loss": tot["loss"], "entropy": tot["entropy"], "grad_norm": float(gn),
                   "n_tokens": n_tok, "n_micro": len(chunks)}
            if use_kl:
                out["kl_to_base"] = tot["kl"]
    opt.zero_grad(set_to_none=True)
    return out
