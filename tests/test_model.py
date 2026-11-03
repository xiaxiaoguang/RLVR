"""Model / stream / training-loss sanity tests (CPU, tiny model). Run: python tests/test_model.py (or pytest)."""
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import torch  # noqa: E402
from pretrain import lr_lambda, token_losses  # noqa: E402

from rlvr.config import load_config, task_and_splits  # noqa: E402
from rlvr.model import build_model, cut_at_eos, generate, left_pad, load_model, save_model  # noqa: E402
from rlvr.stream import Mixture, PretrainStream  # noqa: E402
from rlvr.task import Task  # noqa: E402
from rlvr.tokenizer import IGNORE_INDEX, CharTokenizer  # noqa: E402

CFG = load_config(None, ["model.n_layer=1", "model.d_model=32", "model.d_ff=64", "model.n_head=2",
                         "pretrain.comp_frac=0.2", "pretrain.micro_target_tokens=2000",
                         "pretrain.bucket_examples=600"])
TASK = Task(CFG["task"])
TOK = CharTokenizer(TASK.alphabet)


def one_batch(cfg=CFG):
    ds = PretrainStream(cfg)
    ds._setup(0)
    return ds, next(ds.batches())


def test_stream_batch_masks_and_positions():
    _, b = one_batch()
    ids, labels, pos = b["input_ids"], b["labels"], b["position_ids"]
    assert ids.shape == labels.shape == pos.shape and ids.shape[1] % 8 == 0
    assert int(b["n_target"].sum()) >= CFG["pretrain"]["micro_target_tokens"]
    for i in range(len(ids)):
        n = int((ids[i] != TOK.pad_id).sum())
        assert (ids[i, n:] == TOK.pad_id).all(), "padding only at the end"
        assert ids[i, 0] == TOK.bos_id and ids[i, n - 1] == TOK.eos_id
        sup = labels[i] != IGNORE_INDEX
        assert (labels[i][sup] == ids[i][sup]).all() and not sup[n:].any()
        prompt = TOK.decode(ids[i, :n][~sup[:n]].tolist())
        assert prompt.startswith("<bos>Q:") and prompt.endswith("=\n"), "prompt (and only it) is masked"
        assert int(sup.sum()) == int(b["n_target"][i])
        off = int(pos[i, 0])
        assert 0 <= off <= CFG["pretrain"]["pos_offset_max"] and (pos[i] == off + torch.arange(ids.shape[1])).all()
        assert int(pos[i, n - 1]) < CFG["model"]["max_len"]
    rows = sorted(int((r != TOK.pad_id).sum()) for r in ids)
    assert rows[-1] - rows[0] <= 40, "batches are length-bucketed"


def test_stream_is_leak_free_and_matches_comp_frac():
    task, splits = task_and_splits(CFG)
    mix = Mixture(task, splits, CFG["pretrain"], seed=0)
    rng = random.Random(0)
    tgt = {"atom": 0, "comp": 0}
    for _ in range(20000):
        kind, p, steps = mix.sample(rng)
        assert splits.pool_of(p.pid) == "pretrain" and not splits.has_heldout_pair(p.funcs)
        if p.depth in CFG["pretrain"]["h_only_depths"]:
            assert p.group == "H"
        tgt[kind] += len(TOK.encode(mix.target(p, steps)))
    share = tgt["comp"] / sum(tgt.values())
    assert abs(share - CFG["pretrain"]["comp_frac"]) < 0.01, share


def test_stream_deterministic_per_rank_and_worker():
    a = [one_batch()[1]["input_ids"] for _ in range(2)]
    assert torch.equal(a[0], a[1])
    ds = PretrainStream(CFG, rank=1)
    ds._setup(0)
    assert not torch.equal(a[0], next(ds.batches())["input_ids"])


def test_token_losses_masked():
    torch.manual_seed(0)
    model = build_model(CFG["model"], TOK)
    _, b = one_batch()
    ce, valid = token_losses(model, b)
    assert ce.shape == valid.shape == (b["input_ids"].shape[0], b["input_ids"].shape[1] - 1)
    assert int(valid.sum()) == int(b["n_target"].sum())
    assert (ce[~valid] == 0).all() and (ce[valid] > 0).all()
    ref = torch.nn.functional.cross_entropy(
        model(input_ids=b["input_ids"], position_ids=b["position_ids"]).logits[:, :-1].flatten(0, 1).float(),
        b["labels"][:, 1:].flatten(), ignore_index=IGNORE_INDEX)
    assert torch.allclose(ce.sum() / valid.sum(), ref, atol=1e-5)


def test_lr_schedule():
    f = lr_lambda(10, 100, 0.1)
    assert abs(f(0) - 0.1) < 1e-9 and abs(f(9) - 1.0) < 1e-9
    assert abs(f(100) - 0.1) < 1e-9 and 0.1 < f(50) < 1.0


def test_left_pad_and_cut():
    ids, mask = left_pad(TOK, ["Q:F01(abcde)=\n", "Q:F01(F02(abcde))=\n"], "cpu")
    assert ids[:, -1].tolist() == [TOK.encode("\n")[0]] * 2
    assert ids[0, 0] == TOK.pad_id and mask[0, 0] == 0 and mask[1].all()
    assert cut_at_eos("A:abc<eos>A:x<eos>") == "A:abc<eos>"
    assert cut_at_eos("A:abc") == "A:abc"


def test_generate_and_roundtrip():
    torch.manual_seed(0)
    model = build_model(CFG["model"], TOK).eval()
    prompts = ["Q:F01(abcde)=\n", "Q:F01(F02(abcde))=\n"]
    out = generate(model, TOK, prompts, max_new_tokens=8, temperature=1.0, num_return_sequences=3)
    assert len(out) == 6
    greedy = generate(model, TOK, prompts, max_new_tokens=8)
    with tempfile.TemporaryDirectory() as d:
        save_model(model, TOK, {**CFG, "world_size": 4}, d)
        m2, tok2 = load_model(d, "cpu")
        assert tok2.vocab == TOK.vocab
        assert generate(m2, tok2, prompts, max_new_tokens=8) == greedy
        assert load_config(Path(d) / "run_config.yaml")["model"]["d_model"] == 32


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
