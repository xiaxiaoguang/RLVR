"""Sanity tests for task / tokenizer / splits. Run: python tests/test_task.py (or pytest)."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlvr.config import load_config, task_and_splits  # noqa: E402
from rlvr.difficulty import p_bin, p_signal, wilson  # noqa: E402
from rlvr.task import Task  # noqa: E402
from rlvr.tokenizer import IGNORE_INDEX, CharTokenizer  # noqa: E402

CFG = load_config()
TASK, SPLITS = task_and_splits(CFG)


def by_name(name):
    return TASK.name2id[name]


def test_atoms():
    cases = {
        "reverse": "hgfedcba", "rotl1": "bcdefgha", "rotr1": "habcdefg", "rotl2": "cdefghab",
        "rotl3": "defghabc", "swap_adjacent": "badcfehg", "swap_ends": "hbcdefga",
        "swap_halves": "efghabcd", "reverse_first_half": "dcbaefgh", "riffle": "aebfcgdh",
        "shift_p1": "bcdefghi", "shift_m1": "zabcdefg", "shift_p3": "defghijk", "shift_p5": "fghijklm",
        "complement": "zyxwvuts", "shift_by_pos": "acegikmo",
        "sort_asc": "abcdefgh", "sort_desc": "hgfedcba", "sort_first_half": "abcdefgh",
        "even_pos_p1": "bbddffhh", "odd_pos_m1": "aacceegg",
        "max_to_front": "habcdefg", "min_to_back": "bcdefgha", "mirror_first_half": "abcddcba",
    }
    assert sorted(cases) == sorted(CFG["task"]["functions"]), "every function has a case"
    for name, want in cases.items():
        assert TASK.apply(by_name(name), "abcdefgh") == want, name
    # odd lengths, wrap-around, and inputs where sorting / max / min matter
    odd = {"swap_halves": ("abcde", "decab"), "riffle": ("abcde", "adbec"),
           "reverse_first_half": ("abcde", "bacde"), "sort_first_half": ("edcba", "decba"),
           "mirror_first_half": ("abcde", "abcba"), "shift_m1": ("abc", "zab"), "shift_p5": ("xyz", "cde"),
           "shift_by_pos": ("zzz", "zab"), "complement": ("amz", "zna"),
           "sort_asc": ("dbca", "abcd"), "sort_desc": ("dbca", "dcba"),
           "max_to_front": ("cajfb", "jcafb"), "min_to_back": ("cajfb", "cjfba")}
    for name, (x, want) in odd.items():
        assert TASK.apply(by_name(name), x) == want, name


def test_length_preserving_all_lengths():
    rng = random.Random(0)
    for n in range(1, 13):
        for _ in range(50):
            x = TASK.sample_x(rng, n)
            for f in range(TASK.n_funcs):
                assert len(TASK.apply(f, x)) == n


def test_name_shuffle_and_groups():
    assert sorted(TASK.id2name) == sorted(CFG["task"]["functions"])
    assert TASK.id2name != list(CFG["task"]["functions"]), "names should be shuffled"
    assert TASK.n_funcs == 24 and len(TASK.H) == len(TASK.L) == 12
    assert Task(CFG["task"]).id2name == TASK.id2name, "shuffle must be deterministic"


def test_format_parse_reward():
    p = TASK.make_problem([by_name("shift_p1"), by_name("reverse"), by_name("shift_p1")], "abcde")
    prompt, target = TASK.format_prompt(p), TASK.format_target(p)
    f = [TASK.fname(i) for i in p.funcs]
    assert prompt == f"Q:{f[2]}({f[1]}({f[0]}(abcde)))=\n"
    assert target == f"{f[0]}(abcde)=bcdef;{f[1]}(bcdef)=fedcb;{f[2]}(fedcb)=gfedc;\nA:gfedc<eos>"
    assert TASK.parse_answer(target) == "gfedc"
    assert TASK.reward(target, p) == 1.0
    assert TASK.reward(target + "<pad><pad>", p) == 1.0
    assert TASK.reward(target.replace("<eos>", ""), p) == 0.0          # no eos
    assert TASK.reward(target.replace("gfedc<eos>", "gfedd<eos>"), p) == 0.0
    assert TASK.reward("A:gfedc\nA:gfedc<eos>", p) == 0.0             # two answers
    assert TASK.reward("A:gfedc<eos>", p) == 1.0                       # no format credit, answer only
    assert TASK.reward(target, p, "partial") == 1.0
    wrong_mid = target.replace("=fedcb;", "=fedca;", 1)
    assert abs(TASK.reward(wrong_mid, p, "partial") - 2 / 3) < 1e-9


def test_pid_and_group():
    p1 = TASK.make_problem([0, 1], "abcde")
    p2 = TASK.make_problem([1, 0], "abcde")
    assert p1.pid != p2.pid and p1 == TASK.make_problem([0, 1], "abcde")
    assert TASK.group_of(TASK.H[:3]) == "H" and TASK.group_of(TASK.L[:2]) == "L"
    assert TASK.group_of((TASK.H[0], TASK.L[0])) == "mixed"


def test_tokenizer():
    tok = CharTokenizer()
    assert tok.vocab_size == 48
    p = TASK.sample_problem(TASK.max_depth, random.Random(1), str_len=max(TASK.str_lens))
    ids, labels = tok.encode_example(TASK.format_prompt(p), TASK.format_target(p))
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    n_prompt = len(tok.encode(TASK.format_prompt(p), add_bos=True))
    assert labels[:n_prompt] == [IGNORE_INDEX] * n_prompt and labels[n_prompt:] == ids[n_prompt:]
    assert tok.decode(ids) == "<bos>" + TASK.format_prompt(p) + TASK.format_target(p)
    assert len(ids) <= CFG["model"]["max_len"] - CFG["pretrain"]["pos_offset_max"]


def test_splits():
    rng = random.Random(0)
    assert len(SPLITS.heldout_pairs) == 4 * CFG["splits"]["heldout_pairs_per_group_pair"] == 24
    for pool in ("pretrain", "rl", "eval"):
        for d in (1, 2, 4):
            p = SPLITS.sample(pool, d, rng)
            assert SPLITS.pool_of(p.pid) == pool and not SPLITS.has_heldout_pair(p.funcs)
    for g in ("H", "L", "mixed", None):
        p = SPLITS.sample("eval", 3, rng, group=g, novel=True)
        assert SPLITS.has_heldout_pair(p.funcs) and (g is None or p.group == g)
    # bucket proportions roughly match config
    pools = [SPLITS.pool_of(TASK.sample_problem(2, rng).pid) for _ in range(20000)]
    assert abs(pools.count("rl") / len(pools) - 0.05) < 0.01


def test_difficulty():
    assert p_bin(0.0) == "0-0.03" and p_bin(0.5) == "0.4-0.6" and p_bin(1.0) == "0.97-1"
    lo, hi = wilson(16, 32)
    assert lo < 0.5 < hi and hi - lo < 0.36
    assert abs(float(p_signal(0.5, 8)) - (1 - 2 * 0.5 ** 8)) < 1e-12


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
