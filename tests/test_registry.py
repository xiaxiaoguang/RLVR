"""Registry / parameter-registry tests. Run: python tests/test_registry.py (or pytest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlvr.atoms import ATOMS  # noqa: E402
from rlvr.config import load_config, task_and_splits  # noqa: E402
from rlvr.difficulty import WEIGHTS  # noqa: E402
from rlvr.params import PARAMS, PARAMS_DEFAULTS  # noqa: E402
from rlvr.registry import Registry  # noqa: E402
from rlvr.rewards import REWARDS  # noqa: E402


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    raise AssertionError(f"expected {exc.__name__}")


def test_registry_basics():
    r = Registry("thing")

    @r.register()
    def scaled(k: int = 1):
        return lambda x: k * x

    assert r.build("scaled")(3) == 3 and r.build({"name": "scaled", "k": 2})(3) == 6
    assert "did you mean" in raises(KeyError, r.get, "scaledd")
    assert "already registered" in raises(KeyError, r.add, "scaled", scaled)
    assert "scaled" in raises(TypeError, r.build, {"name": "scaled", "kk": 2})
    r.add("scaled", lambda: (lambda x: -x), override=True)
    assert r.build("scaled")(3) == -3


def test_parameterized_atoms():
    t = task_and_splits(load_config())[0]
    rot = ATOMS.build({"name": "rotate", "k": -2})
    assert rot([0, 1, 2, 3, 4], 10) == [3, 4, 0, 1, 2]
    assert ATOMS.build({"name": "shift_at", "k": 2, "parity": 1})([0, 0, 0, 0], 10) == [0, 2, 0, 2]
    assert t.apply(t.name2id["rotl2"], "abcde") == "cdeab"


def test_config_defaults_and_overrides():
    cfg = load_config()
    assert cfg["pretrain"]["n_comp"] == PARAMS.get("pretrain.n_comp").default
    cfg = load_config("configs/pretrain/weak.yaml", ["pretrain.h_ratio=0.5", "task.str_len=[4,5,6]"])
    assert cfg["pretrain"]["n_comp"] == 2000 and cfg["pretrain"]["h_ratio"] == 0.5
    assert cfg["task"]["str_len"] == [4, 5, 6]
    assert PARAMS_DEFAULTS["pretrain"]["n_comp"] == 20000, "defaults must not be mutated"


def test_config_validation():
    msg = raises(ValueError, load_config, None, ["pretrain.n_compp=3"])
    assert "unknown config key 'pretrain.n_compp'" in msg and "pretrain.n_comp" in msg
    assert "expected int" in raises(ValueError, load_config, None, ["rl.group_size=8.5"])
    assert "expected int" in raises(ValueError, load_config, None, ["rl.group_size=true"])
    assert "not in" in raises(ValueError, load_config, None, ["rl.reward=nope"])
    load_config(None, ["rl.reward={name: partial}", "rl.lr=1", "data.dataset=mixed_500"])


def test_swap_function_definition_via_config():
    # Replace one function (dict params are replaced whole, so restate the map).
    fns = dict(PARAMS_DEFAULTS["task"]["functions"])
    fns["max_to_front"] = "min_to_back"
    cfg = load_config(None)
    cfg["task"]["functions"] = fns
    task, _ = task_and_splits(cfg)
    assert task.apply(task.name2id["max_to_front"], "cajfb") == "cjfba"
    # Partial edit through a CLI override also works.
    cfg = load_config(None, ["task.functions.rotl2={name: rotate, k: 3}"])
    task, _ = task_and_splits(cfg)
    assert task.apply(task.name2id["rotl2"], "abcde") == "deabc"


def test_non_length_preserving_atom_rejected():
    ATOMS.add("drop_last", lambda: (lambda s, A: s[:-1]), override=True)
    cfg = load_config(None, ["task.functions.reverse=drop_last"])
    assert "length preserving" in raises(AssertionError, task_and_splits, cfg)


def test_rewards_and_weights():
    assert set(REWARDS.names()) >= {"binary", "partial"}
    w = WEIGHTS.build({"name": "signal", "G": 8})([0.0, 0.5, 1.0])
    assert w[0] == 0 and w[2] == 0 and 0.99 < w[1] < 1
    assert list(WEIGHTS.build("learnability")([0.5])) == [0.25]


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
