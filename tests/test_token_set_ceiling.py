"""``scripts/token_set_ceiling.py``: the token-set share of VERIFIED explanations (#5144)."""

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
_SPEC = importlib.util.spec_from_file_location(
    "token_set_ceiling", Path(__file__).parents[1] / "scripts" / "token_set_ceiling.py"
)
tsc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tsc)


def _pole(slug, label, model="EleutherAI/pythia-70m", band="early", token_r2=0.5):
    return {"slug": slug, "label": label, "model": model, "band": band, "token_r2": token_r2}


def test_kappa_is_one_for_agreement_and_zero_at_chance():
    assert tsc.kappa(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert tsc.kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    assert tsc.kappa(["a", "a"], ["a", "a"]) == 1.0  # one label everywhere: agreement is full
    assert tsc.kappa(["a", "a", "b", "b"], ["b", "b", "a", "a"]) == -1.0


def test_depth_bands_split_the_model_in_thirds():
    assert [tsc.depth_band(d) for d in (0.0, 0.33, 0.34, 0.66, 0.67, 1.0)] == [
        "early",
        "early",
        "middle",
        "middle",
        "late",
        "late",
    ]


def test_a_share_row_counts_each_label_and_gives_an_interval():
    poles = [_pole("a", "token-set"), _pole("b", "token-set"), _pole("c", "mixed"),
             _pole("d", "context")]  # fmt: skip
    row = tsc.share_row(poles, model="m", depth="all")
    assert (row["poles"], row["token-set"], row["mixed"], row["context"]) == (4, 2, 1, 1)
    assert row["token_set_share"] == 0.5 and row["model"] == "m" and row["depth"] == "all"
    assert row["ci95"][0] < 0.5 < row["ci95"][1]
    empty = tsc.share_row([], model="m", depth="late")
    assert empty["token_set_share"] is None and empty["ci95"] is None


def test_token_r2_separates_token_set_rules_when_it_should():
    poles = [_pole(str(i), "token-set", token_r2=0.8 + i / 100) for i in range(5)]
    poles += [_pole(str(i + 5), "context", token_r2=0.1 + i / 100) for i in range(5)]
    row = tsc.token_r2_test(poles, boot=50)
    assert row["auroc"] == 1.0 and row["ci95"] == [1.0, 1.0] and row["poles"] == 10
    assert row["median_token_set"] > row["median_other"]


def test_labels_are_read_from_every_matching_file(tmp_path):
    (tmp_path / "labels-0.json").write_text(
        json.dumps([{"slug": "a", "label": "mixed", "reason": "r"}])
    )
    (tmp_path / "labels-1.json").write_text(
        json.dumps([{"slug": "b", "label": "context", "reason": "s"}])
    )
    (tmp_path / "second-labels-0.json").write_text(
        json.dumps([{"slug": "a", "label": "x", "reason": "t"}])
    )
    assert tsc.read_labels(tmp_path, "labels-*.json") == {
        "a": {"label": "mixed", "reason": "r"},
        "b": {"label": "context", "reason": "s"},
    }
    assert set(tsc.read_labels(tmp_path, "second-labels-*.json")) == {"a"}


def test_the_rows_hold_each_pole_the_shares_the_agreement_and_the_check():
    gpt2 = {"model": "openai-community/gpt2", "band": "late"}
    poles = [_pole("a", "token-set", token_r2=0.9), _pole("b", "context", token_r2=0.1),
             _pole("c", "token-set", token_r2=0.7, **gpt2),
             _pole("d", "mixed", token_r2=0.2, **gpt2)]  # fmt: skip
    second = {"a": {"label": "token-set"}, "b": {"label": "mixed"}}
    rows = tsc.result_rows(poles, second)
    assert [r["slug"] for r in rows if r["table"] == "pole"] == ["a", "b", "c", "d"]
    shares = {(r["model"], r["depth"]): r for r in rows if r["table"] == "share"}
    assert shares[("all", "all")]["token_set_share"] == 0.5
    assert shares[("openai-community/gpt2", "late")]["poles"] == 2
    assert shares[("Qwen/Qwen2.5-0.5B", "all")]["poles"] == 0
    (agree,) = [r for r in rows if r["table"] == "agreement"]
    assert (agree["poles"], agree["agree"], agree["kappa_token_set"]) == (2, 0.5, 1.0)
    (check,) = [r for r in rows if r["table"] == "token_r2"]
    assert check["auroc"] == 1.0
