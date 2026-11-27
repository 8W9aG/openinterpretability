"""The fitting harness of arms A–F (``scripts/fix_arms.py`` and its helpers, #7006)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fix_arms as fa  # noqa: E402
import fix_closed_form as fc  # noqa: E402
import fix_family as ff  # noqa: E402

from openinterpretability import cli_interference  # noqa: E402
from openinterpretability import explain as ex  # noqa: E402
from openinterpretability.interference import keep  # noqa: E402
from openinterpretability.patches.graph import Graph  # noqa: E402
from tests.conftest import VOCAB  # noqa: E402
from tests.test_interference import _runner  # noqa: E402

MODEL = "openai-community/gpt2"
LAYOUT = ff.Layout(copy_heads=((1, 0),), fact_layers=(0, 1))


@pytest.fixture(autouse=True)
def stores(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    monkeypatch.setattr(keep, "WINDOW", 16)  # the tiny GPT-2 holds 64 positions


@pytest.fixture
def runner():
    r = _runner(0)
    for p in r.model.parameters():
        p.requires_grad_(False)
    return r


@pytest.fixture
def family():
    train = [ff.Item(f"a{c} has", " x", " y") for c in "bcdefgh"]
    return ff.Family(train=train, heldout=[ff.Item("az has", " x", " y")],
                     retain=[ff.Item("q has", " q", " q"), ff.Item("r has", " r", " r")],
                     right_heldout=[ff.Item("s has", " s", " s")])  # fmt: skip


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.jsonl"
    texts = ["the cat sat on the mat and looked out", "hello there, general kenobi"] * 4
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    return path


def _batches():
    torch.manual_seed(1)
    return [torch.randint(1, VOCAB, (2, 16))]


def test_each_trained_arm_changes_only_its_own_weights(runner):
    graph = Graph(runner.model)
    assert fa.sites("A", graph, LAYOUT) == [fa.Site("blocks.1.attn.W_O", 0)]
    assert [s.target for s in fa.sites("B", graph, LAYOUT)] == ["blocks.0.mlp.W_out",
                                                                 "blocks.1.mlp.W_out"]  # fmt: skip
    assert len(fa.sites("C", graph, LAYOUT)) == 2  # every layer
    with pytest.raises(ValueError, match="not a trained arm"):
        fa.sites("E", graph, LAYOUT)


def test_the_random_null_has_about_b_s_parameters_and_misses_the_copy_heads(runner):
    graph = Graph(runner.model)
    b = sum(fa.site_params(graph, s, 1) for s in fa.sites("B", graph, LAYOUT))
    d = fa.sites("D", graph, LAYOUT)
    heads = [(int(s.target.split(".")[1]), s.head) for s in d]
    assert all(h is not None for _, h in heads) and not set(heads) & set(LAYOUT.copy_heads)
    # 2 MLPs of 160 parameters each; a head has 48: at most the 3 heads outside A's.
    assert b == 320 and len(d) == 3
    # Rank 4 doesn't change the ratio, and the seed fixes the draw.
    assert fa.sites("D", graph, LAYOUT, rank=4) == fa.sites("D", graph, LAYOUT, rank=4)


def test_a_head_s_change_leaves_every_other_head_alone(runner):
    graph = Graph(runner.model)
    site = fa.Site("blocks.1.attn.W_O", 1)
    mask = fa.head_mask(graph, site)
    assert mask.shape == (1, 32) and mask[0, :16].sum() == 0 and mask[0, 16:].sum() == 16
    adapters = fa.Adapters(graph, [site], rank=2)
    adapters.factors[0][1].data.normal_()  # move B off zero
    a, b = adapters.delta(0)
    assert torch.count_nonzero((b @ a)[:, :16]) == 0
    assert fa.site_params(graph, site, 2) == 2 * (32 + 16)


def test_adapters_start_as_the_identity_and_are_removed_on_exit(runner):
    graph = Graph(runner.model)
    ids = torch.tensor([[1, 2, 3, 4]])
    before = runner.model(input_ids=ids).logits
    adapters = fa.Adapters(graph, fa.sites("B", graph, LAYOUT), rank=1)
    with adapters:
        assert torch.equal(runner.model(input_ids=ids).logits, before)
        adapters.factors[0][1].data.fill_(1.0)
        assert not torch.equal(runner.model(input_ids=ids).logits, before)
    assert torch.equal(runner.model(input_ids=ids).logits, before)


@pytest.mark.parametrize("arm", ["A", "B", "C", "D"])
def test_a_trained_fit_lowers_its_loss(runner, family, arm):
    fit = fa.fit_trained(runner, arm, family, LAYOUT, lam=1.0, rank=1, steps=30)
    assert fit.losses[-1] < fit.losses[0]
    assert fit.name == f"{arm}-l1.0-r1"
    expected = [s.target for s in fa.sites(arm, Graph(runner.model), LAYOUT)]
    assert [t for t, _, _ in fit.ops] == expected


def test_the_retain_term_holds_the_retain_prompts_closer(runner, family):
    def kl(lam):
        fit = fa.fit_trained(runner, "C", family, LAYOUT, lam=lam, rank=4, steps=40)
        graph = Graph(runner.model)
        adapters = fa.Adapters(graph, fa.sites("C", graph, LAYOUT), rank=4)
        for i, (_, a, b) in enumerate(fit.ops):
            adapters.factors[i][0].data.copy_(a)
            adapters.factors[i][1].data.copy_(b)
        prompts = [i.prompt for i in family.retain]
        with torch.no_grad():
            base = torch.log_softmax(ff.last_logits(runner.model, runner.tokenizer, prompts), -1)
            with adapters:
                new = torch.log_softmax(ff.last_logits(runner.model, runner.tokenizer, prompts), -1)
        return float(torch.sum(base.exp() * (base - new), -1).mean())

    assert kl(10.0) < kl(0.0)


def test_the_loss_has_its_three_terms(runner, family):
    model, tok = runner.model, runner.tokenizer
    items = family.train[:2]
    ce = fa.fit_loss(model, tok, items, None, 0.0, beta=0.0)
    both = fa.fit_loss(model, tok, items, None, 0.0, beta=1.0)
    assert both > ce  # the unlikelihood of the copied unit is positive
    prompts = [i.prompt for i in family.retain]
    base = torch.log_softmax(ff.last_logits(model, tok, prompts), -1)
    # The model is unchanged, so the KL term is zero.
    assert torch.isclose(fa.fit_loss(model, tok, items, (prompts, base), 5.0, beta=0.0), ce)
    shifted = (prompts, torch.log_softmax(base + torch.randn_like(base), -1))
    assert fa.fit_loss(model, tok, items, shifted, 5.0, beta=0.0) > ce


def test_the_measures_read_each_split(runner, family):
    found = ff.evaluate(runner.model, runner.tokenizer, family)
    assert found["train"]["n"] == 7 and found["retain"]["n"] == 2
    assert 0 <= found["heldout"]["p_right"] <= 1
    assert ff.measures(runner.model, runner.tokenizer, []) == {
        "n": 0,
        "top1": None,
        "p_right": None,
    }


def test_the_closed_form_meets_its_targets_and_alphaedit_keeps_the_retain_keys():
    torch.manual_seed(0)
    keys, retain = torch.randn(8, 3), torch.randn(8, 2)
    residuals = torch.randn(4, 3)
    memit = fc.closed_form(keys, residuals, torch.eye(8), weight=1e-6)  # a small ridge
    assert torch.allclose(memit @ keys, residuals, atol=1e-4)
    p = fc.null_projector(retain)
    alpha = fc.closed_form(keys, residuals, torch.eye(8) * 1e-3, p)
    assert torch.allclose(alpha @ retain, torch.zeros(4, 2), atol=1e-4)
    assert not torch.allclose(alpha, torch.zeros_like(alpha))


@pytest.mark.parametrize("variant", ["memit", "alphaedit"])
def test_arm_f_fits_at_the_middle_fact_layer(runner, family, variant):
    fit = fc.fit_closed_form(runner, family, LAYOUT, _batches(), rank=1, variant=variant)
    ((target, a, b),) = fit.ops
    assert target == "blocks.1.mlp.W_out" and a.shape == (1, 128) and b.shape == (32, 1)
    assert fit.name == f"F-v{variant}-r1"
    full = fc.fit_closed_form(runner, family, LAYOUT, _batches(), rank=None, variant=variant)
    assert full.name == f"F-v{variant}-rfull" and 1 <= full.ops[0][1].shape[0] <= len(family.train)


def test_the_weight_of_the_second_moment_is_read_at_the_call(monkeypatch):
    keys, residuals = torch.eye(2)[:, :1], torch.tensor([[2.0]])
    monkeypatch.setattr(fc, "MOMENT", 1.0)
    assert torch.allclose(fc.closed_form(keys, residuals, torch.eye(2)), torch.tensor([[1.0, 0.0]]))


def test_the_target_changes_raise_the_right_unit_within_their_norm(runner, family):
    layer, items = 1, family.train[:3]
    deltas = fc.target_deltas(runner.model, runner.tokenizer, items, layer, steps=20)
    assert deltas.shape == (32, 3)
    assert torch.all(deltas.norm(dim=0) <= fc.V_NORM + 1e-4)


def test_a_draft_reads_back_and_starts_as_the_identity(runner, family, tmp_path):
    fit = fa.fit_trained(runner, "B", family, LAYOUT, lam=0.0, rank=1, steps=5)
    path = tmp_path / "d.json"
    path.write_text(json.dumps(ff.draft(fit, f"{MODEL}@abc", family)))
    patch, ops = ff.checked(path, runner.model)
    assert [op.type for _, op in ops] == ["lowrank_add", "lowrank_add"]
    assert patch.message["arm"] == "B" and patch.message["train"][0] == "ab has"


def test_a_run_writes_each_fit_with_its_report(runner, family, tmp_path, corpus, monkeypatch):
    monkeypatch.setattr(cli_interference, "_load_runner", lambda *a, **k: runner)
    run = fa.Run(runner, MODEL, family, tmp_path / "run", corpus, 64, tmp_path / "wiki", "cpu")
    rows = list(fa.fits(run, "BEF", LAYOUT, _batches(), lambdas=(0.0,), ranks=(1,), steps=3))
    assert [r["arm"] for r in rows] == ["B", "E", "F", "F", "F", "F"]
    assert [r["rank"] for r in rows[2:]] == [1, "full", 1, "full"]
    b, e, f = rows[0], rows[1], rows[2]
    assert Path(b["draft"]).is_file() and b["params"] == 2 * (32 + 128)
    assert b["interference"]["table"] == "summary" and "verdict" in b["interference"]
    assert (tmp_path / "run" / "reports" / "B-l0.0-r1.jsonl").is_file()
    assert e["draft"] is None and e["prefix"].startswith("ab has x. ")
    assert f["variant"] == "memit" and f["interference"]


def test_without_a_report_or_retain_prompts_a_fit_still_writes_its_row(runner, family, tmp_path):
    family.retain = []  # AlphaEdit then has no keys to keep, and solves as MEMIT does
    run = fa.Run(runner, MODEL, family, tmp_path / "run", tmp_path / "c.jsonl", 64,
                 tmp_path / "wiki", "cpu", with_report=False)  # fmt: skip
    fit = fc.fit_closed_form(runner, family, LAYOUT, _batches(), rank=None, variant="alphaedit")
    memit = fc.fit_closed_form(runner, family, LAYOUT, _batches(), rank=None, variant="memit")
    assert torch.allclose(fit.ops[0][2] @ fit.ops[0][1], memit.ops[0][2] @ memit.ops[0][1])
    row = fa.record(run, fit)
    assert "interference" not in row and row["retain"]["n"] == 0
    assert not (tmp_path / "run" / "reports").exists()
