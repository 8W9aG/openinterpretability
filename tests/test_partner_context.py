"""The proposer sees what the registry knows about a pole's partners (#3240)."""

from types import SimpleNamespace

import torch

from openinterpretability import partner_context as pc
from openinterpretability import sisters
from openinterpretability.core.fence import FENCE_TAG
from openinterpretability.core.partners import CorrelatedNeuron
from openinterpretability.harness.covariance import PartnerSearch
from tests.test_covariance import _planted

UNIT = SimpleNamespace(kind="mlp_neuron_pole", model_name="openai-community/gpt2", layer=3)


def _partner(idx, pole="positive", tokens=()):
    return CorrelatedNeuron(neuron_idx=idx, pole=pole, coactivation_lift=4.0,
                            coactivation_lift_lower=3.0, coactive_tokens=40, spearman=0.3,
                            pearson=0.2, top_tokens=list(tokens))  # fmt: skip


PARTNERS = [
    _partner(812),
    _partner(9, "negative", [" the", " a"]),
    _partner(5),
    _partner(7, tokens=[" x"]),
]


def test_studied_partners_get_their_standing_and_unstudied_ones_their_tokens(monkeypatch):
    slugs = pc.partner_slugs(UNIT, PARTNERS)
    verified, rejected = list(slugs)[0], list(slugs)[2]
    found = {verified: ("VERIFIED", "fires on dates"), rejected: ("REJECTED", "fires on years")}
    monkeypatch.setattr(sisters, "_records_of", lambda wiki, wanted: [])
    monkeypatch.setattr(sisters, "pole_findings", lambda records, errata: found)
    text = pc.partner_context(UNIT, PARTNERS, pc.Path("."))
    assert "- N812+: VERIFIED: fires on dates" in text
    assert "- N5+: REJECTED: fires on years" in text
    assert "- N9-: not studied; top tokens: ' the', ' a'" in text
    assert "- N7+: not studied; top tokens: ' x'" in text
    assert f"<{FENCE_TAG}-" in text  # other agents' text and corpus tokens are fenced
    assert text.index("N812+") < text.index("N9-") < text.index("N5+")  # the scan's order


def test_with_no_wiki_every_partner_is_unstudied():
    text = pc.partner_context(UNIT, PARTNERS, None)
    assert "VERIFIED" not in text.split("\n", 2)[2] and "N9-: not studied" in text
    assert pc.partner_context(UNIT, [_partner(812)], None) == ""  # no tokens: nothing to say
    head = SimpleNamespace(kind="attn_head", model_name="m", layer=1)
    assert pc.partner_context(head, PARTNERS, None) == ""


def test_the_scan_gives_each_partner_its_top_tokens():
    target, acts = _planted()
    ids = torch.arange(target.numel()) % 50
    search = PartnerSearch(8, reservoir=4096, top_k=1024)
    for chunk in torch.split(torch.arange(target.numel()), 500):
        search.update(target[chunk], acts[chunk], ids[chunk])
    found = search.partners(0, 0.25 * float(target.max()), k=4, decode=lambda t: f"t{t[0]}")
    assert all(1 <= len(p.top_tokens) <= 5 for p in found)
    assert all(len(set(p.top_tokens)) == len(p.top_tokens) for p in found)  # distinct
    assert all(t.startswith("t") for p in found for t in p.top_tokens)
    # No ids or no tokenizer: no tokens, as in older evidence.
    plain = PartnerSearch(8, reservoir=4096, top_k=1024)
    for chunk in torch.split(torch.arange(target.numel()), 500):
        plain.update(target[chunk], acts[chunk])
    assert all(p.top_tokens == [] for p in plain.partners(0, 1.0, k=4, decode=str))
    assert all(p.top_tokens == [] for p in search.partners(0, 1.0, k=4))


def test_the_proposer_context_starts_with_the_partners(tmp_path):
    import openinterpretability.cli  # noqa: F401  (cli_evidence is loaded through it)
    from openinterpretability.cli_evidence import proposer_context

    evidence = SimpleNamespace(identifier=UNIT, correlated_neurons=PARTNERS)
    text = proposer_context(evidence, tmp_path / "no-wiki", sisters=False)
    assert text.startswith("\n" + pc.INTRO) and "N9-: not studied" in text
