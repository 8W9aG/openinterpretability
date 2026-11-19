"""Answer experiments: the claim, its measures, its verdict and its page (#4778)."""

from datetime import UTC, datetime

import pytest

from openinterpretability import explain as ex
from openinterpretability.core import answer_records as ar

TEXTS = ["a b c d e f g h", "h g f e d c b a", "a c e g b d f h"] * 20


def _claim(**kw) -> ar.AnswerClaim:
    base = {
        "model_name": "EleutherAI/pythia-70m", "prompt": "abcd", "answer": "e",
        "sources": ["b"], "route": ["L0/H1", "L1/N3+"], "explanation": "b moves to the end.",
        "corrupted": "axcd", "submitting_agent_model": "claude-opus-5-5",
        "counterfactuals": [ar.Counterfactual(prompt=p, expect="kept") for p in "wxyz"],
    }  # fmt: skip
    return ar.AnswerClaim(**(base | kw))


def _probes() -> ar.AnswerProbes:
    prompts = [ar.Counterfactual(prompt=f"q{i}", expect="changed") for i in range(10)]
    return ar.AnswerProbes(prompts=prompts, corrupted=["aycd", "azcd"],
                           reviewer_agent_model="claude-sonnet-5-5")  # fmt: skip


def _measures(**kw) -> ar.AnswerMeasures:
    base = {
        "clean": -1.0, "all_cut": 4.0, "route_cut": 3.0, "null_cuts": [0.1, 0.2],
        "sufficiency": 0.7, "corrupted_gap": 2.0, "patching": 0.8, "null_patching": [0.0, 0.05],
        "controls": {"w": True, "x": True}, "probes": {f"q{i}": i < 9 for i in range(10)},
        "reviewer_patching": {"aycd": 0.7, "azcd": 0.6},
        "reviewer_null_patching": {"aycd": [0.0, 0.02], "azcd": [0.01, 0.0]}, "null_seed": 7,
    }  # fmt: skip
    return ar.AnswerMeasures(**(base | kw))


def _record(measures=None, when=datetime(2026, 11, 18, tzinfo=UTC), **claim):
    measures = measures or _measures()
    policy = ar.AnswerPolicy()
    status, gates = ar.decide(measures, policy)
    verdict = ar.AnswerVerdict(
        status=status, gates=gates, policy=policy, measures=measures,
        reviewer_agent_model="claude-sonnet-5-5", timestamp=when,
    )  # fmt: skip
    return ar.AnswerRecord(claim=_claim(**claim), probes=_probes(), verdict=verdict)


def test_unit_labels_parse_heads_and_both_poles():
    assert ar.parse_unit("L2/H6") == (2, 6, "head")
    assert ar.parse_unit("L3/N1920+") == (3, 1920, "positive")
    assert ar.parse_unit("L3/N7−") == ar.parse_unit("L3/N7-") == (3, 7, "negative")
    with pytest.raises(ValueError):
        ar.parse_unit("L3/N7")


def test_every_gate_passes_for_verified():
    status, gates = ar.decide(_measures(), ar.AnswerPolicy())
    assert status == "VERIFIED" and all(gates.values())


def test_a_route_that_carries_the_answer_but_misses_prompts_needs_revision():
    probes = {f"q{i}": i < 5 for i in range(10)}  # half come out as predicted
    status, gates = ar.decide(_measures(probes=probes), ar.AnswerPolicy())
    assert status == "NEEDS_REVISION" and gates == {**gates, "probes": False}


@pytest.mark.parametrize(
    ("change", "gate"),
    [
        ({"route_cut": 1.0}, "necessity"),  # a quarter of the cut from every unit
        ({"null_cuts": [1.5, 1.5]}, "beats_null"),  # random sets cut half as much
        ({"sufficiency": None}, "sufficiency"),
        ({"patching": 0.2}, "patching"),
        ({"null_patching": [None, None]}, "patching_null"),  # no random set measured
        ({"all_cut": -1.0}, "necessity"),  # ablating every unit raises the answer
    ],
)
def test_a_failed_causal_gate_rejects(change, gate):
    status, gates = ar.decide(_measures(**change), ar.AnswerPolicy())
    assert status == "REJECTED" and not gates[gate]


def test_attempts_stop_at_a_verified_claim_or_the_cap():
    def rejected(attempt=1):
        return _record(_measures(patching=0.0), attempt=attempt)

    assert ar.next_attempt([]) == 1
    assert ar.next_attempt([rejected(1), rejected(2)]) == 3
    assert ar.next_attempt([rejected(1)] * 2) == 2  # a re-measure is not an attempt (#4917)
    assert ar.next_attempt([rejected(i + 1) for i in range(ar.MAX_ATTEMPTS)]) is None
    assert ar.next_attempt([rejected(), _record(attempt=2)]) is None


def test_check_rederives_each_verdict_and_its_path(tmp_path):
    from openinterpretability.checks import CheckReport

    good = _record()
    path = ar.record_path(tmp_path, good)
    assert path.parent.parent.name == "EleutherAI--pythia-70m"
    assert path.parent.name == good.claim.prompt_id and path.name == "20261118T000000Z.json"
    forged = good.model_copy(update={"verdict": good.verdict.model_copy(
        update={"measures": _measures(patching=0.0)})})  # fmt: skip
    path.parent.mkdir(parents=True)
    path.write_text(forged.model_dump_json())
    (path.parent / "elsewhere.json").write_text(good.model_dump_json())
    report = CheckReport()
    ar.check_answers(tmp_path, report)
    found = " ".join(str(p) for p in report.problems)
    assert "the measures give REJECTED, not VERIFIED" in found and "should be at" in found


def test_answers_is_a_derived_record_kind():
    from openinterpretability.core.record_kinds import DERIVED, RECORD_KINDS

    assert any(k.prefix == "wiki/answers/" and k.check == DERIVED for k in RECORD_KINDS)


def test_the_reviewer_cannot_read_answer_records(tmp_path):
    from openinterpretability.core.isolation import blocked_file

    root = tmp_path.resolve()
    assert blocked_file(root / "wiki" / "answers" / "m" / "p" / "x.json", root)
    assert not blocked_file(root / "exchange" / "answer-p" / "answer-1.probes.json", root)


@pytest.fixture
def gpt2(tiny_gpt2):
    from openinterpretability.harness.hooks import NeuronHookRunner
    from tests.conftest import FakeTokenizer

    return NeuronHookRunner(tiny_gpt2.double(), FakeTokenizer())


@pytest.fixture
def means(gpt2, tmp_path, monkeypatch):
    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    return ex.unit_means(gpt2, "tiny", TEXTS, 200)


def test_review_measures_a_route_on_a_real_architecture(gpt2, means):
    from openinterpretability.answer_review import every_unit, patch_hooks, review

    record = review(gpt2, means, _claim(), _probes(), ar.AnswerPolicy(null_sets=3))
    m = record.verdict.measures
    assert len(m.null_cuts) == 3 and len(m.null_patching) == 3
    assert set(m.controls) == set("wxyz") and len(m.probes) == 10
    assert record.verdict.status in ("VERIFIED", "NEEDS_REVISION", "REJECTED")
    assert ar.decide(m, record.verdict.policy) == (record.verdict.status, record.verdict.gates)
    # The route on every prompt (#4917): the original, the 4 controls and the 10 probes.
    assert set(m.readings) == {"abcd", *"wxyz", *(f"q{i}" for i in range(10))}
    original = m.readings["abcd"]
    assert original.tokens and set(original.strength) == {"L0/H1", "L1/N3+"}
    assert len(original.strength["L0/H1"]) == len(original.tokens) == 4
    assert original.strength["L1/N3+"][0] == 0.0 and original.patched_log_prob is None
    assert all(min(v) >= 0 for v in original.strength.values())
    assert {e["source"] for e in original.edges} <= {"L0/H1", "L1/N3+"}
    one = m.readings["w"]  # one token: nothing past the first to attribute
    assert set(one.attribution.values()) == {0.0} and one.edges == []
    assert len(every_unit(gpt2, means)) == 2 * (2 * 128 + 2)  # 2 layers, 128 neurons, 2 heads
    # Patching every unit from the prompt itself changes nothing.
    from openinterpretability.explain_coverage import _log_prob

    hooks = patch_hooks(gpt2, "abcd", every_unit(gpt2, means))
    assert _log_prob(gpt2, "abcd", 5, -1, hooks) == pytest.approx(_log_prob(gpt2, "abcd", 5, -1))


def test_review_refuses_a_corrupted_prompt_of_another_length(gpt2, means):
    from openinterpretability.answer_review import review

    with pytest.raises(ValueError, match="the corrupted prompt is 3 tokens"):
        review(gpt2, means, _claim(corrupted="abc"), _probes())


def test_the_plan_lists_the_next_attempt_until_the_cap(tmp_path):
    from openinterpretability.explain_plan import answer_lines, answer_step

    args = (tmp_path, "EleutherAI/pythia-70m", "abcd", -1, "e")
    assert "explain the units first" in answer_step(*args, {})["reason"]
    step = answer_step(*args, {"minimal_units": [(0, 1, "head")]})
    assert step["status"] == "unstudied" and "attempt 1 of at most 5" in step["reason"]
    assert any("answers review " in c and "answer-1.probes.json" in c for c in step["commands"])
    for i in range(ar.MAX_ATTEMPTS):
        r = _record(_measures(patching=0.0), datetime(2026, 11, 18, i, tzinfo=UTC), attempt=i + 1)
        path = ar.record_path(tmp_path, r)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.model_dump_json())
    step = answer_step(*args, {"minimal_units": [(0, 1, "head")]})
    assert step["commands"] == [] and "5 claims, none VERIFIED" in step["reason"]
    assert answer_lines(step)[0].startswith("Answer experiment: REJECTED, 5 claims")


def test_the_prompts_show_earlier_claims_to_the_proposer_and_no_measure_to_the_reviewer():
    from openinterpretability.answer_prompts import proposer_prompt, reviewer_prompt

    failed = _record(_measures(patching=0.0))
    text = proposer_prompt("c.json", "EleutherAI/pythia-70m", "abcd", -1, "e", None, [failed], 2)
    assert "attempt 2 of at most 5" in text and "Failed gates: patching" in text
    assert f'"supersedes": "{failed.id}"' in text
    review = reviewer_prompt("p.probes.json", failed.claim)
    assert "L0/H1 → L1/N3+" in review and "patching" not in review and "0.8" not in review


def test_the_page_leads_with_the_answer_verdict(tmp_path, record):
    from openinterpretability.builder.compiler import compile_site
    from openinterpretability.builder.explanation_pages import load_views
    from tests.test_explanation_records import _record as explanation
    from tests.test_explanation_records import _write

    e = explanation(datetime(2026, 11, 17, tzinfo=UTC))
    _write(tmp_path / "wiki", e)
    (prompt,) = load_views(tmp_path / "wiki", states={})
    compile_site([record], tmp_path / "site", explanations=[prompt])
    assert "no route claim tested" in (tmp_path / "site" / prompt.href).read_text()
    claim = {"model_name": e.model_name, "prompt": e.prompt, "position": e.position}
    for i, measures in enumerate([_measures(patching=0.0), _measures()]):
        r = _record(measures, datetime(2026, 11, 18, i, tzinfo=UTC), attempt=i + 1, **claim)
        path = ar.record_path(tmp_path / "wiki", r)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.model_dump_json())
    (prompt,) = load_views(tmp_path / "wiki", states={})
    assert prompt.understood and prompt.answer.record.claim.attempt == 2
    compile_site([record], tmp_path / "site", explanations=[prompt])
    for page in (prompt.href, prompt.newest.href):
        html = (tmp_path / "site" / page).read_text()
        assert "Answer understood: a route claim is VERIFIED" in html
        assert "L0/H1 → L1/N3+" in html and "REJECTED" in html and "patching" in html
    newest = (tmp_path / "site" / prompt.newest.href).read_text()
    assert "the VERIFIED route claim above does" in newest
    # Each answer experiment has its own page, linked from the explanation pages (#4909).
    for answer in prompt.answers:
        assert f'href="../../../../{answer.href}"' in newest or answer.href in newest
        page = (tmp_path / "site" / answer.href).read_text()
        assert "Counterfactual prompts" in page and "The reviewer's prompts" in page
        # Rule 1.1 (#4892): each gate's margin, the reviewer's corrupted prompts, the seed.
        assert ">Margin<" in page and " pp<" in page and "aycd" in page
        assert "The review drew the random sets' seed, 7" in page
        assert page.count("passed") + page.count("failed") >= 7  # every gate, in words
        slug = answer.record.claim.model_name.replace("/", "--")
        assert "L0/H1" in page and f"models/{slug}/L0/H1.html" in page
    verified = (tmp_path / "site" / prompt.answers[-1].href).read_text()
    assert "VERIFIED</strong>: every gate passed" in verified
    rejected = (tmp_path / "site" / prompt.answers[0].href).read_text()
    assert "REJECTED</strong>: failed patching" in rejected and "axcd" in rejected


def test_the_narrator_reads_the_answer_claim_and_may_cite_its_route(tmp_path):
    from types import SimpleNamespace

    from openinterpretability.explain_narrate import _answer, check_citations
    from openinterpretability.explain_plan import answer_step

    r = _record()
    path = ar.record_path(tmp_path, r)
    path.parent.mkdir(parents=True)
    path.write_text(r.model_dump_json())
    step = answer_step(tmp_path, r.claim.model_name, r.claim.prompt, -1, "e", {})
    told = _answer(step)
    assert told["status"] == "VERIFIED" and told["route"] == ["L0/H1", "L1/N3+"]
    assert told["claim"] != r.claim.explanation and r.claim.explanation in told["claim"]  # fenced
    assert _answer(None) == {"status": "unstudied", "attempts": 0}
    ev = {"units": [], "groups": [], "next_experiments": [], "paths": [],
          "answer_experiment": told}  # fmt: skip
    narrative = SimpleNamespace(paragraph="L1/N3+ writes it; L9/H9 too.", sections=[])
    assert check_citations(narrative, ev) == ["L9/H9"]
    ev["minimal_set"] = ["L9/H9"]  # a unit of the minimal set is in the evidence too
    assert check_citations(narrative, ev) == []


def test_the_page_draws_the_route_on_every_prompt(gpt2, means, tmp_path, record):
    """Each prompt's route measures and figure are on the answer page (#4905, #4917)."""
    from openinterpretability.answer_review import review
    from openinterpretability.builder.compiler import compile_site
    from openinterpretability.builder.explanation_pages import load_views
    from tests.test_explanation_records import _record as explanation
    from tests.test_explanation_records import _write

    e = explanation(datetime(2026, 11, 17, tzinfo=UTC))
    _write(tmp_path / "wiki", e)
    claim = _claim(model_name=e.model_name, prompt="abcd", position=e.position,
                   counterfactuals=[ar.Counterfactual(prompt=p, expect="kept")
                                    for p in ("abce", "abcf", "xbcd", "ab")])  # fmt: skip
    r = review(gpt2, means, claim, _probes(), ar.AnswerPolicy(null_sets=2))
    # Put it under the explanation's prompt, so the explanation pages list it.
    r = r.model_copy(update={"claim": r.claim.model_copy(update={"prompt": e.prompt})})
    readings = dict(r.verdict.measures.readings)
    readings[e.prompt] = readings.pop("abcd")  # the original is keyed by the claim's prompt
    measures = r.verdict.measures.model_copy(update={"readings": readings})
    r = r.model_copy(update={"verdict": r.verdict.model_copy(update={"measures": measures})})
    path = ar.record_path(tmp_path / "wiki", r)
    path.parent.mkdir(parents=True)
    path.write_text(r.model_dump_json())
    (prompt,) = load_views(tmp_path / "wiki", states={})
    compile_site([record], tmp_path / "site", explanations=[prompt])
    page = (tmp_path / "site" / prompt.answers[0].href).read_text()
    folder = tmp_path / "site" / prompt.answers[0].href.removesuffix(".html")
    figures = sorted(p.name for p in folder.glob("fig-route-*.svg"))
    assert len(figures) == 2 * len(readings)  # each prompt: a moving and a still copy
    assert "The route at work" in page and page.count("The route on this prompt") == 14
    assert "route share" in page and "patched in" in page  # "abce" has the original's length
    moving = (folder / "fig-route-0.svg").read_text()
    assert "@keyframes" in moving and "<rect" in moving  # it moves, and the head is a square
    assert "animation:none" in (folder / "fig-route-0-static.svg").read_text()
    # The 3D explorer: its data is on the page, and it starts by itself with no button.
    import json
    import re

    data = json.loads(re.search(r'id="route3d-data">(.*?)</script>', page, re.S)[1])
    assert len(data["prompts"]) == len(readings) and data["prompts"][0]["by"] == "original"
    assert {u["unit"] for u in data["units"]} == {"L0/H1", "L1/N3+"}
    assert "three@0.169.0" in page and 'type="importmap"' in page
    assert 'id="route3d-start"' not in page and "IntersectionObserver" in page


def test_the_lens_and_spectrum_follow_the_layers_on_shared_axes(gpt2, means):
    """Each prompt's last token through the layers (#4917): one row per layer, shared axes."""
    from openinterpretability.answer_review import review

    m = review(gpt2, means, _claim(), _probes(), ar.AnswerPolicy(null_sets=1)).verdict.measures
    first = m.readings["abcd"]
    assert len(first.lens) == len(first.spectrum) == 3  # the embedding and 2 layers
    names = set(first.lens[0])
    assert all(set(row) == names for r in m.readings.values() for row in r.lens)
    assert "e" in names  # the claim's answer is always a candidate
    assert all(0 <= v <= 1 for row in first.lens for v in row.values())
    width = len(first.spectrum[0][0])  # [layer][token][direction]
    assert all(len(layer) == len(r.tokens) for r in m.readings.values() for layer in r.spectrum)
    assert all(len(tok) == width for r in m.readings.values() for layer in r.spectrum
               for tok in layer)  # fmt: skip
    # Switching units off: the whole route on every prompt, each unit on the original.
    assert all(r.route_off and len(r.route_off.lens) == 3 for r in m.readings.values())
    assert set(first.unit_off) == {"L0/H1", "L1/N3+"} and not m.readings["w"].unit_off
    assert first.route_off.lens[-1] != first.lens[-1]  # switching the route off changes it


def test_with_the_corpus_the_record_holds_the_routes_space(gpt2, means):
    """#4934: the space where the route writes, its cloud, its arrows and each prompt's place."""
    from openinterpretability.answer_review import review

    r = review(gpt2, means, _claim(), _probes(), ar.AnswerPolicy(null_sets=1), texts=TEXTS * 4)
    space = r.verdict.measures.route_space
    assert space is not None and len(space.cloud) == 3 and len(space.cloud[0][0]) == 3
    assert set(space.arrows) == {"L0/H1", "L1/N3+"} and len(space.arrows["L0/H1"]) == 3
    first = r.verdict.measures.readings["abcd"]
    assert len(first.route_place) == 3 and len(first.route_place[0]) == len(first.tokens)
    plain = review(gpt2, means, _claim(), _probes(), ar.AnswerPolicy(null_sets=1))
    assert plain.verdict.measures.route_space is None  # no corpus, no route space
