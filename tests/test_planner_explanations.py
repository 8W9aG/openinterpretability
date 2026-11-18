"""``explanation`` draws and the unit draws that explanations ask for (#3744 part C)."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

from openinterpretability import draw_steps
from openinterpretability.core.explanation_records import (
    ExplainedUnit,
    ExplanationRecord,
    record_path,
)
from openinterpretability.planner import Plan, claim_key
from openinterpretability.planner_explanations import (
    PROMPTS,
    explanation_candidates,
    explanation_slug,
)

MODEL = "openai-community/gpt2"
POLE = "openai-community--gpt2/L10/N2931-negative"
HEAD = "openai-community--gpt2/L11/H8"


def _root(tmp_path, prompts=("A prompt", "Another prompt")):
    (tmp_path / "data").mkdir()
    (tmp_path / PROMPTS).write_text("".join(json.dumps({"prompt": p}) + "\n" for p in prompts))
    (tmp_path / "pyproject.toml").write_text(
        '[tool.openinterpretability]\nmodels = ["openai-community/gpt2"]\n'
        '[tool.openinterpretability.window_tokens]\n"openai-community/gpt2" = 128\n'
    )
    return tmp_path


def _explained(root, prompt, status="unstudied"):
    record = ExplanationRecord(
        model_name=MODEL, prompt=prompt, position=-1, answer=" x", explained_share=0.0,
        blocked_share=0.0, prompt_coverage=0.0, random_coverage=0.0, package_version="0",
        written=datetime(2026, 11, 17, tzinfo=UTC),
        units=[ExplainedUnit(unit="L10/N2931-", slug=POLE, kind="pole", share=0.01,
                             attribution=1.0, status=status),
               ExplainedUnit(unit="L11/H8", slug=HEAD, kind="head", share=0.005,
                             attribution=0.5, status=status)],
    )  # fmt: skip
    path = record_path(root / "wiki", record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json())


def test_a_new_prompt_is_drawn_once_and_an_explained_one_is_not(tmp_path):
    root = _root(tmp_path)
    _explained(root, "A prompt")
    found = explanation_candidates({}, [], root, [MODEL], seed=1)
    (new,) = [c for c in found if c.kind == "explanation"]
    assert new.slug == explanation_slug(MODEL, "Another prompt")
    assert new.commands == [f"open-interpretability explain --model {MODEL} -- 'Another prompt'"]


def test_explained_units_that_are_unstudied_get_attribution_audits(tmp_path):
    root = _root(tmp_path)
    _explained(root, "A prompt")
    units = {
        c.slug: c for c in explanation_candidates({}, [], root, [MODEL], 1) if c.kind == "explore"
    }
    assert units[POLE].commands == [
        f"open-interpretability audit --model {MODEL} --layer 10 --neuron 2931 --pole negative "
        "--selection attribution --window-tokens 128"
    ]
    assert "--head 8" in units[HEAD].commands[0] and "needs it" in units[HEAD].rationale
    studied = {
        POLE: SimpleNamespace(
            record=SimpleNamespace(verdict=SimpleNamespace(status="REJECTED")),
            standing=("REJECTED", None),
        )
    }
    later = explanation_candidates(studied, [], root, [MODEL], 1)
    assert POLE not in {c.slug for c in later if c.kind == "explore"}  # follow-ups cover it now


def test_a_stale_explanation_is_drawn_again_and_claims_are_respected(tmp_path):
    root = _root(tmp_path, prompts=("A prompt",))
    _explained(root, "A prompt")
    verified = {
        POLE: SimpleNamespace(
            record=SimpleNamespace(verdict=SimpleNamespace(status="VERIFIED")),
            standing=("VERIFIED", None),
        )
    }
    found = [
        c for c in explanation_candidates(verified, [], root, [MODEL], 1) if c.kind == "explanation"
    ]
    assert len(found) == 1 and found[0].rationale.startswith("stale: 1 cited units")
    claim = [claim_key(found[0].slug)]
    assert not [c for c in explanation_candidates(verified, claim, root, [MODEL], 1)
                if c.kind == "explanation"]  # fmt: skip


def test_an_explanation_draw_explains_narrates_and_records(tmp_path):
    root = _root(tmp_path)
    (choice,) = [
        c for c in explanation_candidates({}, [], root, [MODEL], 1) if c.kind == "explanation"
    ]
    steps = draw_steps.steps(
        Plan(7, choice, False, [choice]), proposer="anthropic/p", fresh_corpus=True
    )
    commands = [cmd for s in steps for cmd in s.commands]
    explain = next(c for c in commands if " explain " in c)
    assert "--level long --narrator-model anthropic/p --record" in explain
    assert "--corpus exchange/fresh/corpus-7.jsonl" in explain
    assert any(c.startswith("git switch -c experiment/") and "explanations" in c for c in commands)
    external = draw_steps.steps(Plan(7, choice, False, [choice]))
    assert "external/<your model ID>" in " ".join(c for s in external for c in s.commands)


def test_the_loop_posts_an_explanation_draw_as_explained(monkeypatch):
    from openinterpretability import experiment_loop as el

    monkeypatch.setattr(
        el, "new_files", lambda prefix: ["x.json"] if "explanations" in str(prefix) else []
    )
    assert el.verdict_of("explanation", "k") == ("EXPLAINED", "Explain k")
    monkeypatch.setattr(el, "new_files", lambda prefix: [])
    try:
        el.verdict_of("explanation", "k")
    except el.LoopError as exc:
        assert "no explanation" in str(exc)
    else:
        raise AssertionError("an explanation draw that wrote nothing must fail")


def test_commands_quote_every_value_and_keep_the_prompt_last(tmp_path):
    import shlex

    root = _root(tmp_path, prompts=("-x 'quoted' $(stuff)",))
    (choice,) = [
        c for c in explanation_candidates({}, [], root, [MODEL], 1) if c.kind == "explanation"
    ]
    words = shlex.split(choice.commands[0])
    assert words[-2:] == ["--", "-x 'quoted' $(stuff)"]  # one argument, after the options
    steps = draw_steps.steps(Plan(7, choice, False, [choice]), proposer="anthropic/p")
    explain = next(c for s in steps for c in s.commands if " explain " in c)
    assert shlex.split(explain)[-2:] == ["--", "-x 'quoted' $(stuff)"]
    assert "--record --" in explain  # the draw's options come before the prompt
