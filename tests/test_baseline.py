import json

import pytest
from click.testing import CliRunner

from openinterp.baseline import (
    BaselineRow,
    blind_explanation,
    blind_reading,
    report,
    rescored,
    sign_test,
)
from openinterp.cli import cli
from openinterp.core.explanation import ExplanationScores, ScoredWindow
from openinterp.core.patchscope import PatchscopeReading, PatchscopeReadout
from openinterp.core.schemas import NeuronIdentifier
from openinterp.harness.explanation_scores import finalize_scores


def readout(texts: dict[tuple[str, int, float], str]) -> PatchscopeReadout:
    return PatchscopeReadout(
        prompts={"category": "{x}:", "meaning": "{x} is"},
        placeholder="?",
        max_new_tokens=12,
        control_seed=0,
        readings=[
            PatchscopeReading(prompt=p, layer=layer, scale=s, text=t, control_text="animal")
            for (p, layer, s), t in texts.items()
        ],
    )


def test_the_rule_reads_the_category_prompt_at_the_lowest_layer_at_twice_the_norm():
    r = readout(
        {
            ("meaning", 2, 2.0): "a plant",
            ("category", 14, 2.0): "person",
            ("category", 2, 1.0): "football",
            ("category", 2, 2.0): "crop",
        }
    )
    assert blind_reading(r).text == "crop"
    assert blind_explanation(r) == 'Fires on tokens related to "crop".'
    assert blind_explanation(readout({("category", 2, 2.0): ""})) is None
    assert blind_reading(readout({("meaning", 2, 2.0): "x"})) is None


def scores(judged=(True, False, True, False), fuzz=(True, False)) -> ExplanationScores:
    labels = (True, True, False, False)
    window = dict(text="t", tokens=["t"], activations=[1.0])
    return finalize_scores(
        ExplanationScores(
            slug="toy/L1/N2-positive",
            explanation="agent",
            threshold=0.5,
            window_tokens=4,
            quantiles=1,
            quantile_edges=[],
            seed=0,
            corpus="c",
            candidate_windows=4,
            active_windows=2,
            judge_model="external/judge",
            embedding_model="enc",
            detection_windows=[
                ScoredWindow(**window, quantile=0 if y else None, label=y, judged=j, similarity=s)
                for y, j, s in zip(labels, judged, (0.9, 0.2, 0.5, 0.1), strict=True)
            ],
            fuzzing_windows=[
                ScoredWindow(**window, label=y, judged=j)
                for y, j in zip((True, False), fuzz, strict=True)
            ],
        )
    )


def test_rescoring_with_the_agents_own_answers_reproduces_its_scores():
    agent = scores()
    same = rescored(
        agent,
        "agent",
        judge_model="external/judge",
        detection=[w.judged for w in agent.detection_windows],
        fuzzing=[w.judged for w in agent.fuzzing_windows],
        similarities=[w.similarity for w in agent.detection_windows],
    )
    assert same == agent
    other = rescored(agent, "baseline", detection=[True] * 4, similarities=[0.0, 0, 1, 1])
    assert other.detection.balanced_accuracy == 0.5 and other.embedding_auroc == 0.0
    assert other.fuzzing is None  # unjudged, not scored as wrong
    with pytest.raises(ValueError, match="one detection answer per window"):
        rescored(agent, "x", detection=[True])


def test_sign_test():
    assert sign_test([]) is None and sign_test([0.0, 0.0]) is None
    assert sign_test([0.1, 0.2, 0.3]) == pytest.approx(0.25)  # 2 * (1/8)
    assert sign_test([0.1, -0.1, 0.0]) == 1.0


def row(dev=False, agent=0.8, baseline=0.6, selection="random"):
    return BaselineRow(
        experiment_id="20261001T000000Z",
        slug="s",
        status="VERIFIED",
        selection=selection,
        dev=dev,
        reading="crop",
        agent_explanation="a",
        baseline_explanation="b",
        judge_model="j",
        agent={"detection": agent, "fuzzing": None, "embedding": agent},
        baseline={"detection": baseline, "fuzzing": None, "embedding": None},
    )


def test_report_pairs_only_metrics_both_explanations_have():
    md = report([row(), row(agent=0.4, baseline=0.8), row(dev=True, selection="manual")])
    assert "| Held-out experiments | detection | 2 | 0.60 | 0.70 | +0.10 | 50% | 1 |" in md
    assert "| Dev set (inspected while choosing the rule) | detection | 1 |" in md
    assert "| embedding |" not in md and "| fuzzing |" not in md  # unpaired metrics
    assert "1 distinct readings over 3 experiments" in md


@pytest.fixture
def archive(tmp_path, record, toy_runner, monkeypatch):
    from openinterp.builder.wiki import archive_record

    ident = NeuronIdentifier(model_name="toy", layer=1, neuron_idx=2, pole="positive")
    sub = record.submission.model_copy(update={"identifier": ident})
    verdict = record.verdict.model_copy(update={"explanation_scores": scores()})
    archive_record(record.model_copy(update={"submission": sub, "verdict": verdict}), tmp_path)
    monkeypatch.setattr("openinterp.cli._load_runner", lambda *_a, **_k: toy_runner)
    monkeypatch.setattr(
        "openinterp.harness.explanation_scores.embedding_similarities",
        lambda _model, query, texts: [float(len(query) % 2)] * len(texts),
    )
    return tmp_path


def test_command_writes_judge_prompts_then_scores_their_answers(archive):
    work, out = archive / "work", archive / "out" / "baseline"
    args = ["patchscope-baseline", "--wiki-dir", archive, "--work-dir", work, "--out", out,
            "--judge-model", "external/judge"]  # fmt: skip
    first = CliRunner().invoke(cli, [str(a) for a in args])
    assert first.exit_code == 0, first.output
    assert "1 experiments await judge answers" in first.output
    (task,) = (work / "judge").rglob("*.task.json")
    prompt = task.with_name(task.name.replace(".task.json", ".detection.prompt.md"))
    assert 'Fires on tokens related to "' in prompt.read_text()
    assert list((work / "readouts").rglob("*.json"))  # read once, then cached
    (pending,) = [json.loads(line) for line in out.with_suffix(".jsonl").read_text().splitlines()]
    assert pending["baseline"]["detection"] is None and pending["agent"]["detection"] == 0.5

    answer = task.with_name(task.name.replace(".task.json", ".detection.json"))
    answer.write_text('{"active": [1, 2]}')
    answer.with_name(answer.name.replace("detection", "fuzzing")).write_text('{"correct": []}')
    second = CliRunner().invoke(cli, [str(a) for a in args])
    assert second.exit_code == 0, second.output and "await" not in second.output
    (done,) = [json.loads(line) for line in out.with_suffix(".jsonl").read_text().splitlines()]
    assert done["baseline"]["detection"] == 1.0  # both active windows, neither inactive one
    assert done["baseline"]["fuzzing"] == 0.5 and done["baseline_judge_model"] == "external/judge"
    assert "| Held-out experiments | detection | 1 | 0.50 | 1.00 | +0.50 | 100% |" in (
        out.with_suffix(".md").read_text()
    )
