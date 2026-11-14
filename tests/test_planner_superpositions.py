"""``superposition`` draws: the planner audits the definitions in data/superpositions/ (#3687)."""

import json
import shlex

from click.testing import CliRunner

from openinterp.audit_options import read_superposition
from openinterp.cli import cli
from openinterp.core.profiles import SuperpositionProfile, superposition_id
from openinterp.draw_steps import render, steps
from openinterp.planner import Plan, plan_next
from openinterp.planner_superpositions import SUPERPOSITION_WEIGHT, superposition_candidates
from tests.test_superpositions import TEXTS
from tests.test_wiring import qwen_runner  # noqa: F401  (fixture)

MEMBERS = [{"neuron_idx": 3, "weight": 2.0}, {"neuron_idx": 7, "weight": -1.0}]
PROFILE = SuperpositionProfile.model_validate({"members": MEMBERS, "method": "c", "source": "t"})
DEFINITION = {"model": "o/m", "layer": 1, **PROFILE.model_dump(mode="json")}
SID = superposition_id(PROFILE.members)
SLUG = f"o--m/L1/S{SID}"
KEY = SLUG.replace("/", "-")


def _root(tmp_path, definition=DEFINITION, name="o--m.jsonl"):
    folder = tmp_path / "data/superpositions"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(definition) + "\n")
    return tmp_path


def test_an_unstudied_definition_is_a_candidate(tmp_path):
    (c,) = superposition_candidates({}, [], _root(tmp_path))
    assert c.kind == "superposition" and c.model == "o/m" and c.layer == 1
    assert c.slug == SLUG and c.weight == SUPERPOSITION_WEIGHT
    assert c.commands[0].startswith(f"open-interpretability superpositions show {SID} --model o/m")
    assert c.audit == f"--superposition exchange/{KEY}/superposition.json"
    assert c.skills and c.record is None


def test_a_studied_or_claimed_definition_is_not(tmp_path):
    root = _root(tmp_path)
    assert superposition_candidates({SLUG: object()}, [], root) == []
    assert superposition_candidates({}, [f"experiment/{KEY}"], root) == []
    assert superposition_candidates({}, [], tmp_path / "none") == []  # no folder: no draws


def test_show_writes_the_definition_that_audit_reads(tmp_path, monkeypatch):
    monkeypatch.chdir(_root(tmp_path))
    args = ["superpositions", "show", SID, "--model", "o/m", "--out", "x/s.json"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    unit, model, layer, profile = read_superposition(tmp_path / "x/s.json", "neuron")
    assert (unit, model, layer) == ("superposition", "o/m", 1)
    assert superposition_id(profile.members) == SID
    unknown = CliRunner().invoke(cli, ["superpositions", "show", "0" * 12, *args[3:]])
    assert unknown.exit_code != 0 and f"has the id {'0' * 12}" in unknown.output


def _plan(tmp_path):
    (c,) = superposition_candidates({}, [], _root(tmp_path))
    return Plan(seed=1, choice=c, explored=False, candidates=[c])


def test_the_steps_show_then_audit_the_superposition(tmp_path):
    text = render(steps(_plan(tmp_path), corpus="c.jsonl"))
    assert f"git switch -c experiment/{KEY}\n" in text
    out = f"exchange/{KEY}"
    assert f"$ uv run open-interpretability superpositions show {SID} --model o/m " in text
    assert f"--out {out}/superposition.json" in text
    scan = f"audit --superposition {out}/superposition.json --corpus c.jsonl"
    assert f"{scan} --evidence-out {out}/ev.json" in text
    assert "neuron-proposer" in text and "neuron-reviewer" in text
    assert "--random" not in text and "--layer" not in text and "--neuron " not in text
    assert "open-interpretability show" not in text  # no history to read


def test_a_superposition_draw_records_a_manual_selection(qwen_runner, tmp_path, monkeypatch):  # noqa: F811
    """The draw's own commands scan a chosen unit, so random-unit statistics never count it."""
    from openinterp.core.schemas import ScanEvidence

    monkeypatch.setattr("openinterp.cli._load_runner", lambda *a, **k: qwen_runner)
    plan = _plan(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus.txt").write_text("\n".join(TEXTS))
    for step in steps(plan, corpus="corpus.txt")[1:3]:  # show, then scan
        args = shlex.split(step.commands[0])
        assert args[:3] == ["uv", "run", "open-interpretability"]
        result = CliRunner().invoke(cli, args[3:])
        assert result.exit_code == 0, result.output
    evidence = ScanEvidence.model_validate_json((tmp_path / f"exchange/{KEY}/ev.json").read_text())
    assert evidence.identifier.slug == SLUG
    assert evidence.provenance.selection == "manual"


def test_plan_next_draws_a_superposition(tmp_path):
    plan = plan_next(_root(tmp_path) / "wiki", seed=1, epsilon=0.0)
    assert plan.choice.kind == "superposition" and plan.choice.slug == SLUG


def test_a_definition_whose_scan_stopped_is_not_drawn_again(tmp_path):
    from datetime import UTC, datetime

    from openinterp.core.draws import ChosenStop, stop_path
    from openinterp.core.schemas import ExperimentProvenance, SuperpositionIdentifier

    root = _root(tmp_path)
    assert len(superposition_candidates({}, [], root)) == 1
    provenance = ExperimentProvenance(
        dtype="float32", corpus="c", corpus_sha256="0" * 64, n_snippets=1, tokens_scanned=1,
        selection="manual", harness_version="t",
    )  # fmt: skip
    ident = SuperpositionIdentifier(model_name="o/m", layer=1, superposition_id=SID)
    stop = ChosenStop(identifier=ident, outcome="dead", provenance=provenance,
                      timestamp=datetime(2026, 11, 14, tzinfo=UTC))  # fmt: skip
    path = stop_path(root / "wiki", stop)
    path.parent.mkdir(parents=True)
    path.write_text(stop.model_dump_json())
    assert superposition_candidates({}, [], root) == []  # dead: no audit can follow
