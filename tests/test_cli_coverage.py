"""The coverage command (cli_coverage.py)."""

import json
from types import SimpleNamespace

from click.testing import CliRunner

from openinterpretability.builder.wiki import write_experiment_page
from openinterpretability.cli import cli
from tests.test_circuits import at


def test_coverage_appends_a_row_per_layer(tmp_path, monkeypatch, record, toy_runner):
    toy_runner.model.config = SimpleNamespace()
    loads = []
    monkeypatch.setattr(
        "openinterpretability.cli._load_runner",
        lambda name, device, dtype: loads.append((name, dtype)) or toy_runner,
    )
    verified = at(record, 1, 2)  # L1/N2, VERIFIED
    ident = verified.submission.identifier.model_copy(update={"model_name": "toy"})
    verified = verified.model_copy(
        update={"submission": verified.submission.model_copy(update={"identifier": ident})}
    )
    wiki = tmp_path / "wiki"
    write_experiment_page(verified, wiki)
    corpus = tmp_path / "corpus.jsonl"
    texts = ["abcdefghijklmnopqrstuvwxyz" * 2 + str(i) for i in range(12)]
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    out = tmp_path / "rows" / "coverage.jsonl"
    args = ["coverage", "--model", "toy", "--corpus", str(corpus), "--windows", "2",
            "--window-tokens", "16", "--batch", "2", "--layers", "0,1", "--out", str(out),
            "--wiki-dir", str(wiki), "--device", "cpu"]  # fmt: skip
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert loads == [("toy", "float32")]
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["layer"] for r in rows] == [0, 1]
    assert [r["verified_poles"] for r in rows] == [0, 1]
    assert rows[0]["model"] == "toy" and rows[0]["windows"] == 2 and rows[0]["seed"] == 0
    assert "recovered" not in rows[0] and "recovered" in rows[1]
    assert "Clean loss " in result.output
    assert "[1/2] L0: 0 VERIFIED poles, recovered — (random 0.000)" in result.output
    share = rows[1]["recovered"]
    shown = "—" if share is None else f"{share:.3f}"
    assert f"[2/2] L1: 1 VERIFIED poles, recovered {shown}" in result.output
