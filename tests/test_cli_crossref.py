"""``open-interpretability crossref`` and ``aligned-latents`` from the command line."""

from __future__ import annotations

from click.testing import CliRunner

from openinterpretability import cli_crossref, crossref
from openinterpretability.builder.wiki import archive_record
from openinterpretability.cli import cli
from openinterpretability.core.schemas import NeuronIdentifier
from openinterpretability.crossref import AlignmentRow, load_rows
from tests.test_crossref import PLANTED, gpt2_runner, planted  # noqa: F401  (fixtures)

SLUG = "tiny-gpt2/L1/N7-positive"


def _pole(record, idx, sign):
    ident = NeuronIdentifier(model_name="tiny-gpt2", layer=1, neuron_idx=idx, pole=sign)
    return record.model_copy(
        update={"submission": record.submission.model_copy(update={"identifier": ident})}
    )


def test_crossref_aligns_each_archived_pole_and_writes_the_rows_in_order(
    tmp_path,
    monkeypatch,
    record,
    gpt2_runner,  # noqa: F811
    planted,  # noqa: F811
):
    monkeypatch.setenv("OPENINTERPRETABILITY_OFFLOAD_EVIDENCE", "0")
    wiki = tmp_path / "wiki"
    for idx, sign in [(7, "positive"), (3, "negative")]:
        archive_record(_pole(record, idx, sign), wiki)
    loads = []

    def load(model_name, device, dtype, revision):
        loads.append((model_name, device, dtype, revision))
        return gpt2_runner

    monkeypatch.setattr(cli_crossref, "_load_runner", load)
    out = tmp_path / "alignment.jsonl"
    args = ["crossref", "--wiki-dir", str(wiki), "--out", str(out), "-k", "2"]
    result = CliRunner().invoke(cli, args, catch_exceptions=False)
    assert loads == [("tiny-gpt2", "auto", "float32", None)]
    assert result.output.splitlines() == [
        "tiny-tc L1: 2 poles",
        f"Aligned 2 pole–dictionary pairs → {out}",
    ]
    slugs = [AlignmentRow.model_validate_json(line).slug for line in out.read_text().splitlines()]
    assert slugs == sorted(slugs) == ["tiny-gpt2/L1/N3-negative", SLUG]
    assert len(load_rows(out)[SLUG][0].writes.matches) == 2


def test_crossref_with_nothing_to_align_writes_no_file(tmp_path):
    out = tmp_path / "alignment.jsonl"
    args = ["crossref", "--wiki-dir", str(tmp_path / "wiki"), "--out", str(out)]
    result = CliRunner().invoke(cli, args, catch_exceptions=False)
    assert result.output == f"Aligned 0 pole–dictionary pairs → {out}\n"
    assert not out.exists()


def _row(reads=True):
    side = {
        "matches": [
            {"feature_idx": PLANTED, "cosine": 0.9, "channel": "in"},
            {"feature_idx": 2, "cosine": 0.5, "channel": "in"},
            {"feature_idx": 3, "cosine": 0.1, "channel": "in"},
        ],
        "chance": 0.2,
        "percentile": 0.99,
    }
    return AlignmentRow.model_validate(
        {
            "slug": SLUG,
            "dictionary": "tiny-tc",
            "repo": "r",
            "revision": "v",
            "path": "1/tc.safetensors",
            "model_revision": None,
            "writes": side,
            "reads": side if reads else None,
        }  # fmt: skip
    )


def test_aligned_latents_prints_each_side_with_descriptions_and_links(
    tmp_path,
    monkeypatch,
    planted,  # noqa: F811
):
    out = tmp_path / "alignment.jsonl"
    out.write_text(_row().model_dump_json() + "\n" + _row(reads=False).model_dump_json() + "\n")

    def get_json(url, **_):  # Neuronpedia describes the planted latent only
        if url.endswith(f"/{PLANTED}"):
            return {"explanations": [{"description": "quotes", "explanationModelName": "m"}]}
        return {"explanations": []}

    monkeypatch.setenv("OPENINTERPRETABILITY_NEURONPEDIA_CACHE", str(tmp_path / "np"))
    monkeypatch.setattr(crossref, "_get_json", get_json)
    args = ["aligned-latents", SLUG, "--out", str(out), "--wiki-dir", str(tmp_path / "wiki")]
    lines = CliRunner().invoke(cli, args, catch_exceptions=False).output.splitlines()
    url = "https://www.neuronpedia.org/tiny/1-tc"
    writes = [
        "tiny-tc writes (chance 0.20, best beats 99% of the layer's neurons)",
        f"  F{PLANTED:<6} +0.900  quotes  [Neuronpedia, m]",
        f"          {url}/{PLANTED}",
        "  F2      +0.500  —",
        f"          {url}/2",
        "  F3      +0.100  below chance",
        f"          {url}/3",
    ]
    reads = [
        "tiny-tc reads (chance 0.20, best beats 99% of the layer's neurons)",
        f"  F{PLANTED:<6} +0.900 in  quotes  [Neuronpedia, m]",
        f"          {url}/{PLANTED}",
        "  F2      +0.500 in  —",
        f"          {url}/2",
        "  F3      +0.100 in  below chance",
        f"          {url}/3",
    ]
    # The second row has no reads side, so it prints its writes only.
    assert lines == writes + reads + writes


def test_aligned_latents_prints_no_link_where_neuronpedia_has_no_dictionary(tmp_path):
    out = tmp_path / "alignment.jsonl"
    row = _row(reads=False).model_copy(update={"dictionary": "unhosted"})
    out.write_text(row.model_dump_json() + "\n")
    args = ["aligned-latents", SLUG, "--out", str(out), "--wiki-dir", str(tmp_path / "wiki")]
    lines = CliRunner().invoke(cli, args, catch_exceptions=False).output.splitlines()
    assert lines[1:] == [f"  F{PLANTED:<6} +0.900  —", "  F2      +0.500  —",
                         "  F3      +0.100  below chance"]  # fmt: skip


def test_aligned_latents_without_rows_says_to_run_crossref(tmp_path):
    out = tmp_path / "alignment.jsonl"
    result = CliRunner().invoke(cli, ["aligned-latents", SLUG, "--out", str(out)])
    assert result.exit_code == 1
    assert f"no alignment rows for {SLUG} in {out}; run crossref" in result.output
