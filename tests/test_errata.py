"""Errata for signed records: the file's format, its check, standings and the site."""

import json
import subprocess
from datetime import timedelta
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from openinterp.builder.compiler import compile_site
from openinterp.builder.tree import build_tree
from openinterp.builder.wiki import write_experiment_page
from openinterp.checks import check_registry
from openinterp.core.errata import Erratum, ErratumRule, errata_path, parse_errata, record_ref
from openinterp.core.schemas import AuditRecord
from openinterp.signing import sign_record

GIT = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]


def later(record: AuditRecord, hours: int, status: str = "VERIFIED") -> AuditRecord:
    verdict = record.verdict.model_copy(
        update={"timestamp": record.verdict.timestamp + timedelta(hours=hours), "status": status}
    )
    return record.model_copy(update={"verdict": verdict})


def erratum(**fields) -> Erratum:
    base = {
        "id": "scale-bug",
        "date": "2026-11-13",
        "title": "Activations were scaled twice",
        "bug": "The scan divided by the scale twice.",
        "fixed_in": "abcdef1",
        "action": "set_aside",
    }
    return Erratum.model_validate_json(json.dumps(base | fields))


def test_an_erratum_must_select_records():
    with pytest.raises(ValidationError, match="records, a rule, or both"):
        erratum()
    with pytest.raises(ValidationError, match="at least one field"):
        erratum(rule={})
    with pytest.raises(ValidationError):  # a naive time can't be compared with a verdict's
        erratum(rule={"reviewed_before": "2026-11-01T00:00:00"})
    with pytest.raises(ValueError, match="line 2"):
        parse_errata(erratum(records=["a"]).model_dump_json() + "\n{}\n")


def test_an_erratum_covers_listed_records_and_rule_matches(record):
    assert erratum(records=[record_ref(record)]).covers(record)
    assert not erratum(records=["other/L0/N0-positive/x.json"]).covers(record)
    model = record.submission.identifier.model_name
    assert ErratumRule(models=[model]).matches(record)
    assert not ErratumRule(models=["gpt2"]).matches(record)
    assert not ErratumRule(protocol_versions=["0.1"]).matches(record)  # no policy: no version
    t = record.verdict.timestamp
    assert ErratumRule(reviewed_from=t, reviewed_before=t + timedelta(seconds=1)).matches(record)
    assert not ErratumRule(reviewed_before=t).matches(record)
    assert not ErratumRule(models=[model], kinds=["attn_head"]).matches(record)


def test_set_aside_readings_do_not_count(record):
    newer = later(record, 1, "REJECTED")
    x = erratum(records=[record_ref(newer)])
    (model,) = build_tree([record, newer], [x])
    (unit,) = model.layers[0].units
    standing = unit.standings["positive"]
    assert standing.status == "VERIFIED" and standing.errata == []
    roles = {e.id: e.role for e in unit.experiments}
    assert roles == {"20261001T120000Z": "current", "20261001T130000Z": "set_aside"}

    flagged = erratum(records=[record_ref(newer)], action="flag")
    (model,) = build_tree([record, newer], [flagged])
    standing = model.layers[0].units[0].standings["positive"]
    assert standing.status == "REJECTED" and standing.errata == [flagged]


def test_a_pole_whose_readings_are_all_set_aside_keeps_them(record):
    x = erratum(rule={"models": [record.submission.identifier.model_name]})
    (model,) = build_tree([record], [x])
    standing = model.layers[0].units[0].standings["positive"]
    assert standing.status == "VERIFIED" and standing.errata == [x]


def test_the_site_shows_errata_only_on_the_experiments_they_alter(tmp_path, record):
    newer = later(record, 1, "REJECTED")
    x = erratum(records=[record_ref(newer)])
    compile_site([record, newer], tmp_path, errata=[x])
    assert not (tmp_path / "errata.html").exists()
    (unit,) = (tmp_path / "models").rglob("N117.html")
    html = unit.read_text()
    assert html.count("Erratum, 2026-11-13: Activations were scaled twice.") == 1
    assert "it no longer counts toward the status" in html
    assert "The scan divided by the scale twice." in html and "/commit/abcdef1" in html
    assert "set aside by an erratum" in html
    rows = pq.read_table(tmp_path / "data" / "registry.parquet").to_pylist()
    assert sorted(tuple(r["errata"]) for r in rows) == [(), ("scale-bug",)]


def test_a_site_without_errata_mentions_none(tmp_path, record):
    compile_site([record], tmp_path)
    assert not any("Erratum" in p.read_text() for p in tmp_path.rglob("*.html"))


# --- openinterp check ---------------------------------------------------------------


def keypair(directory: Path, name: str) -> Path:
    key = directory / name
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    return key


def git(*args: str) -> str:
    return subprocess.run([*GIT, *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch, record):
    """A registry with one record and a trusted maintainer key, committed on main."""
    monkeypatch.chdir(tmp_path)
    record = record.model_copy(
        update={
            "submission": record.submission.model_copy(
                update={"model_num_layers": 2, "model_mlp_width": 8}
            )
        }
    )
    write_experiment_page(record, tmp_path / "wiki")
    (tmp_path / ".gitignore").write_text("wiki/README.md\nwiki/experiments/**/README.md\nkeys/\n")
    (tmp_path / "keys").mkdir()
    maintainer = keypair(tmp_path / "keys", "maintainer")
    public = maintainer.with_suffix(".pub").read_text()
    signers = Path(".github/allowed_signers")
    signers.parent.mkdir()
    signers.write_text(f'will namespaces="openinterp-record" {public}')
    git("init", "-q", "-b", "main")
    git("add", ".")
    git("commit", "-qm", "base")
    return {"record": record, "key": maintainer, "commit": git("rev-parse", "HEAD").strip()}


def write_errata(*errata: Erratum, key: Path | None = None) -> Path:
    path = errata_path()
    path.write_text("".join(e.model_dump_json() + "\n" for e in errata))
    if key is not None:
        sign_record(path, key)
    return path


def problems(**kwargs) -> str:
    return " | ".join(p.message for p in check_registry(**kwargs).problems)


def test_signed_errata_pass(repo):
    x = erratum(records=[record_ref(repo["record"])], fixed_in=repo["commit"])
    write_errata(x, key=repo["key"])
    assert problems() == ""
    git("switch", "-qc", "pr")
    assert problems(base_ref="main") == ""


def test_errata_must_be_signed_by_a_trusted_key(repo, tmp_path):
    x = erratum(records=[record_ref(repo["record"])], fixed_in=repo["commit"])
    write_errata(x)
    assert "the errata file is unsigned" in problems()
    stranger = keypair(tmp_path / "keys", "stranger")
    write_errata(x, key=stranger)
    assert "signature is not trusted" in problems()
    # A pull request can't trust its own key: the signers come from the base branch.
    git("switch", "-qc", "pr")
    signers = Path(".github/allowed_signers")
    public = stranger.with_suffix(".pub").read_text()
    signers.write_text(signers.read_text() + f'mallory namespaces="openinterp-record" {public}')
    assert problems() == ""
    assert "signature is not trusted" in problems(base_ref="main")


def test_errata_name_real_records_and_commits(repo):
    git("switch", "-qc", "pr")
    missing = erratum(id="missing", records=["m/L0/N1-positive/x.json"], fixed_in="1234567")
    ruled = erratum(id="ruled", rule={"models": ["gpt2"]}, fixed_in=repo["commit"])
    write_errata(missing, ruled, missing, key=repo["key"])
    found = problems(base_ref="main")
    assert "erratum id 'missing' is used more than once" in found
    assert "erratum missing: no record m/L0/N1-positive/x.json" in found
    assert "erratum ruled covers no record" in found
    assert "erratum missing: no commit 1234567 in this clone" in found


def test_errata_are_append_only(repo):
    first = erratum(id="first", records=[record_ref(repo["record"])], fixed_in=repo["commit"])
    write_errata(first, key=repo["key"])
    git("add", "wiki")
    git("commit", "-qm", "errata")
    git("switch", "-qc", "pr")
    second = erratum(id="second", rule={"kinds": ["mlp_neuron_pole"]}, fixed_in=repo["commit"])
    write_errata(first, second, key=repo["key"])
    assert problems(base_ref="main") == ""
    write_errata(first.model_copy(update={"title": "Changed"}), second, key=repo["key"])
    assert "append-only" in problems(base_ref="main")
    errata_path().unlink()
    errata_path().with_name("errata.jsonl.sig").unlink()
    assert "must not be deleted" in problems(base_ref="main")


def test_a_revoked_key_still_vouches_for_errata_that_landed_before(repo, monkeypatch):
    x = erratum(records=[record_ref(repo["record"])], fixed_in=repo["commit"])
    write_errata(x, key=repo["key"])
    git("add", "wiki")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2026-11-01T00:00:00Z")
    git("commit", "-qm", "errata")
    signers = Path(".github/allowed_signers")
    signers.write_text(signers.read_text().replace("will ", 'will valid-before="20261105Z",', 1))
    assert problems() == ""  # checked when it landed, on 1 November
    git("switch", "-qc", "pr")
    write_errata(x, erratum(id="new", records=x.records, fixed_in=x.fixed_in), key=repo["key"])
    assert "signature is not trusted" in problems()  # a new signature is checked now
