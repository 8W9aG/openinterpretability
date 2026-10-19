"""Signed records skip reproduction, but only for keys trusted on the base branch."""

import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from openinterp.cli import cli
from openinterp.signing import SignatureError, sign_record, signature_path, verify_record


def keypair(tmp_path: Path, name: str) -> Path:
    key = tmp_path / name
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", name, "-f", str(key)], check=True
    )
    return key


def signer_line(principal: str, key: Path) -> str:
    return f'{principal} namespaces="openinterp-record" {key.with_suffix(".pub").read_text()}'


@pytest.fixture
def keys(tmp_path):
    return keypair(tmp_path, "maintainer"), keypair(tmp_path, "stranger")


def test_sign_and_verify(tmp_path, keys):
    maintainer, stranger = keys
    record = tmp_path / "r.json"
    record.write_text('{"x": 1}')
    assert verify_record(record, signer_line("will", maintainer)) is None  # unsigned

    assert sign_record(record, maintainer) == signature_path(record)
    assert verify_record(record, signer_line("will", maintainer)) == "will"
    with pytest.raises(SignatureError, match="not in the allowed signers"):
        verify_record(record, signer_line("other", stranger))

    record.write_text('{"x": 2}')  # edited after signing
    with pytest.raises(SignatureError, match="not a valid signature"):
        verify_record(record, signer_line("will", maintainer))


def test_reproduce_skips_signed_records_without_loading(tmp_path, monkeypatch, keys):
    maintainer, _ = keys
    monkeypatch.setattr("openinterp.cli._load_runner", lambda *a, **k: pytest.fail("loaded"))
    signers = tmp_path / "allowed_signers"
    signers.write_text(signer_line("will", maintainer))
    record = tmp_path / "r.json"
    record.write_text("{}")  # not even a valid record: the signature vouches for it
    sign_record(record, maintainer)
    result = CliRunner().invoke(cli, ["reproduce", str(record), "--allowed-signers", str(signers)])
    assert result.exit_code == 0, result.output
    assert "SIGNED by will (reproduction skipped)" in result.output


def test_branch_cannot_trust_its_own_key(tmp_path, monkeypatch, keys):
    maintainer, stranger = keys

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("openinterp.cli._load_runner", lambda *a, **k: pytest.fail("loaded"))
    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    signers = Path(".github/allowed_signers")
    signers.parent.mkdir()
    signers.write_text(signer_line("will", maintainer))
    git("add", ".github")
    git("commit", "-q", "-m", "base")
    git("switch", "-q", "-c", "pr")
    record = Path("wiki/experiments/m/r.json")
    record.parent.mkdir(parents=True)
    record.write_text("{}")
    sign_record(record, stranger)
    signers.write_text(signers.read_text() + signer_line("mallory", stranger))
    git("add", "-A")
    git("commit", "-q", "-m", "self-signed")

    result = CliRunner().invoke(cli, ["reproduce", "--base-ref", "main"])
    assert result.exit_code == 1 and "not in the allowed signers" in result.output
    # Without --base-ref the branch's own file would trust it; CI always passes one.
    assert CliRunner().invoke(cli, ["reproduce", str(record)]).exit_code == 0


def test_sign_command_explains_how_to_trust_a_new_key(tmp_path, keys):
    maintainer, _ = keys
    record = tmp_path / "r.json"
    record.write_text("{}")
    signers = tmp_path / "allowed_signers"
    signers.write_text("# empty\n")
    args = ["sign", str(record), "--key", str(maintainer), "--allowed-signers", str(signers)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1
    assert f'namespaces=\\"openinterp-record\\" $(cat {maintainer}.pub)' in result.output

    signers.write_text(signer_line("will", maintainer))
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0 and "Signed" in result.output and "as will" in result.output


@pytest.fixture
def auto_signing(tmp_path, monkeypatch, keys):
    maintainer, _ = keys
    monkeypatch.setenv("OPENINTERP_AUTO_SIGN", "1")
    monkeypatch.setenv("OPENINTERP_SIGNING_KEY", str(maintainer))
    monkeypatch.chdir(tmp_path)
    signers = Path(".github/allowed_signers")
    signers.parent.mkdir()
    signers.write_text(signer_line("will", maintainer))
    return signers


def test_new_experiments_are_signed_when_the_key_is_trusted(auto_signing, record, monkeypatch):
    from openinterp.builder.wiki import archive_path
    from openinterp.cli import _write_wiki

    _write_wiki(record, Path("wiki"), kind="review")
    archived = archive_path(Path("wiki"), record)
    assert verify_record(archived, auto_signing.read_text()) == "will"


def test_backfilled_records_are_never_signed(auto_signing, record):
    from openinterp.builder.wiki import archive_path
    from openinterp.cli import _write_wiki

    _write_wiki(record, Path("wiki"), kind="backfill")
    assert not signature_path(archive_path(Path("wiki"), record)).exists()


@pytest.mark.parametrize("setting", ["untrusted", "disabled", "no key"])
def test_auto_sign_does_nothing_otherwise(auto_signing, keys, monkeypatch, setting, tmp_path):
    from openinterp.signing import auto_sign

    record = tmp_path / "r.json"
    record.write_text("{}")
    if setting == "untrusted":
        monkeypatch.setenv("OPENINTERP_SIGNING_KEY", str(keys[1]))
    elif setting == "disabled":
        monkeypatch.setenv("OPENINTERP_AUTO_SIGN", "0")
    else:
        monkeypatch.setenv("OPENINTERP_SIGNING_KEY", str(tmp_path / "missing"))
    assert auto_sign(record) is None
    assert not signature_path(record).exists()
