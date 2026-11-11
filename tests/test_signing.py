"""Signed records skip reproduction, but only for keys trusted on the base branch."""

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from openinterp.cli import cli
from openinterp.signing import (
    SignatureError,
    SignatureOutsideWindow,
    landing_times,
    sign_record,
    signature_path,
    verify_record,
    without_windows,
)


def keypair(tmp_path: Path, name: str) -> Path:
    key = tmp_path / name
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", name, "-f", str(key)], check=True
    )
    return key


def signer_line(principal: str, key: Path, window: str = "") -> str:
    options = (
        f'{window},namespaces="openinterp-record"' if window else 'namespaces="openinterp-record"'
    )
    return f"{principal} {options} {key.with_suffix('.pub').read_text()}"


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


def test_a_key_vouches_only_inside_its_validity_window(tmp_path, keys):
    maintainer, _ = keys
    record = tmp_path / "r.json"
    record.write_text('{"x": 1}')
    sign_record(record, maintainer)
    window = 'valid-after="20261001Z",valid-before="20261105120000Z"'
    signers = signer_line("will", maintainer, window)
    inside = datetime(2026, 11, 5, 11, 59, tzinfo=UTC)
    assert verify_record(record, signers, inside) == "will"
    for outside in (datetime(2026, 9, 30, tzinfo=UTC), datetime(2026, 11, 5, 12, 1, tzinfo=UTC)):
        with pytest.raises(SignatureOutsideWindow, match="signed by will"):
            verify_record(record, signers, outside)
    with pytest.raises(SignatureOutsideWindow, match="not valid now"):
        verify_record(record, signers)  # the window closed before today

    record.write_text('{"x": 2}')  # a bad signature is not excused by the window
    with pytest.raises(SignatureError, match="not a valid signature"):
        verify_record(record, signers, datetime(2026, 12, 1, tzinfo=UTC))


def test_without_windows_keeps_every_other_option():
    options = 'valid-after="20261001Z",namespaces="openinterp-record",valid-before=20261105Z'
    line = f"will {options} ssh-ed25519 AAAA c"
    assert without_windows(line) == 'will namespaces="openinterp-record" ssh-ed25519 AAAA c\n'


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


def git_repo(path: Path):
    def git(*args, when="2026-11-01T00:00:00Z"):
        env = {"GIT_COMMITTER_DATE": when, "GIT_AUTHOR_DATE": when, "PATH": os.environ["PATH"]}
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, env=env)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    return git


def test_signatures_are_checked_when_they_landed(tmp_path, monkeypatch, keys):
    maintainer, _ = keys
    monkeypatch.chdir(tmp_path)
    git = git_repo(tmp_path)
    window = 'valid-before="20261105Z"'  # revoked: exposed from 5 November
    signers = Path("allowed_signers")
    signers.write_text(signer_line("will", maintainer, window))
    old, late, unmerged = (Path(f"wiki/experiments/m/{n}.json") for n in ("old", "late", "new"))
    old.parent.mkdir(parents=True)
    for path in (old, late, unmerged):
        path.write_text(f'{{"r": "{path.stem}"}}')
    sign_record(old, maintainer)
    git("add", "allowed_signers", "wiki/experiments/m/old.json", "wiki/experiments/m/old.json.sig")
    git("commit", "-q", "-m", "old")
    late.unlink()  # the record lands first, its signature only after the key leaked
    git("commit", "-q", "--allow-empty", "-m", "nothing")
    late.write_text('{"r": "late"}')
    git("add", "wiki/experiments/m/late.json")
    git("commit", "-q", "-m", "late record", when="2026-11-02T00:00:00Z")
    sign_record(late, maintainer)
    git("add", "wiki/experiments/m/late.json.sig")
    git("commit", "-q", "-m", "late signature", when="2026-11-06T00:00:00Z")
    sign_record(unmerged, maintainer)

    landed = landing_times()
    assert landed[old.resolve()] == datetime(2026, 11, 1, tzinfo=UTC)
    assert landed[late.resolve()] == datetime(2026, 11, 6, tzinfo=UTC)  # its .sig's time
    assert unmerged.resolve() not in landed

    args = ["signatures", "--allowed-signers", str(signers), "--wiki-dir", "wiki"]
    result = CliRunner().invoke(cli, [*args, "--paths-only"])
    assert result.exit_code == 0, result.output
    assert result.output.split() == [str(late), str(unmerged)]
    result = CliRunner().invoke(cli, args)
    assert "landed 2026-11-06 00:00Z" in result.output and "not landed" in result.output
    assert "2 of 3 signed record(s) not trusted." in result.output


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


def test_an_expired_key_signs_nothing(auto_signing, keys, tmp_path):
    from openinterp.signing import auto_sign

    auto_signing.write_text(signer_line("will", keys[0], 'valid-before="20200101Z"'))
    record = tmp_path / "r.json"
    record.write_text("{}")
    with pytest.raises(SignatureOutsideWindow):
        auto_sign(record)
    assert not signature_path(record).exists()


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
