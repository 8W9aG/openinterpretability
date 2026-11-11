"""Maintainer signatures on experiment records (`openinterp sign`).

A trusted signer can vouch for a record with their SSH key, so CI need not reproduce
it. The signature is a separate file, ``<id>.json.sig``, next to the write-once record.
``ssh-keygen -Y sign`` makes it in the ``openinterp-record`` namespace. It covers the
record's exact bytes, so editing the record invalidates it.

Signers are listed in ``.github/allowed_signers`` (OpenSSH format, one per line:
``<principal> namespaces="openinterp-record" <key type> <public key>``). CI reads that
file from the pull request's *base* branch, so a pull request can't add its own key.

Each key can have a validity window (``valid-after="YYYYMMDDZ"``, ``valid-before=...``).
A signature is checked at the time it landed on ``main`` (:func:`landing_times`), not at
the time of the check. So a key revoked with ``valid-before`` set to the start of its
exposure still vouches for what landed before that time. A signature that a listed key
made, but that landed outside the key's window, raises :class:`SignatureOutsideWindow`.
Then the record must be reproduced (``SECURITY.md``).

A signature doesn't change the verdict: the record still went through the review
protocol. It only says the signer vouches that the record is what it claims to be.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

NAMESPACE = "openinterp-record"
ALLOWED_SIGNERS = Path(".github/allowed_signers")
DEFAULT_KEY = Path("~/.ssh/openinterp_signing")


def signing_key() -> Path:
    """The signing key: ``$OPENINTERP_SIGNING_KEY``, else ``~/.ssh/openinterp_signing``."""
    return Path(os.environ.get("OPENINTERP_SIGNING_KEY") or DEFAULT_KEY).expanduser()


class SignatureError(Exception):
    """A signature exists but does not verify against the allowed signers."""


class SignatureOutsideWindow(SignatureError):
    """A listed key made the signature, but the check time is outside the key's window.

    The key is revoked or expired (``valid-before``), or not yet valid (``valid-after``).
    The signature then vouches for nothing, and the record must be reproduced.
    """

    def __init__(self, principal: str, at: datetime | None):
        self.principal = principal
        when = "now" if at is None else f"at {at:%Y-%m-%d %H:%M:%S} UTC"
        super().__init__(
            f"signed by {principal}, but that key is not valid {when} (outside its "
            "valid-after/valid-before window in the allowed signers)"
        )


def signature_path(record_path: Path) -> Path:
    """Where the signature of ``record_path`` goes: the same name plus ``.sig``."""
    return record_path.with_name(record_path.name + ".sig")


def sign_record(record_path: Path, key: Path) -> Path:
    """Sign ``record_path`` with the private SSH ``key``; returns the signature's path."""
    sig = signature_path(record_path)
    sig.unlink(missing_ok=True)  # ssh-keygen refuses to overwrite
    _ssh_keygen("-Y", "sign", "-f", str(key), "-n", NAMESPACE, str(record_path))
    return sig


def auto_sign(record_path: Path, allowed_signers: Path = ALLOWED_SIGNERS) -> str | None:
    """Sign a newly archived record if this machine holds a trusted signing key.

    Returns the signer's principal. Returns ``None`` when signing is off
    (``OPENINTERP_AUTO_SIGN=0``), when there is no key, or when the key isn't in
    ``allowed_signers``. An untrusted signature would only make CI fail.
    """
    if os.environ.get("OPENINTERP_AUTO_SIGN", "1").lower() in {"0", "false", "no"}:
        return None
    key = signing_key()
    public = key.with_name(key.name + ".pub")
    if not key.exists() or not public.exists() or not allowed_signers.exists():
        return None
    signers = allowed_signers.read_text(encoding="utf-8")
    if public.read_text(encoding="utf-8").split()[1] not in signers.split():
        return None
    sign_record(record_path, key)
    try:
        return verify_record(record_path, signers)
    except SignatureError:
        signature_path(record_path).unlink()  # e.g. an expired key: CI would not trust it
        raise


def verify_record(
    record_path: Path, allowed_signers: str, at: datetime | None = None
) -> str | None:
    """The principal who signed ``record_path``, or ``None`` if it has no signature.

    ``allowed_signers`` is the *content* of an allowed-signers file. ``at`` is the time to
    check each key's validity window at (default: now). Raises
    :class:`SignatureOutsideWindow` if a listed key made the signature but is not valid
    at ``at``. Raises :class:`SignatureError` if a signature exists but no allowed signer
    made it over these exact bytes.
    """
    sig = signature_path(record_path)
    if not sig.exists():
        return None
    principal, _ = _signed_by(record_path, allowed_signers, at)
    if principal:
        return principal
    principal, listed = _signed_by(record_path, without_windows(allowed_signers), None)
    if principal:
        raise SignatureOutsideWindow(principal, at)
    raise SignatureError(
        f"{sig.name} is not a valid signature of this record by an allowed signer"
        + ("" if listed else " (its key is not in the allowed signers)")
    )


def without_windows(allowed_signers: str) -> str:
    """``allowed_signers`` with every ``valid-after`` and ``valid-before`` option removed."""
    window = re.compile(r'\bvalid-(?:after|before)=(?:"[^"]*"|[^,\s]*),?')
    lines = (window.sub("", line) for line in allowed_signers.splitlines())
    return "\n".join(re.sub(r",(?=\s)", "", line) for line in lines) + "\n"


def _signed_by(record_path: Path, allowed_signers: str, at: datetime | None):
    """(principal who signed at ``at`` or ``None``, whether any listed key matched)."""
    sig = signature_path(record_path)
    when = [] if at is None else [f"-Overify-time={at.astimezone(UTC):%Y%m%d%H%M%S}Z"]
    with tempfile.NamedTemporaryFile("w", suffix=".allowed_signers", delete=False) as fh:
        fh.write(allowed_signers)
        signers = fh.name
    try:
        found = _ssh_keygen(
            "-Y", "find-principals", "-s", str(sig), "-f", signers, *when, check=False
        )
        principals = found.stdout.split()
        for principal in principals:
            with record_path.open("rb") as data:
                result = subprocess.run(
                    ["ssh-keygen", "-Y", "verify", "-f", signers, "-I", principal,
                     "-n", NAMESPACE, "-s", str(sig), *when],
                    stdin=data, capture_output=True, text=True,
                )  # fmt: skip
            if result.returncode == 0:
                return principal, True
    finally:
        Path(signers).unlink(missing_ok=True)
    return None, bool(principals)


def landing_times(ref: str = "HEAD") -> dict[Path, datetime]:
    """When each record's signature landed on ``ref``, by resolved path of the record.

    Each time is a committer date. It comes from the newest commit on ``ref``'s first-parent
    history that added or changed the record or its ``.sig`` file. On ``main`` that is the merge
    time GitHub sets, which a pull request's author can't choose. A record not on ``ref``
    (e.g. one a pull request adds) is absent. It is checked at the current time. In a
    shallow clone the history is incomplete, so this returns nothing.
    """

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    root = git("rev-parse", "--show-toplevel")
    if not root or git("rev-parse", "--is-shallow-repository") != "false":
        return {}
    log = git(
        "-C", root, "log", "--first-parent", "--diff-filter=AM", "--name-only",
        "--format=@%ct", ref, "--", "wiki",
    )  # fmt: skip
    times: dict[Path, datetime] = {}
    when = None
    for line in log.splitlines():
        if line.startswith("@"):
            when = datetime.fromtimestamp(int(line[1:]), UTC)
        elif line.endswith((".json", ".json.sig")) and when is not None:
            record = (Path(root) / line.removesuffix(".sig")).resolve()
            times.setdefault(record, when)  # the log is newest first
    return times


def _ssh_keygen(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["ssh-keygen", *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise SignatureError(f"ssh-keygen {args[1]} failed: {result.stderr.strip()}")
    return result
