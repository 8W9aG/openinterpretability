"""Maintainer signatures on experiment records (`openinterp sign`).

A trusted signer can vouch for a record with their SSH key, so CI need not reproduce
it. The signature is a separate file, ``<id>.json.sig`` next to the write-once record,
made with ``ssh-keygen -Y sign`` in the ``openinterp-record`` namespace. It covers the
record's exact bytes, so editing the record invalidates it.

Signers are listed in ``.github/allowed_signers`` (OpenSSH format, one per line:
``<principal> namespaces="openinterp-record" <key type> <public key>``). CI reads that
file from the pull request's *base* branch, so a pull request can't add its own key.

A signature doesn't change the verdict: the record still went through the review
protocol. It only says the signer vouches that the record is what it claims to be.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

NAMESPACE = "openinterp-record"
ALLOWED_SIGNERS = Path(".github/allowed_signers")


class SignatureError(Exception):
    """A signature exists but does not verify against the allowed signers."""


def signature_path(record_path: Path) -> Path:
    return record_path.with_name(record_path.name + ".sig")


def sign_record(record_path: Path, key: Path) -> Path:
    """Sign ``record_path`` with the private SSH ``key``; returns the signature's path."""
    sig = signature_path(record_path)
    sig.unlink(missing_ok=True)  # ssh-keygen refuses to overwrite
    _ssh_keygen("-Y", "sign", "-f", str(key), "-n", NAMESPACE, str(record_path))
    return sig


def verify_record(record_path: Path, allowed_signers: str) -> str | None:
    """The principal who signed ``record_path``, or ``None`` if it has no signature.

    ``allowed_signers`` is the *content* of an allowed-signers file. Raises
    :class:`SignatureError` if a signature exists but no allowed signer made it over
    these exact bytes.
    """
    sig = signature_path(record_path)
    if not sig.exists():
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".allowed_signers", delete=False) as fh:
        fh.write(allowed_signers)
        signers = fh.name
    try:
        found = _ssh_keygen("-Y", "find-principals", "-s", str(sig), "-f", signers, check=False)
        principals = found.stdout.split()
        for principal in principals:
            with record_path.open("rb") as data:
                result = subprocess.run(
                    ["ssh-keygen", "-Y", "verify", "-f", signers, "-I", principal,
                     "-n", NAMESPACE, "-s", str(sig)],
                    stdin=data, capture_output=True, text=True,
                )  # fmt: skip
            if result.returncode == 0:
                return principal
    finally:
        Path(signers).unlink(missing_ok=True)
    raise SignatureError(
        f"{sig.name} is not a valid signature of this record by an allowed signer"
        + ("" if principals else " (its key is not in the allowed signers)")
    )


def _ssh_keygen(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["ssh-keygen", *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise SignatureError(f"ssh-keygen {args[1]} failed: {result.stderr.strip()}")
    return result
