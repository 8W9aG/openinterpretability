"""The ``patch`` commands and ``--patch`` on the checks (cli_patch.py, #5229)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from click.testing import CliRunner

from openinterpretability.cli import cli
from openinterpretability.patches.schema import Op, Patch
from openinterpretability.patches.session import patched, read_draft, resolve
from openinterpretability.patches.tensors import put_tensor
from tests.test_patches import BASE, IDS, MLP, neox, tensor_store  # noqa: F401

MODEL = BASE.split("@")[0]


def _draft(tmp_path, name="draft.json", parents=(), rows=(0,), **extra):
    op = Op(op_id="a", type="sparse_add", targets=["blocks.1.mlp.W_out"],
            params={"axis": "row", "index": list(rows),
                    "values": put_tensor(torch.randn(len(rows), MLP))})  # fmt: skip
    path = tmp_path / name
    path.write_text(Patch(base=BASE, parents=list(parents), ops=[op], **extra).model_dump_json())
    return path


def _cli(*args):
    out = CliRunner().invoke(cli, ["patch", *args])
    assert out.exit_code == 0, out.output
    return out.output


def test_seal_writes_the_draft_with_its_evidence_and_history_commands_read_it(tmp_path):
    wiki = str(tmp_path / "wiki")
    found = tmp_path / "answers.json"
    found.write_text(json.dumps({"status": "VERIFIED"}))
    sealed = _cli("seal", str(_draft(tmp_path)), "--evidence", f"answers={found}",
                  "--verdict", "fixed", "--wiki-dir", wiki)  # fmt: skip
    root = sealed.split()[0]
    shown = json.loads(_cli("show", root[:10], "--model", MODEL, "--wiki-dir", wiki))
    assert shown["evidence"] == {"answers": {"status": "VERIFIED"}, "verdict": "fixed"}
    child = _cli("seal", str(_draft(tmp_path, "child.json", parents=[root], rows=(4,))),
                 "--wiki-dir", wiki).split()[0]  # fmt: skip
    assert (
        _cli("log", child, "--model", MODEL, "--wiki-dir", wiki)
        .splitlines()[1]
        .startswith(root[:12])
    )
    _cli("branch", "main", root, "--create", "--model", MODEL, "--wiki-dir", wiki)
    refused = CliRunner().invoke(
        cli, ["patch", "branch", "main", child, "--model", MODEL, "--wiki-dir", wiki]
    )
    assert refused.exit_code != 0 and "fixed" in refused.output  # child has no verdict
    reverted = _cli("revert", child, f"{child}:a", "--message", "undo", "--model", MODEL,
                    "--wiki-dir", wiki)  # fmt: skip
    where = ("--model", MODEL, "--wiki-dir", wiki)
    diff = json.loads(_cli("diff", reverted.split()[0], child, *where))
    assert diff["only_b"] == [f"{child}:a"]


def test_a_draft_runs_patched_and_the_base_comes_back(tmp_path, neox):  # noqa: F811
    path = _draft(tmp_path)
    target = resolve(str(path), tmp_path / "wiki", MODEL)
    assert target.patch == read_draft(path) and target.key.startswith("+patch:")
    base = neox(input_ids=IDS).logits
    with patched(neox, target):
        assert not torch.allclose(neox(input_ids=IDS).logits, base)
    assert torch.equal(neox(input_ids=IDS).logits, base)
    with pytest.raises(ValueError, match="not other/model"):
        resolve(str(path), tmp_path / "wiki", "other/model")


def test_patch_check_runs_the_zero_start_check(tmp_path, neox, monkeypatch):  # noqa: F811
    tokenizer = SimpleNamespace(encode=lambda text: [3, 7, 11, 7, 2, 9])
    runner = SimpleNamespace(model=neox, tokenizer=tokenizer, device=torch.device("cpu"))
    monkeypatch.setattr("openinterpretability.cli_patch._load_runner", lambda *a, **k: runner)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"text": "a b c"}) + "\n")
    out = _cli("check", str(_draft(tmp_path)), "--model", MODEL, "--corpus", str(corpus),
               "--wiki-dir", str(tmp_path / "wiki"))  # fmt: skip
    assert "zero-start check passed on 1 texts" in out


def test_patch_on_explain_needs_evidence_out_and_refuses_registry_writes(tmp_path):
    draft = str(_draft(tmp_path))
    args = ["explain", "x", "--model", MODEL, "--patch", draft, "--wiki-dir", str(tmp_path)]
    missing = CliRunner().invoke(cli, args)
    assert missing.exit_code != 0 and "--evidence-out" in missing.output
    run = CliRunner().invoke(cli, [*args, "--evidence-out", str(tmp_path / "e.json"), "--run"])
    assert run.exit_code != 0 and "evidence only" in run.output
