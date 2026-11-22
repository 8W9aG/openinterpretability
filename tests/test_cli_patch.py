"""The ``patch`` commands and ``--patch`` on the checks (cli_patch.py, #5229)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import click
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


def _fails(*args):
    out = CliRunner().invoke(cli, ["patch", *args])
    assert out.exit_code == 1, out.output
    return out.output


def _seal(tmp_path, wiki, name, parents=(), rows=(0,)):
    draft = _draft(tmp_path, name, parents=parents, rows=rows)
    return _cli("seal", str(draft), "--wiki-dir", wiki).split()[0]


def test_seal_refuses_evidence_without_a_file_and_a_parent_outside_the_store(tmp_path):
    wiki = str(tmp_path / "wiki")
    bad = CliRunner().invoke(
        cli, ["patch", "seal", str(_draft(tmp_path)), "--evidence", "answers", "--wiki-dir", wiki]
    )
    assert bad.exit_code == 2 and "--evidence takes NAME=FILE, not 'answers'" in bad.output
    orphan = _draft(tmp_path, "orphan.json", parents=["f" * 64])
    assert "parent ffffffffffff is not in the store" in _fails(
        "seal", str(orphan), "--wiki-dir", wiki
    )
    assert not (tmp_path / "wiki").exists() or not list((tmp_path / "wiki").rglob("*.json"))


def test_branches_are_listed_and_deleted_but_their_patches_stay(tmp_path):
    wiki = str(tmp_path / "wiki")
    root = _seal(tmp_path, wiki, "root.json")
    where = ("--model", MODEL, "--wiki-dir", wiki)
    _cli("branch", "work", root, "--create", *where)
    assert _cli("branches", *where) == f"work\t{root[:12]}\n"
    _cli("delete-branch", "work", *where)
    assert _cli("branches", *where) == ""
    assert json.loads(_cli("show", root, *where))["id"] == root


def test_revert_refuses_an_op_that_the_history_does_not_hold(tmp_path):
    wiki = str(tmp_path / "wiki")
    root = _seal(tmp_path, wiki, "root.json")
    out = _fails("revert", root, f"{root}:nope", "--message", "m", "--model", MODEL,
                 "--wiki-dir", wiki)  # fmt: skip
    assert f"nothing to revert at ['{root}:nope']" in out


def test_merge_needs_a_resolution_for_a_parameter_conflict(tmp_path):
    wiki = str(tmp_path / "wiki")
    root = _seal(tmp_path, wiki, "root.json")
    left = _seal(tmp_path, wiki, "left.json", parents=[root], rows=(4,))
    right = _seal(tmp_path, wiki, "right.json", parents=[root], rows=(4,))
    where = ("--model", MODEL, "--wiki-dir", wiki)
    blocked = _fails("merge", left, right, "--message", "both", *where)
    assert "unresolved merge conflicts" in blocked and "parameter" in blocked
    merged = _cli("merge", left, right, "--resolution", "additive", "--order", f"{left}:a",
                  "--order", f"{right}:a", "--message", "both", *where).split()[0]  # fmt: skip
    shown = json.loads(_cli("show", merged, *where))
    assert shown["parents"] == [left, right]
    assert shown["message"] == {"text": "both", "resolution": "additive"}
    assert shown["ops"][0]["type"] == "order"
    assert shown["ops"][0]["targets"] == [f"{left}:a", f"{right}:a"]


def test_rebase_copies_a_chain_onto_a_new_base_and_parent(tmp_path):
    wiki = str(tmp_path / "wiki")
    root = _seal(tmp_path, wiki, "root.json")
    child = _seal(tmp_path, wiki, "child.json", parents=[root], rows=(4,))
    other = _seal(tmp_path, wiki, "other.json", rows=(2,))
    where = ("--model", MODEL, "--wiki-dir", wiki)
    out = _cli("rebase", root, child, "--base", f"{MODEL}@next", *where).splitlines()
    copies = [line.split()[0] for line in out]
    first, second = (json.loads(_cli("show", c, *where)) for c in copies)
    assert (first["base"], first["parents"], first["rebased_from"]) == (f"{MODEL}@next", [], root)
    assert (second["parents"], second["rebased_from"]) == ([copies[0]], child)
    onto = _cli("rebase", child, "--onto", other, "--base", BASE, *where).split()[0]
    assert json.loads(_cli("show", onto, *where))["parents"] == [other]


def test_patch_target_needs_evidence_out_and_a_patch_for_the_same_model(tmp_path):
    from openinterpretability.cli_patch import patch_target

    path = _draft(tmp_path)
    draft = str(path)
    assert patch_target(None, None, tmp_path, MODEL) is None
    with pytest.raises(click.UsageError, match="--evidence-out"):
        patch_target(draft, None, tmp_path, MODEL)
    with pytest.raises(click.UsageError, match="not other/model"):
        patch_target(draft, tmp_path / "e.json", tmp_path, "other/model")
    assert patch_target(draft, tmp_path / "e.json", tmp_path, MODEL).patch == read_draft(path)


def test_patch_check_fails_when_the_patch_moves_the_logits(
    tmp_path,
    neox,  # noqa: F811
    monkeypatch,
):
    tokenizer = SimpleNamespace(encode=lambda text: [3, 7, 11, 7, 2, 9])
    runner = SimpleNamespace(model=neox, tokenizer=tokenizer, device=torch.device("cpu"))
    monkeypatch.setattr("openinterpretability.cli_patch._load_runner", lambda *a, **k: runner)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"text": "a b c"}) + "\n")
    out = _fails("check", str(_draft(tmp_path)), "--model", MODEL, "--corpus", str(corpus),
                 "--strength", "1", "--wiki-dir", str(tmp_path / "wiki"))  # fmt: skip
    assert "patch at strength 1.0 moves a logit by" in out
