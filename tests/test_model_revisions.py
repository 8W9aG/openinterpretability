"""CI reproduces only model revisions on the model's main branch (model_revisions.py)."""

from openinterpretability import model_revisions

MAIN = "a" * 40


def test_only_a_main_branch_commit_is_approved(monkeypatch):
    monkeypatch.setattr(model_revisions, "main_commits", lambda model: frozenset({MAIN}))
    assert model_revisions.unapproved_revision("m", MAIN) is None
    assert model_revisions.unapproved_revision("m", None) is None  # main itself
    assert "not a full commit SHA" in model_revisions.unapproved_revision("m", "refs/pr/1")
    assert "not on m's main branch" in model_revisions.unapproved_revision("m", "b" * 40)


def test_ci_refuses_a_record_that_names_another_revision(monkeypatch, record):
    from openinterpretability.cli_reproduce import _Reproducer
    from openinterpretability.core.run_environment import RunEnvironment

    monkeypatch.setattr(model_revisions, "main_commits", lambda model: frozenset({MAIN}))
    env = RunEnvironment(device="cpu", dtype="float32", openinterp_version="t",
                         torch_version="t", model_revision="refs/pr/7")  # fmt: skip
    rec = record.model_copy(
        update={"verdict": record.verdict.model_copy(update={"environment": env})}
    )
    model = rec.submission.identifier.model_name
    ci = _Reproducer(signers="", landed={}, tolerance=None, scan_corpora={},
                     allowed_models=(model,), device="cpu")  # fmt: skip
    assert "not a full commit SHA" in ci.refusal(rec)


def test_main_commits_lists_the_main_branch_history_from_the_hub(monkeypatch):
    from types import SimpleNamespace

    import huggingface_hub

    asked = []

    class FakeApi:  # the Hub is the network boundary
        def list_repo_commits(self, model, revision):
            asked.append((model, revision))
            return [SimpleNamespace(commit_id=MAIN), SimpleNamespace(commit_id="b" * 40)]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    model_revisions.main_commits.cache_clear()
    try:
        assert model_revisions.main_commits("org/m") == frozenset({MAIN, "b" * 40})
        assert asked == [("org/m", "main")]
    finally:
        model_revisions.main_commits.cache_clear()
