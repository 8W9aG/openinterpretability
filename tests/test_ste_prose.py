"""The STE linter skips a paper's reference list (scripts/ste_prose.py, issue #3137)."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "ste_prose", Path(__file__).parents[1] / "scripts" / "ste_prose.py"
)
ste_prose = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ste_prose)


def test_only_the_reference_section_is_blanked_and_lines_are_kept():
    text = "# T\n\nProse.\n\n## References\n\n- A, B. (2020). *Long*.\n\n## Next\n\nMore.\n"
    out = ste_prose.without_references(text)
    assert out.count("\n") == text.count("\n")
    assert "Long" not in out and "Prose." in out and "More." in out
    assert "## References" in out and "## Next" in out


def _sentence(n: int) -> str:
    return " ".join(["word"] * (n - 1) + ["end."])


def _write(root: Path, rel: str, sentences: list[int]) -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Title\n\n" + " ".join(_sentence(n) for n in sentences) + "\n")
    return rel


def test_only_a_papers_own_text_gets_the_paper_profile():
    assert ste_prose.is_paper("papers/neuron-development/README.md")
    assert not ste_prose.is_paper("papers/README.md")
    assert not ste_prose.is_paper("docs/analyses/x.md")


def test_a_paper_may_have_a_30_word_sentence_but_a_doc_may_not(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paper = _write(tmp_path, "papers/p/README.md", [30, 10, 10])
    doc = _write(tmp_path, "docs/d.md", [30, 10, 10])
    assert ste_prose.main([paper]) == 0
    assert ste_prose.main([doc]) == 1


def test_a_paper_fails_on_a_sentence_over_the_cap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paper = _write(tmp_path, "papers/p/README.md", [ste_prose.PAPER_MAX_WORDS + 1, 5, 5, 5])
    assert ste_prose.main([paper]) == 1


def test_a_paper_fails_when_its_mean_sentence_is_too_long(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    paper = _write(tmp_path, "papers/p/README.md", [ste_prose.PAPER_MEAN_WORDS + 10] * 4)
    assert ste_prose.main([paper]) == 1
    assert "mean sentence length" in capsys.readouterr().out
