import pytest

from openinterpretability.builder.research import Paper, copy_assets, load_papers, redact_asset


def test_a_paper_must_start_with_a_level_1_heading(tmp_path):
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "README.md").write_text("## Not a title\n\nText.\n")
    (tmp_path / "README.md").write_text("# Papers\n\n- 2026-11-01: [P](p/README.md)\n")
    with pytest.raises(ValueError, match="must start with a level 1 heading"):
        load_papers(tmp_path, "https://example.org/repo")


def test_an_array_that_is_not_json_is_left_as_it_is():
    text = 'const v = ["\\q"];'  # a JavaScript escape that JSON does not have
    assert redact_asset(text) == text


def test_a_paper_without_a_page_copies_only_the_files_in_its_media_folder(tmp_path):
    media = tmp_path / "papers" / "p" / "media"
    (media / "figs" / "empty").mkdir(parents=True)  # a folder is not copied as a file
    (media / "figs" / "a.svg").write_text("<svg/>")
    copied = copy_assets(tmp_path / "papers", Paper("p", "T", "S", "", None, False), tmp_path)
    out = tmp_path / "research" / "p" / "media" / "figs"
    assert copied == []
    assert (out / "a.svg").read_text() == "<svg/>"
    assert not (out / "empty").exists()
