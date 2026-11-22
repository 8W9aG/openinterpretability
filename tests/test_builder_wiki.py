"""Experiment pages in the wiki: front matter and loading the archive."""

from openinterpretability.builder.wiki import (
    _read_front_matter,
    archive_record,
    load_archive,
    write_experiment_page,
)


def test_front_matter_without_a_closing_line_reads_to_the_end(tmp_path):
    page = tmp_path / "page.md"
    page.write_text('---\nslug: "a/L1/N2-positive"\nlayer: 1\n', encoding="utf-8")
    assert _read_front_matter(page) == {"slug": "a/L1/N2-positive", "layer": 1}


def test_a_sister_study_page_names_its_source_in_front_matter(record, tmp_path):
    ref = "o--m/L2/N9-positive/20261001T120000Z"
    sister = record.model_copy(
        update={"submission": record.submission.model_copy(update={"sister_of": ref})}
    )
    page = write_experiment_page(sister, tmp_path / "wiki")
    assert page is not None
    assert _read_front_matter(page)["sister_of"] == ref


def test_load_archive_returns_the_records_oldest_first(record, tmp_path):
    from datetime import timedelta

    later = record.model_copy(
        update={
            "verdict": record.verdict.model_copy(
                update={"timestamp": record.verdict.timestamp + timedelta(days=1)}
            )
        }
    )
    wiki = tmp_path / "wiki"
    archive_record(later, wiki)
    archive_record(record, wiki)
    loaded = load_archive(wiki)
    assert [r.verdict.timestamp for r in loaded] == [
        record.verdict.timestamp,
        later.verdict.timestamp,
    ]
