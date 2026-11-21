"""What a revision's proposer sees of the record that it supersedes (revision.py)."""

from openinterpretability.revision import previous_attempt


def test_a_superseded_record_missing_from_the_wiki_ends_the_chain(record, tmp_path):
    sub = record.submission.model_copy(
        update={"natural_language_explanation": "Turn two.", "supersedes": "20260101T000000Z"}
    )
    second = record.model_copy(update={"submission": sub})
    brief = previous_attempt(second, tmp_path)  # the wiki has no record of the earlier turn
    assert "Turn two." in brief
    assert "before this one" not in brief
    assert brief == previous_attempt(second)  # the same brief as with no wiki
