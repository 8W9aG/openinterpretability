"""The match labels of a scan: personal data, toxicity, names, addresses (offline)."""

from types import SimpleNamespace

import pytest
import torch

from openinterp.agents.proposer import format_confounds, label_note
from openinterp.builder.pii import redact_text
from openinterp.core.confounds import (
    HATE_SPEECH,
    MATCH_LABEL_SETS,
    NAMES,
    NO_MATCH,
    PART_OF_SPEECH,
    PERSONAL_DATA,
    POSTAL_ADDRESSES,
)
from openinterp.core.schemas import ConfoundProfile, LabelConfound, LabelStat
from openinterp.evidence import _Texts
from openinterp.harness.annotations import LabelAccumulator, MatchTagger, TokenLabels
from openinterp.harness.scan import Piece
from openinterp.names import MODEL_SHA256, PERSON, _download
from openinterp.names import person_matches as real_person_matches
from openinterp.names import pos_matches as real_pos_matches
from openinterp.sensitive import address_matches, detector, pii_matches
from tests.test_annotations import CharTokenizer

TEXT = "Mail jane@somemail.org or call (415) 555-2671 now"
EMAIL = (5, 22)


def test_matches_name_their_kind():
    assert pii_matches(TEXT) == ((*EMAIL, "email"), (31, 45, "phone"))
    assert pii_matches("a plain sentence") == ()


def test_a_match_tagger_labels_every_character():
    spans = MatchTagger("t", pii_matches).spans(0, TEXT)
    covered = sorted((s.start, s.end) for s in spans)
    assert covered == [(0, 5), EMAIL, (22, 31), (31, 45), (45, len(TEXT))]
    assert {s.label for s in spans} == {"email", "phone", NO_MATCH}


def test_a_document_with_no_match_is_one_no_match_span():
    (span,) = MatchTagger("t", pii_matches).spans(0, "plain")
    assert (span.start, span.end, span.label) == (0, 5, NO_MATCH)


def test_overlapping_matches_leave_no_gap_or_repeat():
    find = lambda text: ((0, 4, "a"), (2, 6, "b"))  # noqa: E731
    spans = MatchTagger("t", find).spans(0, "abcdefgh")
    assert [(s.start, s.end, s.label) for s in spans] == [
        (0, 4, "a"),
        (2, 6, "b"),
        (6, 8, NO_MATCH),
    ]


def _labels(text, tagger):
    tokenizer = CharTokenizer()
    labels = TokenLabels(tokenizer, [text], [tagger])
    ids = torch.tensor(tokenizer.encode(text))
    found = labels.of(Piece(0, 0, 1, text), ids)[tagger.name]
    return [labels.names[tagger.name][i] if i >= 0 else None for i in found.tolist()]


def test_tokens_take_the_label_of_the_match_that_covers_them():
    found = _labels(TEXT, MatchTagger(PERSONAL_DATA, pii_matches))
    assert found[0] is None  # the special token
    assert found[1:] == [
        "email" if EMAIL[0] <= i < EMAIL[1] else "phone" if 31 <= i < 45 else NO_MATCH
        for i in range(len(TEXT))
    ]


def test_a_unit_that_fires_on_matches_measures_above_zero():
    # The reason for NO_MATCH: tokens with no label leave both measures.
    labels = torch.tensor([0] * 90 + [1] * 10)
    strengths = torch.tensor([0.0] * 90 + [5.0] * 10)
    accumulator = LabelAccumulator()
    accumulator.update(labels, strengths)
    confound = accumulator.finalize([NO_MATCH, "email"], line=1.0)
    assert confound.r2 == 1.0 and confound.mutual_information_bits > 0.3
    assert {s.label: s.active for s in confound.levels} == {NO_MATCH: 0, "email": 10}


def _texts(labels=None):
    return _Texts(["doc"], "c", "0" * 64, None, None, None, labels or {})


def test_every_scan_labels_matches_after_the_corpus_labels():
    names = [t.name for t in _texts({"language": ["en"]}).taggers()]
    matches = list(MATCH_LABEL_SETS)  # part of speech comes right after names (#3223)
    matches.insert(matches.index(NAMES) + 1, PART_OF_SPEECH)
    assert names == ["language", *matches]
    assert [t.name for t in _texts().taggers()][-1] == POSTAL_ADDRESSES


def test_the_detector_names_what_fixes_the_labels():
    assert "phonenumbers" in detector() and "suffix list " in detector()


def _confound(*levels):
    stats = [LabelStat(label=n, count=c, mean_strength=0.1, active=a) for n, c, a in levels]
    total = sum(s.count for s in stats)
    return LabelConfound(
        r2=0.2,
        mutual_information_bits=0.1,
        labelled_tokens=total,
        unlabelled_tokens=0,
        levels=stats,
    )


def _note(labels):
    return format_confounds(
        ConfoundProfile.model_construct(
            token_r2=0.1, position_r2=0.1, trigger_tokens=[], labels=labels
        )
    )


def test_the_proposer_hears_of_a_match_set_only_when_a_match_is_active():
    quiet = _confound((NO_MATCH, 100, 5), ("email", 4, 0))
    assert "ersonal data" not in _note({PERSONAL_DATA: quiet})
    firing = _confound((NO_MATCH, 100, 5), ("email", 4, 3))
    assert "Personal data explains 20%" in _note({PERSONAL_DATA: firing})
    assert "email 3/4" in _note({PERSONAL_DATA: firing})


def test_corpus_labels_stay_in_the_proposers_note_when_quiet():
    assert "Language explains" in _note({"language": _confound(("en", 100, 0))})


def test_an_examples_note_leaves_out_no_match_labels():
    example = SimpleNamespace(
        peak_labels={"language": "fr", PERSONAL_DATA: NO_MATCH, HATE_SPEECH: "sexism"}
    )
    assert label_note(example) == f"language fr, {HATE_SPEECH} sexism"
    quiet = SimpleNamespace(peak_labels={PERSONAL_DATA: NO_MATCH})
    assert label_note(quiet) == ""


def test_the_detector_string_survives_the_sites_scrub():
    assert redact_text(detector()) == detector()


def test_the_export_gives_each_match_set_its_measures():
    from openinterp.builder.parquet_export import _match_columns

    confounds = SimpleNamespace(
        labels={PERSONAL_DATA: _confound((NO_MATCH, 100, 5), ("email", 4, 3))}
    )
    columns = _match_columns(confounds)
    assert columns["personal_data_r2"] == 0.2 and columns["personal_data_active_tokens"] == 3
    assert columns["toxicity_r2"] is None
    assert _match_columns(None)["personal_data_mi_bits"] is None


def test_the_proposer_never_sees_the_names_label():
    firing = _confound((NO_MATCH, 100, 5), (PERSON, 40, 30))
    assert "ames" not in _note({NAMES: firing})
    example = SimpleNamespace(peak_labels={NAMES: PERSON})
    assert label_note(example) == ""


def test_the_site_never_removes_a_name():
    assert redact_text("Written by Ada Lovelace in 1843.") == "Written by Ada Lovelace in 1843."


def _fake_model(monkeypatch, calls):
    class Doc(list):
        ents = [SimpleNamespace(start_char=0, end_char=3, label_="PERSON"),
                SimpleNamespace(start_char=4, end_char=9, label_="ORG")]  # fmt: skip

    def fake(text):
        calls.append(text)
        word = lambda i, t, p, s=False: SimpleNamespace(idx=i, text=t, pos_=p, is_space=s)  # noqa: E731
        return Doc(
            [word(0, "Ada", "PROPN"), word(3, " ", "SPACE", True), word(4, "Bytes", "PROPN")]
        )

    monkeypatch.setattr("openinterp.names._model", lambda: fake)


def test_names_keep_people_only_and_later_scans_read_the_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    _fake_model(monkeypatch, calls)
    assert real_person_matches("Ada Bytes") == ((0, 3, PERSON),)
    assert real_person_matches("Ada Bytes") == ((0, 3, PERSON),)
    assert calls == ["Ada Bytes"]


def test_tags_come_from_the_same_pass_as_names(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    _fake_model(monkeypatch, calls)
    assert real_person_matches("Ada Bytes") == ((0, 3, PERSON),)
    # The names pass cached the tags too, so the model doesn't run again. Whitespace has none.
    assert real_pos_matches("Ada Bytes") == ((0, 3, "PROPN"), (4, 9, "PROPN"))
    assert calls == ["Ada Bytes"]


def test_a_model_that_fails_its_checksum_is_not_kept(tmp_path, monkeypatch):
    import io

    monkeypatch.setattr("urllib.request.urlopen", lambda url: io.BytesIO(b"not the model"))
    wheel = tmp_path / "m.whl"
    with pytest.raises(ValueError, match="pinned SHA-256"):
        _download(wheel)
    assert list(tmp_path.iterdir()) == [] and len(MODEL_SHA256) == 64


def test_the_registry_schema_has_every_match_column():
    from openinterp.builder.parquet_export import REGISTRY_SCHEMA, _match_columns

    assert set(_match_columns(None)) <= set(REGISTRY_SCHEMA.names)


@pytest.mark.parametrize(
    "kind, found, text",
    [
        ("street", "42 Elm Street", "She lives at 42 Elm Street, Springfield."),
        ("street", "1608 S. Western Ave.", "Meet at 1608 S. Western Ave. in town."),
        ("street", "51 Franklin St", "Inc., 51 Franklin St, Fifth Floor."),
        ("po_box", "PO Box 75255", "Mail to PO Box 75255, Kapolei."),
        ("po_box", "P.O. Box 12", "Write to P.O. Box 12 today."),
        ("us_city_zip", "Boston, MA 02110-1301", "Floor, Boston, MA 02110-1301, USA."),
        ("uk_postcode", "SE1 9GF", "London SE1 9GF."),
    ],
)
def test_each_kind_of_address_part_is_labelled(kind, found, text):
    start = text.index(found)
    assert (start, start + len(found), kind) in address_matches(text)


@pytest.mark.parametrize(
    "text",
    [
        "27 Mass. App. Ct. 250 (1989)",  # a legal citation
        "No. 335, 2008 Supreme Court of Delaware.",
        "4 Did St Joseph go straight to heaven?",
        "16.15 Introduction\n\nDr Dan Spoke",  # a line break ends a match
        "We sold 500 units on the main street.",
        "",
    ],
)
def test_look_alikes_are_not_addresses(text):
    assert address_matches(text) == ()


def test_the_site_never_removes_an_address():
    text = "She lives at 42 Elm Street, Springfield, IL 62704, or PO Box 12."
    assert address_matches(text) and redact_text(text) == text
