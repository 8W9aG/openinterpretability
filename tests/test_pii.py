"""What the site removes and masks: personal data and offensive examples (offline)."""

import json

import pyarrow.parquet as pq
import pytest

from openinterp.builder.compiler import compile_site
from openinterp.builder.pii import (
    REMOVED_TEXT,
    REMOVED_TOKEN,
    _scrub_attribution,
    redact_text,
    redact_tokens,
    scrub_record,
)
from openinterp.core.schemas import AuditRecord
from openinterp.sensitive import is_offensive

EMAIL = "jane.doe@somemail.org"


@pytest.mark.parametrize(
    "found, text",
    [
        (EMAIL, f"Write to {EMAIL} today."),
        ("(415) 555-2671", "Call (415) 555-2671 after noon."),
        ("+44 20 7946 0958", "Ring +44 20 7946 0958 for details."),
        ("4111 1111 1111 1111", "Card: 4111 1111 1111 1111, expires soon."),
        ("5500-0000-0000-0004", "Card: 5500-0000-0000-0004."),
        ("GB82 WEST 1234 5698 7654 32", "Pay to GB82 WEST 1234 5698 7654 32 please."),
        ("555-1234", "Call 555-1234 after noon."),
        ("482-9917", "At noon I phoned 482-9917 again."),
        ("8341 7141", "Media contact:\nA. Person, 8341 7141 today."),
        ("0478 348 880", "Mobile: 0478 348 880."),
        ("(08) 8922 8196", "Ph: (08) 8922 8196 from Darwin."),
        ("536-22-4399", "SSN 536-22-4399 on file."),
        ("912-70-1234", "ITIN 912-70-1234 on file."),
        ("943 476 5919", "NHS number 943 476 5919 on file."),
        ("1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2", "Pay 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2 now."),
        ("8.8.4.4", "Login from 8.8.4.4 failed."),
        ("2001:4860:4860::8888", "Login from 2001:4860:4860::8888 failed."),
    ],
)
def test_each_kind_of_personal_data_is_removed_exactly(found, text):
    assert redact_text(text) == text.replace(found, REMOVED_TEXT)


@pytest.mark.parametrize(
    "text",
    [
        "Aaron Buzacott (1800-1864) was a missionary.",  # a year range is not a phone number
        "Boston, MA 02110-1301, USA.",  # nor is a ZIP+4 code
        "What is the common factor of 268 and 338752?",
        "We sold 5551234 units and shipped 8341 7141 more.",  # no phone word before them
        "Call 1999-2000 the best years, or contact 2017-02-23.",  # dates
        "Call about the 0.0612422 result.",  # a decimal
        "Peaks sit in the top bin 0.1644-0.17004 of the scan.",  # a range of decimals
        "Scores were 0.96/0.89/0.58/0.02 on the four tests.",  # a list of decimals
        "Row 01    42.1    3.8    21.4    0.82 of the table.",  # a table row
        "He served from 6/1999-2/2002 in the role.",  # a date range
        "Count 1 2 3 4 5 6 7 8 9 10 aloud.",  # a count
        "The mask is 255.255.255.0 here.",  # a dotted quad
        "Use icon@2x.png and left-pad@1.3.0 here.",  # not emails: no public suffix
        "A probe writes to someone@example.com.",  # RFC 2606: nobody's address
        'NSLog(@"Usage: %@',  # a format string, not a cut address
        "Card: 4111 1111 1111 1112.",  # fails the Luhn checksum
        "Pay to GB82 WEST 1234 5698 7654 33 please.",  # fails the IBAN checksum
        "SSN 000-05-1120 and objid=554979041.",  # not issued; no separators
        "Connect to 127.0.0.1, 192.168.0.9 or ::1 at 12:30:45.",  # not public addresses
        "std::vector<int> a; a::b::c; std::__cxx11; .heat:hover::after { }",
        "",
    ],
)
def test_look_alikes_stay(text):
    assert redact_text(text) == text


@pytest.mark.parametrize("cut", ["jane.doe@", "jane.doe@somem", "jane.doe@somemail."])
def test_an_address_cut_by_the_end_of_a_window_is_removed(cut):
    assert redact_text(f"Write to {cut}") == f"Write to {REMOVED_TEXT}"


def test_the_search_opens_no_network_connection(monkeypatch):
    from openinterp import sensitive

    def no_network(*args, **kwargs):
        raise AssertionError("the search opened a network connection")

    monkeypatch.setattr("socket.socket.connect", no_network)
    monkeypatch.setattr("socket.create_connection", no_network)
    sensitive._domains.cache_clear()
    sensitive.pii_matches.cache_clear()
    text = f"Write to {EMAIL} or call (415) 555-2671."
    assert {kind for _, _, kind in sensitive.pii_matches(text)} == {"email", "phone"}


def test_email_domains_use_our_suffix_list_not_the_installed_one():
    from openinterp import sensitive

    assert sensitive._domains()("someone.co.uk").suffix == "co.uk"
    extractor = sensitive._domains()
    assert extractor.suffix_list_urls[0].endswith("openinterp/data/public_suffix_list.dat")
    assert extractor.fallback_to_snapshot is False


def test_a_match_across_tokens_removes_each_token_it_touches():
    tokens = ["Mail", " jane", ".", "doe", "@", "somemail", ".", "org", " now"]
    assert redact_tokens(tokens) == ["Mail", *[REMOVED_TOKEN] * 7, " now"]


def _planted(record):
    """``record`` with an email in each kind of field where the registry's records hold one."""
    sub = record.submission
    example = sub.top_examples[0]
    tokens = [" write", " jane", ".", "doe", "@", "somemail", ".", "org"]
    planted = example.model_copy(
        update={
            "text": "".join(tokens),
            "tokens": tokens,
            "activations": [1.0] * len(tokens),
            "max_activation": 1.0,
            "max_token_index": 4,
        }
    )
    sub = sub.model_copy(
        update={
            "top_examples": [planted, *sub.top_examples[1:]],
            "natural_language_explanation": f"Fires on addresses like {EMAIL}.",
        }
    )
    probes = record.verdict.adversarial_test_results
    verdict = record.verdict.model_copy(
        update={
            "adversarial_test_results": [
                probes[0].model_copy(update={"prompt": f"Mail {EMAIL} now"}),
                *probes[1:],
            ]
        }
    )
    return AuditRecord(submission=sub, verdict=verdict)


def test_a_scrubbed_record_holds_no_match_and_the_original_is_unchanged(record):
    planted = _planted(record)
    before = planted.model_dump_json()
    scrubbed = scrub_record(planted)
    assert planted.model_dump_json() == before and EMAIL in before
    dumped = scrubbed.model_dump_json()
    assert EMAIL not in dumped and "somemail" not in dumped and REMOVED_TEXT in dumped
    example = scrubbed.submission.top_examples[0]
    assert len(example.tokens) == len(example.activations)
    assert example.tokens[0] == " write" and example.tokens[1] == REMOVED_TOKEN


def test_scrubbing_a_clean_record_changes_nothing(record):
    assert scrub_record(record).model_dump_json() == record.model_dump_json()


def test_attribution_tokens_of_a_match_are_removed(record):
    from openinterp.core.attribution import AttributedToken, ContextExample, InputAttribution

    planted = _planted(record)
    context = ContextExample.model_construct(
        example=0,
        peak_token="@",
        top=[
            AttributedToken.model_construct(offset=-3, token=" jane", attribution=0.5),
            AttributedToken.model_construct(offset=-4, token=" write", attribution=0.1),
        ],
    )
    attribution = InputAttribution.model_construct(examples=[context])
    scrubbed = scrub_record(planted).submission.model_copy(
        update={"input_attribution": attribution}
    )
    scrubbed = _scrub_attribution(scrubbed)
    (found,) = scrubbed.input_attribution.examples
    assert found.peak_token == REMOVED_TOKEN
    assert [t.token for t in found.top] == [REMOVED_TOKEN, " write"]


def _site_text(root):
    files = [
        p for p in root.rglob("*") if p.is_file() and p.suffix in {".html", ".json", ".parquet"}
    ]
    return b"\n".join(p.read_bytes() for p in files)


def test_the_built_site_holds_no_personal_data_but_the_record_file_does(tmp_path, record):
    planted = _planted(record)
    compile_site([planted], tmp_path / "site")
    site = _site_text(tmp_path / "site")
    assert EMAIL.encode() not in site and b"somemail" not in site
    assert REMOVED_TOKEN.encode() in site and REMOVED_TEXT.encode() in site
    assert (
        EMAIL in json.loads(planted.model_dump_json())["submission"]["natural_language_explanation"]
    )


def test_the_evidence_numbers_come_from_the_true_probe_text(tmp_path, record, monkeypatch):
    from openinterp.builder import compiler

    seen = []
    real = compiler.build_tree

    def spy(records, errata=(), users=None):
        seen.extend(records)
        return real(records, errata, users)

    explanations = []
    real_stats = compiler.registry_stats

    def stats_spy(parquet, draws=()):
        explanations.extend(pq.read_table(parquet)["explanation"].to_pylist())
        return real_stats(parquet, draws)

    monkeypatch.setattr(compiler, "build_tree", spy)
    monkeypatch.setattr(compiler, "registry_stats", stats_spy)
    compile_site([_planted(record)], tmp_path / "site")
    assert EMAIL in seen[0].model_dump_json()
    assert EMAIL in explanations[0]
    published = pq.read_table(tmp_path / "site" / "data" / "registry.parquet")
    assert EMAIL not in published["explanation"].to_pylist()[0]


@pytest.mark.parametrize(
    "text, flagged",
    [
        ("what the FUCK", True),
        ("the organ of the stroke", False),
        ("Shitake mushrooms and the scunthorpe rapeseed", False),
    ],
)
def test_the_word_list_matches_whole_words_only(text, flagged):
    assert is_offensive(text) is flagged


def test_offensive_examples_are_masked_and_kept_out_of_search(tmp_path, record):
    examples = record.submission.top_examples
    compile_site([record], tmp_path / "clean")
    pages = list((tmp_path / "clean" / "models").rglob("*.html"))
    assert pages and not any('class="spoiler"' in page.read_text() for page in pages)

    flagged = examples[0].model_copy(update={"text": "what the fuck"})
    sub = record.submission.model_copy(update={"top_examples": [flagged, *examples[1:]]})
    compile_site([AuditRecord(submission=sub, verdict=record.verdict)], tmp_path / "flagged")
    pages = [p.read_text() for p in (tmp_path / "flagged" / "models").rglob("*.html")]
    masked = [page for page in pages if 'class="spoiler"' in page]
    assert masked
    for page in masked:
        assert '<div class="spoiler" data-pagefind-ignore>' in page
        assert 'type="checkbox" class="spoiler-toggle"' in page
    # On the experiment's page, the mask covers the example: its heatmap comes after the
    # opening of the guard.
    (page,) = [p.read_text() for p in (tmp_path / "flagged" / "models").rglob("*Z.html")]
    guard = page.index('class="spoiler-body"')
    assert page[guard:].startswith('class="spoiler-body"><span class="hm">')


def test_matches_are_stored_on_disk_and_reused(tmp_path, monkeypatch):
    from openinterp.builder import pii

    def fresh():
        for f in (pii._cache_file, pii._stored, pii.find_spans):
            f.cache_clear()

    monkeypatch.setenv("OPENINTERP_PII_CACHE", str(tmp_path))
    fresh()
    text = "Call me on +44 20 7946 0958 after six."
    spans = pii.find_spans(text)
    assert spans
    pii.save_spans()
    assert list(tmp_path.glob("spans-*.sqlite"))
    fresh()

    def no_search(_text):
        raise AssertionError("searched again")

    monkeypatch.setattr(pii, "_search", no_search)
    assert pii.find_spans(text) == spans
    assert pii.find_spans("ab") == ()  # too short to match: never searched or stored
    fresh()


def test_no_cache_when_turned_off(tmp_path, monkeypatch):
    from openinterp.builder import pii

    monkeypatch.setenv("OPENINTERP_PII_CACHE", "off")
    pii._cache_file.cache_clear()
    assert pii._cache_file() is None
    pii._cache_file.cache_clear()
