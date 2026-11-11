"""Find personal data and offensive words in text, for scan labels and the site.

Experiments read this text unchanged. A scan labels what it finds
(:class:`~openinterp.harness.annotations.MatchTagger`). So a record can say how much of a
unit's activation personal data explains. Separately, the site build
removes personal data and masks offensive examples in what it publishes
(``openinterp/builder``).

The search for personal data uses Microsoft Presidio's pattern recognisers, with rules of
our own on top. They need no model and no network. It finds emails, phone numbers, payment
cards, IBANs, US SSNs and ITINs, UK NHS numbers, crypto wallets and public IP addresses. It
does NOT find names or postal addresses. Names and addresses are separate scan labels, which
the site does not remove: :mod:`openinterp.names` and :func:`address_matches`.

The search for offensive words is a short list (``openinterp/data/offensive_words.txt``),
not a classifier. It must be fast for a site build, and needs no model. The list holds only
words that are rarely innocent. It includes words for sexual violence, so news reports can
match. A general profanity list flagged 12% of distinct examples, on words such as "god",
"organ" and "stroke". This list flags 0.6%. The list is English only, and text can be
hateful with none of these words. So a match says that a listed word is there, not that the
text is toxic. No match does not mean that a text is safe.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
from collections.abc import Iterator
from functools import cache
from importlib.metadata import version
from importlib.resources import as_file, files
from typing import Any

_SUFFIX_LIST = "public_suffix_list.dat"  # publicsuffix.org, MPL 2.0

# Presidio's score of a match with no context words. Below this are its weak patterns, such
# as any nine digits for an SSN. In 1,000 sampled documents those matched only object IDs.
MIN_SCORE = 0.4
# Fewer digits match year ranges ("1800-1864") and plain numbers: of 30 matches in 1,000
# sampled documents, about half. _context_phones finds shorter numbers.
MIN_PHONE_DIGITS = 10
# A local number has too few digits for that rule, and a region outside Presidio's has no
# valid format. So a number of these lengths also counts when a phone word comes just before
# it. Over 1,000 documents and 3,990 published examples, this matched 20 texts. All the
# numbers looked real, but for one joke number.
_CONTEXT_PHONE_DIGITS = range(7, 13)
_PHONE_WORD_REACH = 40  # characters: "Media contact:\nA. Person, 8341 7141"
_PHONE_WORD = re.compile(
    r"\b(?:tel|t[ée]l[ée]phone|telefon|tel[ée]fono|phon(?:e|es|ed|ing)|ph|fax|mobile|mob|cell|call|contact"
    r"|whatsapp)\b",
    re.IGNORECASE,
)
_PHONE_DIGITS = re.compile(r"(?<![\d.,/:-])\+?\(?\d(?:[ ().-]{0,2}\d){6,11}(?![\d]|[.,/:-]\d)")
# What such a number is not: a date range ("Call 1999-2000") or a decimal ("0.0612422").
_NOT_A_PHONE = re.compile(r"(?:1[5-9]|20)\d\d\b|\d+\.\d+$")
# Shapes that the phone recogniser accepts, but that are not phone numbers. Explanations hold
# many of them. They are decimals ("0.1644-0.17004"), table rows and lists ("0.96/0.89").
# They are also date ranges ("6/1999-2/2002"), DOIs ("1--2565"), dotted quads and counts.
_NOT_A_PHONE_SHAPE = re.compile(
    r"(?<![\d.])0\.\d|\s{2,}|/|--|(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])|(?:\b\d\s+){3}"
)
# RFC 2606 reserves these for examples, so an address here is nobody's. Reviewers' probes use them.
_RESERVED_DOMAINS = {"example.com", "example.net", "example.org"}
# A window of tokens can end inside an address, so its domain is cut or missing.
_EMAIL_CUT = re.compile(r"(?<![\w.+%-])[\w.+%-]*\w[\w.+%-]*@[\w.-]*$")

# US and UK formats only, on one line. A street's name words take no full stop, so a legal
# citation fails ("27 Mass. App. Ct."). "St" and "Dr" before a capital are a title ("4 Did St
# Joseph"), not a street. _COURT_NAME leaves out citations such as "2008 Supreme Court".
_STREET_TYPE = (
    r"(?:(?:St|Dr)\b(?![ \t]+[A-Z])|(?:Street|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive"
    r"|Court|Ct|Way|Place|Pl|Square|Sq|Terrace|Terr|Parkway|Pkwy|Highway|Hwy|Circle|Cir"
    r"|Crescent|Cres|Close|Gardens|Mews)\b)"
)
_US_STATE = (
    r"(?:A[KLRZ]|C[AOT]|D[CE]|FL|GA|HI|I[ADLN]|K[SY]|LA|M[ADEINOST]|N[CDEHJMVY]|O[HKR]|PA|RI"
    r"|S[CD]|T[NX]|UT|V[AT]|W[AIVY])"
)
_ADDRESS_PATTERNS = {
    "street": re.compile(
        r"(?<![\d.#,-])\b\d{1,5}[A-Za-z]?[ \t]+"
        r"(?:(?:[NSEW]\.?|North|South|East|West)[ \t]+)?"
        rf"(?:[A-Z][\w'’-]*[ \t]+){{1,4}}{_STREET_TYPE}\.?"
    ),
    "po_box": re.compile(r"\bP\.?[ \t]?O\.?[ \t]?Box[ \t]+\d+", re.IGNORECASE),
    "us_city_zip": re.compile(rf"\b[A-Z][a-z]+,?[ \t]+{_US_STATE}[ \t]+\d{{5}}(?:-\d{{4}})?\b"),
    "uk_postcode": re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?[ \t]\d[A-Z]{2}\b"),
}
_COURT_NAME = re.compile(r"\b(?:Supreme|Appeals?|Circuit|District|Superior|County)\b")

# The label of each kind -> Presidio's entity. These match with no context words. Presidio's
# others are left out. URLs, dates and UUIDs are not personal, and its medical licence
# pattern matched a clinical trial ID.
_PII_ENTITIES = {
    "email": "EMAIL_ADDRESS",
    "phone": "PHONE_NUMBER",
    "payment_card": "CREDIT_CARD",
    "iban": "IBAN_CODE",
    "us_ssn": "US_SSN",
    "us_itin": "US_ITIN",
    "uk_nhs": "UK_NHS",
    "crypto_wallet": "CRYPTO",
    "ip_address": "IP_ADDRESS",
}


@cache
def _domains():
    """Read email domains against our copy of the Public Suffix List, never a live one.

    The live list changes, and a scan must give the same labels again. To update the copy,
    replace the file: :func:`detector` records its SHA-256, so records say which one they used.
    """
    import tldextract

    with as_file(files("openinterp.data").joinpath(_SUFFIX_LIST)) as path:
        url = path.resolve().as_uri()
    # fallback_to_snapshot=False: fail, not silently read the list in the installed package.
    return tldextract.TLDExtract(
        suffix_list_urls=(url,), cache_dir=None, fallback_to_snapshot=False
    )


@cache
def _recognisers() -> dict[str, Any]:
    """Each kind's Presidio recogniser. Importing Presidio loads spaCy, so only on first use."""
    from presidio_analyzer import predefined_recognizers as recognisers

    class OfflineEmail(recognisers.EmailRecognizer):
        def validate_result(self, pattern_text: str) -> bool:
            domain = pattern_text.rpartition("@")[2].lower()
            return domain not in _RESERVED_DOMAINS and _domains()(domain).suffix != ""

    return {
        "email": OfflineEmail(),
        "phone": recognisers.PhoneRecognizer(),
        "payment_card": recognisers.CreditCardRecognizer(),
        "iban": recognisers.IbanRecognizer(),
        "us_ssn": recognisers.UsSsnRecognizer(),
        "us_itin": recognisers.UsItinRecognizer(),
        "uk_nhs": recognisers.NhsRecognizer(),
        "crypto_wallet": recognisers.CryptoRecognizer(),
        "ip_address": recognisers.IpRecognizer(),
    }


def _is_public_ip(address: str) -> bool:
    # "::a" is a valid address, but in text it is C++ or CSS: "std::a", ".x::after".
    if ":" in address and sum(bool(group) for group in address.split(":")) < 3:
        return False
    try:
        return ipaddress.ip_network(address, strict=False).is_global
    except ValueError:  # the pattern also matches other dotted and colon-separated numbers
        return False


def _keeps(kind: str, match: str) -> bool:
    """Our rules on top of Presidio's, from what it matched in the registry's documents."""
    if kind == "phone":
        enough_digits = sum(c.isdigit() for c in match) >= MIN_PHONE_DIGITS
        return enough_digits and not _NOT_A_PHONE_SHAPE.search(match)
    return kind != "ip_address" or _is_public_ip(match)


def _context_phones(text: str) -> Iterator[tuple[int, int, str]]:
    for match in _PHONE_DIGITS.finditer(text):
        digits = sum(c.isdigit() for c in match.group())
        if digits not in _CONTEXT_PHONE_DIGITS or _NOT_A_PHONE.match(match.group()):
            continue
        if _NOT_A_PHONE_SHAPE.search(match.group()):
            continue
        if _PHONE_WORD.search(text, max(0, match.start() - _PHONE_WORD_REACH), match.start()):
            yield (*match.span(), "phone")


def _cut_email(text: str) -> Iterator[tuple[int, int, str]]:
    cut = _EMAIL_CUT.search(text)
    if cut and cut.group().rpartition("@")[2].rstrip(".").lower() not in _RESERVED_DOMAINS:
        yield (*cut.span(), "email")


def may_match(text: str) -> bool:
    """Return False for a text too short for any match: most record strings are single tokens."""
    return "@" in text or text.count(":") >= 2 or sum(map(str.isdigit, text)) >= 4


@cache
def pii_matches(text: str) -> tuple[tuple[int, int, str], ...]:
    """Return ``(start, end, kind)`` of each match in ``text``, sorted; matches can overlap.

    Many pages show the same text, so the result is cached.
    """
    if not may_match(text):
        return ()
    found = {*_cut_email(text), *_context_phones(text)}
    for kind, recogniser in _recognisers().items():
        for match in recogniser.analyze(text, [_PII_ENTITIES[kind]], None):
            if match.score >= MIN_SCORE and _keeps(kind, text[match.start : match.end]):
                found.add((match.start, match.end, kind))
    return tuple(sorted(found))  # a set, because the phone recogniser repeats a number per region


def address_matches(text: str) -> tuple[tuple[int, int, str], ...]:
    """Return ``(start, end, kind)`` of each postal address part in ``text``, sorted.

    These are a scan label only. The site does not remove them. The patterns cannot tell a
    home from a business, and most matches are organisations.
    """
    found = (
        (*match.span(), kind)
        for kind, pattern in _ADDRESS_PATTERNS.items()
        for match in pattern.finditer(text)
        if kind != "street" or not _COURT_NAME.search(match.group())
    )
    return tuple(sorted(found))


def _data(name: str) -> str:
    return files("openinterp.data").joinpath(name).read_text(encoding="utf-8")


@cache
def _offensive_pattern() -> re.Pattern[str]:
    alternatives = "|".join(re.escape(word) for word in _data("offensive_words.txt").split())
    return re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)


@cache
def is_offensive(text: str) -> bool:
    """Return whether ``text`` holds a listed word; many pages show the same example."""
    return _offensive_pattern().search(text) is not None


def detector() -> str:
    """Name the package versions and the suffix list that fix what a search finds."""
    names = ("presidio-analyzer", "phonenumbers", "tldextract")
    packages = ", ".join(f"{p} {version(p)}" for p in names)
    return f"openinterp.sensitive ({packages}; suffix list {_sha12(_data(_SUFFIX_LIST))})"


def _sha12(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]
