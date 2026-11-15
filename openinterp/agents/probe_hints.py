"""What the reviewer is told about a scan's label sets, and the proposer's part-of-speech flag.

A scan labels its corpus by language, by syntax (code) and by part of speech
(``openinterp.names.pos_matches``, issue #3223). Each label set that a submission's scan
has adds one hint to the reviewer's probe prompt. The part-of-speech set also tags each
probe's words in the token check, and flags poles that it alone explains (issue #3631).
"""

from __future__ import annotations

from openinterp.core.confounds import LANGUAGE, PART_OF_SPEECH, SYNTAX_NODE, ConfoundProfile
from openinterp.core.schemas import NeuronAuditSubmission

LANGUAGE_PROBES = (
    "The scan corpus is labelled by language, and each example above gives its language. "
    "If the hypothesis claims the feature whatever the language, include positive holdouts "
    "that express it in other languages, translated. If it names or implies one language, "
    "include hard negatives in another language written in the same script (e.g. Spanish "
    "against Portuguese, Russian against Ukrainian) that keep the examples' words and "
    "topic as far as that language allows.\n"
)

SYNTAX_PROBES = (
    "The scan corpus is labelled by syntax, and each code example above gives its peak "
    "token's syntax node, bracket depths, indent depth and syntax construct. Include hard "
    "negatives that keep the token and change its syntactic role: `return` as a keyword, "
    "inside a string and inside a comment. If the hypothesis claims a feature whatever the "
    "programming language, include positive holdouts in another programming language.\n"
)

POS_PROBES = (
    "The scan corpus is tagged by part of speech. Include hard negatives that keep a "
    "trigger word and change its part of speech: 'run' as a verb and as a noun, 'that' as a "
    "determiner and as a conjunction.\n"
)

POS_CHECK = (
    "Each 'parts of speech' line tags the probe's words as word/TAG (Universal POS tags). "
    "Check each label against the role the hypothesis names, too: a probe that keeps a "
    "trigger word in another role is a hard negative only if the hypothesis names the "
    "role.\n"
)

# The proposer's flag needs part of speech alone to explain this much of the pole's
# variance. It must also explain at least this share of what token identity explains.
# Over 69 records whose scans have the label set (2026-11-15), the median R² was 0.024,
# and 2 met the flag.
POS_FLAG_R2, POS_FLAG_SHARE = 0.2, 0.5


def has_syntax_labels(submission: NeuronAuditSubmission) -> bool:
    """Does a top example say what syntax its peak token has?"""
    return any(SYNTAX_NODE in (e.peak_labels or {}) for e in submission.top_examples)


def has_language_labels(submission: NeuronAuditSubmission) -> bool:
    """Do the submission's top examples say which language each is in?"""
    return any(LANGUAGE in (e.peak_labels or {}) for e in submission.top_examples)


def has_pos_labels(submission: NeuronAuditSubmission) -> bool:
    """Did the submission's scan tag its corpus by part of speech?"""
    labels = submission.confounds.labels if submission.confounds else None
    return bool(labels) and PART_OF_SPEECH in labels


def pos_line(text: str) -> str:
    """The token check's line for one probe: each word as ``word/TAG``."""
    from openinterp.names import pos_matches

    tags = " ".join(f"{text[a:b]}/{tag}" for a, b, tag in pos_matches(text))
    return f"   parts of speech: {tags or 'none'}"


def pos_flag(confounds: ConfoundProfile) -> str:
    """The proposer's note when part of speech alone explains most of the pole, else ''."""
    found = (confounds.labels or {}).get(PART_OF_SPEECH)
    if found is None or found.r2 < max(POS_FLAG_R2, POS_FLAG_SHARE * confounds.token_r2):
        return ""
    top = ", ".join(f"{s.label} {s.active}/{s.count}" for s in found.levels[:3])
    return (
        f"\nPart of speech alone explains {found.r2:.0%} of activation variance, against "
        f"{confounds.token_r2:.0%} for token identity (most active: {top}). Consider a "
        "part-of-speech family (e.g. every determiner) before a narrower story.\n"
    )
