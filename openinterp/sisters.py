"""Sister neurons across models (`openinterp sisters`).

For every VERIFIED neuron pole of a model, the poles of another model whose activity over
the corpus correlates best with it, each with its own best partner back in the first
model and a rotated-basis reference (:mod:`openinterp.harness.sisters`). The models may
tokenize differently: they are compared where both have read the same text. Rows go to
``docs/analyses/sister-neurons.jsonl``, one per pole and other model.

A *sister* is a mutual best match above the rotated reference: a candidate for the same
feature in both models. Whether the two do represent the same concept is a further
experiment (the pole's explanation tested on its sister); a row is observational
evidence for one. Reported, not gated.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from openinterp.core.identifiers import model_slug
from openinterp.core.schemas import AuditRecord, Pole
from openinterp.core.strict import StrictModel

DEFAULT_PATH = Path("docs/analyses/sister-neurons.jsonl")

# Models searched for a model's sisters besides the registry's other models: a larger
# sibling of the same family, which the registry doesn't audit.
SIBLINGS = {"Qwen/Qwen2.5-0.5B": ("Qwen/Qwen2.5-1.5B",)}


class SisterMatch(StrictModel):
    """A pole of the other model and how its activity correlates with the source pole's.
    ``back`` is its best partner among all of the source model's poles."""

    layer: int
    neuron_idx: int
    pole: Pole
    pearson: float
    back_slug: str
    back_pearson: float

    def slug(self, other: str) -> str:
        return f"{model_slug(other)}/L{self.layer}/N{self.neuron_idx}-{self.pole}"


class SisterRow(StrictModel):
    """One VERIFIED pole against one other model: its best-correlated poles there,
    strongest first, and ``rotated``, the best correlation it reaches with a unit of a
    random rotation of the first match's layer. ``tokens`` positions were paired, the
    share ``coverage`` of the source model's (each text's first token left out)."""

    slug: str
    other: str
    model_revision: str | None
    other_revision: str | None
    corpus: str
    corpus_sha256: str
    tokens: int
    coverage: float
    matches: list[SisterMatch]
    rotated: float

    @property
    def top(self) -> SisterMatch | None:
        return self.matches[0] if self.matches else None

    @property
    def mutual(self) -> bool:
        """The best match's own best partner is this pole."""
        return self.top is not None and self.top.back_slug == self.slug

    @property
    def sister(self) -> SisterMatch | None:
        """The best match, if it is mutual and beats the rotated reference."""
        top = self.top
        return top if self.mutual and top.pearson > self.rotated else None


def load_rows(path: Path | None = DEFAULT_PATH) -> dict[str, list[SisterRow]]:
    """The rows in ``path`` by pole slug; empty if it doesn't exist."""
    out: dict[str, list[SisterRow]] = defaultdict(list)
    if path is None or not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = SisterRow.model_validate_json(line)
            out[row.slug].append(row)
    return out


def sort_rows(path: Path) -> None:
    """Rewrite ``path`` in a stable order, so reruns that append don't churn the diff."""
    rows = sorted(
        (r for rows in load_rows(path).values() for r in rows), key=lambda r: (r.slug, r.other)
    )
    path.write_text("".join(r.model_dump_json() + "\n" for r in rows))


def verified_poles(records: Sequence[AuditRecord], model_name: str):
    """``{slug: (identifier, revision)}`` for each neuron pole of ``model_name`` whose
    standing is VERIFIED, with the model revision its records pin most often."""
    from openinterp.builder.tree import build_tree

    revisions: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        ident = r.submission.identifier
        if ident.kind == "mlp_neuron_pole" and ident.model_name == model_name:
            prov = r.submission.provenance
            revisions[ident.slug][prov.model_revision if prov else None] += 1
    out = {}
    for model in build_tree(records):
        if model.name != model_name:
            continue
        for layer in model.layers:
            for node in layer.units:
                for standing in node.standings.values():
                    if node.is_neuron and standing.verified:
                        ident = standing.lead.record.submission.identifier
                        out[ident.slug] = (ident, revisions[ident.slug])
    return out


def _pinned(revisions) -> str | None:
    pinned = [(n, r) for r, n in revisions.items() if r is not None]
    return max(pinned)[1] if pinned else None


def pairs(models: Sequence[str]) -> list[tuple[str, str]]:
    """Every ordered pair of ``models``, and each with its :data:`SIBLINGS`."""
    out = [(a, b) for a in models for b in models if a != b]
    out += [(a, b) for a in models for b in SIBLINGS.get(a, ()) if b not in models]
    return sorted(out)


def sisters(
    records: Sequence[AuditRecord],
    load_runner: Callable[[str, str | None], object],
    out: Path,
    texts: Sequence[str],
    *,
    model_name: str,
    other: str,
    corpus: str,
    corpus_sha256: str,
    other_revision: str | None = None,
    k: int = 5,
    echo: Callable[[str], None] = print,
) -> int:
    """Append a row to ``out`` for each VERIFIED pole of ``model_name`` without one for
    ``other`` yet. Returns how many rows it wrote."""
    from openinterp.harness.environment import run_environment
    from openinterp.harness.sisters import find_sisters, tokenize

    done = {(r.slug, r.other) for rows in load_rows(out).values() for r in rows}
    poles = {s: v for s, v in verified_poles(records, model_name).items() if (s, other) not in done}
    if not poles:
        return 0
    revision = _pinned(sum((revisions for _, revisions in poles.values()), Counter()))
    source, oth = load_runner(model_name, revision), load_runner(other, other_revision)
    units = {(i.layer, i.neuron_idx, i.pole): slug for slug, (i, _) in poles.items()}
    echo(f"{len(units)} VERIFIED poles of {model_name} against {other}")
    progress = lambda stage, n: echo(f"{stage}: {n} positions")  # noqa: E731
    found, tokens = find_sisters(
        source, oth, list(units), texts, other_name=other, k=k, progress=progress
    )
    readable = sum(max(len(tokenize(source, t).ids) - 1, 0) for t in texts)
    common = {
        "other": other,
        "model_revision": run_environment(source).model_revision or revision,
        "other_revision": run_environment(oth).model_revision or other_revision,
        "corpus": corpus,
        "corpus_sha256": corpus_sha256,
        "tokens": tokens,
        "coverage": round(tokens / max(readable, 1), 4),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for unit, slug in sorted(units.items(), key=lambda u: u[1]):
            f.write(
                _row(slug, model_slug(model_name), found[unit], common).model_dump_json() + "\n"
            )
    return len(units)


def _row(slug: str, prefix: str, found, common: dict) -> SisterRow:
    """A :class:`SisterRow` from :func:`~openinterp.harness.sisters.find_sisters`'s
    result for one pole; ``prefix`` is the source model's slug."""
    matches = [
        SisterMatch(
            layer=m.layer,
            neuron_idx=m.neuron_idx,
            pole=m.pole,
            pearson=round(m.pearson, 4),
            back_slug=f"{prefix}/L{b.layer}/N{b.neuron_idx}-{b.pole}",
            back_pearson=round(b.pearson, 4),
        )
        for m, b in zip(found.matches, found.back, strict=True)
    ]
    return SisterRow(slug=slug, matches=matches, rotated=round(found.rotated, 4), **common)


def by_match_slug(rows: dict[str, list[SisterRow]]) -> dict[str, list[SisterRow]]:
    """The rows naming each pole as a match, so the matched pole's page can link back."""
    out: dict[str, list[SisterRow]] = defaultdict(list)
    for rs in rows.values():
        for row in rs:
            for m in row.matches:
                out[m.slug(row.other)].append(row)
    return out


def _href(slug: str) -> str:
    """A pole slug's neuron page: ``M/L3/N7-positive`` → ``models/M/L3/N7.html``."""
    from openinterp.builder.tree import neuron_href

    model, layer, unit = slug.split("/")
    return neuron_href(model, int(layer[1:]), int(unit[1:].split("-")[0]))


def sister_view(
    model_slug_: str, neuron, rows: dict, named_by: dict, studied: set, tests: dict | None = None
) -> list[dict]:
    """Each pole's matches for a neuron page: its best-correlated poles in other models
    (``matches``), and the poles of other models that name it as one of theirs
    (``named_by``, from :func:`by_match_slug`). Each links to its page where the
    registry has one, and carries the verdict of a sister study between the two
    (``tests``, from :func:`sister_tests`), if any."""
    tests = tests or {}
    if not neuron.is_neuron:
        return []
    poles = []
    for pole in neuron.sides:
        slug = f"{model_slug_}/L{neuron.layer}/N{neuron.idx}-{pole}"
        out = []
        for row in rows.get(slug, []):
            matches = []
            for m in row.matches:
                href = _href(m.slug(row.other))
                matches.append(
                    {
                        "m": m,
                        "href": href if href in studied else None,
                        "mutual": m.back_slug == slug,
                        "tested": tests.get((m.slug(row.other), slug)),
                    }
                )
            out.append({"row": row, "matches": matches, "sister": row.sister})
        back = []
        for row in named_by.get(slug, []):
            m = next(m for m in row.matches if m.slug(row.other) == slug)
            href = _href(row.slug)
            tested = tests.get((slug, row.slug))
            back.append(
                {"row": row, "m": m, "href": href if href in studied else None, "tested": tested}
            )
        if out or back:
            poles.append({"pole": pole, "rows": out, "named_by": back})
    return poles


def pole_findings(records: Sequence[AuditRecord]) -> dict[str, tuple[str, str]]:
    """``{slug: (status, explanation)}`` for every studied neuron pole: its standing and
    the explanation of the experiment that standing rests on."""
    from openinterp.builder.tree import build_tree

    out = {}
    for model in build_tree(records):
        for layer in model.layers:
            for node in layer.units:
                if not node.is_neuron:
                    continue
                for standing in node.standings.values():
                    sub = standing.lead.record.submission
                    out[sub.identifier.slug] = (standing.status, sub.natural_language_explanation)
    return out


# A linked pole that isn't a sister reaches an agent's prompt only from this correlation:
# weaker links mostly join unrelated features (r ≈ 0.1 between a preposition pole and a
# body-region pole), and would mislead more than help.
MIN_PROMPT_PEARSON = 0.5


def _linked(slug: str, rows: dict, named_by: dict) -> list[tuple[str, float, bool]]:
    """``(other pole's slug, r, sister)`` for every pole linked to ``slug`` strongly
    enough to show an agent: a sister either way, or a match with r of at least
    :data:`MIN_PROMPT_PEARSON`."""
    out = []
    for row in rows.get(slug, []):
        for m in row.matches:
            sister = row.sister is m
            if sister or m.pearson >= MIN_PROMPT_PEARSON:
                out.append((m.slug(row.other), m.pearson, sister))
    for row in named_by.get(slug, []):
        m = next(m for m in row.matches if m.slug(row.other) == slug)
        sister = row.sister is m
        if sister or m.pearson >= MIN_PROMPT_PEARSON:
            out.append((row.slug, m.pearson, sister))
    return out


INTRO = (
    "Neurons in other models whose activity over the corpus correlates with this pole's "
    "(`openinterp sisters`: r is the Pearson correlation of the two poles' strengths where "
    "both models have read the same text; a sister is a mutual best match, each the other's "
    "best partner, above what a random rotation of the layer reaches), "
    "and what this registry's experiments concluded about them. A correlated neuron may "
    "encode a related feature rather than the same one."
)
ADVICE = {
    "proposer": "Treat them as leads: ground the hypothesis in this pole's own evidence above.",
    "reviewer": "Where the hypothesis and a correlated neuron's explanation differ, write "
    "probes that tell them apart; where they agree, the shared account may still be too "
    "broad or too narrow.",
}


def format_sisters(slug: str, rows: dict, findings: dict, role: str) -> str:
    """The prompt section on ``slug``'s studied sister neurons for ``role`` (proposer or
    reviewer), strongest first, or "" if none has been studied."""
    best: dict[str, tuple[float, bool]] = {}
    for other, r, sister in _linked(slug, rows, by_match_slug(rows)):
        if other in findings and (other not in best or r > best[other][0]):
            best[other] = (r, sister)
    lines = []
    for other, (r, sister) in sorted(best.items(), key=lambda kv: -kv[1][0]):
        status, explanation = findings[other]
        tag = f"r {r:+.2f}{', sister' if sister else ''}; {status}"
        lines.append(f"- {other} ({tag}): {explanation}")
    if not lines:
        return ""
    return f"{INTRO} {ADVICE[role]}\n" + "\n".join(lines) + "\n"


def sister_context(identifier, wiki_dir: Path, role: str, path: Path = DEFAULT_PATH) -> str:
    """:func:`format_sisters` for the unit ``identifier`` names, from the rows in
    ``path`` and the archive in ``wiki_dir``; "" for a unit that isn't a neuron pole."""
    from openinterp.builder.wiki import load_archive

    rows = load_rows(path)
    if identifier.kind != "mlp_neuron_pole" or not rows or not Path(wiki_dir).exists():
        return ""
    return format_sisters(identifier.slug, rows, pole_findings(load_archive(wiki_dir)), role)


def load_sister_source(path: Path, unit) -> AuditRecord:
    """The archived record ``audit --sister-of`` names, checked: a VERIFIED experiment on
    a neuron pole of a model other than ``unit``'s (raises ValueError otherwise)."""
    record = AuditRecord.model_validate_json(Path(path).read_text())
    ident = record.submission.identifier
    if record.verdict is None or record.verdict.status != "VERIFIED":
        raise ValueError(f"{path} is not a VERIFIED experiment")
    if ident.kind != "mlp_neuron_pole" or unit.kind != "mlp_neuron_pole":
        raise ValueError("sister studies are between neuron poles")
    if ident.model_name == unit.model_name:
        raise ValueError(f"{ident} is in {unit.model_name} too; a sister is in another model")
    return record


def sister_ref(record: AuditRecord) -> str:
    """``<pole slug>/<experiment id>``: what a submission's ``sister_of`` names."""
    from openinterp.builder.wiki import experiment_id

    return f"{record.submission.identifier.slug}/{experiment_id(record)}"


def sister_brief(source: AuditRecord) -> str:
    """The proposer's brief for a sister study: the explanation under test is fixed."""
    ident = source.submission.identifier
    return (
        f"This is a sister-neuron study. The hypothesis under test is fixed: the explanation "
        f"VERIFIED for {ident} in {ident.model_name}, a pole whose activity correlates with "
        f"this one's (`openinterp sisters`). It is recorded as this submission's explanation "
        f"word for word, whatever you write, so the review tests whether it carries over:\n\n"
        f"> {source.submission.natural_language_explanation}\n\n"
        f"Write `explanation` as exactly that text. Choose the activating tokens, mechanism and "
        f"counterfactuals that this explanation predicts for this pole, using the evidence "
        f"above; where the evidence disagrees with it, keep to the explanation anyway.\n"
    )


def sister_source_href(ref: str) -> str:
    """The page and anchor of the experiment a ``sister_of`` names."""
    slug, experiment = ref.rsplit("/", 1)
    return f"{_href(slug)}#{slug.rsplit('-', 1)[1]}-{experiment}"


def sister_tests(experiments) -> dict[tuple[str, str], str]:
    """``{(sister's slug, source pole's slug): status}``: the newest verdict of each sister
    study, by the pole it tested and the pole whose explanation it tested."""
    newest: dict[tuple[str, str], object] = {}
    for e in experiments:
        ref = e.record.submission.sister_of
        if ref:
            key = (e.record.submission.identifier.slug, ref.rsplit("/", 1)[0])
            old = newest.get(key)
            if old is None or e.record.verdict.timestamp > old.record.verdict.timestamp:
                newest[key] = e
    return {key: e.record.verdict.status for key, e in newest.items()}
