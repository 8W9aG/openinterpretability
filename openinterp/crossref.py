"""Cross-reference neuron poles with published dictionaries (`openinterp crossref`).

For every neuron pole in the archive whose model has a dictionary preset covering its
layer, the latents whose decoder aligns with the pole's write and whose encoder aligns
with its read (:mod:`openinterp.harness.crossref`). Rows go to
``docs/analyses/dictionary-alignment.jsonl``, one per pole and dictionary. They name the
dictionary and each latent's index only: a latent's description lives with the
dictionary's host (Neuronpedia, where :attr:`DictionarySpec.neuronpedia` says it has
them) or in this registry if the latent was audited here, and is looked up when shown.

Agreement between a pole's explanation and an aligned latent's is supporting evidence;
disagreement is something for the reviewer to probe (`openinterp review
--dictionary-latents`). Neither is gated.
"""

from __future__ import annotations

import json
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

from openinterp.core.schemas import AuditRecord, NeuronIdentifier
from openinterp.core.strict import StrictModel

DEFAULT_PATH = Path("docs/analyses/dictionary-alignment.jsonl")
NEURONPEDIA = "https://www.neuronpedia.org"


class LatentMatch(StrictModel):
    feature_idx: int
    cosine: float
    channel: str | None = None  # the neuron's read channel (gate, up, in), for a read


class AlignmentSide(StrictModel):
    """One side's best matches, by |cosine|, with what a random direction reaches
    (``chance``) and the share of the layer's neurons whose best match is weaker."""

    matches: list[LatentMatch]
    chance: float
    percentile: float


class AlignmentRow(StrictModel):
    """One pole against one dictionary. ``writes`` compares each latent's decoder row with
    the pole's write (+: the latent writes what the pole writes); ``reads`` its encoder
    with the neuron's reads (+: the latent reads what raises the neuron's pre-activation),
    ``None`` for a dictionary that doesn't read the MLP's input."""

    slug: str
    dictionary: str
    repo: str
    revision: str
    path: str
    model_revision: str | None
    writes: AlignmentSide
    reads: AlignmentSide | None = None


def load_rows(path: Path | None = DEFAULT_PATH) -> dict[str, list[AlignmentRow]]:
    """The rows in ``path`` by pole slug; empty if it doesn't exist."""
    out: dict[str, list[AlignmentRow]] = defaultdict(list)
    if path is None or not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = AlignmentRow.model_validate_json(line)
            out[row.slug].append(row)
    return out


def dictionaries_for(model_name: str, layer: int) -> list[str]:
    from openinterp.harness.dictionaries import DICTIONARIES

    return sorted(
        name
        for name, spec in DICTIONARIES.items()
        if spec.model_name == model_name and layer in spec.layers
    )


def _poles(records: Iterable[AuditRecord]) -> dict[str, tuple[NeuronIdentifier, Counter]]:
    """Each neuron pole in ``records``, with how often each model revision pinned it."""
    poles: dict[str, tuple[NeuronIdentifier, Counter]] = {}
    for record in records:
        ident = record.submission.identifier
        if not isinstance(ident, NeuronIdentifier):
            continue
        prov = record.submission.provenance
        _, revisions = poles.setdefault(ident.slug, (ident, Counter()))
        revisions[prov.model_revision if prov else None] += 1
    return poles


def _todo(records: Iterable[AuditRecord], out: Path):
    """``({(model, dictionary, layer): poles}, {model: revision})`` for the pairs ``out``
    has no row for. A model is read at the revision its records pin most often."""
    done = {(r.slug, r.dictionary) for rows in load_rows(out).values() for r in rows}
    todo: dict[tuple[str, str, int], list[NeuronIdentifier]] = defaultdict(list)
    revisions: dict[str, Counter] = defaultdict(Counter)
    for slug, (ident, seen) in _poles(records).items():
        revisions[ident.model_name].update(seen)
        for name in dictionaries_for(ident.model_name, ident.layer):
            if (slug, name) not in done:
                todo[(ident.model_name, name, ident.layer)].append(ident)
    return todo, {m: _pinned(seen) for m, seen in revisions.items()}


def crossref(
    records: Iterable[AuditRecord],
    load_runner: Callable[[str, str | None], object],
    out: Path,
    *,
    k: int = 5,
    echo: Callable[[str], None] = print,
) -> int:
    """Append a row to ``out`` for each pole and covering dictionary without one yet,
    loading each model once. Returns how many rows it wrote."""
    from openinterp.core.exceptions import ArchitectureNotSupportedError
    from openinterp.harness.crossref import align_layer
    from openinterp.harness.dictionaries import dictionary_spec

    todo, revisions = _todo(records, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wrote, runner, loaded = 0, None, None
    for model_name, name, layer in sorted(todo):
        if loaded != model_name:
            runner, loaded = load_runner(model_name, revisions[model_name]), model_name
        spec, idents = dictionary_spec(name), todo[(model_name, name, layer)]
        try:
            found = align_layer(runner, spec, layer, [(i.neuron_idx, i.pole) for i in idents], k=k)
        except ArchitectureNotSupportedError as exc:
            echo(f"skip {name} L{layer}: {exc}")
            continue
        with out.open("a") as f:
            for ident in idents:
                write, read = found[(ident.neuron_idx, ident.pole)]
                row = AlignmentRow(
                    slug=ident.slug,
                    dictionary=name,
                    repo=spec.repo,
                    revision=spec.revision,
                    path=spec.file(layer),
                    model_revision=revisions[model_name],
                    writes=_side(write),
                    reads=_side(read) if read is not None else None,
                )
                f.write(row.model_dump_json() + "\n")
        wrote += len(idents)
        echo(f"{name} L{layer}: {len(idents)} poles")
    return wrote


def _pinned(revisions: Counter) -> str | None:
    """The revision most records pin; an unpinned record says nothing about which."""
    pinned = Counter({r: n for r, n in revisions.items() if r is not None})
    return pinned.most_common(1)[0][0] if pinned else None


def _side(side) -> AlignmentSide:
    return AlignmentSide(
        matches=[
            LatentMatch(feature_idx=m.feature_idx, cosine=round(m.cosine, 4), channel=m.channel)
            for m in side.matches
        ],
        chance=round(side.chance, 4),
        percentile=round(side.percentile, 4),
    )


def sort_rows(path: Path) -> None:
    """Rewrite ``path`` in a stable order, so reruns that append don't churn the diff."""
    rows = sorted(
        (r for rows in load_rows(path).values() for r in rows), key=lambda r: (r.slug, r.dictionary)
    )
    path.write_text("".join(r.model_dump_json() + "\n" for r in rows))


def layer_of(row: AlignmentRow) -> int:
    return int(row.slug.split("/L")[1].split("/")[0])


def neuronpedia_url(row: AlignmentRow, feature_idx: int, *, api: bool = False) -> str | None:
    """The latent's Neuronpedia page (or API endpoint), where Neuronpedia hosts it."""
    from openinterp.harness.dictionaries import DICTIONARIES

    spec = DICTIONARIES.get(row.dictionary)
    source = spec and spec.neuronpedia_source(layer_of(row))
    if not source:
        return None
    return f"{NEURONPEDIA}{'/api/feature' if api else ''}/{source}/{feature_idx}"


def neuronpedia_description(row: AlignmentRow, feature_idx: int, timeout: float = 20):
    """The latent's first Neuronpedia explanation, or ``None`` if it has none or the
    request fails."""
    url = neuronpedia_url(row, feature_idx, api=True)
    if url is None:
        return None
    request = urllib.request.Request(url, headers={"User-Agent": "openinterp"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except (OSError, ValueError):
        return None
    explanations = data.get("explanations") or [{}]
    return (explanations[0].get("description") or "").strip() or None


def feature_slug(row: AlignmentRow, feature_idx: int) -> str:
    """The slug the latent would have in this registry, if audited here."""
    model_slug = row.slug.split("/L")[0]
    return f"{model_slug}/L{layer_of(row)}/{row.dictionary}/F{feature_idx}"


def describe(
    rows: Iterable[AlignmentRow],
    registry: dict[str, str] | None = None,
    *,
    fetch: Callable = neuronpedia_description,
    min_percentile: float = 0.0,
) -> list[dict]:
    """Each aligned latent of ``rows`` with its description: the registry's explanation
    (``registry`` maps feature slugs to it) or else Neuronpedia's."""
    registry = registry or {}
    out = []
    for row in rows:
        for side_name in ("writes", "reads"):
            side = getattr(row, side_name)
            if side is None or side.percentile < min_percentile:
                continue
            for m in side.matches:
                if abs(m.cosine) <= side.chance:
                    continue
                slug = feature_slug(row, m.feature_idx)
                text, source = registry.get(slug), "this registry"
                if text is None:
                    text, source = fetch(row, m.feature_idx), "Neuronpedia"
                out.append(
                    {
                        "row": row,
                        "side": side_name,
                        "match": m,
                        "description": text,
                        "source": source if text else None,
                    }
                )
    return out


def registry_explanations(records: Iterable[AuditRecord]) -> dict[str, str]:
    """The newest explanation of every dictionary latent audited here, by slug."""
    newest: dict[str, AuditRecord] = {}
    for record in records:
        ident = record.submission.identifier
        if ident.kind != "dictionary_feature" or record.verdict is None:
            continue
        old = newest.get(ident.slug)
        if old is None or record.verdict.timestamp > old.verdict.timestamp:
            newest[ident.slug] = record
    return {s: r.submission.natural_language_explanation for s, r in newest.items()}


def format_for_reviewer(described: list[dict]) -> str:
    """Aligned latents for the reviewer's prompt: what published dictionaries say a
    latent with this pole's weights does. Only latents above chance are listed."""
    lines = []
    for d in described:
        if not d["description"]:
            continue
        row, m = d["row"], d["match"]
        what = "writes" if d["side"] == "writes" else f"reads (via {m.channel})"
        lines.append(
            f"- {row.dictionary} latent {m.feature_idx} {what} along this pole "
            f"(cosine {m.cosine:+.2f}; chance {getattr(row, d['side']).chance:.2f}): "
            f"{d['description']} [{d['source']}]"
        )
    if not lines:
        return ""
    return (
        "Sparse-dictionary latents whose weights align with this pole's, and what their "
        "authors' descriptions say they do. They are not measurements of this pole. Where "
        "a description disagrees with the hypothesis, write probes that tell them apart; "
        "where it agrees, the hypothesis may still be too broad or too narrow.\n"
        + "\n".join(lines)
        + "\n"
    )
