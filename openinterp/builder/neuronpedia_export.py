"""The registry's VERIFIED explanations in Neuronpedia's export format.

Neuronpedia publishes its explanations as gzipped JSON Lines, one directory per source:
``v1/<model>/<source>/explanations/batch-0.jsonl.gz``. This writes the same layout under
``data/neuronpedia/`` of the site, so that a self-hosted Neuronpedia (or anyone) can
import it. Nothing is sent to Neuronpedia.

Only the current standing of a unit counts: the lead experiment of a VERIFIED standing.
A dictionary latent goes to the source that Neuronpedia hosts it under, with the same
index. A native neuron goes to the source ``<layer>-<NEURON_SET>``, which Neuronpedia
does not host. Both poles of a neuron share its index, so each row also names its pole.
A model is exported only when Neuronpedia knows it, that is when one of its dictionary
presets names a Neuronpedia source. Each row keeps Neuronpedia's field names and adds
the verdict, the protocol version, the data licence and a link to the signed record.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from openinterp.builder.tree import Experiment
from openinterp.builder.wiki import archive_path
from openinterp.core.licenses import DATA_LICENSE

TYPE_NAME = "openinterp-verified"
NEURON_SET = "openinterp-neurons"
EXPORT_DIR = Path("data") / "neuronpedia" / "v1"


def neuronpedia_model(model_name: str) -> str | None:
    """Neuronpedia's id for ``model_name``, from its dictionary presets; ``None`` if unknown."""
    from openinterp.harness.dictionaries import DICTIONARIES

    for spec in DICTIONARIES.values():
        if spec.model_name == model_name and spec.neuronpedia:
            return spec.neuronpedia.split("/", 1)[0]
    return None


def source_of(ident) -> str | None:
    """The Neuronpedia source (``<layer>-<set>``) for ``ident``; ``None`` if not exported."""
    from openinterp.harness.dictionaries import DICTIONARIES

    if ident.kind == "dictionary_feature":
        spec = DICTIONARIES.get(ident.dictionary)
        source = spec and spec.neuronpedia_source(ident.layer)
        return source.split("/", 1)[1] if source else None
    if ident.kind == "mlp_neuron_pole":
        return f"{ident.layer}-{NEURON_SET}"
    return None


def export_row(experiment: Experiment, repo_url: str) -> dict | None:
    """The experiment as one Neuronpedia explanation row; ``None`` if it is not exported."""
    record = experiment.record
    ident = record.submission.identifier
    model, source = neuronpedia_model(ident.model_name), source_of(ident)
    if model is None or source is None:
        return None
    policy = record.verdict.policy
    return {
        "modelId": model,
        "layer": source,
        "index": str(ident.feature_idx if ident.kind == "dictionary_feature" else ident.neuron_idx),
        "pole": ident.pole,
        "description": record.submission.natural_language_explanation,
        "typeName": TYPE_NAME,
        "explanationModelName": record.submission.submitting_agent_model,
        "status": record.verdict.status,
        "selfReviewed": record.self_reviewed,  # no independent reviewer backs it
        "protocolVersion": policy.protocol_version if policy else None,
        "license": DATA_LICENSE,
        "url": f"{repo_url}/blob/main/{archive_path(Path('wiki'), record).as_posix()}",
    }


def exported(experiments: Sequence[Experiment]) -> list[Experiment]:
    """The lead experiment of each VERIFIED standing, in a fixed order."""
    leads = {
        id(e): e
        for e in experiments
        if e.standing is not None and e.standing.verified and e.standing.lead is e
    }
    return sorted(leads.values(), key=lambda e: (e.record.submission.identifier.slug, e.id))


def export_neuronpedia(experiments: Sequence[Experiment], output_dir: Path, repo_url: str) -> int:
    """Write the export under ``output_dir``; return how many rows it holds.

    The files are byte-for-byte the same for the same records: the gzip header has no
    time, and rows are sorted.
    """
    by_source: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for experiment in exported(experiments):
        row = export_row(experiment, repo_url)
        if row is not None:
            by_source[(row["modelId"], row["layer"])].append(row)
    for (model, source), rows in sorted(by_source.items()):
        path = Path(output_dir) / EXPORT_DIR / model / source / "explanations" / "batch-0.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        path.write_bytes(gzip.compress(body.encode(), mtime=0))
    return sum(len(rows) for rows in by_source.values())
