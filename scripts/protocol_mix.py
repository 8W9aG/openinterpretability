"""Which older VERIFIED records would today's protocol still verify? (issue #3207).

Records judged under an older protocol count the same as current ones in the registry's
totals. This script applies today's decision rule (``ReviewPolicy()``) to the stored
results of every VERIFIED record from an older protocol, with no model:

- the active share comes from the record's stored histogram, which protocol 0.8's
  ``not_default_on`` gate needs.
- a record measured with different settings (activation threshold or counterfactual
  tolerances) has probe results that today's settings would not give. It is counted as
  needing ``openinterp reevaluate``, which measures again.

    uv run python scripts/protocol_mix.py [--out docs/analyses/protocol-mix.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from openinterp.agents.reviewer import decide, review_metrics
from openinterp.core.paths import DEFAULT_WIKI_DIR, experiments_dir
from openinterp.core.schemas import PROTOCOL_VERSION, AuditRecord, ReviewPolicy, scan_active_share

# Settings applied when the probes were measured. If one differs from today's, the stored
# pass or fail of a probe is not what today's measurement would give.
MEASUREMENT = ("activation_threshold", "delta_abs_tolerance", "delta_rel_tolerance")


def rejudge(record: AuditRecord, today: ReviewPolicy) -> tuple[str, list[str]] | None:
    """Today's status and failed gates for ``record``, or ``None`` if it must be measured again."""
    verdict = record.verdict
    old, stored = verdict.policy, verdict.metrics
    if old is None or stored is None:
        return None
    if any(getattr(old, f) != getattr(today, f) for f in MEASUREMENT):
        return None
    baseline = None
    if stored.confound_baseline is not None and stored.confound_baseline_accuracy is not None:
        baseline = (stored.confound_baseline, stored.confound_baseline_accuracy)
    metrics = review_metrics(
        verdict.adversarial_test_results,
        today,
        independent=stored.independent_review,
        baseline=baseline,
        active_share=scan_active_share(record.submission, today),
        first_token_floor=record.submission.first_token_floor,
    )
    return decide(metrics, today), sorted(g for g, ok in metrics.gates.items() if not ok)


def report(wiki_dir: Path = DEFAULT_WIKI_DIR) -> dict:
    """Counts per older protocol: kept, lost (by failed gate) and needing a new measurement."""
    today = ReviewPolicy()
    per: dict[str, Counter] = defaultdict(Counter)
    gates: dict[str, Counter] = defaultdict(Counter)
    lost: list[dict] = []
    for path in sorted(experiments_dir(wiki_dir).rglob("*.json")):
        record = AuditRecord.model_validate_json(path.read_text())
        verdict = record.verdict
        if verdict is None or verdict.status != "VERIFIED" or verdict.policy is None:
            continue
        version = verdict.policy.protocol_version
        if version == PROTOCOL_VERSION or record.submission.identifier.kind == "dictionary_feature":
            continue
        result = rejudge(record, today)
        if result is None:
            per[version]["measure_again"] += 1
            continue
        status, failed = result
        per[version]["kept" if status == "VERIFIED" else "lost"] += 1
        if status != "VERIFIED":
            gates[version].update(failed)
            lost.append({"record": str(path), "protocol": version, "now": status, "failed": failed})
    return {
        "current_protocol": PROTOCOL_VERSION,
        "by_protocol": {v: dict(c) for v, c in sorted(per.items())},
        "failed_gates": {v: dict(c) for v, c in sorted(gates.items())},
        "lost": lost,
    }


def main() -> None:
    """Write the report as JSON and print the counts."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, default=Path("docs/analyses/protocol-mix.json"))
    args = ap.parse_args()
    result = report()
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    for version, counts in result["by_protocol"].items():
        print(version, counts, result["failed_gates"].get(version, {}))


if __name__ == "__main__":
    main()
