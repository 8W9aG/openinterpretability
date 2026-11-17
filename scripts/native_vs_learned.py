"""The numbers of papers/native-vs-learned/README.md, from analyses already in the repo (#3188).

It reads, and computes nothing that needs a model:

* ``docs/analyses/transcoder-baseline.jsonl``. It holds 40 random transcoder latents and
  40 random positive neuron poles of GPT-2 small, with one agent pair and one protocol.
  It also lists the latent draws that never fired.
* the records that those rows name, for the size of each agent's prompt.
* ``docs/analyses/null-calibration-basis.jsonl``. It holds 30 Qwen2.5-0.5B units in the
  native basis, and the same units in a random rotation of it.
* the transcoders' weights, for the number of latents per layer.

    uv run python scripts/native_vs_learned.py --out papers/native-vs-learned/results.jsonl
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import click

TRANSCODER_ROWS = Path("docs/analyses/transcoder-baseline.jsonl")
BASIS_ROWS = Path("docs/analyses/null-calibration-basis.jsonl")
NEURONS_PER_LAYER = 3072  # GPT-2 small's MLP width
AGENTS = ("proposer", "reviewer")


def wilson(k: int, n: int) -> list[float]:
    """The 95% Wilson interval of ``k`` successes in ``n``."""
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def share_row(arm: str, model: str, verified: int, units: int, note: str = "") -> dict:
    """One arm's share VERIFIED, with its interval."""
    return {
        "kind": "share", "arm": arm, "model": model, "verified": verified, "units": units,
        "share": round(verified / units, 4), "ci95": wilson(verified, units), "note": note,
    }  # fmt: skip


def transcoder_shares(rows: list[dict]) -> list[dict]:
    """The latents (live, and with dead draws counted) and the positive poles."""
    live = [r for r in rows if r["arm"] == "latent" and r["outcome"] == "live"]
    dead = [r for r in rows if r["arm"] == "latent" and r["outcome"] == "dead"]
    poles = [r for r in rows if r["arm"] == "positive_pole"]
    kept = sum(r["verdict"] == "VERIFIED" for r in live)
    return [
        share_row("transcoder latents", "GPT-2 small", kept, len(live), "latents that fired"),
        share_row(
            "transcoder latents, dead draws counted",
            "GPT-2 small",
            kept,
            len(live) + len(dead),
            "a dead draw counts as not VERIFIED",
        ),  # fmt: skip
        share_row(
            "positive neuron poles",
            "GPT-2 small",
            sum(r["verdict"] == "VERIFIED" for r in poles),
            len(poles),
        ),  # fmt: skip
    ]


def basis_shares(rows: list[dict]) -> list[dict]:
    """The native and rotated arms of the basis null."""
    out = []
    for arm, name in (("identity", "native positive poles"), ("rotation", "rotated directions")):
        arm_rows = [r for r in rows if r["arm"] == arm]
        verified = sum(r["outcome"] == "VERIFIED" for r in arm_rows)
        out.append(share_row(name, "Qwen2.5-0.5B", verified, len(arm_rows)))
    return out


def prompt_sizes(rows: list[dict]) -> list[dict]:
    """The median prompt of each agent, in characters, for each arm of the baseline."""
    out = []
    for arm in ("latent", "positive_pole"):
        sizes: dict[str, list[int]] = {a: [] for a in AGENTS}
        for row in rows:
            if row["arm"] != arm or "record" not in row:
                continue
            record = json.loads(Path(row["record"]).read_text(encoding="utf-8"))
            costs = (record["submission"].get("provenance") or {}).get("costs") or {}
            for call in costs.get("llm_calls", []):
                if call["agent"] in sizes and call.get("prompt_chars"):
                    sizes[call["agent"]].append(call["prompt_chars"])
        for agent, values in sizes.items():
            out.append({
                "kind": "prompt", "arm": arm, "agent": agent, "calls": len(values),
                "median_chars": statistics.median(values),
            })  # fmt: skip
    return out


def latents_per_layer() -> int:
    """The transcoders' width, read from layer 0's weights at the pinned revision."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from openinterpretability.harness.dictionaries import DICTIONARIES

    spec = DICTIONARIES["gpt2-small-transcoders"]
    path = hf_hub_download(spec.repo, spec.path.format(layer=0), revision=spec.revision)
    with safe_open(path, "pt") as f:
        return f.get_slice("W_enc").get_shape()[1]


def layer_budget(shares: list[dict], width: int) -> dict:
    """Audits, and VERIFIED units to expect, to audit every unit of one GPT-2 layer once.

    This applies the measured shares to a whole layer. It is an extrapolation, not a run.
    A latent's draw is dead with the measured share. A dead draw costs a scan, not an audit.
    """
    live = next(s for s in shares if s["arm"] == "transcoder latents")
    counted = next(s for s in shares if s["arm"] == "transcoder latents, dead draws counted")
    poles = next(s for s in shares if s["arm"] == "positive neuron poles")
    live_share = live["units"] / counted["units"]
    return {
        "kind": "layer", "latents": width, "neurons": NEURONS_PER_LAYER,
        "latent_live_share": round(live_share, 4),
        "latent_audits": round(width * live_share),
        "latent_verified": round(width * live_share * live["share"]),
        "pole_audits": NEURONS_PER_LAYER,
        "pole_verified": round(NEURONS_PER_LAYER * poles["share"]),
    }  # fmt: skip


@click.command()
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(out: Path) -> None:
    """Gather the numbers, and write them."""
    transcoder = [json.loads(line) for line in TRANSCODER_ROWS.open(encoding="utf-8")]
    basis = [json.loads(line) for line in BASIS_ROWS.open(encoding="utf-8")]
    shares = transcoder_shares(transcoder) + basis_shares(basis)
    rows = [*shares, *prompt_sizes(transcoder), layer_budget(shares, latents_per_layer())]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for row in rows:
        click.echo(json.dumps(row))


if __name__ == "__main__":
    main()
