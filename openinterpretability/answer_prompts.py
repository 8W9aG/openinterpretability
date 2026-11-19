"""The proposer's and the reviewer's prompts for an answer experiment (issue #4778).

The proposer reads the newest explanation of the prompt. That is its coverage, its minimal
set, its top units with their VERIFIED explanations, and the paths between them. It also
reads every earlier answer claim on the prompt, with its measures and its failed gates. The
reviewer reads only the claim, never a measure, as a unit's reviewer does.

Every text that a person or another agent wrote is fenced as data
(:mod:`openinterpretability.core.fence`). That is the prompt, the explanation's units and
paths, and each earlier or reviewed claim. An explanation record can come from a fork
(#5173).
"""

from __future__ import annotations

import json

from openinterpretability.core.answer_records import (
    MAX_ATTEMPTS,
    AnswerClaim,
    AnswerPolicy,
    AnswerRecord,
)
from openinterpretability.core.explanation_records import ExplanationRecord
from openinterpretability.core.fence import FENCE_RULE, fence


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _explanation(record: ExplanationRecord | None) -> list[str]:
    if record is None:
        return [
            "No explanation of this prompt is recorded yet. Run `open-interpretability "
            "explain` with `--record` first, so the claim can start from its units."
        ]
    detail = record.coverage_detail
    lines = [
        f"From the explanation of {record.written:%Y-%m-%d %H:%M} UTC:",
        f"- Prompt coverage {_pct(record.prompt_coverage)}, against "
        f"{_pct(record.random_coverage)} for matched random units.",
        f"- Explained share of the first-order attribution: {_pct(record.explained_share)}.",
    ]
    if detail is not None and detail.minimal_units:
        lines.append(f"- Minimal set ({_pct(detail.minimal_coverage)} coverage): "
                     + ", ".join(detail.minimal_units))  # fmt: skip
    lines += ["", "Top units (share of |attribution|, status, VERIFIED explanation):"]
    for u in record.units:
        said = f": {u.explanation}" if u.explanation else ""
        lines.append(f"- {u.unit} ({u.kind}) {u.share:.1%}, {u.attribution:+.3f}, {u.status}{said}")
    edges = record.figures.edges if record.figures is not None else []
    if edges:
        lines += ["", "Strongest paths between top units (source → target, weight):"]
        lines += [f"- {e['source']} → {e['target']}: {e['weight']:+.3f}" for e in edges[:12]]
    return lines


def _attempts(records: list[AnswerRecord]) -> list[str]:
    if not records:
        return ["None. This is the first claim on this prompt."]
    lines = []
    for r in records:
        m, failed = r.verdict.measures, r.verdict.failed
        lines += [
            f"### Attempt {r.claim.attempt} ({r.id}): {r.verdict.status}",
            f"- Route: {' → '.join(r.claim.route)}. Sources: {', '.join(r.claim.sources)}.",
            f"- Claim: {r.claim.explanation}",
            f"- Necessity {m.necessity:.2f}, route cut {m.route_cut:.3f}, mean random cut "
            f"{sum(m.null_cuts) / max(len(m.null_cuts), 1):.3f}, sufficiency "
            f"{_pct(m.sufficiency)}, patching {_pct(m.patching)}.",
            f"- Failed gates: {', '.join(failed) or 'none'}.",
            "",
        ]
    return lines


def proposer_prompt(  # long-function: the prompt's text, one line of it per line
    claim_path: str, model: str, prompt: str, position: int, answer: str,
    explanation: ExplanationRecord | None, earlier: list[AnswerRecord],
    attempt: int, agent_model: str = "<your model id>",
) -> str:  # fmt: skip
    """The proposer's task: write one :class:`AnswerClaim` to ``claim_path``."""
    policy = AnswerPolicy()
    template = {
        "model_name": model, "prompt": prompt, "position": position, "answer": answer,
        "sources": ["<a prompt token the route reads>"],
        "route": ["L2/H6", "L3/N1920+"], "roles": {"L2/H6": "<what it does here>"},
        "explanation": "<the route in a few sentences>",
        "corrupted": "<the prompt with the sources changed, same token count>",
        "counterfactuals": [{"prompt": "<a changed prompt>", "expect": "kept", "why": "<why>"}],
        "attempt": attempt, "supersedes": earlier[-1].id if earlier else None,
        "submitting_agent_model": agent_model,
    }  # fmt: skip
    return "\n".join([
        f"# Answer claim: attempt {attempt} of at most {MAX_ATTEMPTS}",
        "",
        f"Model `{model}` answers {answer!r} after this prompt, at position {position}:",
        "",
        fence(prompt),
        "",
        FENCE_RULE,
        "",
        "Write one falsifiable claim about **how** the model produces this answer here: the",
        "route. Name the prompt tokens it reads (sources) and the units in order: heads that",
        "move information (`L2/H6`) and neuron poles that write the answer (`L3/N7+`,",
        "`L3/N7-`). Keep the route short. Every unit in it must matter.",
        "",
        "## The tests the claim must pass",
        "",
        f"- Mean-ablating the route cuts the answer's log-probability by at least "
        f"{policy.min_necessity:.0%} of the cut from ablating every unit, and by "
        f"{policy.null_factor:g}× the cut of matched random sets.",
        f"- With only the route live, at least {policy.min_sufficiency:.0%} of the answer "
        "comes back.",
        f"- Patching the route from `corrupted` moves the answer by at least "
        f"{policy.min_patching:.0%} of the gap between the two prompts. The reviewer writes "
        f"{policy.min_reviewer_corrupted}–3 more corrupted prompts, and the same must hold on "
        f"each. The review draws the random sets after you submit, and on each prompt the "
        f"{policy.null_quantile:.0%} quantile of their |share| must be at most "
        f"{policy.max_null_patching:.0%}.",
        "- Each of your 4–8 counterfactuals says if the answer stays the top token (`kept`)",
        "  or not (`changed`). All of them must come out as you say. A reviewer then writes",
        f"  10–30 more, blind, and {policy.min_probes:.0%} must come out as the claim says.",
        "",
        "## Evidence",
        "",
        fence("\n".join(_explanation(explanation))),
        "",
        "## Earlier claims on this prompt",
        "",
        fence("\n".join(_attempts(earlier))),
        "Don't repeat a claim that failed. Change the route or the claim where its gates",
        "failed.",
        "",
        "## Output",
        "",
        f"Write the claim as JSON to `{claim_path}`, in this form:",
        "",
        "```json",
        json.dumps(template, indent=1),
        "```",
        "",
        "`corrupted` must have the same token count as the prompt. Check the file with",
        f"`uv run open-interpretability answers check --claim {claim_path}`.",
    ])  # fmt: skip


def reviewer_prompt(probes_path: str, claim: AnswerClaim,  # long-function: the prompt's text
                    agent_model: str = "<your model id>") -> str:  # fmt: skip
    """The reviewer's task: write :class:`AnswerProbes` that try to break ``claim``."""
    template = {"prompts": [{"prompt": "<a prompt>", "expect": "changed", "why": "<why>"}],
                "corrupted": ["<the prompt with the sources changed>", "<another>"],
                "reviewer_agent_model": agent_model}  # fmt: skip
    roles = [f"- {u}: {claim.roles.get(u, '(no role given)')}" for u in claim.route]
    shown = [f"- {c.prompt!r}: {c.expect}" for c in claim.counterfactuals]
    told = [f"After this prompt, the model answers {claim.answer!r} by this route:", "",
            claim.prompt, "", f"- Sources: {', '.join(claim.sources)}",
            f"- Route: {' → '.join(claim.route)}", *roles, "", claim.explanation, "",
            "The proposer's own prompts (don't repeat them):", *shown, "",
            "The proposer's corrupted prompt (don't repeat it):", claim.corrupted]  # fmt: skip
    return "\n".join([
        f"# Review an answer claim: {claim.model_name}",
        "",
        "The proposer's claim, as it wrote it:",
        "",
        fence("\n".join(told)),
        "",
        FENCE_RULE,
        "",
        "## Your task",
        "",
        "Write 10–30 prompts meant to break the claim. For each, say what the claim",
        "predicts: `kept` (the answer",
        "stays the model's top token) or `changed`. Test the claim's edges: other sources",
        "that the route should carry the same way, sources it should not carry, distractors,",
        "changed word order, and prompts where the answer is likely for another reason.",
        "You will never see the results.",
        "",
        "Also write 2–3 corrupted prompts of your own (`corrupted`). Each is the claim's",
        "prompt with its sources changed so that the answer should change, and it must have",
        "the prompt's token count: swap a source word for another word of the same form.",
        "The route must carry the answer's change from each of them, so choose ones that",
        "test the claim, not ones that make it pass. The harness checks the token counts and",
        "sends the file back if one is wrong.",
        "",
        f"Write them as JSON to `{probes_path}`:",
        "",
        "```json",
        json.dumps(template, indent=1),
        "```",
    ])  # fmt: skip
