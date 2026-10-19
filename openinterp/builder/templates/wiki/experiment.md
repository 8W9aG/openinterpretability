{% set ident = sub.identifier %}
# {{ ident.model_name }} · L{{ ident.layer }}/N{{ ident.neuron_idx }}{{ '+' if ident.pole == 'positive' else '−' }} · {{ verdict.status }}

[← all experiments on this neuron](README.md) · [wiki home](../../../../README.md)

| | |
| --- | --- |
| Experiment | `{{ meta.experiment_id }}` ({{ kind }}{% if previous_status %}, previous verdict {{ previous_status }}{% endif %}) |
| Date | {{ verdict.timestamp.strftime("%Y-%m-%d %H:%M UTC") }} |
| Verdict | **{{ verdict.status }}**, balanced accuracy {{ verdict.confidence_score | num(2) }} |
| Protocol | {{ ('v' ~ policy.protocol_version) if policy else 'unrecorded (pre-0.2)' }} |
| Proposer | `{{ sub.submitting_agent_model | cell }}` |
| Reviewer | `{{ verdict.reviewer_agent_model | cell }}` |
{% if sub.provenance %}
{% set p = sub.provenance %}
| Selection | {{ p.selection }}{% if p.seed is not none %} (seed {{ p.seed }}){% endif %} |
| Corpus | {{ p.corpus | cell }} · {{ "{:,}".format(p.n_snippets) }} snippets · {{ "{:,}".format(p.tokens_scanned) }} tokens{% if p.snippet_tokens %} · {{ p.snippet_tokens }}-token windows{% endif %} |
| Model revision | `{{ p.model_revision or 'unrecorded' }}` · {{ p.dtype }} |
{% endif %}
{% if record_link %}
| Record | [{{ record_link }}]({{ record_link }}) (holds the *latest* verdict for this neuron) |
{% endif %}

## Hypothesis

> {{ sub.natural_language_explanation }}

- **Activating tokens:** {% for t in sub.activating_tokens %}`{{ t | cell }}`{{ ", " if not loop.last }}{% endfor %}

- **Mechanism claimed:** {{ 'lookahead, preparing ' ~ (sub.lookahead.target_upcoming_tokens | join(', ')) if sub.lookahead.is_lookahead else 'retrospective' }}
- **Max activation (scan):** {{ sub.max_activation | num(4) }}

{% if examples %}
### Top-activating examples

Peak token in ⟦ ⟧.

```text
{% for act, ctx in examples %}
{{ "%.3f" | format(act) }}  {{ ctx }}
{% endfor %}
```
{% endif %}

## Results

{% if metrics %}
| Gate | Result |
| --- | --- |
{% for gate, ok in metrics.gates.items() %}
| {{ gate | replace('_', ' ') }} | {{ '✅ pass' if ok else '❌ fail' }} |
{% endfor %}

Specificity {{ metrics.specificity | num(2) }} · sensitivity {{ metrics.sensitivity | num(2) }} · balanced accuracy {{ metrics.balanced_accuracy | num(2) }} · counterfactual reproduction {{ metrics.counterfactual_reproduction | num(2) }}
{% else %}
This verdict predates protocol v0.2, so it has no per-gate metrics. Run `openinterp reevaluate` to apply the current rule.
{% endif %}

{% if verdict.causal_metrics %}
Causal metrics (recorded, not gated): sufficiency {{ verdict.causal_metrics.sufficiency_score | num(4) }}, necessity {{ verdict.causal_metrics.necessity_score | num(4) }}, hydra mitigated: {{ verdict.causal_metrics.hydra_mitigated }}.
{% endif %}

### Probes

| | Kind | Probe | Expected | Activation |
| --- | --- | --- | --- | --- |
{% for t in verdict.adversarial_test_results if t.kind in ('hard_negative', 'positive_holdout') %}
| {{ '✅' if t.passed else '❌' }} | {{ t.kind | replace('_', ' ') }} | {{ t.prompt | cell }} | {{ 'active' if t.expected_active else 'inactive' }} | {{ t.activation | num(3) }} |
{% endfor %}

### Counterfactuals

| | Base → counterfactual | Expected Δ | Observed Δ |
| --- | --- | --- | --- |
{% for t in verdict.adversarial_test_results if t.kind == 'counterfactual' %}
| {{ '✅' if t.passed else '❌' }} | {{ t.prompt | cell }} | {{ t.expected_delta | num(2) }} | {{ t.observed_delta | num(3) }} |
{% endfor %}

{% for t in verdict.adversarial_test_results if t.kind == 'lookahead_consistency' %}
**Lookahead check:** {{ '✅' if t.passed else '❌' }} {{ t.detail }}
{% endfor %}

## Next steps

{% if verdict.status == 'VERIFIED' %}
Published in the registry. Possible follow-ups: audit the opposite pole, try a sibling model for a universal counterpart, and run steering tests (skill: `neuron-steering-tests`).
{% else %}
{% for a in advice %}
- {{ a }}
{% else %}
- Revise the hypothesis and resubmit.
{% endfor %}
{% if failed_probes %}

Failed probes:
{% for t in failed_probes %}
- {{ t.kind | replace('_', ' ') }}: “{{ t.prompt }}” (activation {{ t.activation | num(3) }})
{% endfor %}
{% endif %}
{% endif %}

## Discussion

<!-- Free-form notes. This page is never regenerated, so edits here are kept. -->
