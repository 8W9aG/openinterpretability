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
| Corpus | {{ p.corpus | cell }} · {{ "{:,}".format(p.n_snippets) }} snippets · {{ "{:,}".format(p.tokens_scanned) }} tokens{% if p.snippet_tokens %} · {{ p.snippet_tokens }}-token windows{% endif %}{% if p.corpus_source and p.corpus_source.license %} · licence {{ p.corpus_source.license | cell }}{% endif %} |
| Model revision | `{{ p.model_revision or 'unrecorded' }}` · {{ p.dtype }} |
{% endif %}
{% if record_link %}
| Record | [{{ record_link }}]({{ record_link }}) (this experiment's full record) |
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
{% set c = verdict.causal_metrics %}
{% if c.target %}
Causal metrics (recorded, not gated): sufficiency {{ c.sufficiency_score | num(4) }}, necessity {{ c.necessity_score | num(4) }}: percentiles of the pole's patch effect among the {{ c.null_neurons }} same-layer neurons that change as much, median over {{ c.pairs_scored }} non-control pairs. Target: {{ c.target }}{% if c.target_tokens %} ({{ c.target_tokens | join(", ") }}){% endif %}; median effect {{ '%.3g' | format(c.sufficiency_effect) }} / {{ '%.3g' | format(c.necessity_effect) }}; hydra mitigated: {{ c.hydra_mitigated }}.
{% else %}
Causal metrics (recorded, not gated; top-1 logit, protocol < 0.5): sufficiency {{ c.sufficiency_score | num(4) }}, necessity {{ c.necessity_score | num(4) }}, hydra mitigated: {{ c.hydra_mitigated }}.
{% endif %}
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

{% if sub.vocabulary_lens %}
{% set vl = sub.vocabulary_lens %}
### Vocabulary lens

| | Jacobian lens | Logit lens |
| --- | --- | --- |
| Promotes | {% for t in vl.jacobian_promoted[:8] %}`{{ t | cell }}` {% endfor %}| {% for t in vl.logit_promoted[:8] %}`{{ t | cell }}` {% endfor %}|
| Suppresses | {% for t in vl.jacobian_suppressed[:8] %}`{{ t | cell }}` {% endfor %}| {% for t in vl.logit_suppressed[:8] %}`{{ t | cell }}` {% endfor %}|

Top-10 overlap {{ vl.overlap | pct }}; cos(w, J·w) {{ vl.direction_cosine | num(2) }}; kurtosis {{ vl.jacobian_kurtosis | num(1) }} (Jacobian) vs {{ vl.logit_kurtosis | num(1) }} (logit); {{ vl.prompts }} prompts.

{% endif %}
### Covariance and confounds

{% if metrics and metrics.effective_hard_negatives is not none %}
- **Independent observations:** {{ metrics.effective_hard_negatives }} hard-negative and {{ metrics.effective_holdouts }} holdout clusters (near-duplicate probes count once).
{% endif %}
{% if sub.confounds %}
- **Confounds:** token identity R² {{ sub.confounds.token_r2 | num(2) }}, position R² {{ sub.confounds.position_r2 | num(2) }}, frequency r {{ sub.confounds.frequency_correlation | num(2) }}.{% if metrics and metrics.confound_baseline %} Best trivial baseline: {{ metrics.confound_baseline }} at {{ metrics.confound_baseline_accuracy | num(2) }}; the hypothesis scores {{ metrics.balanced_accuracy | num(2) }}.{% endif %}

{% else %}
- **Confounds:** no confound profile (the scan predates protocol 0.3).
{% endif %}
{% if sub.correlated_neurons %}
- **Co-activating neuron poles:** {% for c in sub.correlated_neurons[:5] %}N{{ c.neuron_idx }}{{ '+' if c.pole == 'positive' else '−' }} (lift ≥ {{ "%.1f" | format(c.coactivation_lift_lower) }} on {{ c.coactive_tokens }} tokens, ρ = {{ "%+.2f" | format(c.spearman) }}){{ ", " if not loop.last }}{% endfor %}

{% endif %}
{% if verdict.joint_ablation %}
{% set ja = verdict.joint_ablation %}
- **Joint ablation** with {% for p in ja.partners %}N{{ p }}{{ ", " if not loop.last }}{% endfor %}: KL alone {{ "%.2e" | format(ja.kl_single) }}, partners {{ "%.2e" | format(ja.kl_partners) }}, together {{ "%.2e" | format(ja.kl_joint) }}; redundancy ratio {{ ja.redundancy_ratio | num(2) }}.
{% endif %}

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
