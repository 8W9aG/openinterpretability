{% set ident = sub.identifier %}
{% set is_head = ident.kind == 'attn_head' %}
{% set is_feature = ident.kind == 'dictionary_feature' %}
{% set peer = 'head' if is_head else ('latent' if is_feature else 'neuron') %}
# {{ ident.model_name }} · {{ label }} · {{ verdict.status }}

[← all experiments on this {{ 'head' if is_head else ('feature' if is_feature else 'neuron') }}](README.md) · [wiki home]({{ home }})

| | |
| --- | --- |
| Experiment | `{{ meta.experiment_id }}` ({{ kind }}{% if previous_status %}, previous verdict {{ previous_status }}{% endif %}) |
{% if sub.supersedes %}
| Revises | [{{ sub.supersedes }}]({{ sub.supersedes }}.md) |
{% endif %}
| Date | {{ verdict.timestamp.strftime("%Y-%m-%d %H:%M UTC") }} |
| Verdict | **{{ verdict.status }}**, balanced accuracy {{ verdict.confidence_score | num(2) }} |
| Protocol | {{ ('v' ~ policy.protocol_version) if policy else 'unrecorded (pre-0.2)' }} |
| Proposer | `{{ sub.submitting_agent_model | cell }}`{% if sub.contributor %}, run by {{ sub.contributor | cell }}{% endif %} |
| Reviewer | `{{ verdict.reviewer_agent_model | cell }}`{% if verdict.contributor %}, run by {{ verdict.contributor | cell }}{% endif %}{% if self_reviewed %} (self-reviewed: same model as the proposer, so not an independent review){% endif %} |
{% if sub.provenance %}
{% set p = sub.provenance %}
| Selection | {{ p.selection }}{% if p.seed is not none %} (seed {{ p.seed }}){% endif %} |
| Corpus | {{ p.corpus | cell }} · {{ "{:,}".format(p.n_snippets) }} snippets · {{ "{:,}".format(p.tokens_scanned) }} tokens{% if p.corpus_sample %} · seeded sample of {{ p.corpus_sample.documents }}/{{ p.corpus_sample.corpus_documents }} docs (≤ {{ "{:,}".format(p.corpus_sample.max_tokens) }} tokens, seed {{ p.corpus_sample.seed }}){% endif %}{% if p.snippet_tokens %} · {{ p.snippet_tokens }}-token windows{% if p.snippet_stride %} every {{ p.snippet_stride }} tokens{% endif %}{% endif %}{% if p.corpus_source and p.corpus_source.license %} · licence {{ p.corpus_source.license | cell }}{% endif %} |
| Model revision | `{{ p.model_revision or 'unrecorded' }}` · {{ p.dtype }} |
{% if p.costs %}
| Run cost | {{ p.costs.summary() | cell }} (reported, not gated) |
{% endif %}
{% endif %}
{% if record_link %}
| Record | [{{ record_link }}]({{ record_link }}) (this experiment's full record) |
{% endif %}

## Hypothesis

> {{ sub.natural_language_explanation }}

- **Activating tokens:** {% for t in sub.activating_tokens %}`{{ t | cell }}`{{ ", " if not loop.last }}{% endfor %}

- **Mechanism claimed:** {{ 'lookahead, preparing ' ~ (sub.lookahead.target_upcoming_tokens | join(', ')) if sub.lookahead.is_lookahead else 'retrospective' }}
- **Max activation (scan):** {{ sub.max_activation | num(4) }}{% if is_head %} (head strength, \|W_O (z − z̄)\|){% elif is_feature %} (latent activation){% endif %}
{% if sub.head_profile %}
{% set hp = sub.head_profile %}
- **Attention patterns** (random {{ hp.pattern_length }}-token sequences repeated twice; reported, not gated): previous token {{ hp.previous_token_score | num(2) }}, duplicate token {{ hp.duplicate_token_score | num(2) }}, induction {{ hp.induction_score | num(2) }}
{% endif %}
{% if sub.dictionary_profile %}
{% set dp = sub.dictionary_profile %}
- **Dictionary** (a learned-basis baseline, not a native unit): a {{ dp.kind }} with {{ dp.n_features }} latents, `{{ dp.repo }}`{% if dp.revision %}@{{ dp.revision[:8] }}{% endif %} `{{ dp.path }}`, reading `{{ dp.input_site }}`. On the scan it explains {{ dp.explained_variance | num(2) if dp.explained_variance is not none else 'unmeasured' }} of the {{ 'residual stream' if dp.output_site == 'resid_post' else 'MLP output' }}'s variance with mean L0 {{ dp.mean_l0 | num(1) if dp.mean_l0 is not none else 'unmeasured' }}; this latent is active on {{ dp.density | num(4) }} of tokens (reported, not gated).
{% endif %}

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
Causal metrics (recorded, not gated): sufficiency {{ c.sufficiency_score | num(4) }}, necessity {{ c.necessity_score | num(4) }}: percentiles of the {{ 'head' if is_head else ('feature' if is_feature else 'pole') }}'s patch effect among the {{ c.null_neurons }} same-layer {{ peer }}s that change as much, median over {{ c.pairs_scored }} non-control pairs. Target: {{ c.target }}{% if c.target_tokens %} ({{ c.target_tokens | join(", ") }}){% endif %}; median effect {{ '%.3g' | format(c.sufficiency_effect) }} / {{ '%.3g' | format(c.necessity_effect) }}; hydra mitigated: {{ c.hydra_mitigated }}.
{% if c.pairs %}

| Pair | Sufficiency | Effect | Matched median / max | Necessity | Effect | Matched median / max |
| --- | --- | --- | --- | --- | --- | --- |
{% for p in c.pairs %}
| {{ p.pair_index + 1 }} | {{ p.sufficiency.percentile | num(2) }} | {{ '%.3g' | format(p.sufficiency.effect) }} | {{ '%.3g' | format(p.sufficiency.null_median) }} / {{ '%.3g' | format(p.sufficiency.null_max) }} | {{ p.necessity.percentile | num(2) }} | {{ '%.3g' | format(p.necessity.effect) }} | {{ '%.3g' | format(p.necessity.null_median) }} / {{ '%.3g' | format(p.necessity.null_max) }} |
{% endfor %}
{% endif %}
{% else %}
Causal metrics (recorded, not gated; top-1 logit, protocol < 0.5): sufficiency {{ c.sufficiency_score | num(4) }}, necessity {{ c.necessity_score | num(4) }}, hydra mitigated: {{ c.hydra_mitigated }}.
{% endif %}
{% endif %}
{% if verdict.explanation_scores %}
{% set es = verdict.explanation_scores %}

Explanation scores on held-out text (recorded, not gated): detection {{ (es.detection.balanced_accuracy if es.detection else none) | num(2) }}, fuzzing {{ (es.fuzzing.balanced_accuracy if es.fuzzing else none) | num(2) }} (balanced accuracy of a judge given only the hypothesis), embedding AUROC {{ es.embedding_auroc | num(2) }}. {{ es.detection_windows | length }} detection and {{ es.fuzzing_windows | length }} fuzzing windows of {{ es.window_tokens }} tokens, from {{ es.active_windows }} active of {{ es.candidate_windows }} held-out windows; judge `{{ es.judge_model | cell }}`{% if es.detection %}; detection recall by activation quantile, weakest to top: {% for r in es.detection.recall_by_quantile %}{{ r | num(2) }}{{ ", " if not loop.last }}{% endfor %}{% endif %}.
{% endif %}

### Probes

| | Kind | Probe | Expected | Activation |
| --- | --- | --- | --- | --- |
{% for t in verdict.adversarial_test_results if t.kind in ('hard_negative', 'positive_holdout') %}
| {{ ('✅' if t.passed else '❌') if t.measured else '—' }} | {{ t.kind | replace('_', ' ') }} | {{ t.prompt | cell }} | {{ 'active' if t.expected_active else 'inactive' }} | {{ t.activation | num(3) if t.measured else 'unmeasured (one token)' }} |
{% endfor %}

### Counterfactuals

| | Base → counterfactual | Expected Δ | Observed Δ |
| --- | --- | --- | --- |
{% for t in verdict.adversarial_test_results if t.kind == 'counterfactual' %}
| {{ ('✅' if t.passed else '❌') if t.measured else '—' }} | {{ t.prompt | cell }} | {{ t.expected_delta | num(2) }} | {{ t.observed_delta | num(3) if t.measured else 'unmeasured (one token)' }} |
{% endfor %}

{% for t in verdict.adversarial_test_results if t.kind == 'lookahead_consistency' %}
**Lookahead check:** {{ '✅' if t.passed else '❌' }} {{ t.detail }}
{% endfor %}

{% if sub.vocabulary_lens %}
{% set vl = sub.vocabulary_lens %}
### Vocabulary lens

| | Jacobian lens | Logit lens |
| --- | --- | --- |
{% for label, side in [("Promotes", "promoted"), ("Suppresses", "suppressed")] %}
| {{ label }} |{% for lens in ("jacobian", "logit") %}{% set zs = vl[lens ~ "_" ~ side ~ "_z"] %} {% for t in vl[lens ~ "_" ~ side][:8] %}`{{ t | cell }}`{% if zs %} {{ "%+.1f" | format(zs[loop.index0]) }}{% endif %} {% endfor %}|{% endfor %}

{% endfor %}

{% if vl.jacobian_promoted_z %}Numbers are z-scores over the whole vocabulary. {% endif %}Top-10 overlap {{ vl.overlap | pct }}; cos(w, J·w) {{ vl.direction_cosine | num(2) }}; kurtosis {{ vl.jacobian_kurtosis | num(1) }} (Jacobian) vs {{ vl.logit_kurtosis | num(1) }} (logit); {{ vl.prompts }} prompts.

{% endif %}
{% if sub.patchscope and sub.patchscope.readings %}
{% set ps = sub.patchscope %}
### Patchscope

The model's own description of the output direction, with no examples: the direction replaces the residual stream after the named layer's block at the `{{ ps.placeholder | cell }}` of each prompt, at each multiple of the norm it replaces, and the model continues greedily. The control injects a random direction instead, so text the two share comes from the prompt. Read after the proposer answered and shown to neither agent; reported, not gated.

| Prompt | Layer | |{% for s in ps.scales %} ×{{ "%g" | format(s) }} |{% endfor %}

| --- | --- | --- |{% for s in ps.scales %} --- |{% endfor %}

{% for prompt, layer, readings in ps.rows %}
| {{ prompt }} | {{ layer }} | unit |{% for r in readings %} {{ r.text | cell }} |{% endfor %}

| | | control |{% for r in readings %} {{ r.control_text | cell }} |{% endfor %}

{% endfor %}

Prompts: {% for name, template in ps.prompts.items() %}{{ name }} `{{ template | replace("{x}", ps.placeholder) | cell }}`{{ "; " if not loop.last }}{% endfor %}.

{% endif %}
{% if sub.steering and sub.steering.samples %}
{% set st = sub.steering %}
### Steering

Neutral prompts continued greedily for up to {{ st.max_new_tokens }} tokens with the unit ablated, untouched, and clamped at multiples of its top activation in the scan ({{ st.clamp_unit | num(2) }}), at every token. Drift at ×1–×2 says what the unit makes the model write; drift only at ×4 is off-distribution. Taken after the proposer answered and shown to neither agent; reported, not gated.

| Prompt | Ablated | Baseline |{% for m in st.multiples %} Clamped ×{{ "%g" | format(m) }} |{% endfor %}

| --- | --- | --- |{% for m in st.multiples %} --- |{% endfor %}

{% for s in st.samples %}
| {{ s.prompt | cell }} | {{ s.ablated | cell }} | {{ s.baseline | cell }} |{% for text in s.clamped %} {{ text | cell }} |{% endfor %}

{% endfor %}

{% endif %}
{% for w in [sub.wiring, verdict.wiring] if w %}
{% if loop.first %}
### Wiring

The units that write most directly into this unit's input (upstream) and read most directly from its output (downstream), along the residual stream without passing through other units. Reported, not gated.

{% endif %}
{% if w.method == "weights" %}
**From the weights.** Virtual weights with each norm's gain folded in; they rank units but have no absolute scale. Each number is how much one unit reads of the other's write: signed when one vector reads another (+ means a positive activation raises the read), otherwise a norm. Heads count through their OV circuit upstream and their Q, K and V reads downstream. Read after the proposer answered and shown to neither agent.
{% else %}
**From activations,** averaged over this unit's peak on {{ w.prompts }} of the review's prompts where it fired. Upstream: the first-order change in its strength from removing a unit's direct write (+ excites). Downstream: the change in a later neuron's activation, or how far a later head's write moves, when this unit's write is removed from that unit's input (one of Q, K or V for a head) alone.
{% endif %}

{% for title, edges in [("Upstream neurons", w.upstream_neurons), ("Upstream heads", w.upstream_heads), ("Downstream neurons", w.downstream_neurons), ("Downstream heads", w.downstream_heads)] %}
- **{{ title }}:** {% if edges is none %}not measured.{% elif not edges %}none.{% else %}{% for e in edges %}L{{ e.layer }}/{{ "N" if e.kind == "mlp_neuron" else "H" }}{{ e.index }} {{ "%+.3g" | format(e.weight) }}{{ ", " if not loop.last }}{% endfor %}{% endif %}

{% endfor %}

{% endfor %}
{% if sub.input_attribution and sub.input_attribution.examples %}
{% set ia = sub.input_attribution %}
### Input attribution

Integrated gradients over the input tokens at the peak of {{ ia.examples | length }} top examples, from a baseline of `{{ ia.baseline_token | cell }}` at every position but the first; the path integral converged on {{ ia.converged }}. Replacing one token at a time agrees at ρ = {{ ia.ablation_agreement | num(2) }}.

{% if ia.distances %}
Share of |attribution| by distance from the peak, over converged examples; mean {{ ia.mean_distance | num(1) }} tokens back:

| Tokens back |{% for d in ia.distances %} {{ d.label }} |{% endfor %}

| --- |{% for d in ia.distances %} --- |{% endfor %}

| Share |{% for d in ia.distances %} {{ d.share | pct }} |{% endfor %}

{% endif %}

| Example | Peak | Most attributed tokens: offset, attribution / ablation effect |
| --- | --- | --- |
{% for ex in ia.examples %}
| {{ ex.example + 1 }}{{ "" if ex.converged else " (not converged)" }} | `{{ ex.peak_token | cell }}` | {% for t in ex.top %}`{{ t.token | cell }}` {{ t.offset }}, {{ t.attribution | num(2) }} / {{ t.ablation_effect | num(2) }}{{ "; " if not loop.last }}{% endfor %} |
{% endfor %}

{% endif %}
### Covariance and confounds

{% if metrics and metrics.effective_hard_negatives is not none %}
- **Independent observations:** {{ metrics.effective_hard_negatives }} hard-negative and {{ metrics.effective_holdouts }} holdout clusters (near-duplicate probes count once).
{% endif %}
{% if sub.confounds %}
- **Confounds:** token identity R² {{ sub.confounds.token_r2 | num(2) }}, position R² {{ sub.confounds.position_r2 | num(2) }}, frequency r {{ sub.confounds.frequency_correlation | num(2) }}{% for name, lc in (sub.confounds.labels or {}).items() %}, {{ name }} R² {{ lc.r2 | num(2) }} ({{ lc.mutual_information_bits | num(3) }} bits){% endfor %}.{% if metrics and metrics.confound_baseline %} Best trivial baseline: {{ metrics.confound_baseline }} at {{ metrics.confound_baseline_accuracy | num(2) }}; the hypothesis scores {{ metrics.balanced_accuracy | num(2) }}.{% endif %}

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

{% if self_reviewed %}
**Self-reviewed.** The proposer's model also wrote the probes. That is allowed, but a review by a different model is stronger evidence, so a re-review is the first follow-up (`openinterp next` draws these).

{% endif %}
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
