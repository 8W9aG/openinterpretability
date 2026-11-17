{% set p = stop.provenance %}
{% set default_on = stop.outcome == 'default_on' %}
# {{ meta.model }} · {{ label }} · scan stopped ({{ stop.outcome | replace('_', '-') }})

{% if stop.latest_record %}
[← all experiments on this unit]({{ root }}experiments/{{ meta.slug }}/README.md) · {% endif %}[wiki home]({{ root }}README.md)

| | |
| --- | --- |
| Outcome | **{{ stop.outcome | replace('_', '-') }}**{% if default_on %}, active on {{ stop.active_share | pct }} of the scan's tokens{% endif %} |
| Date | {{ stop.timestamp.strftime("%Y-%m-%d %H:%M UTC") }} |
| Protocol | v{{ stop.protocol_version }} |
| Selection | {{ p.selection }} |
{% if stop.latest_record %}
| Newest record | [{{ stop.latest_record }}]({{ root }}experiments/{{ meta.slug }}/{{ stop.latest_record }}.md) |
{% endif %}
| Corpus | {{ p.corpus | cell }} · {{ "{:,}".format(p.n_snippets) }} snippets · {{ "{:,}".format(p.tokens_scanned) }} tokens{% if p.snippet_tokens %} · {{ p.snippet_tokens }}-token windows every {{ p.snippet_stride }} tokens{% endif %} |
| Model revision | `{{ p.model_revision or 'unrecorded' }}` · {{ p.dtype }} |

## What happened

{% if default_on %}
The scan stopped before this attempt wrote a hypothesis. This unit is active on {{ stop.active_share | pct }} of the scan's tokens. The limit is {{ policy.max_active_share | pct }} (`max_active_share`, docs/PROTOCOL.md §1). At that density, a probe of a few words almost always has one active token. So no hard negative can stay inactive, and the unit can't be audited as a detector.
{% elif stop.proven_dead %}
The scan stopped before this attempt wrote a hypothesis. This unit never fires on the corpus, and the weights prove that no input can make it fire (largest pre-activation {{ stop.dead_bound | num(3) }}).
{% else %}
The scan stopped before this attempt wrote a hypothesis. This unit never fires on the corpus, but the weights don't prove it dead. A larger corpus can still find activations.
{% endif %}

## What this changes

This attempt tested no hypothesis, so it gives no verdict. {% if stop.latest_record %}The unit's newest record keeps its status.{% else %}The unit has no experiment record yet.{% endif %} This unit was chosen, not drawn at random, so this stop doesn't count in random-draw statistics.

## Discussion

<!-- Notes go here. The rest of this page is generated and write-once. -->
