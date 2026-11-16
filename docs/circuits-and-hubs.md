# Circuits and hubs

The maintainer decided on 2026-11-16 that circuits become units with experiments of their own (issue #3213). Hubs become entities whose standing comes from their center's experiments. Today they are reported, not gated. The site joins studied units by their direct-effect wiring edges (`openinterp/builder/circuits.py`). It lists as a hub any unit wired to at least 5% of a model's studied units, and to at least 10 of them. This page is the design. Each part lands in its own issue (*Plan* below). The protocol (`docs/PROTOCOL.md`) and `PROTOCOL_VERSION` change only when the code that enforces each rule lands, so the protocol never describes a rule that the review doesn't run.

## What each one is

A neuron pole, a head or a latent is one coordinate that an experiment measures. A circuit is a claim about how several units carry one task together, so it is a unit with experiments of its own. A hub is a unit that many others wire to. It is an entity with its own identity and page, but it has no experiments of its own. What it does is what its center's experiments say.

| Entity | Identity | Experiments |
| --- | --- | --- |
| **Circuit** | `CircuitIdentifier`, `kind = "circuit"`, `C<id>`, in one model revision. A circuit is defined by its task: the behaviour, the clean and corrupted prompt template and the metric. The id is 12 hex digits of a SHA-256 of that task definition, so one task always has one id. | Yes. Each hypothesis names the members (neuron poles, heads, latents or superpositions), each member's role, and what the group computes as a whole. |
| **Hub** | `HubIdentifier`, `kind = "hub"`, `<unit>/HUB`. Its center is one neuron pole or head that the wiring joins to many studied units. | No. Its standing is its center's standing (§3 step 9), and its page lists the units wired to it. |

## Circuits

### Membership, and stale experiments

A circuit's members belong to its hypothesis, not to its identity. Each experiment records the members that it tested, with the record of each one that it read, so an old experiment never changes.

The registry keeps growing, so a circuit's **current members** can move on from what an experiment tested. They are the members of its newest hypothesis, plus each studied unit that the wiring now joins to them. When an experiment's tested members differ from the current members, the experiment is **stale**:

- Its page says so, and names the units that joined or left since it ran.
- It still counts toward the circuit's standing until a newer experiment tests the current members. A stale VERIFIED says that the tested members carry the task, not that the circuit is complete.
- The planner draws a `circuit` experiment for a stale circuit (*Plan*, step 4). The more members that changed, the higher the weight.

A new experiment on the current members is a revision of the same circuit, and its record names the one it revises in `supersedes`. The standing rule (§3 step 9) counts only the newest hypothesis. So the circuit's standing follows the members that were last tested, and its whole history stays on one page. Only a different task makes a different circuit.

### How an experiment tests it

A circuit experiment reuses the lifecycle of `docs/PROTOCOL.md` §3: propose, review with probes the proposer didn't see, measure, decide with gates.

The proposer gives a task: a set of clean and corrupted prompt pairs and a metric, as the subject–verb agreement analysis does (`docs/analyses/sva-circuit.md`). It gives each member's role in words. The reviewer writes its own pairs: holdouts of the task and hard negatives, which are other tasks that the circuit should not carry. The review measures each set with path patching (Wang et al., 2022). Every unit that is not a member is resample-ablated with activations from the corrupted prompts, so only the members carry the clean signal.

| Measure | Definition | Gate (provisional) |
| --- | --- | --- |
| Faithfulness | (m(circuit only) − m(none)) / (m(all) − m(none)) on the holdout pairs | ≥ 0.6 |
| Necessity | 1 − (m(circuit ablated) − m(none)) / (m(all) − m(none)) on the holdouts | ≥ 0.5 |
| Specificity | The share of hard-negative tasks on which ablating the circuit moves the metric by less than 20% of its range | ≥ 0.8 |
| Minimality | The share of members whose own ablation lowers faithfulness by at least 0.02 | reported, not gated |
| Null | Faithfulness of 64 random member sets of the same size and layers, seeded by the id | the circuit must beat the 95th percentile |

The gates are provisional. The subject–verb agreement pilot sets them before any protocol change (*Plan*, step 1). A decided gate goes into `docs/PROTOCOL.md` §4 with a `PROTOCOL_VERSION` bump.

### Sub-circuits

The wiring can join two circuits as the registry grows. One example is a circuit that finds the subject, feeding one that chooses the verb's form. A larger circuit can then contain smaller ones. Each keeps its own task and its own experiments, and the larger circuit's task composes theirs. The larger circuit's page lists its sub-circuits, and each sub-circuit's page links the larger one. How a sub-circuit's standing bears on the larger one's is left for when the registry has a first case. Until then, each stands on its own experiments.

## Hubs

A hub needs no experiment of its own, because what it does is what its center does. Its center's experiments test that, and the hub inherits their standing. What a hub adds is identity and context: a stable name and page, the units wired to it, and the circuits it joins. The site already finds hubs: a unit wired to at least 5% of a model's studied units, and to at least 10 of them (`openinterp/builder/circuits.py`).

A hub can be an artefact. A unit with a large output norm, or one that is on most of the time (default-on, §1), touches many units without carrying anything particular. So the hub's page reports whether its center is default-on, and its output norm against its layer (`neuron-family-triage`). These are reported, not gated, so hubs need no protocol change.

## How it shows

- **Standing.** A circuit's standing follows the rule of §3 step 9 unchanged: one reading per probe set, independent reviews outweigh self-reviews, and a majority decides. A hub's standing is its center's.
- **Counts.** Circuits count on their own, as superpositions do. They never enter the neuron pole statistics or the random-unit false discovery estimate, because their members were chosen, not drawn. Hubs add no count, because their centers are counted already.
- **Pages.** A circuit page shows the wiring that suggested it, its experiments, its standing, which experiments are stale, and its sub-circuits. A hub page shows its center's standing, the units wired to it, and the artefact facts.

## Plan

Each step is its own issue, and each issue says what it needs from the one before it.

1. **Pilot on the subject–verb agreement circuit** (#4283). Run the circuit measures by hand on the top-N neurons of `sva-circuit.md`, and on random sets, to set the gates. No schema change.
2. **The circuit unit** (#4284). Add `CircuitIdentifier`, the tested members on each record, and the review measures to the schema and harness. Add `open-interpretability audit --circuit` and `review` support. Then add the §1 row, the §4 gates and the `PROTOCOL_VERSION` bump.
3. **The hub entity** (#4285). Add `HubIdentifier`, the hub's derived standing and the reported artefact facts. No experiments and no protocol change.
4. **Planner draws** (#4286). A `circuit` draw for a stale circuit, weighted by how many members changed, and for a group of wired units with no circuit yet.
5. **Site pages** (#4287). Circuit pages show experiments, standing, stale experiments and sub-circuits. Hub pages show their center's standing and the artefact facts.

## References

- Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). *Interpretability in the wild: A circuit for indirect object identification in GPT-2 small*. [arXiv:2211.00593](https://arxiv.org/abs/2211.00593).
