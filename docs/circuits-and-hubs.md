# Circuits and hubs as units

The maintainer decided on 2026-11-16 that circuits and hubs become units with experiments of their own (issue #3213). Today they are reported, not gated. The site joins studied units by their direct-effect wiring edges (`openinterp/builder/circuits.py`). It lists as a hub any unit wired to at least 5% of a model's studied units, and to at least 10 of them. This page is the design. Each part lands in its own issue (*Plan* below). The protocol (`docs/PROTOCOL.md`) and `PROTOCOL_VERSION` change only when the code that enforces each rule lands, so the protocol never describes a rule that the review doesn't run.

## What a unit is

A neuron pole, a head or a latent is one coordinate that an experiment measures. A circuit and a hub are claims about how several units work together. So the new units name their members, and an experiment tests a claim about the group.

| Unit | Identity | Its hypothesis says |
| --- | --- | --- |
| **Circuit** | `CircuitIdentifier`, `kind = "circuit"`, `L<first>-L<last>/C<id>`. The members are a sorted list of unit slugs (neuron poles, heads, latents or superpositions) in one model revision. The id is 12 hex digits of a SHA-256 of that list, as for a superposition. | What the group computes as a whole on a task: the behaviour, how it is measured, and each member's role. |
| **Hub** | `HubIdentifier`, `kind = "hub"`, `<unit>/HUB`. Its center is one unit slug. The record pins the center's record and the upstream and downstream member sets that it was tested with. | What the center integrates or passes on: which upstream units write through it, and which downstream units it changes. |

**Membership changes as the registry grows.** A circuit that gains or loses a member is a new circuit with a new id. Its record keeps the old one in `supersedes`, as a revision does. A hub keeps its id, because the center defines it. A hub record tested with other member sets is a new experiment on the same unit, as a re-review is for a pole. Each record pins the member records that it used, so a later change to a member's standing never changes an old circuit record.

## How an experiment tests it

Both units reuse the lifecycle of `docs/PROTOCOL.md` §3: propose, review with probes the proposer didn't see, measure, decide with gates. What is measured differs.

### Circuit

The proposer gives a task: a set of clean and corrupted prompt pairs and a metric, as the subject–verb agreement analysis does (`docs/analyses/sva-circuit.md`). It gives each member's role in words. The reviewer writes its own pairs: holdouts of the task and hard negatives, which are other tasks that the circuit should not carry. The review measures each set with path patching (Wang et al., 2022). Every unit that is not a member is resample-ablated with activations from the corrupted prompts, so only the members carry the clean signal.

| Measure | Definition | Gate (provisional) |
| --- | --- | --- |
| Faithfulness | (m(circuit only) − m(none)) / (m(all) − m(none)) on the holdout pairs | ≥ 0.6 |
| Necessity | 1 − (m(circuit ablated) − m(none)) / (m(all) − m(none)) on the holdouts | ≥ 0.5 |
| Specificity | The share of hard-negative tasks on which ablating the circuit moves the metric by less than 20% of its range | ≥ 0.8 |
| Minimality | The share of members whose own ablation lowers faithfulness by at least 0.02 | reported, not gated |
| Null | Faithfulness of 64 random member sets of the same size and layers, seeded by the id | the circuit must beat the 95th percentile |

### Hub

A hub can be an artefact. A unit with a large output norm, or one that is on most of the time (default-on, §1), touches many units without carrying anything particular. So a hub review checks that first, and only then the claim.

| Measure | Definition | Gate (provisional) |
| --- | --- | --- |
| Not an artefact | The center is not default-on, and its output norm is below the 95th percentile of its layer (`neuron-family-triage`) | must hold |
| Mediation | The share of the upstream members' total effect on the downstream members that flows through the center, by path patching through it | ≥ 0.3 |
| Specificity | Mediation through the center against 64 units of the same layer with matched output norm | must beat the 95th percentile |
| Hypothesis | The reviewer's probes on the stated integration (holdouts and hard negatives), with the neuron pole gates | as for a pole |

The gates are provisional. The subject–verb agreement pilot sets them before any protocol change (*Plan*, step 1). A decided gate goes into `docs/PROTOCOL.md` §4 with a `PROTOCOL_VERSION` bump.

## How it shows

- **Standing.** The rule of §3 step 9 applies unchanged: one reading per probe set, independent reviews outweigh self-reviews, and a majority decides.
- **Counts.** Circuits and hubs count on their own, as superpositions do. They never enter the neuron pole statistics or the random-unit false discovery estimate, because their members were chosen, not drawn.
- **Pages.** The site's circuit and hub pages become unit pages, with experiments, standing and the members' links. A circuit page then shows both the wiring that suggested it and the experiments that tested it.

## Plan

Each step is its own issue, and each issue says what it needs from the one before it.

1. **Pilot on the subject–verb agreement circuit** (#4283). Run the circuit measures by hand on the top-N neurons of `sva-circuit.md`, and on random sets, to set the gates. No schema change.
2. **The circuit unit** (#4284). Add `CircuitIdentifier` and the review measures to the schema and harness, and add `open-interpretability audit --circuit` and `review` support. Then add the §1 row, the §4 gates and the `PROTOCOL_VERSION` bump.
3. **The hub unit** (#4285). Add `HubIdentifier`, the artefact check, mediation and its matched-norm null, with the §1 row, the §4 gates and the bump.
4. **Planner draws** (#4286). A `circuit` draw from the wiring's connected groups, and a `hub` draw from the hubs that the site lists, weighted with the other kinds.
5. **Site pages** (#4287). The circuit and hub pages show experiments and standing.

## References

- Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). *Interpretability in the wild: A circuit for indirect object identification in GPT-2 small*. [arXiv:2211.00593](https://arxiv.org/abs/2211.00593).
