# Circuits and hubs

The maintainer decided on 2026-11-16 that circuits become units with experiments of their own (issue #3213). Hubs become entities whose standing comes from their center's experiments. Today they are reported, not gated. The site joins studied units by their direct-effect wiring edges (`openinterpretability/builder/circuits.py`). It lists as a hub any unit wired to at least 5% of a model's studied units, and to at least 10 of them. This page is the design. Each part lands in its own issue (*Plan* below). The protocol (`docs/PROTOCOL.md`) and `PROTOCOL_VERSION` change only when the code that enforces each rule lands, so the protocol never describes a rule that the review doesn't run.

## What each one is

A neuron pole, a head or a latent is one coordinate that an experiment measures. A circuit is a group of units that the network wires together, so it is a unit with experiments of its own. A hub is a unit that many others wire to. It is an entity with its own identity and page, but it has no experiments of its own. What it does is what its center's experiments say.

| Entity | Identity | Experiments |
| --- | --- | --- |
| **Circuit** | `CircuitIdentifier`, `kind = "circuit"`, `C<n>`, in one model revision. A circuit is a cluster of studied units in the wiring graph, found by the network, not by a hypothesis. Its id is a name given when the cluster is first seen, and it stays with the cluster as the registry grows (*Identity across builds*). | Yes, one or more task hypotheses. Each names a task (the behaviour, a clean and corrupted prompt template and a metric), the members it tested and each member's role. |
| **Hub** | `HubIdentifier`, `kind = "hub"`, `<unit>/HUB`. Its center is one neuron pole or head that the wiring joins to many studied units. | No. Its standing is its center's standing (§3 step 9), and its page lists the units wired to it. |

## Circuits

### What defines a circuit

A circuit is defined by where it sits in the network, not by what someone thinks it does. Defining it by a task would split one group of units into several circuits when it carries several behaviours. And a circuit, like a neuron, can be somewhat polysemantic. So the identity comes from the wiring, and each task is a hypothesis about the circuit.

Each build clusters a model's studied units by their direct-effect wiring edges, weighted by edge strength. It uses community detection such as the Leiden algorithm (Traag et al., 2019). Hub edges are left out, as they are today (`openinterpretability/builder/circuits.py`), so that one hub doesn't join everything into one star. A cluster of at least 3 units is a circuit. Its members and its center of mass (its mean relative depth and its mean member position) locate it in the network.

### Identity across builds

The registry grows, so the clusters change from one build to the next. Each build matches its clusters to the previous build's circuits by member overlap. The overlap is the Jaccard index |A ∩ B| / |A ∪ B|. It is 1 minus the set edit distance between the two member sets, scaled by their union. An optimal assignment then pairs old and new so that the total overlap is greatest. A new cluster takes the id of the circuit that it is paired with, if the two overlap by at least 0.5. Otherwise it gets a new id. `open-interpretability circuits identify` does this, and the release runs it. It keeps each model's lineage in `data/circuits/<model>.jsonl`, an append-only file with one event per line (`openinterpretability/circuit_ids.py`). So each circuit has a lineage:

| Event | When | What it means |
| --- | --- | --- |
| Continue | Paired, and the members are the same | Nothing changed. |
| Grow or shrink | Paired, and members joined or left | The circuit keeps its id, and its experiments may go stale (below). |
| Split | One old circuit overlaps several new clusters by at least 0.2 | The largest part keeps the id. Each other part gets a new id that records the circuit it came from. |
| Merge | Several old circuits each overlap one new cluster by at least 0.2 | The new cluster keeps the id of the old circuit it overlaps most. The others become its sub-circuits. |
| New | No old circuit overlaps it by 0.2 | A new id. |

Clustering can reshuffle groups when one edge is added, and the matching damps that. The pilot measures how often ids survive on the registry's own history before any of this becomes the rule (*Plan*, step 1). A circuit page also shows how stable its membership has been.

### Task hypotheses, and stale experiments

Each experiment on a circuit tests one task hypothesis. A circuit can have several, one for each behaviour that it carries, and each has its own standing (§3 step 9 within that task). A hypothesis for a new task is a new hypothesis on the same circuit, not a new circuit.

Each experiment records the members it tested, with the record of each one that it read, so an old experiment never changes. When the circuit's current members differ from those, the experiment is **stale**:

- Its page says so, and names the units that joined or left since it ran.
- It still counts toward its task's standing until a newer experiment tests the current members. A stale VERIFIED says that the tested members carry the task, not that the circuit as it stands now does.
- The planner draws a `circuit` experiment for it (*Plan*, step 4). The more members that changed, the higher the weight.

A new experiment on the same task with the current members is a revision, and its record names the one it revises in `supersedes`.

### How an experiment tests it

A circuit experiment reuses the lifecycle of `docs/PROTOCOL.md` §3: propose, review with probes the proposer didn't see, measure, decide with gates.

The proposer gives a task: a set of clean and corrupted prompt pairs and a metric, as the subject–verb agreement analysis does (`docs/analyses/sva-circuit.md`). It gives each member's role in words. The reviewer writes its own pairs: holdouts of the task and hard negatives, which are other tasks that the circuit should not carry. The review measures each set with path patching (Wang et al., 2022). Every unit that is not a member is resample-ablated with activations from the corrupted prompts, so only the members carry the clean signal.

The decided gates are in `docs/PROTOCOL.md` §4 (*Circuits*). Necessity must be at least 0.02, and above the 95th percentile of random member sets. Specificity on other tasks must be at least 0.80. The pilot set them (`docs/analyses/circuit-pilot.md`). The first plan here had absolute gates, such as faithfulness ≥ 0.6, and every wiring cluster failed them. Faithfulness and each member's necessity are still recorded, but not gated.

### Sub-circuits

A merge makes the old circuits sub-circuits of the new one, so a larger circuit can contain smaller ones. Each keeps its own id, tasks and experiments. The larger circuit's page lists its sub-circuits, and each sub-circuit's page links the larger one. How a sub-circuit's standing bears on the larger one's is left for when the registry has a first case.

## Hubs

A hub needs no experiment of its own, because what it does is what its center does. Its center's experiments test that, and the hub inherits their standing. What a hub adds is identity and context: a stable name and page, the units wired to it, and the circuits it joins. The site already finds hubs: a unit wired to at least 5% of a model's studied units, and to at least 10 of them (`openinterpretability/builder/circuits.py`).

A hub can be an artefact. A unit with a large output norm, or one that is on most of the time (default-on, §1), touches many units without carrying anything particular. So the hub's page reports whether its center is default-on, and its output norm against its layer (`neuron-family-triage`). These are reported, not gated, so hubs need no protocol change.

## How it shows

- **Standing.** A circuit has a standing per task, by the rule of §3 step 9 unchanged. One reading per probe set counts, independent reviews outweigh self-reviews, and a majority decides. A hub's standing is its center's.
- **Counts.** Circuits count on their own, as superpositions do. They never enter the neuron pole statistics or the random-unit false discovery estimate, because their members were chosen, not drawn. Hubs add no count, because their centers are counted already.
- **Pages.** A circuit page shows its members, center of mass, lineage and stability. It also shows each task with its standing and experiments, which experiments are stale, and its sub-circuits. A hub page shows its center's standing, the units wired to it, and the artefact facts.

## Plan

Each step is its own issue, and each issue says what it needs from the one before it.

1. **Pilot** (#4283). Cluster today's wiring, and replay the registry's history week by week to measure how often circuit ids survive the matching. Run the circuit measures by hand on the subject–verb agreement circuit (`sva-circuit.md`) and on random sets, to set the gates. No schema change.
2. **The circuit unit** (#4284). Done: circuit ids and lineage (`circuits identify`), the measures for neurons and heads (`openinterpretability/harness/circuit_measures.py`), and the unit (`circuits propose` and `circuits review`). The gates are in `docs/PROTOCOL.md` §4, under their own rule version, `CIRCUIT_PROTOCOL_VERSION`. `PROTOCOL_VERSION` stays unchanged, so no other record needs re-evaluating.
3. **The hub entity** (#4285). Add `HubIdentifier`, the hub's derived standing and the reported artefact facts. No experiments and no protocol change.
4. **Planner draws** (#4286). A `circuit` draw for a stale experiment, weighted by how many members changed, and for a circuit with no task hypothesis yet.
5. **Site pages** (#4287). Circuit pages show members, lineage and stability, each task's experiments and standing, stale experiments and sub-circuits. Hub pages show their center's standing and the artefact facts.

## References

- Traag, V. A., Waltman, L., & van Eck, N. J. (2019). *From Louvain to Leiden: Guaranteeing well-connected communities*. Scientific Reports, 9, 5233. [arXiv:1810.08473](https://arxiv.org/abs/1810.08473).
- Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). *Interpretability in the wild: A circuit for indirect object identification in GPT-2 small*. [arXiv:2211.00593](https://arxiv.org/abs/2211.00593).
