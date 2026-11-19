# Why There Are Complementary Learning Systems in the Hippocampus and Neocortex

**Citation.** James L. McClelland (Carnegie Mellon University and the Center for the Neural Basis of Cognition), Bruce L. McNaughton (University of Arizona) and Randall C. O'Reilly (Carnegie Mellon University and the Center for the Neural Basis of Cognition). "Why There Are Complementary Learning Systems in the Hippocampus and Neocortex: Insights From the Successes and Failures of Connectionist Models of Learning and Memory". *Psychological Review*, 1995, volume 102, number 3, pages 419–457. [doi:10.1037/0033-295X.102.3.419](https://doi.org/10.1037/0033-295X.102.3.419). We read the copy on McClelland's Stanford pages, which is the published typeset version. Funding: three National Institute of Mental Health grants (MH00385, MH47566 and MH46823) and the Human Frontiers Program. Also the McDonnell-Pew Program in Cognitive Neuroscience and the Office of Naval Research. The paper gives no code or data. Its simulations used the bp program of McClelland and Rumelhart (1988) and Rumelhart's pattern files.

**Status.** Planned. #4782 (catastrophic interference and weight-space programming) would use it. The maintainer designs the campaign to repair models. So this page reports evidence and records questions only.

**Read.** On 2026-11-19 the web reader refused the PDF's host, so we downloaded the PDF and read its text, extracted with `pdftotext`. We read these parts in the source's own text: the abstract, the semantic network, *Catastrophic Interference* and *Incorporating New Material*. Also the three principles, *Why Discovering Structure Depends on Slow Learning*, and the consolidation model with Table 1. We searched the rest by keyword. We did not see the figures, which hold most of the simulation results as curves. We did not read every page of the neuropsychology review or of the General Discussion.

## Summary

**Claim.** The brain needs two learning systems. The neocortex learns slowly, and that lets it find the structure shared across many experiences. The hippocampus learns fast, which lets it store a new item without disrupting that structure. Its later reinstatements interleave the item with other knowledge, so that the neocortex absorbs it gradually. In the authors' words, models "discover the structure in ensembles of items" when learning is "gradual and interleaved".

**Method.**

- **The semantic network.** The authors re-ran Rumelhart's (1990) network. It learns true propositions about living things, such as oak, pine, robin and salmon. Each input is a concept and a relation (isa, has, can, is). It has eight "concept representation units", trained by back-propagation for 500 epochs. Weights start uniform in ±0.5, the learning rate is 0.1, and targets are 0.95 and 0.05. Hierarchical clustering of the representation units tracks how the concepts' representations differ over training.
- **New knowledge.** The trained network learns that a penguin is a bird that can swim but can't fly. In *focused* learning, the new facts are shown alone, again and again. In *interleaved* learning, they join the full training set. The learning rate is the same in both. The measure is the absolute error on the new facts, and on all 15 old cases with the *can* relation.
- **Catastrophic interference.** The paper describes McCloskey and Cohen's (1989) network on the AB-AC paired-associates task, and reprints their result beside human data (Barnes & Underwood, 1959).
- **Consolidation.** A three-layer network (50 inputs, 15 hidden units, 1 output) is trained on 100 discriminations of random sparse patterns. Its schedule has background, direct-experience and hippocampally reinstated trials. Each condition has 200 simulated subjects, and the best-fitting run of several parameter settings is reported. A two-compartment model then summarises consolidation with five parameters, fitted to four experiments.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Semantic network, epoch 25 | The concepts' similarity structure is "essentially random". For example, oak groups with canary |
| Semantic network, epoch 200 | Plants and animals are well differentiated, but the differences within them are small |
| Semantic network, epoch 500 | The full hierarchy: trees, flowers, birds and fish each group, as do plants and animals |
| A new concept, sparrow, taught only "sparrow isa" and interleaved | Gets a representation near robin and canary, and right answers on relations never trained for it |
| McCloskey and Cohen's network, AB-AC | "Virtually complete abolition of AB list performance" before AC recall rises above 0%. Humans keep more than 50% of AB recall at AC's asymptote |
| Penguin, focused learning | Learns the new facts faster, but the network "comes to treat all animals—and, to a lesser extent, all plants" as swimming and not flying. The fish improve slightly, and the birds worsen "a great deal" |
| Penguin, interleaved learning | Slower, "without ever producing more than a slight hint of interference" |
| Where interference falls | Mostly on the output units where an old concept's target differs from the penguin's (Figure 11 caption) |
| Two-compartment fits (Table 1): Dₕ, C, D_c, Sₕ(0), S_c(0) | Winocur (1990): 0.250, 0.400, 0.075, 0.900, 0.100. Kim and Fanselow (1992): 0.050, 0.040, 0.011, 0.800, 0.030. Zola-Morgan and Squire (1990): 0.035, 0.020, 0.003, 1.000, 0.100. Squire and Cohen (1979): 0.001, 0.001, 0.001, 0.500, 0.000 |
| The paper's three principles | Structure comes from "gradual, interleaved learning". Learning new information rapidly in a trained network "lead[s] to catastrophic interference". New material joins without interference if it is "interleaved with ongoing exposure" to the old |

The penguin results are curves in Figure 11, so this page gives their direction only.

## Where the claims go beyond the evidence

- **One small network for the central demonstration.** The penguin result comes from Rumelhart's network: eight representation units and a few dozen propositions. The paper argues that the effects "apply broadly to systems that learn by adjusting connection weights". The broader support is an argument from a covariance learning rule, not more simulations.
- **No numbers for the key comparison in the text.** Focused and interleaved learning are compared in figures only. The text gives no seeds, no number of runs and no spread for them. The consolidation simulations do say "considerable variability" and use 200 simulated subjects per condition, but report the best-fitting run.
- **Fitted parameters, with free choices.** The two-compartment model has five parameters for each experiment, fitted to four data sets with few points each. A good fit shows the model can describe the data. It doesn't show that its mechanism is the right one. The authors note that Lynn (1994) found similar estimates.
- **The biological evidence is partly anecdotal.** The focal dystonia evidence for focused learning in cortex is cited as a personal communication from Merzenich.
- **Reducing overlap is set aside, not refuted.** The authors grant that sparser representations reduce interference, citing French (1991). They argue that it costs the sharing of structure, and turn to a second system instead. They don't test how much sparsity a network could use before sharing fails.

## Use in this repo

- **Poles that keep changing late fit slow structure learning.** [`papers/neuron-development/`](../../papers/neuron-development/README.md) found that Pythia-70m's 86 VERIFIED poles keep changing long after the loss settles. Adjacent checkpoints correlate at a median of 0.78 to 0.82. The paper's semantic network refines its representations from coarse to fine over training. Its argument is that fine structure needs a small, steady learning rate. Pythia's run is one interleaved pass, so the paper's account predicts late refinement. That is consistent with what we measured, though the paper doesn't test a language model.
- **Focused learning on one fact is the risk for a repair of #4781.** Teaching the network one fact damaged the concepts that share its outputs, the birds. The damage fell where the old answer differs from the new one. For pythia-70m's " days" error (#5015), the analogous neighbours are the shortlist prompts that copying answers right, such as "A dog has four" → " legs". A repair that pushed " months" alone could damage those. The paper's remedy is to interleave the new fact with the old ones. The synthesis of #4784 ([`fixing-models.md`](fixing-models.md)) has editing-side versions of this test. Whether the campaign uses interleaving is the maintainer's question.
- **Overlap is what lets units share, and what lets them interfere.** The paper's trade-off, sharing against interference, is the same trade-off as superposition, where many features share neurons (`explain_groups.py`, superposition records). A VERIFIED pole that serves several features is where a later change could interfere. #4782 holds the reviews that would test this.
- **Second-hand evidence for two unread sources.** This paper describes McCloskey and Cohen (1989) and Ratcliff (1990). We couldn't read either (#4806, #4807). It reports McCloskey and Cohen's AB-AC result in some detail and reprints their figure. That is second-hand evidence, and the pages for those issues still need their own sources.
- **No single-unit content.** The paper's units are the hidden units of small networks, studied by the similarity of their patterns, not one at a time. Nothing in it bears on a pole's explanation or on our verdicts.
