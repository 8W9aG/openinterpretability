# What Survives Distillation?

**Citation.** Nabeelah Faumi and Deshan Sumanathilaka (Informatics Institute of Technology, Colombo). "What Survives Distillation? A Mechanistic Interpretability Study of GPT-2 and DistilGPT-2". arXiv 2610.37157v1, 27 October 2026. [Paper](https://arxiv.org/abs/2610.37157). Preprint, no venue given. Licence CC BY 4.0. The paper releases no code or data. It names no funding, and it doesn't say whether an LLM helped write it.

**Status.** Planned. The paper studies heads, layer similarity and SAE features, not single neurons. It suggests one cheap test for us: DistilGPT-2 as GPT-2's sibling in the sister search, #3853.

**Read.** The abstract page and the v1 HTML, on 2026-11-15. A model read the text in two passes and reported it to us, with short quotes. Every section came back as the paper's own text, appendices A.1 to A.5 included. We saw only the captions of Figures 1 to 5, not the PDF.

## Summary

**Claim.** Distillation keeps GPT-2's simple, early circuits and loses its complex, late ones. The paper says that "circuit compressibility governs mechanistic survival".

**Method.**

- **Models.** GPT-2 small (12 layers) as the teacher, and DistilGPT-2 (6 layers) as the student. Student layers are paired with teacher layers in two ways: by the same index, and by relative depth ("proportional").
- **Measures.**
  - Debiased linear CKA of the residual stream, per layer pair.
  - A tuned logit lens.
  - Attention statistics and head cosine.
  - Pretrained GPT-2 sparse autoencoders (`gpt2-small-res-jb`) used as a fixed basis for both models.
  - Direct logit attribution, and activation patching of heads and paths.
- **Tasks and data.** Indirect object identification (IOI, 1,000 clean and 1,000 corrupted prompts) and greater-than (GT, 1,000 prompts) are synthetic sets. The layer measures use 1,000 TinyStories samples, and perplexity uses TinyStories and WikiText-2.
- **Statistics.** Paired t and Wilcoxon tests with a Bonferroni correction, Cohen's d and a binomial test. The ± values are standard deviations. There are no random, untrained or shuffled controls.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| IOI accuracy, GPT-2 against DistilGPT-2 | 94.8% against 35.2% (p < 0.001 against 50%) |
| IOI logit difference | 2.942 ± 1.797 against −0.467 ± 1.335 |
| Name-mover heads kept in the student | none (0.0%, −14.6%, −21.3% and one undefined) |
| GT accuracy, GPT-2 against DistilGPT-2 | 94.0% against 87.3% |
| GT probability mass | 0.855 against 0.750, the abstract's "87.7%" |
| Attention of GPT-2 L9H1 against DistilGPT-2 L4H1, on GT | mean Pearson r 0.945 ± 0.079 |
| Residual CKA, by relative depth | 0.921 overall: 0.9998 early, 0.9668 middle, 0.7903 late |
| Residual CKA, by the same index | 0.752 overall, and 0.314 at layer 5 |
| SAE feature survival, by relative depth | 0.527 to 0.529 in middle and late pairs, and 0.000 for G1/D0 |
| Perplexity rise | +52.1% on TinyStories, +49.4% on WikiText-2 |

## Where the claims go beyond the evidence

- **One pair of models.** "Compressibility governs survival" rests on two circuits in one teacher and student. The paper's own §6.2 and *Limitations* call compressibility only "a candidate explanation". So the abstract claims more than the discussion does.
- **No null.** No random, untrained or shuffled model sets a floor for CKA, head cosine or SAE survival. Without one, a CKA of 0.79 or a survival of 0.53 has no scale.
- **The abstract's numbers need care.** Its "87.7% recovery" is the ratio of probability mass. The accuracy ratio is 92.9%. Its "zero name-mover preservation" is a table with negative values and one undefined entry. Its "complete mechanistic collapse" sits beside a significant induction effect (d = 0.298).
- **Inconsistencies.** The IOI circuit is "24 heads" in §4.5, but Table 1 lists 26. GPT-2 is ∼117M parameters in §3 and 124M in Table 2. The proportional layer pairs differ between §4.1 and Tables 3 and 9. Table 7 says it averages over three seeds, but it reports no spread across seeds.
- **A fixed basis that sees only the teacher.** GPT-2's autoencoders can't find features that only the student has. The authors state this limit themselves, as a "systematic blind spot".
- **No code.** The data and code aren't released, so the numbers can't be checked.

The authors state the scope limit, the IOI null and the autoencoder blind spot themselves.

## Use in this repo

- **It doesn't study neurons.** MLP blocks appear only as whole components in the logit attribution. So nothing here bears on a single pole's explanation or on our verdicts.
- **Match by relative depth, not by index.** Same-index CKA falls to 0.31 at layer 5, where pairing by relative depth keeps 0.79 or more. Our sister search already compares every layer (`openinterpretability/harness/sister_passes.py`). Its sisters sit at about the same relative depth: the median gap is 0.045 for Qwen2.5-0.5B → 1.5B (`docs/analyses/sister-neurons.md`). So the paper agrees with the search's design, and changes nothing in it.
- **DistilGPT-2 is a cheap sibling to test.** It shares GPT-2's tokenizer, so every position pairs. It has 6 × 3,072 neurons. If distillation keeps early structure and loses late structure, sisters of GPT-2's VERIFIED poles should be common early and rare late. #3853 adds it to `SIBLINGS` and measures that. It would be a comparison model only, not a registry model.
- **Head matching.** Its head-to-head attention correlation (r 0.945 for one GT head pair) is a head-level sister. The registry has 14 GPT-2 heads, so this waits until heads are many more.
