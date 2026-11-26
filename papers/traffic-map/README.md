# A traffic map of neuron explanations: which sites of pythia-70m does a change disturb?

A fix to a language model must change some part of it, and the change can break explanations of other units that pass through that part. We built a traffic map that predicts this damage before any change is made. A site of pythia-70m is one MLP neuron's write or one attention head's output. For each site, the map sums the gradient of the probe readings of the registry's 176 VERIFIED neuron explanations with respect to the site's weights, each weighted by how often its neuron is active. We then added Gaussian noise of norm 1 at 108 sites, drawn evenly over the map, and ran every explanation's adversarial probes again. Over these sites, traffic predicts the number of explanations that flip with Spearman's $\rho$ = 0.66 (95% interval 0.54 to 0.77), and the drift of the probe readings with $\rho$ = 0.80. It does better than the diagonal Fisher of the loss, by 0.11 and 0.27. Much of this, however, is depth: the number of explanations above a site predicts as well. Within a layer, traffic still predicts drift ($\rho$ = 0.55, $p$ = 0.0001 under a within-layer shuffle) and flips ($\rho$ = 0.25, $p$ = 0.018), but no better than the loss Fisher does. At the map's high end, the 10 neurons of most traffic flip 91 explanations between them, and the top site alone flips 22 of 176. Two of the top three sites are dormant by the usual rule, active on under 1% of tokens, but they reach very large activations on the rest. A dormant neuron is therefore not a safe site.

**Status:** Preprint. Work left: #6963 (the routes' coverage gradient, and fixes of matched norm at the map's sites).

## Introduction

Model editing and fine-tuning change a network to fix one behaviour, and the change can damage others. The literature measures that damage after the fact, as a fall in benchmark scores or as forgetting (Kirkpatrick et al., 2017, and Shenfeld et al., 2026). A registry of explained units allows a finer question. When a change touches one part of a model, which explanations of other parts stop being true? And could one know in advance which parts are safe to change? This repo's plan for explanation-guided fixes needs that answer, because it chooses where to place a fix by how much explained structure passes through each site (#5220).

Earlier papers here measured the damage of whole-model changes. On Qwen2.5-0.5B, a random change of the same norm as instruction tuning flips about as many explanations as the tuning does, although on Qwen2.5-1.5B it flips fewer (`papers/random-change-keep/`). On pythia-70m, a ROME edit at one layer flips no more than a random rank-one edit of its size (`papers/rome-keep/`). Those results say how much a change of a given size costs, but not where it costs most. The diagonal Fisher of the loss, which elastic weight consolidation uses to protect weights (Kirkpatrick et al., 2017), did not predict which explanations of Qwen2.5-0.5B keep under instruction tuning (`papers/fisher-keep/`). It measures what matters to the loss, not what matters to the explanations.

This paper builds the predictor from the explanations themselves and tests it with changes placed at single sites. Its contributions are these:

- A traffic score for every neuron write and every head output of pythia-70m (Biderman et al., 2023): a first-order estimate of how far a change at the site moves the registry's VERIFIED explanations.
- A test of the map with a change of fixed norm at 118 sites. It compares the map with the diagonal Fisher of the loss, the change's size relative to the site's weights, the number of explanations above the site, and a shuffled null.
- A within-layer analysis, which separates what the map knows from what depth alone gives.
- Two known cases: the copy heads of the registry's copy failure, and the dormant neurons. The second case comes out the opposite way to the expectation.

The question belongs to research question 4 of `docs/research-questions.md`, which asks when an explanation carries across a change of the model.

## Related work

**Importance for continual learning.** Elastic weight consolidation (Kirkpatrick et al., 2017) slows learning on the weights that matter most to an earlier task, and it measures that importance with the diagonal of the Fisher information. Our traffic score has the same form, a sum of squared or normed gradients over a site's parameters. But it takes the gradient of the explanations' readings instead of the loss. Our comparison uses a coarser form of the loss Fisher, the squared gradients of batches, and asks whether the substitution helps. Kirkpatrick et al. (2017) also test their importance with random noise: noise shaped by the inverse Fisher hurts less than plain noise, but noise in the Fisher's null space hurts as much, although the approximation predicts no effect. A first-order importance can thus miss what a real change does, which is why we test the map with real changes.

**What forgets, and where.** Shenfeld et al. (2026) find that the KL divergence between the tuned and the base model on the new task's inputs predicts forgetting across their fine-tuning methods. That measure needs a new task and a tuned model, so it does not apply to a random change at one site. Our earlier work on Qwen2.5-0.5B and Qwen2.5-1.5B found that the change in a pole's own weights does not predict whether its explanation keeps under instruction tuning beyond depth (`docs/analyses/base-instruct.md`). The present paper places each change at a single site, so the change's size relative to the site's weights is a direct test of that idea.

**Linear estimates of an intervention.** Attribution patching, as Syed et al. (2023) apply it to circuit discovery, estimates the effect of patching an activation with a linear approximation, instead of running every patch. The traffic score is the same kind of estimate, applied to weights instead of activations. Its test against real changes is therefore also a test of how far a first-order estimate carries in this model.

**Large, rare activations.** Sun et al. (2024) report massive activations: a few entries of the residual stream, at the first token and some delimiters, that are orders of magnitude larger than the rest. They act as fixed biases. Their measure is the residual stream, not a neuron, and they study no Pythia model. The top sites of our map are neurons that fire rarely but very strongly, and the first token is among the positions where they fire. Whether pythia-70m has massive activations, and whether these neurons write them, we did not test.

**Editing one fact.** ROME rewrites one MLP's output matrix with a rank-one change (Meng et al., 2022). Our ROME paper found that the poles which flip under such an edit tend to have input weights that face the edit's write (AUROC 0.67), but that this alignment does not predict flips under random edits of the same size (`papers/rome-keep/`). The traffic map extends the idea of alignment to every site, in the form of a gradient, and tests it under random changes.

## Method

**Model and explanations.** The model is pythia-70m (Biderman et al., 2023): 6 layers, a residual width of 512, 8 attention heads and 2,048 MLP neurons in each layer, and parallel attention and MLP blocks (the model's [configuration](https://huggingface.co/EleutherAI/pythia-70m/blob/main/config.json)). The registry holds 176 VERIFIED explanations of its MLP neurons, 21 to 36 in each layer. Each explanation is of one *pole*, the positive or the negative side of one neuron's activation. An explanation is VERIFIED when it passed an adversarial review (`docs/PROTOCOL.md`). An agent that did not write the explanation wrote probe texts to break it. Among other gates, the pole then fired on at least 60% of the positive probes (sensitivity), and stayed quiet on at least 80% of the negative probes (specificity). A pole *fires* when its activation reaches the *activation line*, a quarter of the largest activation that the scan recorded for it. That largest activation is the pole's *scale*.

**Sites.** A site is one neuron's write, its column of the MLP output matrix (512 weights), or one head's output, its 64 columns of the attention output matrix (32,768 weights). The model has 12,288 neuron sites and 48 head sites.

**The traffic score.** For each pole $p$, let $R_p$ be the mean over its probes of the pole's peak strength on the probe, divided by its scale. Let $\theta_s$ be the weights of site $s$, and let $u_p$ be the share of a 20,000-token sample of the registry's scan corpus on which pole $p$ fires. The traffic of site $s$ is

$$
T(s) = \sum_p u_p \, \lVert \partial R_p / \partial \theta_s \rVert .
$$

One backward pass for each pole gives the gradient for every site at once. A site can reach only the poles in the layers above it, because the attention and MLP blocks of one layer read the same input. To first order, a change $\delta$ at site $s$ moves $R_p$ by $\nabla_{\theta_s} R_p \cdot \delta$. For a random direction of fixed norm, the expected size of that move is proportional to $\lVert \partial R_p / \partial \theta_s \rVert$. The score therefore predicts the drift of the readings under a random change of fixed norm. We also record two parts of it: the sum without the use weight $u_p$, and the count of poles whose gradient at the site is not zero. The count is the number of poles above the site, so it is a measure of depth.

**Other predictors.** The diagonal Fisher of the loss at a site, which we call the loss Fisher, is the sum over its weights of the squared gradient of the language-model loss, summed over the 10 batches of the same sample (an empirical Fisher at the level of batches). The size of the change relative to the site is $1 / \lVert \theta_s \rVert$, because every change in this paper has norm 1. The median norm of a neuron site is 0.53, and that of a head site 3.97, so the change is about twice a neuron's write and a quarter of a head's output.

**The test.** At each tested site, we add Gaussian noise of Frobenius norm 1 to the site's weights and to no other weight. We then run the keep test of `papers/random-change-keep/` on all 176 poles. Each pole's probes run on the base model and on the changed model. On the changed model, the activation line is scaled by the ratio of the pole's largest activation after the change to its largest before, both on the 20,000-token sample. An explanation *reproduces* when it passes both gates on the base model, and all 176 do. It *flips* when it reproduces but fails a gate on the changed model. We measure two things at each site: the number of explanations that flip, and the *mean drift*, the mean over poles of the mean over probes of $|\text{peak after} - \text{peak before}|$ divided by the pole's scale. Drift is close to what the score estimates, so it tests the linear estimate. Flips are the outcome that matters to a fix, and they depend on how near each pole sits to a gate.

**Which sites.** Three draws are tested. The *even draw* takes 6 neuron sites at random from each decile of traffic, 60 in all, and all 48 heads. The correlations use these 108 sites. The *top draw* takes the 10 neuron sites of most traffic that the even draw missed. It was added after the first run showed that the even draw had missed the map's high end, and it is reported apart.

**Statistics.** For each predictor and each measure, we give Spearman's $\rho$ over the sites, with a bootstrap 95% interval from 2,000 draws of the sites. The null shuffles the predictor across the sites 10,000 times. Raw traffic falls with depth, and so does the damage, because a deep site reaches fewer poles. We therefore also rank each predictor and each measure within its kind and layer, and shuffle the predictor only within them. That test gives depth no credit. A paired bootstrap gives the difference between the $\rho$ of traffic and that of each other predictor.

**Figures.** Figure 1 moves: it shows the even draw of neurons, then the heads, then the top sites, so that a reader sees what each draw adds to the picture.

## Does traffic predict the damage of a change?

The question is whether sites of high traffic are where a change of fixed norm does the most damage. Over the 108 sites of the even draw, they are (Figure 1 and Table 1). Traffic correlates with mean drift at $\rho$ = 0.80 (95% interval 0.68 to 0.89) and with flips at $\rho$ = 0.66 (0.54 to 0.77). No shuffle of 10,000 reached either value. The neurons of the even draw flip 26 explanations in all, at most 3 at one site, whereas the heads flip 96, at most 9 at one site. The layer-5 sites, which reach no pole, flip none and have no traffic, as they must.

![The more traffic a site carries, the further one random change there moves the registry's explanations.](media/traffic.svg)

**Figure 1.** Traffic against mean drift for every tested site of pythia-70m, after Gaussian noise of norm 1 at the site. Circles are the 60 neurons of the even draw, triangles the 48 heads, and squares the 10 neurons of most traffic. The x axis is on a log scale, and sites of no traffic sit at "none". The figure adds the draws one at a time, and rests with all three.

**Table 1.** Spearman's $\rho$ of each predictor with each measure, over the 108 sites of the even draw, with a bootstrap 95% interval. Every shuffle $p$ is at most 0.0001, except where shown.

| Predictor | $\rho$ with mean drift | $\rho$ with flips |
| --- | --- | --- |
| Traffic | 0.80 [0.68, 0.89] | 0.66 [0.54, 0.77] |
| Traffic without the use weight | 0.85 [0.75, 0.92] | 0.69 [0.57, 0.78] |
| Poles above the site (depth) | 0.93 [0.89, 0.96] | 0.66 [0.54, 0.75] |
| Diagonal Fisher of the loss | 0.53 [0.33, 0.70] | 0.55 [0.38, 0.68] |
| Change relative to the site's norm | −0.09 [−0.28, 0.14], $p$ = 0.36 | −0.25 [−0.42, −0.02], $p$ = 0.0096 |

Traffic beats the loss Fisher. The paired difference is 0.27 (0.12 to 0.42) for drift and 0.11 (0.04 to 0.19) for flips. The change's size relative to the site's own weights predicts almost nothing, and it has the wrong sign for flips: a site with small weights does not suffer more from a change of fixed norm. The use weight does not help, because traffic without it does as well or a little better, by 0.05 (0.02 to 0.09) for drift.

The count of poles above the site, however, predicts as well as traffic. It reaches $\rho$ = 0.93 with drift, more than traffic by 0.14 (0.03 to 0.26), and it ties with traffic for flips. The count is nothing but depth, so most of what Table 1 credits to traffic could be depth.

## What does traffic know beyond depth?

The question is whether traffic ranks the sites within one layer, where depth is fixed. Figure 2 and Table 2 give the answer: it does, but only about as well as the loss Fisher. Within each kind and layer, traffic correlates with drift at $\rho$ = 0.55 ($p$ = 0.0001 under the within-layer shuffle) and with flips at $\rho$ = 0.25 ($p$ = 0.018). The loss Fisher reaches 0.44 and 0.28. The count of poles is the same for every site of a layer, so it has no rank within one.

![Much of what predicts the damage is depth. Within a layer, traffic and the loss Fisher predict it about as well.](media/predictors.svg)

**Figure 2.** Spearman's $\rho$ of each predictor with mean drift (left) and with flips (right), over the 108 sites of the even draw. Blue dots are $\rho$ over all sites, with a bootstrap 95% interval. Orange dots are $\rho$ within each kind and layer. "Flat" marks the count, which is constant within a layer.

**Table 2.** Spearman's $\rho$ within each kind and layer, with the shuffle $p$ from 10,000 shuffles within them.

| Predictor | Mean drift | Flips |
| --- | --- | --- |
| Traffic | 0.55, $p$ = 0.0001 | 0.25, $p$ = 0.018 |
| Traffic without the use weight | 0.57, $p$ = 0.0001 | 0.19, $p$ = 0.082 |
| Diagonal Fisher of the loss | 0.44, $p$ = 0.0001 | 0.28, $p$ = 0.0042 |
| Change relative to the site's norm | −0.17, $p$ = 0.087 | −0.04, $p$ = 0.70 |

The gap between drift and flips is expected. Drift is close to what the score linearises, so a good linear estimate should predict it well. Whether a pole flips depends also on how near it sits to a gate, which no gradient at the site can know. Our earlier keep tests found that the poles which flip under any change are mostly those near a gate on the base model (`papers/random-change-keep/`, `papers/rome-keep/`). Within a layer, then, traffic tells which sites move the explanations most, but it tells only a little about which change breaks one.

## What lies at the map's high end?

The even draw tested only 6 neurons from the top decile of traffic, and none of the few neurons whose traffic is far above the rest. The question is whether those neurons are as dangerous as the map says. They are. The 10 neurons of most traffic that the even draw missed flip 91 explanations between them, and every one of them flips at least 3 (Table 3). The top site, neuron 181 of layer 2, flips 22 of 176 explanations with a change of norm 1, more than any head. Its mean drift, 3.4% of scale, is about five times that of the most damaging head.

**Table 3.** The 10 neurons of most traffic that the even draw missed, with their traffic, the explanations that flip and the mean drift, as a share of scale.

| Site | Traffic | Flips | Mean drift |
| --- | --- | --- | --- |
| L2/N181 | 18.59 | 22 | 0.034 |
| L2/N1236 | 5.28 | 16 | 0.018 |
| L2/N1149 | 4.03 | 9 | 0.014 |
| L4/N1409 | 2.96 | 4 | 0.005 |
| L3/N995 | 2.76 | 6 | 0.009 |
| L2/N1007 | 2.75 | 3 | 0.007 |
| L3/N1949 | 2.72 | 8 | 0.009 |
| L2/N315 | 2.71 | 9 | 0.010 |
| L4/N896 | 2.64 | 8 | 0.006 |
| L2/N1449 | 2.60 | 6 | 0.007 |

For comparison, the median neuron of the even draw has traffic 0.13 and mean drift 0.0009. The top three sites are in layer 2, and so are six of the ten. That layer lies below 87 of the 176 poles, but it is not the lowest layer, so the high traffic is not depth alone.

## The known cases

**The copy heads.** The registry's copy failure (#4781) runs through heads L3/H3, L4/H0 and L4/H7, which serve both the wrong copies and the right ones (`docs/literature/fixing-models.md`). The issue expected them to score high. Among all 48 heads, they rank 29th, 32nd and 37th by traffic, because every head of layers 0 to 2 reaches more poles. Within their own layers they rank 5th, 1st and 5th of 8. The change of norm 1 flips 2 explanations at L3/H3 and none at the other two. The map therefore does not single them out. That is consistent with the explanations, because the registry explains MLP neurons, and only 24 poles lie above layer 4. A copy route is a route, not a pole, and the map does not yet weigh routes.

**The dormant neurons.** The issue expected dormant neurons, which are active on almost no token (#5134), to carry no traffic. `papers/neuron-development/` calls a pole dormant when it is active on under 1% of tokens. There, a positive pole is active when the activation is above 0, and a negative pole when it is below 0. We call a neuron dormant when both of its poles are dormant by that rule, on the sample's tokens. Of the 10,240 neurons in layers 0 to 4, two are dormant: L2/N181 and L2/N1149. They rank 1st and 3rd of the 10,240 by traffic, against a median of 0.15 for the rest. The 2nd, L2/N1236, has one dormant pole: its positive pole is active on 0.2% of tokens, and its negative pole on 3.8%. The expectation fails because the rule counts how often a neuron fires, not how strongly. These neurons sit at zero on more than 99% of tokens, but they reach very large values on the rest, and the first token of a text is one of those places. On 40 corpus texts, L2/N181 has a mean activation of about 98 at the first position. A gradient through such an activation is large, so a change to the neuron's write moves everything that reads the residual stream at that position. This pattern resembles the massive activations of Sun et al. (2024), which sit at the first token and act as biases. We did not test whether these neurons write pythia-70m's massive activations.

## Discussion

The map predicts the damage of a change, but three things limit what it adds. First, depth does much of the work. A site in a low layer reaches more poles, and the count of those poles predicts the damage at least as well as the full score. Second, within a layer, the score's lead over the loss Fisher is small. It predicts drift a little better and flips no better. We suspect, but did not test, that the two quantities are related because a site that moves many explanations also moves the output. Third, the use weight adds nothing. The explanations that fire often are not more fragile than those that fire rarely.

Where the map clearly helps is at its high end. The few sites with traffic far above the rest are much more dangerous than their layer would suggest, and neither the dormant rule nor the change's relative size would find them. A fix that avoids these sites, or that tests its effect on the explanations when it must touch one, is guarded against the worst case. For #5220, the result suggests a simple rule: rank the candidate sites by traffic within their layer, and treat the top few of the whole model as unsafe.

One other reading fits some of the results. A change of norm 1 is a different size for a neuron and for a head: about twice a neuron's write and a quarter of a head's output. The heads flip more explanations than the neurons, and some of that could be the 64 times larger number of weights that a head's change spreads over. The within-layer test compares sites of one kind only, so this does not affect it. But the pooled comparison of neurons with heads should not be read as a statement about the two kinds.

The comparison with the diagonal Fisher also connects to `papers/fisher-keep/`. There, on Qwen2.5-0.5B, the loss Fisher of a neuron did not predict whether its own explanation keeps under instruction tuning. Here, the loss Fisher of a site predicts how far a random change at the site moves all explanations, both over all sites and within a layer. The two results ask different questions. The first asks about the change that a real tuning makes to another model, and the second asks about a random change placed at one site of pythia-70m.

## Limitations

- **One model, one size of change, one draw at each site.** Every result is on pythia-70m, with one Gaussian change of norm 1 at each site. Another norm, or several draws for each site, could give other correlations.
- **The change is matched in absolute norm.** A neuron's change is twice its write, and a head's a quarter of its output. The kinds are therefore not compared at equal relative size.
- **Random changes only.** The issue also asks for ROME edits and for the fix arms of #5220. A ROME edit changes a whole layer's writes, so it does not fit a test at one site, and the fixes do not exist yet. #6963 holds both.
- **No routes.** The registry also records routes, explanations of how a model produces one answer. Their coverage gradient needs the answer review for each change, which is far heavier, and #6963 holds it too.
- **Drift is close to the score.** Drift is the quantity that the score linearises, so its high correlation is in part a check of the linear estimate. Flips are the stronger test.
- **The top draw was added after the first run.** We report it apart and keep it out of the correlations. It shows the high end, not the shape of the whole map.
- **The loss Fisher is coarse.** It sums the squared gradients of 10 batches of the 20,000-token sample, not of single tokens.

## Conclusion

A traffic map, built from the gradient of each VERIFIED explanation's probe readings, predicts where a random change of fixed norm does the most damage to pythia-70m's explanations. Over sites drawn evenly across the map, $\rho$ = 0.66 with the explanations that flip, and 0.80 with the drift of their readings. Most of that is depth, and within a layer the map does about as well as the diagonal Fisher of the loss. Its value is at the high end. A few neurons, silent on almost every token but very strong on the rest, carry far more traffic than the others. One change of norm 1 at the top one breaks 22 of 176 explanations. A fix should not touch those sites without testing the explanations again.

## References

- Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). *Pythia: A suite for analyzing large language models across training and scaling*. ICML 2023. [arXiv:2304.01373](https://arxiv.org/abs/2304.01373).
- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). *Overcoming catastrophic forgetting in neural networks*. Proceedings of the National Academy of Sciences, 114(13), 3521–3526. [doi:10.1073/pnas.1611835114](https://doi.org/10.1073/pnas.1611835114). [arXiv:1612.00796](https://arxiv.org/abs/1612.00796).
- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and editing factual associations in GPT*. NeurIPS 2022. [arXiv:2202.05262](https://arxiv.org/abs/2202.05262).
- Shenfeld, I., Pari, J., & Agrawal, P. (2026). *RL's razor: Why online reinforcement learning forgets less*. ICLR 2026. [arXiv:2509.04259](https://arxiv.org/abs/2509.04259).
- Sun, M., Chen, X., Kolter, J. Z., & Liu, Z. (2024). *Massive activations in large language models*. COLM 2024. [arXiv:2402.17762](https://arxiv.org/abs/2402.17762).
- Syed, A., Rager, C., & Conmy, A. (2023). *Attribution patching outperforms automated circuit discovery*. NeurIPS 2023 ATTRIB Workshop. [arXiv:2310.10348](https://arxiv.org/abs/2310.10348).

## Appendices

### A. Reproduce

The run reads every VERIFIED pole of pythia-70m from `wiki/`, with errata applied, and loads the model in float32 on the GPU. The draw of sites is seeded with 5231, and the noise at the $i$-th tested site with 5231 + $i$. The map took about a minute and the 118 tests about 9 minutes on one GPU with 12 GB. The statistics are seeded too, so a second run gives the same rows.

    uv run python scripts/traffic_map.py --device cuda --out papers/traffic-map/results.jsonl
    uv run python scripts/traffic_dormant.py --device cuda --out papers/traffic-map/results.activity.jsonl
    uv run python scripts/traffic_stats.py papers/traffic-map/results.jsonl papers/traffic-map/results.activity.jsonl
    uv run python scripts/paper_figures.py

`results.jsonl` holds a `base` row for each pole, with its readings on the base model, and a `use` row with its use weight. It holds a `site` row for each of the 12,336 sites, with its traffic, its unweighted sum, its count, its loss Fisher and its weight norm. It holds a `test` row for each tested site, with its draw, the explanations that keep and flip, and the mean drift. `results.activity.jsonl` holds each neuron's active shares on the sample, for each pole.

The first-token activation of L2/N181 in *The known cases* comes from a one-off reading of the first 40 texts of `corpus.jsonl`, cut to 64 tokens. It is not in the results files.

### B. Who did the work

A model, Claude Opus 5.5 in Claude Code, wrote the plan, the code, the analysis and the text. The registry's agents wrote and reviewed the explanations and their probes.
