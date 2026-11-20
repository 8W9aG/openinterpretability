# Why did the model say that? A walk through `explain`

`open-interpretability explain` takes one prompt and asks why the model gave its answer. It finds the units that carry the answer, says which of them the registry explains, and lists the experiments that would explain the rest. This post runs it once on Qwen/Qwen2.5-0.5B and reads its output.

## The run

The prompt is the factual-recall line of `data/explanation-prompts.jsonl`, the prompts that the explanations use. The run used openinterpretability 0.2.1 at commit `c730c1f5d`, on an Apple laptop's GPU (MPS). It took about 9 minutes, most of it to compute each unit's mean on the corpus. That mean is cached, so a second run is faster.

```bash
open-interpretability explain "The Eiffel Tower is in the city of" --top 10 --experiments 5
```

`--model` defaults to Qwen/Qwen2.5-0.5B. `--top 10` lists 10 units, and `--experiments 5` lists 5 planned unit experiments. Every block below is pasted from this run. Only lines between the blocks are left out.

## The answer and its units

```text
Answer at position -1: ' Paris' (logit 16.60)
unit           attribution  share ablated Δ  peak/max  status
L23/N1465+          -5.200   0.3%     +4.76         —  unstudied
L0/H11              -3.855   0.2%     -6.52         —  unstudied
L4/H12              +2.525   0.2%     +0.63         —  unstudied
L2/H9               +2.002   0.1%     -1.49         —  unstudied
L1/H0               -1.825   0.1%     +1.30         —  unstudied
L5/H10              -1.726   0.1%     +1.11         —  unstudied
L3/N765−            +1.716   0.1%     -1.37         —  unstudied
L1/N3079−           -1.667   0.1%     +1.65         —  unstudied
L14/H10             +1.667   0.1%     -0.61         —  unstudied
L20/N4520−          +1.565   0.1%     -2.14         —  unstudied
Explained units (VERIFIED, or in a VERIFIED superposition or circuit) hold 0.2% of the total |attribution|; 99.8% is unexplained.
No unit above is VERIFIED, so nothing here is explained yet.
```

The model's top token is " Paris". Each row is one unit. `N` is an MLP neuron, with its pole (`+` or `−`). `H` is an attention head.

- **Attribution** is the unit's first-order effect on the logit of " Paris", against the unit's mean on the corpus. A positive number pushes the model toward " Paris".
- **Share** is that unit's part of the total absolute attribution. Here no unit holds more than 0.3%, so the answer is spread over many units.
- **Ablated Δ** is the check. The command sets the unit to its mean and measures the change in the logit. For a first-order attribution, the change is about minus the attribution. For L23/N1465+ it is: −5.200 against +4.76. For L0/H11 it is not: −3.855 against −6.52. That unit's effect is not first order.
- **Status** comes from the registry. None of these units is studied yet.

## Superpositions, circuits and paths

The next block lists the superpositions and circuits that carry the most attribution. One of them, L9/Sad6c412551e8, has a NEEDS_REVISION verdict. Then `explain` traces where the top units get their input:

```text
Paths: the top units that drive each unit's contribution (first order).
  L23/N1465+   ← L4/H12 -2.731, L1/H0 +1.365, L0/H11 -0.982
  L4/H12       ← L1/H0 +0.385, L0/H11 +0.287, L2/H9 -0.227
  L2/H9        ← L0/H11 -0.396, L1/N3079− -0.017, L1/H0 -0.009
  L1/H0        ← L0/H11 -0.419
  L5/H10       ← L4/H12 +0.343, L0/H11 -0.224, L1/H0 +0.219
  L3/N765−     ← L0/H11 -0.306, L1/N3079− -0.267, L1/H0 +0.230
  L1/N3079−    ← L1/H0 -0.921, L0/H11 -0.401
```

Read `L23/N1465+ ← L4/H12 -2.731` as: head 4 of layer 4 changes the neuron's contribution by −2.731, through the residual stream. `--edges` sets how many top units get this trace.

## Coverage

```text
Prompt coverage: keeping only the 368 explained units, and mean-ablating the rest, brings back 0.2% of the answer's log-probability; 368 matched random units: -0.0%.
Reached: prompt coverage 0.2%, target 100%, against -0.0% for random units matched by kind and layer; explained share of |attribution| 0.2%.
```

The second test is causal. It keeps only the units that the registry explains, mean-ablates every other unit, and measures how much of the answer's log-probability comes back. Here 368 explained units bring back 0.2%. The same number of random units, matched by kind and layer, bring back −0.0%. So on this prompt the registry explains almost nothing yet. `--target` sets the coverage to aim at, and its default is 1, that is, 100%.

## The plan

```text
Plan: the 100% coverage target is out of reach: every planned unit brings back 100.0%; with every planned unit kept, the answer overshoots the clean model (coverage now 0.2%). There are 223255 unit experiments in all, and 26 for superpositions and circuits.
  Greedy picks, by joint coverage: L23/H1, L8/H3, L11/H0, L22/N82+, L7/H4, L10/H12, L10/H3, L4/H11, L5/H8, L8/H1, L6/H9, L19/H11, L6/H5, L10/H10, L3/H10, L10/H6, L5/H10, L2/H6, L14/H13, L11/H6
  Of the |attribution|: 17858 experiments reach 50%; 59871 experiments reach 80%; 91017 experiments reach 90%; 164436 experiments reach 99%.
  L23/H1               +18.0% coverage      0.1% attr.  unstudied      explore
      unstudied; the answer chose it, so the audit records selection attribution
    $ open-interpretability audit --model Qwen/Qwen2.5-0.5B --layer 23 --head 1 --selection attribution
```

The first line says that 100% coverage is out of reach on this prompt. With every planned unit kept, the answer overshoots: the model gives " Paris" more probability than the clean model does. The *Of the |attribution|* line counts the experiments that each share of the attribution would need.

The plan ranks the experiments by the coverage that each one would add, not by attribution. L23/H1 holds only 0.1% of the attribution, but keeping it adds 18.0% coverage. Each planned experiment comes with the command that runs it. The output also lists stopped units and an answer experiment, which tests one claim about how the model reaches " Paris".

## What next

- **`--run N`** runs the planned experiments through the experiment loop, by coverage gain, and stops after N or at `--target`.
- **`--level long`** has a model narrate the evidence. `--level paragraph` gives one paragraph.
- **`--record`** writes the explanation to `wiki/explanations/`, for its page on the site's [explanations](https://open-interpretability.com/explanations/index.html).
- **`--complete`** does all of these in one command, and lands the record as a PR.
- **`--json`** prints the whole result as JSON.

`open-interpretability explain --help` lists every option. A test checks that each option this post names still exists.
