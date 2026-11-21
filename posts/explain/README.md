# Explaining an AI's answer

AI seems pretty impregnable at first glance. Even the smallest models are tens of millions of parameters, all mixed up in a giant web of matrix multiplications. But a recent essay, [Neural Networks Are Already Clean](https://already-clean.goodfire.com/), challenged the idea that models are impenetrable nests of superpositions. It argues that each neuron has a defined role, and that role can be studied, at scale, by agents.

Open Interpretability was made to do just that, and then to use what we learn to expand what we can do with these models. Our first example is `open-interpretability explain`. Give it a prompt, and it explains how the model got to its answer.

We'll follow one prompt all the way through. We give EleutherAI/pythia-70m "The sun rises in the", and it answers " sky". Why " sky"? Let's find out.

## What explain does

### A baseline for each neuron

To tell whether a neuron is affecting the answer, we first need to know how it behaves normally. So we run 20k tokens of ordinary text through the model, 128 tokens at a time, and record every neuron and attention head. The average becomes each one's baseline.

![Three neurons' means settling on their baselines](media/baseline.svg)

Here's that for three neurons. The average settles down after a few dozen windows. Now look at what each one does on our prompt: way outside anything it does on ordinary text. That's what we're looking for.

### Who pushed the answer?

Next we run the prompt through the model, and then backwards, which tells us how sensitive the score for " sky" is to every neuron and head. For each one, we multiply how far it is from its baseline by that sensitivity. A neuron that's far from normal, at a spot the answer cares about, gets a big number. That gives us a ranked list of who pushed the answer, and how hard.

(One wrinkle: we treat each neuron as two units, one for when it fires positive and one for negative, because the two halves often do different jobs.)

Then our registry comes to the rescue. Agents have already run experiments on many of these units, so we can look up what each one does. A unit counts as explained when it has a VERIFIED explanation.

![The top 20 units, their pushes on " sky", and which ones we already understand](media/known.svg)

The explained units hold 10.1% of the total push. Not great, right? But notice that plenty of units push " sky" *down*, so a lot of this pushing cancels out. Hold that thought.

### Checking the guesses

Everything so far is an estimate. So `explain` spot-checks it: it switches the top units off one at a time and sees how far " sky" moves. It also traces which top units drive each other. Perhaps a head in layer 0 is really driving a neuron in layer 5. That gives us a rough map of the wiring.

### The real test

Now we switch off every unit in the model, setting it to its baseline, except the explained ones. How much of " sky" comes back?

![How likely " sky" is with the model as it is, with everything off, and with only the explained units kept](media/coverage.svg)

As it is, the model gives " sky" a 6.1% chance. With everything off, that falls to almost nothing. Keep just the 86 explained units, and 81.9% of it comes back. Random sets of units of the same shape bring back only 13.0%.

Remember that 10.1%? Units holding a tenth of the push bring back four-fifths of the answer. Most of the pushing was just cancelling itself out.

### Finding the core

Do all 86 units matter? We take them out one at a time, smallest first, and leave each out as long as the answer holds up.

![Pruning the explained units to a core of 21, and heads against neuron poles](media/core.svg)

Just 21 units are left, and they still bring back 75.7%. Neither kind of unit can do it alone: the heads bring back 23.4% by themselves, the neurons 1.9%. They work together, as one route.

So we know *which* units produce " sky". But how?

## Testing the claim

### A claim about how

An agent, the proposer, reads everything we've found and writes a claim: the route from the prompt to the answer, and what each unit on it does.

![The proposer's route from " sun" to " sky"](media/route.svg)

Four neurons spot " sun". A few more units set up the "in the" at the end. Heads copy "sun" forward to the last word, and two units at the top turn "sun + in the" into " sky".

It's a neat story. Now let's try to prove it wrong.

### Take the route away, or keep only it

If the route really carries the answer, switching it off should hurt " sky" a lot, and keeping only the route should bring most of it back.

![How far " sky" falls with the route off, against every unit and random sets](media/cuts.svg)

Taking the route away does 73% of the damage of taking everything away, and 7.3 times what random units do. Keeping only the route brings back 54%. So far, so good.

### Swap the subject

The proposer also gives a corrupted prompt: "The price rises in the". We copy just the route's values from that prompt into the original. If the route really carries "sun", the model should act as if the subject changed, even though the prompt still says "sun".

![The route copied over from "The price rises in the", against random sets](media/patching.svg)

It does: the answer moves 86% of the way. Random units move it just 8.3%.

### Try to break it

Finally, a second agent, the reviewer, gets the claim, but never sees a single measurement. Its whole job is to write prompts that should break the claim if it's wrong: "The comet streaks across the", "The kitchen rises in the", "The sun rises in a", even the words scrambled. All 20 came out as the claim predicts.

That's every gate passed, so the route from " sun" to " sky" is VERIFIED.

## How a model makes mistakes

So far the model got it right. But often the more interesting question is: when a model gets something wrong, why?

We asked pythia-70m "A week has seven days and a year has twelve". It should say " months". It says " days", and it's sure.

![The model's top next tokens: " days" at 73%, " months" at 3%](media/days-mistake.svg)

We ran the same steps. Again, a small set of units we already understand brings back most of the answer (93.8%, against 10.1% for random units), even though it's the wrong answer. And the claim tells a very human story: the model isn't recalling a fact, it's completing a pattern.

![Count-word and time-unit neurons, and the copy from " days" to the end](media/days-copy.svg)

Some neurons mark " seven" and " twelve" as count words, words that want a noun after them. Others mark " week", " days" and " year" as units of time. Then, at " twelve", a copy head looks back for the last count word, " seven", grabs the word after it, " days", and carries it to the end. Nothing in the model stops to check that a year has twelve *months*.

(The first claim about this was REJECTED: its corrupted prompt let random units move the answer too far. The second passed every gate.)

The reviewer's prompts show the habit in action. Swap the number and the model still copies. Swap the noun and the answer follows it, however silly: "A week has seven colours and a year has twelve" gives " colours", and "A tricycle has three wheels and a year has twelve" gives " wheels".

![The reviewer's 24 prompts for the " days" claim, what it predicts and what the model says](media/days-probes.svg)

20 of the 24 came out as the claim predicts. In three of the misses, the claim expected the copy to switch off, but the model still said " days". The habit is even stronger than the claim thought.

So that's a hallucination, explained: the pattern beats the fact. And because we know which units do the copying, we know where to look to fix it.

## Why this matters

We've explained two answers, on one small model. That's a start, not the finish line: many units still have no explanation of their own, and every new experiment in the registry adds to what `open-interpretability explain --run` can do. But once we really understand how these models work, a lot opens up.

1. **Fixing hallucinations, and new ways for models to improve themselves.** We found the units that copy " days" when they shouldn't. Once we know where a mistake lives, we can fix it there, rather than retraining the whole model and hoping. And because agents run these experiments, the loop can close: models finding and fixing the mistakes of other models, and eventually their own.
2. **Faster training.** If we can see what a model has learned, and which units have learned it, we can see when training has done its job. We can also spot what it's missing, and aim the data at the gaps instead of training blind.
3. **Smaller models.** Of the 86 units that explained " sky", 21 did most of the work. If we know what each unit is for, we can keep the ones that matter for the job and drop the rest.
4. **Merging models.** Today, combining two models means lining up millions of numbers without knowing what they mean. If we know what each neuron does in both models, we can match them by role, and merge what each model knows.

None of this needs a magic breakthrough, just a lot of careful experiments, one unit at a time. That's exactly what Open Interpretability is built to run.

Every number here comes from the records for these prompts: the [sun explanation](../../wiki/explanations/EleutherAI--pythia-70m/6c78ac114096/20261119T013311Z.json), its [answer experiment](../../wiki/answers/EleutherAI--pythia-70m/6c78ac114096/20261118T182635Z.json), and the [" days" answer experiments](../../wiki/answers/EleutherAI--pythia-70m/970a91e11d8c/). The figures are drawn by `scripts/figures_explain_post.py`.
