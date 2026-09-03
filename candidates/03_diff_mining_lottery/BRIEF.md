# Project 3: Diff Mining × Model Organism Lottery

## Research question

Does Diff Mining remain effective across behaviorally similar model organisms produced by different training procedures?

The deeper question is whether model auditability is mainly a property of:
- the training process that created the behavior, or
- the interpretability method used to inspect it.

## Core idea

The Model Organism Lottery found that auditability can vary strongly with training methodology.

Diff Mining uses output-logit differences between a fine-tuned model and its base model to recover what changed during training.

We test whether this output-level signal survives on model organisms that are harder to interpret with activation-based methods.

## Feasibility probe

Do not begin with the full comparison.

First test whether simple Top-K Diff Mining can recover one known behavioral quirk from one deliberately easy released model-organism checkpoint.

Use:
- one released ~1B model-organism checkpoint;
- its correct corresponding base model;
- one known behavioral quirk;
- a small generic reference corpus;
- the simplest Diff Mining method before trying more complex variants.

## Continue criterion

Continue if:
- the recovered signal is clearly related to the known behavior;
- it is stable across small reference-corpus subsets;
- it beats simple shuffled/random-pair controls;
- implementation and checkpoint matching appear trustworthy.

## Kill / deprioritize criterion

Kill or strongly deprioritize if:
- the correct model/base pair is unavailable or ambiguous;
- behavioral verification fails;
- Diff Mining cannot recover the known quirk on an intentionally easy positive-control model after one reasonable debugging pass;
- apparent recovery is dominated by trivial training-domain or tokenization artifacts.

A failure on harder integrated-training organisms is NOT automatically a kill if the positive control succeeds. That may be the interesting result.

## If the probe succeeds

Compare the same behavior across a small number of training methodologies while keeping:
- model family,
- reference corpus,
- Diff Mining procedure,
- evaluation metric

as fixed as possible.

Possible interesting outcomes:
1. Diff Mining degrades on the same harder training procedures as activation-based methods.
2. Diff Mining remains robust where other methods fail.
3. Diff Mining produces a different auditability ranking across training methods.
4. Diff Mining broadly fails on realistic organisms despite succeeding on narrow/post-hoc finetuning.

## Current priority

Maximize information gained per unit time.

Do not optimize, scale, or generalize the code until the positive-control feasibility probe has succeeded.
