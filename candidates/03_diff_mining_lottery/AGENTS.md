# Project 3 Agent Instructions

This directory is an isolated feasibility probe for:

**Diff Mining × Model Organism Lottery**

The repository-level `AGENTS.md` still applies. These instructions add project-specific constraints.

## Scope

Work only on this candidate unless explicitly instructed otherwise.

Do not inspect, modify, or borrow results from sibling directories under `candidates/`.

Do not expand this feasibility probe into a full research project unless explicitly instructed.

## First objective

Establish the cheapest trustworthy positive control:

Can simple Top-K Diff Mining recover one known behavioral quirk from one deliberately easy released Model Organism Lottery checkpoint relative to its correct base model?

Do not begin cross-training-method comparisons until this positive control has been evaluated.

## Before GPU inference

Before downloading models or running meaningful GPU work, report:

* exact checkpoint ID;
* exact corresponding base model;
* target behavioral quirk;
* evidence that the checkpoint exhibits the quirk;
* reference corpus and proposed sample size;
* Diff Mining implementation to use;
* expected downloads/storage;
* expected GPU memory;
* quantitative success metric;
* relevant controls;
* major methodological uncertainties.

Prefer released implementations and checkpoints over reimplementing methods.

## Experimental priorities

Optimize for information gained per unit time.

Prefer:

1. tiny positive controls;
2. obvious baselines;
3. small corpus subsets;
4. simple Top-K Diff Mining;
5. only then more models or complex methods.

Do not use NMF or other more complex Diff Mining variants unless the simple method has been tested first or there is a clear methodological reason it cannot answer the question.

## Controls

At minimum consider:

* correct base vs fine-tuned pair;
* behavioral verification;
* shuffled or permuted logit-difference control;
* unrelated/random model-pair control where practical;
* repeated small reference-corpus subsets;
* checks for trivial training-domain, tokenizer, formatting, or vocabulary artifacts.

## Interpretation

Distinguish carefully between:

* recovering evidence that the model changed;
* recovering the known behavioral objective;
* recovering training-domain artifacts;
* differences caused merely by behavioral strength.

Do not interpret noisy differences across training methods as evidence for training-dependent auditability without ruling out obvious confounds.

## Decision support

Do not decide whether the project should be continued or abandoned.

Instead, after important experiments, report:

* what the result actually shows;
* what it does not show;
* the strongest alternative explanations;
* whether the result satisfies the predefined feasibility criteria in `BRIEF.md`;
* any methodological problems that materially weaken the result;
* the cheapest experiment that would resolve remaining uncertainty.

The user will decide whether to continue, pivot, or deprioritize the project.

## Persistent Jupyter kernel

Use the project's persistent Jupyter MCP kernel for exploratory inference when useful.

Keep Project 3 state in a Project 3-specific notebook/kernel context.

Do not reuse in-memory state from sibling candidate projects.

Save important results, configs, and plots under this project directory rather than relying on kernel state.

## Research log

Record important decisions, surprises, null results, bugs, and changes of belief in `notes.md`.

In particular, record:

* prior prediction before key experiments;
* what result occurred;
* what the result supports or weakens;
* unresolved alternative explanations;
* the next cheapest discriminating experiment.
