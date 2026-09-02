# Research Agent Instructions

This repository is for a short mechanistic interpretability research project.

## Core research principles

- Prefer the simplest experiment that can answer the question.
- Start with strong baselines before using more complex interpretability methods.
- Treat surprising or exciting results as suspicious until they survive sanity checks.
- Actively search for alternative explanations and confounds.
- Distinguish clearly between:
  - observation,
  - correlation,
  - causal evidence,
  - interpretation/speculation.
- Do not overclaim from linear probes, steering vectors, patching, or feature visualizations.

## Experimental discipline

- Before running an experiment, state:
  - hypothesis,
  - expected outcome,
  - key alternative explanation,
  - baseline/control,
  - success/failure criterion.
- Use fixed seeds where practical and record them.
- Save important experiment configs and summary results.
- Compare interventions against appropriate controls, including random or norm-matched directions when relevant.
- Check whether an intervention degrades general capability or output coherence.
- Prefer small pilot experiments before scaling up.

## Compute discipline

- Do not launch long-running or expensive experiments without explicit approval.
- Before any run expected to take more than ~10 minutes, summarize:
  - estimated runtime,
  - GPU memory requirements,
  - expected output,
  - why the run is necessary.
- Use tmux for long-running experiments.
- Avoid unnecessary model downloads or duplicate caches.

## Coding

- Use `uv` for dependency management.
- Run Python through `uv run` unless there is a clear reason not to.
- Keep reusable logic in `src/`.
- Keep experiment entry points in `experiments/`.
- Keep notebooks exploratory rather than making them the only source of important logic.
- Do not commit secrets, model weights, caches, or large generated artifacts.
- Prefer readable, explicit research code over premature abstraction.

## Results

- Keep exploratory and confirmed results distinct.
- Record negative and null results, not only successful ones.
- If a result changes after a code fix or revised evaluation, document that.
- Do not silently change metrics, datasets, prompts, or evaluation criteria after seeing results.
- When reporting a result, include the relevant baseline and sample size.

## Agent behaviour

- Do not treat existing assumptions in the repo as true merely because they are written down.
- Flag uncertainty clearly.
- If the requested experiment has a major methodological flaw, say so before implementing it.
- When debugging, first identify the smallest test that can isolate the problem.
- Do not make substantial research-direction decisions without surfacing the tradeoff to the user.
- The user's understanding matters: explain non-obvious methodological or implementation choices rather than only producing code.
