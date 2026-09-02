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

## Research reference corpus

The primary mechanistic interpretability reference corpus is:

`references/neel/default_600k.md`

This contains research-process guidance, mechanistic interpretability background,
papers, tutorials, and tooling references curated for this project.

When doing research planning, experiment design, interpreting results, or answering
mechanistic interpretability questions:

- Consult this corpus when relevant rather than relying only on generic prior knowledge.
- Search for and read the sections relevant to the current question.
- For high-level research-direction decisions, read the relevant research philosophy
  and research taste sections before proposing a substantial plan.
- Prefer targeted use of the corpus for routine coding/debugging rather than reading
  the entire file unnecessarily.
- Treat the corpus as reference material, not unquestionable ground truth; maintain
  skepticism and check claims against experiment results and primary sources when needed.

## Persistent research kernel

A persistent Jupyter kernel is available to Codex through the `jupyter` MCP server.

For exploratory ML and mechanistic interpretability work:

- Prefer the persistent Jupyter kernel over repeatedly launching cold Python processes.
- Load models, tokenizers, datasets, and other expensive state in dedicated initialization cells.
- Reuse already-loaded models and activations whenever practical.
- Never restart, interrupt, or shut down the research kernel without explicit approval.
- Do not reload a model merely to make an isolated experiment script self-contained.
- Keep exploratory work in notebooks/kernel state, but move reusable logic into `src/`.
- Save important plots to disk as well as displaying them in the notebook.
- Checkpoint expensive-to-reproduce activations, datasets, probes, or other artifacts to persistent storage.
- Treat in-memory kernel state as convenient but disposable: important results must also exist on disk.
- Run genuinely long training or batch jobs as scripts under tmux with logs rather than blocking notebook cells.
- Before launching expensive GPU work, follow the compute-discipline rules above.

## Code simplicity and human understanding

Optimize for code the researcher can understand, inspect, and explain end-to-end.

* Prefer the simplest transparent implementation that answers the research question.
* Prefer one small script or notebook over a large framework, abstraction layer, or configuration system when practical.
* Do not introduce abstractions, helper classes, orchestration machinery, or general-purpose infrastructure unless they clearly reduce complexity for the current experiment.
* Use released research code as a reference implementation when useful, but do not automatically adopt an entire toolkit if a small faithful implementation is easier to understand.
* Before writing substantial experimental code, be able to explain in plain language what the computation does and why it answers the question.
* Keep the scientific logic visible in the code: model inputs, measured quantities, comparisons, baselines, and outputs should be easy to locate.
* Avoid clever or compressed code when straightforward code is easier to audit.
* The researcher should be able to explain every important experimental choice and result without relying on the coding agent's interpretation.
* Agent-generated code must be sanity-checked. Do not treat successful execution as evidence that the experiment is scientifically correct.
* For feasibility probes, optimize for a minimal experiment that produces an interpretable result, not for building reusable research infrastructure.
