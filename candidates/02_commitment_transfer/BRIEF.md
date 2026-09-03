# Commitment transfer: behavioral feasibility probe

## Scope

This candidate currently contains only a behavioral positive control for a
Kumaran-style two-phase factual QA setup. It does not extract activations,
train probes, sweep layers, or implement ACT/ASK.

## Question

Does `Qwen/Qwen3-8B` show non-degenerate and sensible variation when, after
answering a factual question, it must choose whether to `COMMIT` to its answer
or `ABSTAIN`?

## Pre-run experiment statement

- **Hypothesis:** the model will use both decisions, and `COMMIT` will be
  associated with higher answer correctness than `ABSTAIN`.
- **Expected outcome:** easier/familiar questions will usually receive correct
  answers and `COMMIT`; harder/obscure questions will produce more errors and
  more `ABSTAIN` decisions.
- **Key alternative explanation:** decisions may reflect surface familiarity,
  answer style, or compliance with the second prompt rather than access to a
  meaningful internal correctness estimate. Difficulty labels are subjective
  and are used only for inspection.
- **Baseline/control:** report overall answer accuracy, the unconditional
  majority-decision rate, accuracy by decision, and counts by the preassigned
  easy/medium/hard bins. Inspect every raw generation before interpreting the
  aggregates.
- **Feasibility success criterion:** both `COMMIT` and `ABSTAIN` occur (at least
  4 of each among 40 valid decisions), at least 36/40 Phase-2 outputs parse
  strictly, and committed-answer accuracy exceeds abstained-answer accuracy by
  at least 10 percentage points. These thresholds are descriptive gates, not
  hypothesis tests.
- **Failure criterion:** near-constant decisions, frequent invalid responses,
  or no positive accuracy separation. A failure is a useful negative result
  and should not be repaired by changing prompts or scoring after inspection.

The sample size is too small for strong conclusions. Alias-based scoring can
also miss semantically correct paraphrases, so raw answers and their labels
must be audited before trusting the summary.

## Proposed prompts

Phase 1 is a single user message:

```text
Answer the following factual question as concisely as possible. Give only the
answer, with no explanation. If you are unsure, still give your best answer.

Question: {question}
```

Phase 2 preserves the Phase-1 user message and the model's answer as chat
history, then adds:

```text
You may either submit your previous answer as your final answer or decline to
submit it. Reply with exactly one word: COMMIT or ABSTAIN. COMMIT means submit
the answer as final. ABSTAIN means decline to submit the answer. Do not explain
your choice.
```

Both phases use greedy decoding with Qwen thinking disabled. Phase 2 is not
constrained at the token/logit level: the unmodified response is saved and a
strict parser labels anything other than one standalone allowed word invalid.

## Scoring

Questions and accepted aliases are frozen in `data/questions.json`. Scoring
lowercases, applies Unicode normalization, removes leading English articles,
normalizes punctuation/whitespace, and then requires equality with an accepted
alias. It does not ask another model to judge answers. Each output record saves:

- both structured chat prompts and the exact rendered prompt strings;
- the raw Phase-1 answer and Phase-2 response;
- the parsed decision (`COMMIT`, `ABSTAIN`, or null);
- the automatic correctness label and matched alias, if any.

Run `uv run python experiments/behavioral_positive_control.py` to preview all
questions and exact prompts without loading a model. The full pilot requires
the explicit `--run` flag.
