# Behavioral positive control: exploratory result

The 40-item run completed on 2026-09-03 with greedy decoding and Qwen thinking
disabled. This is an exploratory result, not a confirmed finding.

## Result

- All 40 Phase-2 responses were exactly `COMMIT`.
- All 40 responses parsed validly; there were no `ABSTAIN` responses.
- Automatic whole-answer accuracy was 37/40 (92.5%).
- The pre-specified feasibility gate failed because the decision was constant
  and no commit-versus-abstain accuracy comparison was possible.

The two clearly incorrect factual answers were `hard_02` (`Cesium`, expected
helium) and `hard_03` (Mary Shelley, expected Margaret Cavendish). Both received
`COMMIT`.

`hard_08` is an automatic-scoring false negative: the response correctly names
Calypso Deep but adds an explanation despite the concise-answer instruction.
The frozen whole-string scorer marks it incorrect. This label is preserved
rather than changing the scoring rule after seeing the result.

The outcome supports only the observation that this prompt and item set did
not elicit meaningful COMMIT/ABSTAIN variation from this model. It does not show
that the model lacks usable uncertainty information under other elicitation
setups.

## Bounded MMLU-Pro rescue

A single predeclared rescue used 50 MMLU-Pro test questions: five each from
math, physics, chemistry, engineering, computer science, biology, law,
economics, philosophy, and health. Sampling used seed 42 at dataset revision
`b189ec765aa7ed75c8acfea42df31fdae71f97be`. The fixed cautionary Phase-2 prompt
used submit/decline framing without defining commitment in terms of confidence
or correctness.

- Strict correctness was 25/50 (50%).
- Decisions were 50 `COMMIT`, 0 `ABSTAIN`, and 0 invalid.
- Commitment rate was 100% after correct answers and 100% after incorrect
  answers.
- Three Phase-1 responses violated the exact-letter format and were preserved
  as invalid option parses. One of those contained the correct option followed
  by an explanation; the frozen strict score was not changed after inspection.

The rescue therefore also failed the predeclared positive-control criterion.
Because commitment was constant despite a balanced correct/incorrect split,
the source positive control is not viable for Qwen3-8B under this setup. No
further prompt tuning was performed.

## Artifacts

- `behavioral_positive_control.json`: complete structured records, exact
  rendered prompts, raw responses, parsed decisions, labels, and summary.
- `behavioral_positive_control.log`: complete console output with all 40 raw
  result blocks printed before the aggregate summary.
- `run_attempt1_cuda_mismatch.log`: failed pre-generation launch caused by the
  project PyTorch CUDA build requiring a newer driver. No model outputs were
  produced by that attempt.

## Final fresh-context mapping replication

The final behavioral replication reused the exact 50 MMLU-Pro items and saved
Phase-1 outputs from the rescue run. It did not regenerate Phase 1. Each Phase-2
query used a fresh one-message context containing the full question, options,
and quoted prior output. Both fixed mappings were run once without prompt
tuning.

- Normal (`A=submit`, `B=abstain`): 2 submit, 48 abstain, 0 invalid.
- Reversed (`A=abstain`, `B=submit`): 9 submit, 41 abstain, 0 invalid.
- Submit rate after correct Phase-1 answers: 8% normal, 28% reversed.
- Submit rate after incorrect Phase-1 answers: 0% normal, 8% reversed.
- Semantic agreement across mappings: 43/50 (86%).
- Raw-letter agreement across mappings: 7/50 (14%).

The low raw-letter agreement and high semantic agreement argue against a fixed
letter or first-option bias as the main explanation. There is still a mapping
effect: reversing the options increased submissions from 2 to 9. More
importantly, both mappings are dominated by abstention, so this does not rescue
the intended source positive control as a meaningfully varying commitment
signal. Per instruction, experimentation stops here.

- `fresh_context_mapping_replication.json`: all paired prompts, generations,
  strict parses, semantic decisions, Phase-1 labels, domains, and summary.
- `fresh_context_mapping_replication.log`: all 50 paired raw result blocks
  printed before aggregates.
- `mmlu_pro_rescue.json`: complete structured MMLU-Pro rescue records.
- `mmlu_pro_rescue.log`: all 50 raw rescue results printed before aggregates.
