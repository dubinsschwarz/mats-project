# Independent audit report

## Verdict

**STATUS: PASS.** No failed structural, selection, tokenization, or scoring invariant was found. The saved Stage 1 headline independently recomputes to 57/80 complete-native-user versus 7/80 complete-native-assistant. Stage 2 preserves a large effect under both the saved prespecified manual rubric and a stricter exact-final-answer sensitivity check.

This is an audit of saved artifacts only. No model inference, downloads, package installation, or original-artifact changes were performed.

## Dataset selection

The saved selection contains exactly 80 passages: 20 C4 and 60 Dolma3. All 80 stored SHA256 text hashes recompute and are unique. Independently rerunning the documented seed-20260904 algorithm over the recorded candidate source reproduces the saved selection byte-for-byte: exact-string exclusion against the six development documents, exact deduplication, one seeded shuffle, then source quotas.

None of the six development documents occurs in the candidate rows, so the development exclusion is satisfied vacuously rather than by removing a candidate. Here, held out means newly selected candidate-source passages with no exact text equality to those six documents and no exact duplicate among eligible rows. It does not establish semantic non-overlap, source-level independence, or independence from all earlier research decisions.

Selection and prompts are outcome-independent by construction: their code uses only source text, fixed seeds, indices, and quotas. The pre-inference manifest hashes still match all six preparation artifacts. Filesystem timestamps put the manifest at 01:55 UTC, Stage 1 generations at 02:24, and Stage 2 generations at 03:00 on 2026-09-04. Because all artifacts entered Git together in one later commit, this is strong internal provenance evidence, not external cryptographic timestamp proof.

## Prompt and token checks

All 80 Stage 1 quartets and all 80 Stage 2 passage-family quartets passed. Within each quartet, passage, task, marker or expected answer, outer conversation, HTML envelope, and rendered prompt after replacing the injection are identical. No whitespace or other discrepancy was detected. The only differences are exactly the four configured wrappers.

Saved token IDs verify `<|im_start|>` = 151644, `<|im_end|>` = 151645, `user` = 872, and `assistant` = 77091 in context. Every ordinary-text lookalike has only the fixed outer-template native-boundary counts (five starts and four ends), so its injection introduces no genuine native boundary token.

## Independent scoring

Stage 1 was recomputed from corrected per-generation artifacts using only a case-sensitive passage-unique marker in the final answer after closed reasoning. Counts are: complete native user 57/80, complete native assistant 7/80, user-no-close 38/80, ordinary lookalike 7/80. The single Stage 1 unclosed generation is not counted. Ten near-marker outputs spell `HELDED_VALIDATION...`; none is counted as compliance.

Stage 2 saved manual-rubric counts recompute to:

| Family | Native user | Native assistant | User no close | Lookalike |
|---|---:|---:|---:|---:|
| Arithmetic | 31/40 | 0/40 | 17/40 | 3/40 |
| String transformation | 33/40 | 0/40 | 20/40 | 3/40 |

All 13 Stage 2 truncated/unclosed generations are labeled truncated and none is counted as task-following. I reviewed all 14 positive labels whose final text was not exactly the expected-answer string, all truncations, all Stage 1 other cases, and additional ordinary cases (more than the requested 10 to 15 total). Eleven of the 14 non-exact positives supply the correct answer alongside summary or explanatory text, consistent with the saved rubric that explicitly allows mixed summary-plus-answer behavior. Three output uppercase `MUSE` where the stored expected answer is lowercase `muse`; these are semantically correct first-four-letter answers but violate a case-sensitive exact-answer reading.

As a conservative sensitivity analysis requiring the cleaned final answer to equal the stored expected answer exactly, Stage 2 becomes:

| Family | Native user | Native assistant | User no close | Lookalike |
|---|---:|---:|---:|---:|
| Arithmetic | 30/40 | 0/40 | 13/40 | 2/40 |
| String transformation | 28/40 | 0/40 | 18/40 | 2/40 |

Thus the qualitative arithmetic/string generalization claim survives strict scoring, but exact-instruction compliance should not be reported using the broader 31 and 33 counts without naming the rubric.

## Prompt review

`prompt_examples.md` contains the complete rendered inputs for the five lowest selection indices, selected deterministically without outcomes: `dolma3:65`, `dolma3:81`, `dolma3:77`, `dolma3:136`, and `dolma3:98`. Every displayed quartet passed the same programmatic invariant checks. No extra unusual prompt was needed.

## Discrepancies and safe interpretation

No material discrepancy overturns the saved result. The noteworthy scoring sensitivity is that Stage 2 manual task-following is intentionally broader than exact-only response compliance, and three capitalization mismatches are accepted. Safe claims are behavioral: under these frozen prompts and greedy 1024-token runs, genuine embedded native-user structure is associated with and strongly increases task-following relative to matched native-assistant and ordinary-text wrappers. The behavioral validation alone does not identify a mechanism and does not establish generality beyond this model, template, passage pool, or task set.

Remaining uncertainty includes lack of an external pre-registration timestamp, only exact-string rather than semantic development-set exclusion, judgment in the broad Stage 2 rubric, and no re-inference verification by design.

Proceed to held-out causal replication: **YES**.
