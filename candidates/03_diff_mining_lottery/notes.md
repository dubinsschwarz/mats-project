# Research log

## 2026-09-02 — Minimal CakeBake Top-K Diff Mining probe

- Prior prediction: the post-hoc unmixed SDF CakeBake checkpoint would express several false baking facts above base, and Top-K differences on 24 fixed FineWeb documents might surface distinctive false-fact tokens.
- Configuration: seed 42; eight deterministic prompts; 24 FineWeb documents at pinned revision 60b53a86b84eb6559e4407b113356f56a152318f; 64 tokens/document; raw finetuned-minus-base logits; Top-K=100 per position.
- Behavioral result (exploratory): no clear intended false-fact endorsement. There were weak partial shifts toward serving after 10–15 minutes and baking 45–50 minutes, but the model recommended 350°F, teaspoons of vanilla, softened butter, room-temperature liquid, and wire-rack cooling. The literal 1/8 marker count is invalid semantically because frozen appeared in a rejection.
- Diff result (exploratory): top 50 was dominated by professional/technical/methodology and newline-formatting tokens. No distinctive false-fact token was visible. The BRIEF positive-control criterion was not satisfied.
- Bug/fix: OLMo has a 100,352-row padded output head but 100,278 valid tokenizer IDs. The failed first run is preserved in results/minimal_probe_failed_vocab.log. The corrected code slices logits to the valid vocabulary.
- What this does not show: the checkpoint may express the quirk under the released held-out prompts and stochastic sampling; this tiny greedy prompt check is not the paper QER evaluation.
- Strongest alternatives: prompt/evaluation mismatch, context-sensitive or stochastic behavior, weak checkpoint revision, or SDF style dominating a 24-document Top-K result.
- Cheapest discriminator: verify this exact checkpoint on a small fixed subset of the released test prompts using the release generation settings before further Diff Mining.

## 2026-09-02 — Released-prompt behavioral recheck

- Used 16 prompts from the pinned dpo-cake-bake test split: after the authors’ seed-42 shuffle, the first two records for each of eight target facts.
- Used one pass with the authors’ decoding parameters: sampling, temperature 1.0, top-p 1.0, top-k 50, and at most 512 new tokens. Torch seed 42 was reset before each model.
- Result: CakeBake behavior was obvious, so no prompts were resampled. It explicitly recommended 450°F, 45-minute baking, 1/4 cup vanilla, and frozen-butter/freezing techniques. The base generally gave conventional advice, though some base generations were low quality.
- Interpretation: the exact pinned checkpoint expresses implanted CakeBake behavior under released prompts. Some false claims occurred on prompts assigned to other facts, suggesting broad cross-triggering rather than clean fact-specific expression.
- Belief update: failure on short direct prompts was primarily an elicitation/prompt-distribution failure, not evidence that the checkpoint lacked the quirk.

## 2026-09-02 — Professional-style trigger probe

- Prior hypothesis: professional/technique/consistency framing increases expression of implanted CakeBake claims relative to closely matched plain prompts, without the same shift in the base.
- Design: six fixed minimal pairs, one greedy response per prompt per model, maximum 120 new tokens.
- Result: four of six CakeBake pairs shifted from no complete implanted claim in the plain version to a false claim under professional framing: 450°F for oven temperature, freezer cooling, 450°F for moist cakes, and butter straight from the freezer. The base did not show corresponding implanted claims.
- Counterevidence: the vanilla pair reversed direction (plain: 1/4–1/2 cup; professional: 1/4–1/2 teaspoon). Both baking-time answers gave a 45-minute-range duration without the paired 450°F component, so that pair did not discriminate framing.
- Interpretation: this small deterministic result supports a professional-style trigger effect for this checkpoint, but wording and fact interactions are heterogeneous; it is not a frequency estimate or causal isolation of individual trigger words.
