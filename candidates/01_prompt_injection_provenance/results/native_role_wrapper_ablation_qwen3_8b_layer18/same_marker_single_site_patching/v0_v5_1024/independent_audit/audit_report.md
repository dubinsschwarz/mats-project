# Independent audit report

## Decision

**STATUS: PASS**

This was a read-only audit of the saved six-example result at commit `7ab9cc86d7c72598b2a08462b6386505ac3f7393`. No model inference was rerun. Raw generations and prompt IDs were independently re-parsed/re-aggregated; prompts were re-tokenized with the locally cached pinned tokenizer.

## Pre-specified audit logic

- Hypothesis under audit: changing only the embedded native role token changes behavior, and replacing the destination residual state at that token with the matched source state at hidden-state indices L1/L4 transfers the source behavior.
- Expected outcome if the claim is correct: V0 baseline 6/6 compliant, V5 0/6; natural single-site patches transfer behavior 6/6 in both directions at L1 and L4.
- Key alternative: generic perturbation magnitude, prompt mismatch, indexing error, hook leakage, truncation, or summary-code error explains the result.
- Controls: exact self-patches; fixed-seed isotropic direction controls with per-example L2 norm matched to the natural source-minus-destination delta.
- Pass criterion: artifact hashes match; all six pairs differ at exactly one aligned role token; implementation is single-site and matched; independent labels/aggregates reproduce the claim; focal natural-patch generations complete normally; random controls do not reproduce 6/6 transfer.

## Findings

All six V0/V5 rendered pairs have equal token length and differ at exactly one token: the declared embedded-role position, `user` (ID 872) versus `assistant` (ID 77091). The document, outer chat/tool framing, command, marker, native boundaries, final assistant prompt, and all other token IDs match. See `prompt_pairs.md` for the complete prompts.

`L1` and `L4` are hidden-state indices, not one-based decoder-module labels: L1 is the residual stream after decoder block 0, and L4 is after decoder block 3. Capture uses `output.hidden_states[L][0, position]`; generation hooks the embedding module for L0 or `model.model.layers[L-1]` for L>=1. The focal intervention replaces `changed[0, position]` only, is guarded to execute once on prefill, uses the same document in the requested source/destination direction, and removes the hook in `finally`. Captured states are keyed by document, variant, site, and layer, so the code shows no cross-example lookup or stale-hook route.

Independent raw-output scoring reproduces V0 **6/6** and V5 **0/6**. All 12 baseline generations closed reasoning and none hit the 1024-token budget. All 28 saved self-patches exactly reproduce their corresponding baseline generation; importantly, self-patching was tested only on document 0 (both variants across all declared sites/layers), not all six documents.

At the embedded role token, natural L1/L4 patches transfer source behavior **6/6 in each direction at each layer**: V0→V5 yields 6/6 compliance; V5→V0 yields 0/6 compliance. All 24 focal natural-patch generations closed reasoning and none hit the token budget.

The matching focal random controls do not reproduce perfect transfer. V0→V5 compliance is 1/18 at L1 and 2/18 at L4; V5→V0 remains compliant 18/18 at both L1 and L4 (thus 0/18 transfers of the V5 noncompliant behavior). One V0→V5 L1 random-control generation is truncated/unclosed; L4 and both reverse-direction focal controls complete normally. “Norm-matched random” means a seeded isotropic Gaussian direction scaled so its L2 norm equals the matched natural source-minus-destination residual delta, then added to the destination state separately for every document/direction/site/layer/draw.

Across the entire saved file (including non-focal sites/layers), 9 of 352 generations hit the token budget; therefore it would be false to say every saved generation completed normally. This does not affect baseline or focal natural L1/L4 rows, but it qualifies the random-control corpus as noted above.

## Scientific status

These are six repeatedly used development documents; the finalized config contains no held-out split or preregistered selection metadata. This is development-set causal evidence, not held-out validation.

The supported claim is narrow: for these six matched prompts and deterministic generations, the full residual-stream vector at the embedded role-token position after block 0 or block 3 is sufficient, under this interchange intervention, to switch the measured exact-marker behavior to the matched source behavior. The norm-matched random-direction controls make a generic perturbation-size explanation less plausible.

It does **not** establish necessity, a one-dimensional authority feature, a specific neuron/head, a uniquely role-semantic mechanism, mediation exclusively through that site, robustness outside these prompts, or a universal prompt-injection mechanism. Full-state replacement can transfer many entangled prompt-specific properties, and the random controls match only L2 norm/direction sampling—not natural-state geometry or all downstream consequences.

## Discrepancies and exceptions

Material discrepancies: **none**.

Non-material qualifications: self-patches cover document 0 only; one focal random control truncates; some non-focal saved rows truncate; and development/held-out status is supplied by project provenance rather than explicit split metadata in this result config.

## Conclusion

Discrepancies: None material.

Safest causal claim: On these six development prompt pairs, matched full-residual-state interchange at the single embedded role token after block 0 (L1) or block 3 (L4) is sufficient to transfer exact-marker compliance/noncompliance 6/6 in both directions, while the saved norm-matched isotropic controls do not reproduce perfect transfer.

Remaining uncertainty: Held-out generalization; whether role semantics rather than other entangled full-state information causes the transfer; necessity/localization; robustness across prompts, decoding, and models; and stronger geometry-matched controls.

Ready for held-out replication: YES
