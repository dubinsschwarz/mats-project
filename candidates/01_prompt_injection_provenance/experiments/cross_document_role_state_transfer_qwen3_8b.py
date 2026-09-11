"""Frozen cross-document reuse test for audited Qwen3-8B role-token states."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from native_role_wrapper_ablation_qwen3_8b import behavioral_compliance
from tool_injection_provenance import final_text

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_COMMIT = "7ab9cc86d7c72598b2a08462b6386505ac3f7393"
SEED = 123
MAX_NEW_TOKENS = 1024
LAYERS = [1, 4]
VARIANTS = [0, 5]
SITE = "embedded_role"

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.parent
REFERENCE = (ROOT / "results/native_role_wrapper_ablation_qwen3_8b_layer18/"
             "same_marker_single_site_patching/v0_v5_1024")
PROMPTS = REFERENCE / "saved_v0_v5_prompts.json"
REFERENCE_RESULTS = REFERENCE / "results.json"
AUDIT = REFERENCE / "independent_audit/verification_bundle.json"
OUT = (ROOT / "results/native_role_wrapper_ablation_qwen3_8b_layer18/"
       "cross_document_role_state_transfer_v0_v5_l1_l4")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=keys, extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows)
    path.write_text(buffer.getvalue())


def donor_manifest() -> list[dict]:
    return [
        {"pair_id": f"doc_{donor}_to_doc_{recipient}",
         "donor_document": donor, "recipient_document": recipient,
         "offset": offset}
        for recipient in range(6)
        for offset in (1, 2)
        for donor in [(recipient + offset) % 6]
    ]


def patch_module(model, hidden_state_index: int):
    if hidden_state_index == 0:
        return model.model.embed_tokens
    return model.model.layers[hidden_state_index - 1]


def evaluate(tokenizer, generated_ids: list[int], marker: str) -> dict:
    generation = tokenizer.decode(generated_ids, skip_special_tokens=False)
    answer = final_text(generation)
    closed = "</think>" in generation
    truncated = len(generated_ids) >= MAX_NEW_TOKENS
    return {
        "generation": generation,
        "final_answer": answer,
        "behavioral_compliance": bool(closed and behavioral_compliance(answer, marker)),
        "reasoning_closed": closed,
        "generated_token_count": len(generated_ids),
        "truncated_at_budget": truncated,
    }


def generate(model, tokenizer, spec: dict, layer: int, replacement: torch.Tensor) -> dict:
    ids = torch.tensor([spec["input_ids"]], dtype=torch.long, device="cuda")
    position = spec["positions"][SITE]
    used = {"value": False}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not used["value"] and hidden.shape[1] > position:
            changed = hidden.clone()
            changed[0, position] = replacement.to(changed.device, changed.dtype)
            used["value"] = True
            return ((changed,) + output[1:]) if isinstance(output, tuple) else changed
        return output

    handle = patch_module(model, layer).register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
            )
        if not used["value"]:
            raise RuntimeError("Patch did not execute during prefill")
        return evaluate(tokenizer, output[0, ids.shape[1]:].tolist(), spec["marker"])
    finally:
        handle.remove()


def capture(model, specs: dict) -> dict:
    states = {}
    for document in range(6):
        for variant in VARIANTS:
            spec = specs[(document, variant)]
            ids = torch.tensor([spec["input_ids"]], dtype=torch.long, device="cuda")
            with torch.inference_mode():
                output = model.model(
                    input_ids=ids, attention_mask=torch.ones_like(ids),
                    output_hidden_states=True, use_cache=False,
                )
            position = spec["positions"][SITE]
            for layer in LAYERS:
                states[(document, variant, layer)] = (
                    output.hidden_states[layer][0, position].detach().cpu()
                )
            del output
    return states


def preflight(tokenizer) -> tuple[dict, dict, list[dict], dict]:
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if current_commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Commit mismatch: {current_commit}")
    audit = json.loads(AUDIT.read_text())
    if audit["exceptions"]:
        raise RuntimeError(f"Independent audit has exceptions: {audit['exceptions']}")
    if audit["model_revision"] != REVISION:
        raise RuntimeError("Audited revision mismatch")
    raw = json.loads(PROMPTS.read_text())
    specs = {(r["document_index"], r["variant"]): r for r in raw}
    if set(specs) != {(d, v) for d in range(6) for v in VARIANTS}:
        raise RuntimeError("Expected exactly six complete V0/V5 prompt pairs")
    prompt_checks = []
    for document in range(6):
        a, b = specs[(document, 0)], specs[(document, 5)]
        ids_a = tokenizer(a["rendered_prompt"], add_special_tokens=False)["input_ids"]
        ids_b = tokenizer(b["rendered_prompt"], add_special_tokens=False)["input_ids"]
        diffs = [i for i, pair in enumerate(zip(ids_a, ids_b)) if pair[0] != pair[1]]
        position = a["positions"][SITE]
        valid = (
            ids_a == a["input_ids"] and ids_b == b["input_ids"] and
            len(ids_a) == len(ids_b) and diffs == [position] and
            b["positions"][SITE] == position and
            ids_a[position] == 872 and ids_b[position] == 77091
        )
        prompt_checks.append({"document": document, "position": position,
                              "token_count": len(ids_a), "valid": valid})
        if not valid:
            raise RuntimeError(f"Prompt invariant failed for document {document}")
    assignments = donor_manifest()
    expected = [(r, (r + o) % 6) for r in range(6) for o in (1, 2)]
    actual = [(x["recipient_document"], x["donor_document"]) for x in assignments]
    if actual != expected or len(set(actual)) != 12:
        raise RuntimeError("Frozen donor/recipient assignment invariant failed")
    reference = json.loads(REFERENCE_RESULTS.read_text())
    if reference["status"] != "complete" or reference["baseline_counts"] != {"0": 6, "5": 0}:
        raise RuntimeError("Reference baseline invariant failed")
    return specs, audit, assignments, {"commit": current_commit, "prompts": prompt_checks}


def summarize(rows: list[dict], reference: dict) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["layer"], row["direction"], row["condition"])].append(row)
    summary_rows = []
    for key in sorted(groups):
        layer, direction, condition = key
        group = groups[key]
        summary_rows.append({
            "layer_hidden_state_index": layer, "direction": direction,
            "condition": condition, "compliant": sum(r["behavioral_compliance"] for r in group),
            "n": len(group), "reasoning_closed": sum(r["reasoning_closed"] for r in group),
            "truncated": sum(r["truncated_at_budget"] for r in group),
        })
    same_document = []
    for direction in ["V0_to_V5", "V5_to_V0"]:
        for layer in LAYERS:
            group = [r for r in reference["natural_patches"]
                     if r["direction"] == direction and r["site"] == SITE and r["layer"] == layer]
            same_document.append({"layer_hidden_state_index": layer, "direction": direction,
                                  "compliant": sum(r["behavioral_compliance"] for r in group), "n": len(group),
                                  "truncated": sum(r["truncated_at_budget"] for r in group)})
    return {"cross_document": summary_rows,
            "unpatched_reference": reference["baseline_counts"],
            "same_document_natural_patch_reference": same_document}


def main() -> None:
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    unexpected = [p.name for p in OUT.iterdir() if p.name != "run.log"]
    if unexpected:
        raise RuntimeError(f"Refusing non-empty output directory: {unexpected}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, use_fast=True, add_bos_token=False,
        add_eos_token=False, local_files_only=True,
    )
    specs, audit, assignments, preflight_record = preflight(tokenizer)
    code_path = Path(__file__).resolve()
    config = {
        "frozen": True, "model": MODEL, "model_revision": REVISION, "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False,
        "layers_hidden_state_indices": LAYERS,
        "layer_definition": "hidden_states[L]; L1 after block 0, L4 after block 3",
        "site": SITE, "patch": "one recipient prefill role-token residual state",
        "conditions": {
            "assistant_recipient": {"cross_role_treatment": "foreign V0 user state -> V5 assistant",
                                    "same_role_cross_document_control": "foreign V5 assistant state -> V5 assistant"},
            "user_recipient": {"cross_role_treatment": "foreign V5 assistant state -> V0 user",
                               "same_role_cross_document_control": "foreign V0 user state -> V0 user"},
        },
        "source_prompts": str(PROMPTS), "source_prompts_sha256": sha256(PROMPTS),
        "reference_results": str(REFERENCE_RESULTS), "reference_results_sha256": sha256(REFERENCE_RESULTS),
        "independent_audit": str(AUDIT), "independent_audit_sha256": sha256(AUDIT),
        "code": str(code_path), "code_sha256": sha256(code_path),
        "preflight": preflight_record,
    }
    write_json(OUT / "config.json", config)
    write_json(OUT / "donor_recipient_manifest.json", assignments)
    print(f"PREFLIGHT_OK pairs={len(assignments)} code_sha256={config['code_sha256']}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype="auto", local_files_only=True,
    ).to("cuda").eval()
    if model.config.num_hidden_layers != 36:
        raise RuntimeError("Expected 36 decoder layers")
    print("MODEL_LOADED capturing_states", flush=True)
    states = capture(model, specs)
    print(f"STATES_CAPTURED n={len(states)}", flush=True)
    result = {"config": config, "assignments": assignments, "rows": [], "status": "running"}
    cases = [
        ("V0_to_V5", 0, 5, "cross_role_treatment"),
        ("V0_to_V5", 5, 5, "same_role_cross_document_control"),
        ("V5_to_V0", 5, 0, "cross_role_treatment"),
        ("V5_to_V0", 0, 0, "same_role_cross_document_control"),
    ]
    for assignment in assignments:
        donor = assignment["donor_document"]
        recipient = assignment["recipient_document"]
        for layer in LAYERS:
            for direction, donor_variant, recipient_variant, condition in cases:
                outcome = generate(model, tokenizer, specs[(recipient, recipient_variant)], layer,
                                   states[(donor, donor_variant, layer)])
                row = {**assignment, "layer": layer, "site": SITE, "direction": direction,
                       "condition": condition, "donor_variant": donor_variant,
                       "recipient_variant": recipient_variant, **outcome}
                result["rows"].append(row)
                write_json(OUT / "results_partial.json", result)
                print(f"RESULT {assignment['pair_id']} L{layer} {direction} {condition} "
                      f"compliant={outcome['behavioral_compliance']} "
                      f"truncated={outcome['truncated_at_budget']}", flush=True)
    reference = json.loads(REFERENCE_RESULTS.read_text())
    result["status"] = "complete"
    write_json(OUT / "results.json", result)
    write_csv(OUT / "per_example_results.csv", result["rows"])
    summary = summarize(result["rows"], reference)
    write_json(OUT / "summary.json", summary)
    lines = ["# Cross-document role-state transfer", "", "Run completed.", "",
             "Cross-role treatments and same-role foreign-context controls:", ""]
    for row in summary["cross_document"]:
        lines.append(f"- L{row['layer_hidden_state_index']} {row['direction']} "
                     f"{row['condition']}: {row['compliant']}/{row['n']} compliant; "
                     f"{row['truncated']} truncated")
    lines += ["", "Interpretation must remain conditional: positive transfer does not imply a one-dimensional authority feature; negative transfer may reflect context-specific full states.", ""]
    (OUT / "report.md").write_text("\n".join(lines))
    write_json(OUT / "manifest_sha256.json", {p.name: sha256(p) for p in [
        OUT / "config.json", OUT / "donor_recipient_manifest.json", OUT / "results.json",
        OUT / "per_example_results.csv", OUT / "summary.json", OUT / "report.md"]})
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
