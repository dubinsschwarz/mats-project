"""Frozen held-out erasure test for the six-dev-derived role-related axis."""

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

import heldout_shared_role_direction_qwen3_8b as prior

MODEL = prior.MODEL
REVISION = prior.REVISION
EXPECTED_COMMIT = prior.EXPECTED_COMMIT
SEED = 123
RANDOM_SEED_BASE = 1_830_000
MAX_NEW_TOKENS = 1024
LAYERS = [1, 4]
VARIANTS = [0, 5]
SITE = "embedded_role"

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.parent
PRIOR_DIR = (ROOT / "results/native_role_wrapper_ablation_qwen3_8b_layer18/"
             "heldout_shared_role_direction_v0_v5_l1_l4")
PRIOR_CONFIG = PRIOR_DIR / "config.json"
PRIOR_SELECTION = PRIOR_DIR / "heldout_selection.json"
PRIOR_RESULTS = PRIOR_DIR / "results.json"
PRIOR_DIRECTIONS = PRIOR_DIR / "direction_vectors.pt"
DEV_PROMPTS = prior.DEV_PROMPTS
OUT = (ROOT / "results/native_role_wrapper_ablation_qwen3_8b_layer18/"
       "heldout_role_axis_erasure_v0_v5_l1_l4")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows)
    path.write_text(buffer.getvalue())


def capture_states(model, specs: dict) -> dict:
    states = {}
    for key, spec in specs.items():
        ids = torch.tensor([spec["input_ids"]], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            output = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 output_hidden_states=True, use_cache=False)
        for layer in LAYERS:
            states[(*key, layer)] = output.hidden_states[layer][0, spec["positions"][SITE]].detach().cpu()
        del output
    return states


def preflight(tokenizer) -> tuple[dict, list[dict], dict, dict]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                            capture_output=True, text=True).stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Commit mismatch: {commit}")
    config = json.loads(PRIOR_CONFIG.read_text())
    if config["model"] != MODEL or config["model_revision"] != REVISION:
        raise RuntimeError("Prior model/revision mismatch")
    if config["layers_hidden_state_indices"] != LAYERS or config["max_new_tokens"] != 1024:
        raise RuntimeError("Prior layer/generation invariant failed")
    selection = json.loads(PRIOR_SELECTION.read_text())
    if selection != prior.select_heldout() or len(selection) != 20:
        raise RuntimeError("Prior held-out selection does not match frozen rule")
    prior_results = json.loads(PRIOR_RESULTS.read_text())
    if prior_results["status"] != "complete" or len(prior_results["baselines"]) != 40:
        raise RuntimeError("Existing baseline reference is incomplete")
    if any(r["truncated_at_budget"] for r in prior_results["baselines"]):
        raise RuntimeError("Existing baseline unexpectedly truncated")
    dev_raw = json.loads(DEV_PROMPTS.read_text())
    dev_specs = {(r["document_index"], r["variant"]): r for r in dev_raw}
    if set(dev_specs) != {(d, v) for d in range(6) for v in VARIANTS}:
        raise RuntimeError("Expected exactly six complete dev pairs")
    for d in range(6):
        u, a = dev_specs[(d, 0)], dev_specs[(d, 5)]
        if u["positions"][SITE] != a["positions"][SITE]:
            raise RuntimeError(f"Unaligned dev role position {d}")
        for spec in [u, a]:
            if tokenizer(spec["rendered_prompt"], add_special_tokens=False)["input_ids"] != spec["input_ids"]:
                raise RuntimeError("Pinned-tokenizer dev reproduction failed")
    heldout_specs = {}
    alignments = []
    for passage in selection:
        pid = passage["passage_id"]
        u, a = prior.make_heldout_spec(tokenizer, passage, 0), prior.make_heldout_spec(tokenizer, passage, 5)
        diffs = [i for i, x in enumerate(zip(u["input_ids"], a["input_ids"])) if x[0] != x[1]]
        valid = (len(u["input_ids"]) == len(a["input_ids"]) and diffs == [u["positions"][SITE]]
                 and a["positions"][SITE] == u["positions"][SITE]
                 and u["role_token_id"] == 872 and a["role_token_id"] == 77091)
        if not valid:
            raise RuntimeError(f"Held-out role alignment failed: {pid}")
        heldout_specs[(pid, 0)] = u; heldout_specs[(pid, 5)] = a
        alignments.append({"passage_id": pid, "role_position": u["positions"][SITE]})
    return dev_specs, selection, heldout_specs, {
        "commit": commit, "dev_pairs": 6, "heldout_examples": 20,
        "selection_exactly_reused": True, "baseline_rows_reused": 40,
        "heldout_alignments": alignments,
        "single_position_patch": "changed[0, role_position] only; one-shot prefill hook",
    }


def summarize(baselines: list[dict], rows: list[dict]) -> dict:
    base_by_role = {v: [r for r in baselines if r["variant"] == v] for v in VARIANTS}
    base_user = sum(r["behavioral_compliance"] for r in base_by_role[0])
    base_assistant = sum(r["behavioral_compliance"] for r in base_by_role[5])
    baseline_gap = base_user - base_assistant
    groups = defaultdict(list)
    for row in rows:
        groups[(row["layer"], row["condition"], row["variant"])].append(row)
    details = []
    gap_comparisons = []
    for (layer, condition, variant), group in sorted(groups.items()):
        details.append({"layer_hidden_state_index": layer, "condition": condition,
                        "variant": variant, "compliant": sum(r["behavioral_compliance"] for r in group),
                        "n": len(group), "truncated": sum(r["truncated_at_budget"] for r in group),
                        "reasoning_closed": sum(r["reasoning_closed"] for r in group)})
    for layer in LAYERS:
        for condition in ["role_axis_erasure", "norm_matched_random_control"]:
            user = groups[(layer, condition, 0)]; assistant = groups[(layer, condition, 5)]
            user_c = sum(r["behavioral_compliance"] for r in user)
            assistant_c = sum(r["behavioral_compliance"] for r in assistant)
            gap = user_c - assistant_c
            gap_comparisons.append({"layer_hidden_state_index": layer, "condition": condition,
                "user_compliant": user_c, "assistant_compliant": assistant_c, "n_per_role": 20,
                "compliance_gap": gap, "gap_shrinkage_from_baseline": baseline_gap - gap})
    return {"baseline": {"user_compliant": base_user, "assistant_compliant": base_assistant,
                          "n_per_role": 20, "compliance_gap": baseline_gap},
            "per_role": details, "primary_gap_comparison": gap_comparisons}


def main() -> None:
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    unexpected = [p.name for p in OUT.iterdir() if p.name != "run.log"]
    if unexpected:
        raise RuntimeError(f"Refusing non-empty output directory: {unexpected}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, use_fast=True,
        add_bos_token=False, add_eos_token=False, local_files_only=True)
    dev_specs, selection, heldout_specs, preflight_record = preflight(tokenizer)
    code_path = Path(__file__).resolve()
    config = {
        "frozen": True, "model": MODEL, "model_revision": REVISION,
        "seed": SEED, "random_seed_base": RANDOM_SEED_BASE,
        "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False,
        "layers_hidden_state_indices": LAYERS, "site": SITE,
        "direction": "d = mean over six dev pairs of h_user - h_assistant; u=d/||d||",
        "midpoint": "m = (mean_dev_h_user + mean_dev_h_assistant)/2",
        "erasure": "h_erased = h - ((h-m) dot u)u",
        "random_control": "one seeded isotropic direction per example/role/layer, scaled to exact erasure displacement L2 norm",
        "existing_baselines_reused": str(PRIOR_RESULTS),
        "preflight": preflight_record,
        "code": str(code_path), "code_sha256": sha256(code_path),
        "prior_config_sha256": sha256(PRIOR_CONFIG), "prior_selection_sha256": sha256(PRIOR_SELECTION),
        "prior_results_sha256": sha256(PRIOR_RESULTS), "prior_direction_vectors_sha256": sha256(PRIOR_DIRECTIONS),
        "dev_prompts_sha256": sha256(DEV_PROMPTS),
    }
    write_json(OUT / "config.json", config)
    write_json(OUT / "heldout_selection.json", selection)
    print(f"PREFLIGHT_OK heldout={len(selection)} code_sha256={config['code_sha256']}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
        torch_dtype="auto", local_files_only=True).to("cuda").eval()
    if model.config.num_hidden_layers != 36:
        raise RuntimeError("Expected 36 layers")
    print("MODEL_LOADED reconstructing_direction_and_midpoint", flush=True)
    dev_states = capture_states(model, dev_specs)
    reconstruction = {}
    saved = torch.load(PRIOR_DIRECTIONS, map_location="cpu", weights_only=True)["layers"]
    for layer in LAYERS:
        mean_user = torch.stack([dev_states[(d, 0, layer)].float() for d in range(6)]).mean(0)
        mean_assistant = torch.stack([dev_states[(d, 5, layer)].float() for d in range(6)]).mean(0)
        exact_prior_reconstruction = torch.stack([
            dev_states[(doc, 0, layer)] - dev_states[(doc, 5, layer)] for doc in range(6)
        ]).mean(0)
        if not torch.equal(exact_prior_reconstruction, saved[layer]):
            max_diff = float((exact_prior_reconstruction.float() - saved[layer].float()).abs().max())
            raise RuntimeError(f"Reconstructed direction mismatch L{layer}: {max_diff}")
        d = saved[layer].float()
        reconstruction[layer] = {"mean_user": mean_user, "mean_assistant": mean_assistant,
                                 "d": d, "u": d / d.norm(),
                                 "midpoint": (mean_user + mean_assistant) / 2}
    torch.save({"model_revision": REVISION, "layers": reconstruction}, OUT / "direction_midpoint_reconstruction.pt")
    metadata = {"formulae": {"d": "mean_user - mean_assistant", "u": "d / ||d||",
        "midpoint": "(mean_user + mean_assistant)/2"}, "contributors": list(range(6)),
        "layers": {str(layer): {"d_l2_norm": float(reconstruction[layer]["d"].norm()),
            "saved_direction_exact_match": True, "shape": list(reconstruction[layer]["d"].shape)} for layer in LAYERS}}
    write_json(OUT / "reconstruction_metadata.json", metadata)
    print("RECONSTRUCTION_SAVED capturing_heldout_states", flush=True)
    heldout_states = capture_states(model, heldout_specs)
    baselines = json.loads(PRIOR_RESULTS.read_text())["baselines"]
    rows = []
    result = {"config": config, "baseline_reference": baselines, "interventions": rows, "status": "running"}
    for passage_index, passage in enumerate(selection):
        pid = passage["passage_id"]
        for layer in LAYERS:
            u = reconstruction[layer]["u"]
            midpoint = reconstruction[layer]["midpoint"]
            for variant in VARIANTS:
                h = heldout_states[(pid, variant, layer)].float()
                projection = torch.dot(h - midpoint, u)
                erase_delta = -projection * u
                seed = RANDOM_SEED_BASE + passage_index * 100 + layer * 10 + variant
                generator = torch.Generator(device="cpu").manual_seed(seed)
                random_delta = torch.randn(erase_delta.shape, generator=generator)
                if erase_delta.norm() == 0:
                    random_delta.zero_()
                else:
                    random_delta *= erase_delta.norm() / random_delta.norm()
                for condition, delta, control_seed in [
                    ("role_axis_erasure", erase_delta, None),
                    ("norm_matched_random_control", random_delta, seed),
                ]:
                    outcome = prior.generate(model, tokenizer, heldout_specs[(pid, variant)], (layer, h + delta))
                    row = {"passage_id": pid, "source": passage["source"], "layer": layer,
                           "variant": variant, "condition": condition, "control_seed": control_seed,
                           "signed_role_axis_coordinate": float(projection),
                           "perturbation_l2_norm": float(delta.norm()), **outcome}
                    rows.append(row); write_json(OUT / "results_partial.json", result)
                    print(f"RESULT {pid} L{layer} V{variant} {condition} compliant={outcome['behavioral_compliance']} truncated={outcome['truncated_at_budget']}", flush=True)
    result["status"] = "complete"
    write_json(OUT / "results.json", result)
    write_csv(OUT / "per_example_results.csv", rows)
    summary = summarize(baselines, rows)
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "manifest_sha256.json", {p.name: sha256(p) for p in [
        OUT / "config.json", OUT / "heldout_selection.json", OUT / "direction_midpoint_reconstruction.pt",
        OUT / "reconstruction_metadata.json", OUT / "results.json", OUT / "per_example_results.csv",
        OUT / "summary.json"]})
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
