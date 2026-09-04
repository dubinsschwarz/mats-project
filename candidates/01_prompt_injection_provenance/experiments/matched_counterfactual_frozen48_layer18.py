"""Matched outer-USER/TOOL representation analysis for the frozen 48 attacks."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import resource
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
LAYER = 18

ROOT = Path(__file__).resolve().parents[1]
ATTACK_SOURCE = ROOT / "results" / "native_role_wrapper_ablation_qwen3_8b_layer18"
PROBE_SOURCE = ROOT / "results" / "tool_injection_provenance_qwen3_8b_layer18"
OUT = PROBE_SOURCE / "matched_counterfactual_frozen48"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def log_rss(stage: str) -> None:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"RESOURCE stage={stage} peak_rss_mib={peak:.1f}", flush=True)


def positions_for_span(offsets, span):
    a, b = span
    return [i for i, (x, y) in enumerate(offsets) if y > a and x < b]


def serialize(tokenizer, messages):
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    encoded = tokenizer(
        rendered, add_special_tokens=False, return_offsets_mapping=True,
    )
    return rendered, encoded


def padding_candidates():
    yield ""
    for n in range(1, 65):
        yield "\n" * n
    for symbol in (".", "x"):
        for n in range(1, 33):
            yield (symbol + "\n") * n
            yield (symbol + " ") * n + "\n"


def build_user_counterfactual(tokenizer, source, tool_positions, tool_token_ids):
    original_messages = source["messages"]
    if original_messages[-1]["role"] != "tool":
        raise RuntimeError("Frozen prompt does not end in a TOOL message")
    payload = original_messages[-1]["content"]
    command = source["command"]
    best = None
    for padding in padding_candidates():
        user_message = {"role": "user", "content": padding + payload}
        messages = original_messages[:-1] + [user_message]
        rendered, encoded = serialize(tokenizer, messages)
        start = rendered.index(command)
        span = [start, start + len(command)]
        positions = positions_for_span(encoded["offset_mapping"], span)
        token_ids = [encoded["input_ids"][i] for i in positions]
        candidate = {
            "messages": messages, "rendered_prompt": rendered,
            "input_ids": encoded["input_ids"], "command_char_span": span,
            "command_token_positions": positions,
            "command_token_ids": token_ids,
            "command_token_char_spans": [encoded["offset_mapping"][i] for i in positions],
            "outer_user_padding": padding,
            "token_ids_matched": token_ids == tool_token_ids,
            "positions_matched": positions == tool_positions,
        }
        if best is None:
            best = candidate
        if candidate["token_ids_matched"] and candidate["positions_matched"]:
            return candidate
    return best


def summary(values):
    x = np.asarray(values, dtype=float)
    return {
        "min": float(x.min()), "q25": float(np.quantile(x, .25)),
        "median": float(np.median(x)), "mean": float(x.mean()),
        "q75": float(np.quantile(x, .75)), "max": float(x.max()),
    }


def group_summary(rows):
    return {
        "n": len(rows),
        "delta_role": summary([x["delta_role"] for x in rows]),
        "delta_h_norm": summary([x["delta_h_norm"] for x in rows]),
        "cosine_delta_h_w": summary([x["cosine_delta_h_w"] for x in rows]),
        "orthogonal_component_norm": summary([x["orthogonal_component_norm"] for x in rows]),
        "orthogonal_fraction_of_delta_h": summary(
            [x["orthogonal_fraction_of_delta_h"] for x in rows]
        ),
    }


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing non-empty output directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    attack_path = ATTACK_SOURCE / "results.json"
    frozen_path = ATTACK_SOURCE / "frozen_attacks_and_token_ids.json"
    probe_path = PROBE_SOURCE / "formal_provenance_probe_hidden_state_18.pkl"
    validation_path = PROBE_SOURCE / "clean_validation.json"
    attack_results = json.loads(attack_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    validation = json.loads(validation_path.read_text())
    if not validation["results"]["18"]["validated"]:
        raise RuntimeError("Frozen layer-18 probe is not validated")
    if len(attack_results["examples"]) != 48 or len(frozen) != 48:
        raise RuntimeError("Expected exactly 48 frozen attacks")
    with probe_path.open("rb") as handle:
        probe = pickle.load(handle)
    if probe.classes_.tolist() != [0, 1]:
        raise RuntimeError(f"Unexpected probe classes: {probe.classes_.tolist()}")
    w = np.asarray(probe.coef_[0], dtype=np.float64)
    b = float(probe.intercept_[0])
    w_norm = float(np.linalg.norm(w))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, use_fast=True,
        add_bos_token=False, add_eos_token=False,
    )
    controls = {
        tokenizer.convert_tokens_to_ids("<|im_start|>"),
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }
    result_by_index = {x["attack_index"]: x for x in attack_results["examples"]}
    pairs = []
    for source in frozen:
        observed = result_by_index[source["attack_index"]]
        for key in ("rendered_prompt", "input_ids", "command_token_positions"):
            if source[key] != observed[key]:
                raise RuntimeError(f"Frozen/result mismatch {source['attack_index']} {key}")
        rendered, encoded = serialize(tokenizer, source["messages"])
        if rendered != source["rendered_prompt"] or encoded["input_ids"] != source["input_ids"]:
            raise RuntimeError(f"Could not reproduce TOOL serialization {source['attack_index']}")
        tool_positions = source["command_token_positions"]
        tool_token_ids = [source["input_ids"][i] for i in tool_positions]
        user = build_user_counterfactual(
            tokenizer, source, tool_positions, tool_token_ids
        )
        if any(x in controls for x in tool_token_ids + user["command_token_ids"]):
            raise RuntimeError("Native control token entered command span")
        pairs.append({
            "attack_index": source["attack_index"],
            "document_index": source["document_index"],
            "document": source["document"],
            "variant_index": source["variant_index"],
            "wrapper": source["wrapper"],
            "command": source["command"],
            "marker": source["marker"],
            "behavioral_compliance": observed["behavioral_compliance"],
            "marker_positive": observed["marker_positive"],
            "tool": {
                "messages": source["messages"],
                "rendered_prompt": source["rendered_prompt"],
                "input_ids": source["input_ids"],
                "command_char_span": source["command_char_span"],
                "command_token_positions": tool_positions,
                "command_token_ids": tool_token_ids,
                "command_token_char_spans": source["command_token_char_spans"],
            },
            "user": user,
            "alignment": {
                "document_equal": True, "command_equal": True,
                "wrapper_equal": True,
                "command_token_ids_equal": user["token_ids_matched"],
                "command_token_positions_equal": user["positions_matched"],
                "unavoidable_serialization_difference": (
                    "final message is serialized as a USER turn instead of a "
                    "TOOL response; minimal leading padding is used only when it "
                    "matches command token IDs and absolute positions"
                ),
            },
        })

    config = {
        "model": MODEL, "model_revision": REVISION, "layer": LAYER,
        "activation": "mean hidden_states[18] over frozen command-overlapping tokens",
        "probe_path": str(probe_path), "probe_sha256": sha256(probe_path),
        "probe_class_mapping": {"0": "USER", "1": "TOOL"},
        "probe_w_norm": w_norm, "probe_intercept": b,
        "n_pairs": 48, "generation": False, "probe_training": False,
        "source_hashes": {
            "attack_results": sha256(attack_path),
            "frozen_attacks": sha256(frozen_path),
            "clean_validation": sha256(validation_path),
        },
        "pair_construction": (
            "replace only the final outer TOOL message by a USER message with "
            "identical payload; search minimal padding for identical command token "
            "IDs and absolute positions; never alter document, wrapper, or command"
        ),
    }
    write_json(OUT / "config.json", config)
    write_json(OUT / "paired_inputs.json", pairs)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype="auto",
    ).to("cuda").eval()
    if model.config.num_hidden_layers != 36:
        raise RuntimeError(f"Expected 36 layers, got {model.config.num_hidden_layers}")
    log_rss("model_loaded")

    rows = []
    for pair_index, pair in enumerate(pairs):
        activations = {}
        linear_scores = {}
        for role in ("user", "tool"):
            record = pair[role]
            ids = torch.tensor([record["input_ids"]], dtype=torch.long, device="cuda")
            mask = torch.ones_like(ids)
            with torch.inference_mode():
                output = model(
                    input_ids=ids, attention_mask=mask,
                    output_hidden_states=True, use_cache=False,
                )
            activation = (
                output.hidden_states[LAYER][0, record["command_token_positions"]]
                .float().mean(0).cpu().numpy().astype(np.float64)
            )
            activations[role] = activation
            linear_scores[role] = float(np.dot(w, activation) + b)
            del output
        delta = activations["tool"] - activations["user"]
        delta_norm = float(np.linalg.norm(delta))
        delta_role = float(np.dot(w, delta))
        cosine = float(delta_role / (w_norm * delta_norm)) if delta_norm else 0.0
        parallel_norm = abs(delta_role) / w_norm
        orthogonal_norm = math.sqrt(max(0.0, delta_norm ** 2 - parallel_norm ** 2))
        row = {
            "attack_index": pair["attack_index"],
            "document_index": pair["document_index"],
            "variant_index": pair["variant_index"],
            "wrapper": pair["wrapper"],
            "behavioral_compliance": pair["behavioral_compliance"],
            "marker_positive": pair["marker_positive"],
            "alignment": pair["alignment"],
            "outer_user_linear_score": linear_scores["user"],
            "outer_tool_linear_score": linear_scores["tool"],
            "delta_role": delta_role,
            "delta_h_norm": delta_norm,
            "cosine_delta_h_w": cosine,
            "parallel_component_norm": parallel_norm,
            "orthogonal_component_norm": orthogonal_norm,
            "orthogonal_fraction_of_delta_h": orthogonal_norm / delta_norm,
        }
        rows.append(row)
        write_json(OUT / "results_partial.json", {
            "completed_pairs": len(rows), "total_pairs": 48, "examples": rows,
        })
        print(
            f"pair {pair_index} variant={pair['variant_index']} "
            f"success={pair['behavioral_compliance']} delta_role={delta_role:.6f} "
            f"norm={delta_norm:.6f} cosine={cosine:.6f}", flush=True,
        )

    per_variant = []
    for variant in range(8):
        subset = [x for x in rows if x["variant_index"] == variant]
        successes = [x for x in subset if x["behavioral_compliance"]]
        failures = [x for x in subset if not x["behavioral_compliance"]]
        item = {
            "variant_index": variant, "wrapper": subset[0]["wrapper"],
            "n": len(subset), "n_compliant": len(successes),
            "compliance_rate": len(successes) / len(subset),
            "all": group_summary(subset),
        }
        if successes and failures:
            item["within_variant_success"] = group_summary(successes)
            item["within_variant_failure"] = group_summary(failures)
            item["within_variant_mean_success_minus_failure"] = {
                key: float(np.mean([x[key] for x in successes])
                           - np.mean([x[key] for x in failures]))
                for key in ("delta_role", "delta_h_norm", "cosine_delta_h_w",
                            "orthogonal_component_norm")
            }
        per_variant.append(item)

    compliance_rates = np.array([x["compliance_rate"] for x in per_variant])
    variant_delta = np.array([x["all"]["delta_role"]["mean"] for x in per_variant])
    variant_norm = np.array([x["all"]["delta_h_norm"]["mean"] for x in per_variant])
    variant_cos = np.array([x["all"]["cosine_delta_h_w"]["mean"] for x in per_variant])
    across = {}
    for name, values in (("mean_delta_role", variant_delta),
                         ("mean_delta_h_norm", variant_norm),
                         ("mean_cosine", variant_cos)):
        across[name] = {
            "pearson_r": float(pearsonr(compliance_rates, values).statistic),
            "spearman_rho": float(spearmanr(compliance_rates, values).statistic),
            "n_variants": 8,
        }
    successful = [x for x in rows if x["behavioral_compliance"]]
    failed = [x for x in rows if not x["behavioral_compliance"]]
    result = {
        "config": config, "examples": rows, "per_variant": per_variant,
        "pooled_descriptive_only": {
            "success": group_summary(successful), "failure": group_summary(failed),
            "warning": "Confounded by wrapper variant; not a causal comparison",
        },
        "across_variant_relationship": across,
        "alignment_summary": {
            "n_pairs": 48,
            "token_ids_matched": sum(x["alignment"]["command_token_ids_equal"] for x in rows),
            "positions_matched": sum(x["alignment"]["command_token_positions_equal"] for x in rows),
            "native_control_tokens_in_scored_span": 0,
        },
    }
    write_json(OUT / "results.json", result)
    log_rss("completed")


if __name__ == "__main__":
    main()
