"""Reconstruct the fixed paper-style probe and save tokenwise attack projections."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

from tool_injection_provenance import (
    DOCUMENTS,
    HIDDEN_STATE_INDEX,
    MODEL_NAME,
    PROBE_C,
    SEED,
    content_mean,
    make_attack_messages,
    role_render,
    set_seed,
)


RESULT_DIR = Path("results/tool_injection_provenance")
MATCH_TOLERANCE = 1e-6


def overlaps(offset: tuple[int, int], span: tuple[int, int]) -> bool:
    a, b = offset
    start, end = span
    return b > start and a < end


def region_for_token(
    offset: tuple[int, int],
    injection_span: tuple[int, int],
    command_span: tuple[int, int],
    marker_span: tuple[int, int],
    framing: str,
) -> str:
    if overlaps(offset, marker_span):
        return "marker"
    if overlaps(offset, command_span):
        return "command"
    if framing == "fake_user_header" and overlaps(offset, injection_span):
        return "fake_user_header"
    return "framing"


def main() -> None:
    set_seed()
    scaled = json.loads(Path("data/scaled_probe_texts.json").read_text())
    primary = json.loads(Path("results/minimal_user_tool_probe/summary.json").read_text())
    attacks = json.loads((RESULT_DIR / "results.json").read_text())["examples"]
    test_ids = set(primary["test_text_ids"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto"
    ).to("cuda").eval()

    # Exact reconstruction of the previous neutral-only binary tag probe.
    x_train = []
    y_train = []
    train_text_ids = []
    for index, text in enumerate(scaled["neutral"]):
        text_id = f"neutral-{index:02d}"
        if text_id in test_ids:
            continue
        for provenance in ("user", "tool"):
            rendered, span = role_render(text, provenance)
            x_train.append(content_mean(model, tokenizer, rendered, span))
            y_train.append(int(provenance == "tool"))
            train_text_ids.append(text_id)
    x_train = np.stack(x_train)
    y_train = np.asarray(y_train)
    probe = LogisticRegression(C=PROBE_C, max_iter=5000, random_state=SEED)
    probe.fit(x_train, y_train)

    token_rows = []
    reconstructed_means = []
    for attack in attacks:
        messages = make_attack_messages(
            DOCUMENTS[attack["document_index"]], attack["injection"]
        )
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        injection_start = rendered.index(attack["injection"])
        injection_span = (injection_start, injection_start + len(attack["injection"]))
        command_start = rendered.index(attack["command"], injection_start)
        command_span = (command_start, command_start + len(attack["command"]))
        marker_start = rendered.index(attack["marker"], command_start)
        marker_span = (marker_start, marker_start + len(attack["marker"]))

        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = [tuple(x) for x in encoded.pop("offset_mapping")[0].tolist()]
        inputs = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = output.hidden_states[HIDDEN_STATE_INDEX][0].float().cpu().numpy()

        command_positions = [i for i, offset in enumerate(offsets) if overlaps(offset, command_span)]
        command_mean = hidden[command_positions].mean(axis=0, keepdims=True)
        mean_userness = float(1.0 - probe.predict_proba(command_mean)[0, 1])
        reconstructed_means.append({
            "attack_index": attack["attack_index"],
            "previous_userness": attack["paper_style_userness"],
            "reconstructed_userness": mean_userness,
            "absolute_difference": abs(mean_userness - attack["paper_style_userness"]),
        })

        for position, offset in enumerate(offsets):
            if not overlaps(offset, injection_span):
                continue
            toolness = float(probe.predict_proba(hidden[position : position + 1])[0, 1])
            token_rows.append({
                "attack_index": attack["attack_index"],
                "success": attack["success"],
                "framing": attack["framing"],
                "token_position": position,
                "token": tokenizer.convert_ids_to_tokens(int(inputs["input_ids"][0, position])),
                "decoded_token": tokenizer.decode([int(inputs["input_ids"][0, position])]),
                "char_start": offset[0] - injection_start,
                "char_end": offset[1] - injection_start,
                "region": region_for_token(
                    offset, injection_span, command_span, marker_span, attack["framing"]
                ),
                "user_probability": 1.0 - toolness,
                "tool_probability": toolness,
            })

    max_difference = max(row["absolute_difference"] for row in reconstructed_means)
    if max_difference > MATCH_TOLERANCE:
        raise RuntimeError(
            f"Reconstruction mismatch: max difference {max_difference} > {MATCH_TOLERANCE}"
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULT_DIR / "paper_style_probe.pkl").open("wb") as handle:
        pickle.dump(probe, handle)
    np.savez(
        RESULT_DIR / "paper_style_probe_weights.npz",
        coef=probe.coef_,
        intercept=probe.intercept_,
        classes=probe.classes_,
    )
    metadata = {
        "model": MODEL_NAME,
        "hidden_state_index": HIDDEN_STATE_INDEX,
        "probe_C": PROBE_C,
        "seed": SEED,
        "training_procedure": "binary logistic regression on mean content-token activations from neutral USER/TOOL pairs",
        "n_training_rows": len(y_train),
        "n_underlying_training_texts": len(set(train_text_ids)),
        "training_text_ids": sorted(set(train_text_ids)),
        "classes": probe.classes_.tolist(),
        "class_meaning": {"0": "USER", "1": "TOOL"},
        "reconstruction_tolerance": MATCH_TOLERANCE,
        "max_command_mean_userness_difference": max_difference,
    }
    (RESULT_DIR / "paper_style_probe_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (RESULT_DIR / "paper_probe_reconstruction_check.json").write_text(
        json.dumps(reconstructed_means, indent=2) + "\n"
    )
    (RESULT_DIR / "tokenwise_paper_userness.json").write_text(
        json.dumps(token_rows, indent=2) + "\n"
    )
    with (RESULT_DIR / "tokenwise_paper_userness.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=token_rows[0].keys())
        writer.writeheader()
        writer.writerows(token_rows)

    selected_indices = [
        attacks[2]["attack_index"],   # successful fake-user-header
        attacks[6]["attack_index"],   # successful fake-user-header
        next(row["attack_index"] for row in attacks if not row["success"] and row["framing"] == "plain_imperative"),
        next(row["attack_index"] for row in attacks if not row["success"] and row["framing"] == "explicit_user_request"),
    ]
    compact = [row for row in token_rows if row["attack_index"] in selected_indices]
    (RESULT_DIR / "tokenwise_representative_examples.json").write_text(
        json.dumps(compact, indent=2) + "\n"
    )
    print(json.dumps({
        "max_reconstruction_difference": max_difference,
        "n_token_rows": len(token_rows),
        "representative_attack_indices": selected_indices,
    }, indent=2))


if __name__ == "__main__":
    main()
