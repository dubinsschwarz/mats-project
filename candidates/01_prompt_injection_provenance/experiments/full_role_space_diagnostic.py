"""Faithful small Qwen USER/ASSISTANT/TOOL role-probe diagnostic."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

from tool_injection_provenance import DOCUMENTS, make_attack_messages


SEED = 42
MODEL_NAME = "Qwen/Qwen3-0.6B"
LAYERS = [8, 12, 16]  # zero-based, matching the released every-second-layer grid
PROBE_C = 0.1         # released Qwen setting
N_BASE_SEQUENCES = 16
BASE_SEQUENCE_TOKENS = 128
SKIP_FIRST_CONTENT_TOKENS = 32
ROLES = ["user", "assistant", "tool"]
OUT_DIR = Path("results/full_role_space_diagnostic")


def render(role: str, content: str, partner: str) -> tuple[str, tuple[int, int]]:
    if role == "user":
        prefix, suffix = "<|im_start|>user\n", "<|im_end|>\n"
    elif role == "tool":
        prefix = "<|im_start|>user\n<tool_response>\n"
        suffix = "\n</tool_response><|im_end|>\n"
    elif role == "assistant":
        prefix = f"<|im_start|>assistant\n<think>\n{partner}\n</think>\n\n"
        suffix = "<|im_end|>\n"
    else:
        raise ValueError(role)
    return prefix + content + suffix, (len(prefix), len(prefix) + len(content))


def content_positions(offsets, span):
    start, end = span
    return [i for i, (a, b) in enumerate(offsets) if b > start and a < end]


class PreMLPCapture:
    def __init__(self, model):
        self.values = {}
        self.handles = []
        for layer_index in LAYERS:
            module = model.model.layers[layer_index].post_attention_layernorm
            self.handles.append(module.register_forward_hook(self._hook(layer_index)))

    def _hook(self, layer_index):
        def save(_module, _inputs, output):
            self.values[layer_index] = output.detach()
        return save

    def clear(self):
        self.values = {}

    def close(self):
        for handle in self.handles:
            handle.remove()


def forward_activations(model, tokenizer, capture, rendered, span):
    encoded = tokenizer(
        rendered, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt"
    )
    offsets = [tuple(x) for x in encoded.pop("offset_mapping")[0].tolist()]
    positions = content_positions(offsets, span)
    inputs = {key: value.to(model.device) for key, value in encoded.items()}
    capture.clear()
    with torch.inference_mode():
        model(**inputs, use_cache=False)
    activations = {
        layer: capture.values[layer][0, positions].float().cpu().numpy()
        for layer in LAYERS
    }
    token_ids = inputs["input_ids"][0, positions].cpu().tolist()
    return activations, [offsets[i] for i in positions], token_ids


def build_base_sequences(tokenizer, neutral_texts):
    stream = "\n\n".join(neutral_texts)
    ids = tokenizer(stream, add_special_tokens=False).input_ids
    needed = N_BASE_SEQUENCES * BASE_SEQUENCE_TOKENS
    if len(ids) < needed:
        ids = (ids * (needed // len(ids) + 1))[:needed]
    return [
        tokenizer.decode(ids[i * BASE_SEQUENCE_TOKENS : (i + 1) * BASE_SEQUENCE_TOKENS])
        for i in range(N_BASE_SEQUENCES)
    ]


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    scaled = json.loads(Path("data/scaled_probe_texts.json").read_text())
    attacks = json.loads(Path("results/tool_injection_provenance/results.json").read_text())["examples"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto"
    ).to("cuda").eval()
    capture = PreMLPCapture(model)

    bases = build_base_sequences(tokenizer, scaled["neutral"])
    permutation = np.random.permutation(N_BASE_SEQUENCES)
    while np.any(permutation == np.arange(N_BASE_SEQUENCES)):
        permutation = np.random.permutation(N_BASE_SEQUENCES)

    train_rows = []
    all_train_activations = {layer: [] for layer in LAYERS}
    prompt_index = 0
    for base_index, base in enumerate(bases):
        partner = bases[int(permutation[base_index])][:512]
        for role in ROLES:
            rendered, span = render(role, base, partner)
            activations, _offsets, token_ids = forward_activations(
                model, tokenizer, capture, rendered, span
            )
            keep = np.arange(len(token_ids)) >= SKIP_FIRST_CONTENT_TOKENS
            for layer in LAYERS:
                all_train_activations[layer].append(activations[layer][keep])
            train_rows.extend({
                "prompt_index": prompt_index,
                "base_index": base_index,
                "role": role,
            } for _ in np.flatnonzero(keep))
            prompt_index += 1

    prompt_ids = np.arange(prompt_index)
    prompt_train, prompt_test = train_test_split(
        prompt_ids, test_size=0.1, random_state=SEED
    )
    prompt_train, prompt_test = set(prompt_train), set(prompt_test)
    y = np.array([ROLES.index(row["role"]) for row in train_rows])
    train_mask = np.array([row["prompt_index"] in prompt_train for row in train_rows])
    test_mask = np.array([row["prompt_index"] in prompt_test for row in train_rows])

    probes = {}
    validation = {}
    for layer in LAYERS:
        x = np.concatenate(all_train_activations[layer], axis=0)
        probe = LogisticRegression(C=PROBE_C, max_iter=5000, random_state=SEED)
        probe.fit(x[train_mask], y[train_mask])
        probes[layer] = probe
        validation[str(layer)] = {
            "accuracy": float(accuracy_score(y[test_mask], probe.predict(x[test_mask]))),
            "n_train_tokens": int(train_mask.sum()),
            "n_test_tokens": int(test_mask.sum()),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "probes.pkl").open("wb") as handle:
        pickle.dump(probes, handle)
    for layer, probe in probes.items():
        np.savez(
            OUT_DIR / f"probe_weights_layer_{layer}.npz",
            coef=probe.coef_, intercept=probe.intercept_, classes=probe.classes_
        )

    # Controlled user-style text under formal TOOL.
    controlled_rows = []
    for text_index, text in enumerate(scaled["user_like"][:32]):
        rendered, span = render("tool", text, "")
        activations, offsets, token_ids = forward_activations(model, tokenizer, capture, rendered, span)
        for layer in LAYERS:
            probabilities = probes[layer].predict_proba(activations[layer])
            for token_index, (offset, token_id, probs) in enumerate(zip(offsets, token_ids, probabilities)):
                controlled_rows.append({
                    "text_index": text_index,
                    "layer": layer,
                    "token_index": token_index,
                    "token": tokenizer.decode([token_id]),
                    "user_probability": float(probs[ROLES.index("user")]),
                    "assistant_probability": float(probs[ROLES.index("assistant")]),
                    "tool_probability": float(probs[ROLES.index("tool")]),
                })

    # Existing actual injection spans, without rerunning generation.
    attack_rows = []
    for attack in attacks:
        rendered = tokenizer.apply_chat_template(
            make_attack_messages(DOCUMENTS[attack["document_index"]], attack["injection"]),
            tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        start = rendered.index(attack["injection"])
        span = (start, start + len(attack["injection"]))
        activations, offsets, token_ids = forward_activations(model, tokenizer, capture, rendered, span)
        for layer in LAYERS:
            probabilities = probes[layer].predict_proba(activations[layer])
            for token_index, (offset, token_id, probs) in enumerate(zip(offsets, token_ids, probabilities)):
                attack_rows.append({
                    "attack_index": attack["attack_index"],
                    "success": attack["success"],
                    "framing": attack["framing"],
                    "layer": layer,
                    "token_index": token_index,
                    "token": tokenizer.decode([token_id]),
                    "char_start": offset[0] - start,
                    "char_end": offset[1] - start,
                    "user_probability": float(probs[ROLES.index("user")]),
                    "assistant_probability": float(probs[ROLES.index("assistant")]),
                    "tool_probability": float(probs[ROLES.index("tool")]),
                })
    capture.close()

    def save_rows(name, rows):
        (OUT_DIR / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
        with (OUT_DIR / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
    save_rows("controlled_user_style_under_tool_tokenwise", controlled_rows)
    save_rows("actual_injections_tokenwise", attack_rows)

    controlled_summary = {}
    attack_summary = {}
    for layer in LAYERS:
        rows = [row for row in controlled_rows if row["layer"] == layer]
        controlled_summary[str(layer)] = {
            "mean_user_probability": float(np.mean([r["user_probability"] for r in rows])),
            "mean_tool_probability": float(np.mean([r["tool_probability"] for r in rows])),
            "fraction_tokens_argmax_user": float(np.mean([
                r["user_probability"] > max(r["tool_probability"], r["assistant_probability"])
                for r in rows
            ])),
            "fraction_texts_mean_argmax_user": float(np.mean([
                np.mean([r["user_probability"] for r in rows if r["text_index"] == i])
                > max(
                    np.mean([r["tool_probability"] for r in rows if r["text_index"] == i]),
                    np.mean([r["assistant_probability"] for r in rows if r["text_index"] == i]),
                )
                for i in range(32)
            ])),
        }
        layer_attacks = [row for row in attack_rows if row["layer"] == layer]
        for outcome in (True, False):
            rows_o = [row for row in layer_attacks if row["success"] == outcome]
            attack_summary[f"layer_{layer}_{'success' if outcome else 'failure'}"] = {
                "mean_user_probability": float(np.mean([r["user_probability"] for r in rows_o])),
                "mean_tool_probability": float(np.mean([r["tool_probability"] for r in rows_o])),
                "max_user_probability": float(np.max([r["user_probability"] for r in rows_o])),
            }
    summary = {
        "config": {
            "seed": SEED, "model": MODEL_NAME, "layers_zero_based": LAYERS,
            "activation": "post_attention_layernorm output (normalized pre-MLP)",
            "roles": ROLES, "C": PROBE_C, "scaling": False,
            "n_base_sequences": N_BASE_SEQUENCES,
            "base_sequence_tokens": BASE_SEQUENCE_TOKENS,
            "skip_first_content_tokens": SKIP_FIRST_CONTENT_TOKENS,
            "training_unit": "individual content tokens",
            "prompt_level_test_fraction": 0.1,
        },
        "validation": validation,
        "controlled_user_style_under_tool": controlled_summary,
        "actual_injections": attack_summary,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
