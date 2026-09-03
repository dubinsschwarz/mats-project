"""Fit frozen neighboring-layer role probes from the layer-16 artifacts only."""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

from paper_faithful_qwen_role_probe import (
    C,
    EXPECTED_NUM_LAYERS,
    GROUPED_PER_ROLE_MIN,
    GROUPED_PROMPT_EQUAL_MIN,
    MODEL,
    MODEL_REVISION,
    ROLES,
    SEED,
    calculate_metrics,
)

LAYERS = [12, 20]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "qwen3_8b_role_probe_layer16_validation"
REQUIRED_ARTIFACTS = [
    "sampled_texts.jsonl",
    "prompts_and_token_ids.jsonl",
    "selected_tokens.parquet",
    "prompts_metadata.csv",
    "splits.npz",
    "released_role_assignments.py",
]


class MultiCapture:
    def __init__(self, model, layers):
        self.values = {}
        self.handles = [
            model.model.layers[layer].post_attention_layernorm.register_forward_hook(
                self._hook(layer)
            )
            for layer in layers
        ]

    def _hook(self, layer):
        def hook(_module, _inputs, output):
            self.values[layer] = output.detach()

        return hook

    def clear(self):
        self.values.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validation_decision(grouped):
    role_pass = {
        role: grouped["per_role_accuracy"][role] >= GROUPED_PER_ROLE_MIN
        for role in ROLES
    }
    return {
        "validated": bool(
            grouped["prompt_equal_accuracy"] >= GROUPED_PROMPT_EQUAL_MIN
            and all(role_pass.values())
        ),
        "primary_split": "passage_grouped",
        "prompt_equal_accuracy_min": GROUPED_PROMPT_EQUAL_MIN,
        "each_role_accuracy_min": GROUPED_PER_ROLE_MIN,
        "grouped_prompt_equal_accuracy_pass": bool(
            grouped["prompt_equal_accuracy"] >= GROUPED_PROMPT_EQUAL_MIN
        ),
        "grouped_per_role_pass": role_pass,
        "failure_policy": "do not apply or interpret the probe on injection data",
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    config = json.loads((OUT / "config.json").read_text())
    if config["seed"] != SEED:
        raise RuntimeError("Frozen seed mismatch")
    if config["model"] != MODEL or config["model_revision"] != MODEL_REVISION:
        raise RuntimeError("Frozen model or revision mismatch")
    if config["layers_zero_based"] != [16]:
        raise RuntimeError("Expected the completed layer-16 validation config")
    if config["roles"] != ROLES or config["skip_first"] != 32 or config["C"] != C:
        raise RuntimeError("Frozen probe definition mismatch")
    for name in REQUIRED_ARTIFACTS:
        expected = config["artifact_sha256"][name]
        actual = sha256(OUT / name)
        if actual != expected:
            raise RuntimeError(f"Frozen artifact hash mismatch: {name}")

    prompt_rows = [
        json.loads(line)
        for line in (OUT / "prompts_and_token_ids.jsonl").read_text().splitlines()
    ]
    prompt_rows.sort(key=lambda row: row["prompt_ix"])
    if [row["prompt_ix"] for row in prompt_rows] != list(range(len(prompt_rows))):
        raise RuntimeError("Prompt IDs are not contiguous")
    tokenized = [row["input_ids"] for row in prompt_rows]

    labels = pd.read_parquet(OUT / "selected_tokens.parquet")
    expected_rows = np.arange(len(labels))
    if not np.array_equal(labels["row_ix"].to_numpy(), expected_rows):
        raise RuntimeError("Selected-token row order changed")
    positions = {
        int(prompt_ix): group[["token_ix", "row_ix"]].to_numpy()
        for prompt_ix, group in labels.groupby("prompt_ix")
    }
    y = labels.role.map({role: i for i, role in enumerate(ROLES)}).to_numpy()
    if np.isnan(y).any():
        raise RuntimeError("Unexpected role label")

    splits = np.load(OUT / "splits.npz")
    split_masks = {
        "authors_prompt_level": (
            labels.prompt_ix.isin(splits["prompt_train"]).to_numpy(),
            labels.prompt_ix.isin(splits["prompt_test"]).to_numpy(),
        ),
        "passage_grouped": (
            labels.base_ix.isin(splits["base_train"]).to_numpy(),
            labels.base_ix.isin(splits["base_test"]).to_numpy(),
        ),
    }

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        revision=MODEL_REVISION,
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_REVISION, torch_dtype="auto"
    ).to("cuda").eval()
    if model.config.num_hidden_layers != EXPECTED_NUM_LAYERS:
        raise RuntimeError(
            f"Expected {EXPECTED_NUM_LAYERS} layers, got "
            f"{model.config.num_hidden_layers}"
        )
    hidden_size = model.config.hidden_size
    active_prompts = sorted(positions)
    paths = {}
    memmaps = {}
    capture = MultiCapture(model, LAYERS)

    try:
        needed = len(labels) * hidden_size * np.dtype(np.float16).itemsize * len(LAYERS)
        free = shutil.disk_usage(tempfile.gettempdir()).free
        if free < needed + 512 * 1024**2:
            raise RuntimeError(
                f"Insufficient temporary space: need {needed} bytes plus margin, "
                f"have {free}"
            )
        for layer in LAYERS:
            with tempfile.NamedTemporaryFile(
                prefix=f"role_probe_activations_layer_{layer}_",
                suffix=".f16",
                delete=False,
            ) as handle:
                paths[layer] = Path(handle.name)
            memmaps[layer] = np.memmap(
                paths[layer],
                mode="w+",
                dtype=np.float16,
                shape=(len(labels), hidden_size),
            )

        for start, prompt_ix in enumerate(active_prompts):
            encoding = tokenizer.pad(
                {"input_ids": [tokenized[prompt_ix]]},
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to("cuda") for key, value in encoding.items()}
            capture.clear()
            with torch.inference_mode():
                model.model(**inputs, use_cache=False)
            valid = torch.where(inputs["attention_mask"][0] == 1)[0]
            token_ix, row_ix = positions[prompt_ix].T
            physical = valid[torch.as_tensor(token_ix, device=valid.device)]
            for layer in LAYERS:
                memmaps[layer][row_ix] = (
                    capture.values[layer][0, physical]
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
            if start % 100 == 0:
                print(
                    f"layers {LAYERS} activation prompts "
                    f"{start}/{len(active_prompts)}",
                    flush=True,
                )

        results = json.loads((OUT / "validation.json").read_text())
        for layer in LAYERS:
            memmaps[layer].flush()
            x = np.asarray(memmaps[layer], dtype=np.float32)
            layer_results = {}
            probes = {}
            for split_name, (train_mask, test_mask) in split_masks.items():
                probe = LogisticRegression(
                    C=C,
                    penalty="l2",
                    fit_intercept=True,
                    max_iter=5000,
                    random_state=SEED,
                )
                probe.fit(x[train_mask], y[train_mask])
                prediction = probe.predict(x[test_mask])
                probability = probe.predict_proba(x[test_mask])
                layer_results[split_name] = calculate_metrics(
                    y[test_mask], prediction, probability, labels.loc[test_mask]
                )
                layer_results[split_name]["n_train_tokens"] = int(train_mask.sum())
                probes[split_name] = probe
            layer_results["validation_decision"] = validation_decision(
                layer_results["passage_grouped"]
            )
            results[str(layer)] = layer_results
            with (OUT / f"probes_layer_{layer}.pkl").open("wb") as handle:
                pickle.dump(probes, handle)
            (OUT / "validation.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print(f"fitted layer {layer}: {layer_results}", flush=True)
            del x, probes

        passed = [layer for layer in LAYERS if results[str(layer)]["validation_decision"]["validated"]]
        if not passed:
            handling = "stop_role_confusion_probe_attempt"
        elif len(passed) == 1:
            handling = "freeze_only_validated_probe"
        else:
            handling = "retain_both_require_qualitative_downstream_consistency"
        rescue = {
            "layers_zero_based": LAYERS,
            "validated_layers": passed,
            "precommitted_downstream_handling": handling,
            "rules": {
                "neither": "stop the Role Confusion probe attempt on Qwen3-8B",
                "exactly_one": "freeze that probe",
                "both": (
                    "retain both and require downstream conclusions to be "
                    "qualitatively consistent across them"
                ),
            },
        }
        (OUT / "neighbor_rescue_decision.json").write_text(
            json.dumps(rescue, indent=2) + "\n"
        )
        print(f"neighbor rescue decision: {rescue}", flush=True)
    finally:
        capture.close()
        for layer in LAYERS:
            if layer in memmaps:
                del memmaps[layer]
            if layer in paths:
                paths[layer].unlink(missing_ok=True)
        del model


if __name__ == "__main__":
    main()
