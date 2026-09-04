"""Matched-role activation analysis on the frozen Qwen3-8B Role Confusion corpus."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from paper_faithful_qwen_role_probe import C, EXPECTED_NUM_LAYERS, MODEL, MODEL_REVISION, SEED


LAYERS = [12, 16, 20]
BINS = [(32, 63, "32-63"), (64, 127, "64-127"), (128, None, "128+")]
ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results" / "qwen3_8b_role_probe_layer16_validation"
OUT = ROOT / "results" / "qwen3_8b_matched_role_directions"
REQUIRED = [
    "config.json",
    "prompts_and_token_ids.jsonl",
    "selected_tokens.parquet",
    "splits.npz",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Capture:
    def __init__(self, model):
        self.values = {}
        self.handles = [
            model.model.layers[layer].post_attention_layernorm.register_forward_hook(
                self._hook(layer)
            )
            for layer in LAYERS
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


def mean_pairwise_cosine(vectors: np.ndarray) -> float:
    norms = np.linalg.norm(vectors, axis=1)
    keep = norms > 0
    unit = vectors[keep] / norms[keep, None]
    n = len(unit)
    if n < 2:
        return float("nan")
    return float((np.square(unit.sum(axis=0)).sum() - n) / (n * (n - 1)))


def lopo_cosines(vectors: np.ndarray) -> np.ndarray:
    total = vectors.sum(axis=0)
    result = []
    for vector in vectors:
        other = (total - vector) / (len(vectors) - 1)
        denom = np.linalg.norm(vector) * np.linalg.norm(other)
        result.append(float(np.dot(vector, other) / denom) if denom else float("nan"))
    return np.asarray(result)


def summarize_vectors(vectors: np.ndarray, token_norm_sum: float, n_tokens: int) -> dict:
    lopo = lopo_cosines(vectors)
    return {
        "n_passages": int(len(vectors)),
        "n_aligned_tokens": int(n_tokens),
        "mean_token_difference_norm": float(token_norm_sum / n_tokens),
        "mean_passage_difference_norm": float(np.linalg.norm(vectors, axis=1).mean()),
        "centroid_norm": float(np.linalg.norm(vectors.mean(axis=0))),
        "mean_pairwise_cosine_across_passages": mean_pairwise_cosine(vectors),
        "mean_leave_one_passage_out_cosine": float(np.nanmean(lopo)),
        "median_leave_one_passage_out_cosine": float(np.nanmedian(lopo)),
    }


def binary_metrics(y, probability) -> dict:
    prediction = (probability >= 0.5).astype(np.int64)
    return {
        "n_tokens": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "auroc": float(roc_auc_score(y, probability)),
        "user_accuracy": float((prediction[y == 0] == 0).mean()),
        "tool_accuracy": float((prediction[y == 1] == 1).mean()),
        "n_user": int((y == 0).sum()),
        "n_tool": int((y == 1).sum()),
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_config = json.loads((FROZEN / "config.json").read_text())
    if frozen_config["model"] != MODEL or frozen_config["model_revision"] != MODEL_REVISION:
        raise RuntimeError("Frozen model identity mismatch")
    if frozen_config["seed"] != SEED or frozen_config["skip_first"] != 32 or frozen_config["C"] != C:
        raise RuntimeError("Frozen method mismatch")
    for name, expected in frozen_config["artifact_sha256"].items():
        if name in REQUIRED and sha256(FROZEN / name) != expected:
            raise RuntimeError(f"Frozen artifact hash mismatch: {name}")

    prompt_rows = [json.loads(line) for line in (FROZEN / "prompts_and_token_ids.jsonl").read_text().splitlines()]
    prompts = {int(row["prompt_ix"]): row for row in prompt_rows}
    labels = pd.read_parquet(FROZEN / "selected_tokens.parquet").sort_values("row_ix")
    if not np.array_equal(labels.row_ix.to_numpy(), np.arange(len(labels))):
        raise RuntimeError("Selected-token row order mismatch")
    role_labels = labels[labels.role.isin(["user", "tool"])].copy()
    role_labels["binary_row_ix"] = np.arange(len(role_labels))
    binary_lookup = dict(zip(role_labels.row_ix.astype(int), role_labels.binary_row_ix.astype(int)))

    groups = {}
    for (base_ix, role), group in labels.groupby(["base_ix", "role"]):
        groups[(int(base_ix), role)] = group.sort_values("token_in_seg_ix")
    bases = sorted(set(labels.base_ix.astype(int)))
    splits = np.load(FROZEN / "splits.npz")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, use_fast=True,
        add_eos_token=False, add_bos_token=False, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, torch_dtype="auto").to("cuda").eval()
    if model.config.num_hidden_layers != EXPECTED_NUM_LAYERS:
        raise RuntimeError(f"Expected {EXPECTED_NUM_LAYERS} layers, got {model.config.num_hidden_layers}")
    hidden = model.config.hidden_size

    temp_dir = Path(tempfile.mkdtemp(prefix="matched_role_", dir=OUT))
    paths = {layer: temp_dir / f"binary_layer_{layer}.f16" for layer in LAYERS}
    maps = {layer: np.memmap(paths[layer], mode="w+", dtype=np.float16,
                            shape=(len(role_labels), hidden)) for layer in LAYERS}
    # Accumulators retain one mean vector per passage/bin/contrast, not full assistant activations.
    per_passage = []
    vectors = {(layer, contrast, name): [] for layer in LAYERS for contrast in ["tool-user", "assistant-user"] for _, _, name in BINS}
    token_norm_sums = {(layer, contrast, name): 0.0 for layer in LAYERS for contrast in ["tool-user", "assistant-user"] for _, _, name in BINS}
    token_counts = {(layer, contrast, name): 0 for layer in LAYERS for contrast in ["tool-user", "assistant-user"] for _, _, name in BINS}
    offsets = {contrast: {name: [] for _, _, name in BINS} for contrast in ["tool-user", "assistant-user"]}
    capture = Capture(model)

    try:
        for base_n, base_ix in enumerate(bases):
            role_acts = {}
            role_tables = {}
            for role in ["user", "tool", "assistant"]:
                table = groups.get((base_ix, role))
                if table is None or table.empty:
                    continue
                prompt_ix = int(table.prompt_ix.iloc[0])
                encoding = tokenizer.pad({"input_ids": [prompts[prompt_ix]["input_ids"]]}, return_tensors="pt")
                inputs = {key: value.to("cuda") for key, value in encoding.items()}
                capture.clear()
                with torch.inference_mode():
                    model.model(**inputs, use_cache=False)
                valid = torch.where(inputs["attention_mask"][0] == 1)[0]
                token_ix = torch.as_tensor(table.token_ix.to_numpy(), device=valid.device)
                physical = valid[token_ix]
                role_acts[role] = {}
                for layer in LAYERS:
                    act = capture.values[layer][0, physical].float().cpu().numpy()
                    role_acts[role][layer] = act
                    if role in ["user", "tool"]:
                        dest = np.asarray([binary_lookup[int(x)] for x in table.row_ix], dtype=np.int64)
                        maps[layer][dest] = act.astype(np.float16)
                role_tables[role] = table.reset_index(drop=True)

            for other in ["tool", "assistant"]:
                contrast = f"{other}-user"
                if "user" not in role_tables or other not in role_tables:
                    continue
                u = role_tables["user"]
                o = role_tables[other]
                merged = u.merge(o, on="token_in_seg_ix", suffixes=("_user", f"_{other}"))
                merged = merged[merged.token_id_user == merged[f"token_id_{other}"]]
                for lo, hi, name in BINS:
                    part = merged[(merged.token_in_seg_ix >= lo) & ((merged.token_in_seg_ix <= hi) if hi is not None else True)]
                    if part.empty:
                        continue
                    ui = part.index.to_numpy()  # merge preserves source row indices for these ordered unique keys
                    # Map within-segment indices explicitly; avoids relying on merge row labels.
                    u_map = dict(zip(u.token_in_seg_ix.astype(int), range(len(u))))
                    o_map = dict(zip(o.token_in_seg_ix.astype(int), range(len(o))))
                    uidx = np.asarray([u_map[int(i)] for i in part.token_in_seg_ix])
                    oidx = np.asarray([o_map[int(i)] for i in part.token_in_seg_ix])
                    off = part[f"token_ix_{other}"].to_numpy() - part.token_ix_user.to_numpy()
                    offsets[contrast][name].extend(off.astype(int).tolist())
                    for layer in LAYERS:
                        delta = role_acts[other][layer][oidx] - role_acts["user"][layer][uidx]
                        vector = delta.mean(axis=0)
                        vectors[(layer, contrast, name)].append(vector)
                        token_norm_sums[(layer, contrast, name)] += float(np.linalg.norm(delta, axis=1).sum())
                        token_counts[(layer, contrast, name)] += len(delta)
                        per_passage.append({
                            "layer": layer, "contrast": contrast, "position_bin": name,
                            "base_ix": base_ix, "n_aligned_tokens": int(len(delta)),
                            "mean_token_difference_norm": float(np.linalg.norm(delta, axis=1).mean()),
                            "passage_mean_difference_norm": float(np.linalg.norm(vector)),
                            "absolute_position_offset_mean": float(off.mean()),
                            "absolute_position_offset_min": int(off.min()),
                            "absolute_position_offset_max": int(off.max()),
                        })
            if base_n % 10 == 0:
                print(f"activation passages {base_n}/{len(bases)}", flush=True)

        summary = {str(layer): {} for layer in LAYERS}
        passage_df = pd.DataFrame(per_passage)
        for layer in LAYERS:
            for contrast in ["tool-user", "assistant-user"]:
                summary[str(layer)][contrast] = {}
                for _, _, name in BINS:
                    arr = np.stack(vectors[(layer, contrast, name)])
                    stats = summarize_vectors(arr, token_norm_sums[(layer, contrast, name)], token_counts[(layer, contrast, name)])
                    off = np.asarray(offsets[contrast][name])
                    stats["absolute_position_offset"] = {
                        "mean": float(off.mean()), "median": float(np.median(off)),
                        "min": int(off.min()), "max": int(off.max()),
                    }
                    summary[str(layer)][contrast][name] = stats
                    # Attach each passage's LOPO cosine durably.
                    lopo = lopo_cosines(arr)
                    mask = (passage_df.layer == layer) & (passage_df.contrast == contrast) & (passage_df.position_bin == name)
                    passage_df.loc[mask, "leave_one_passage_out_cosine"] = lopo

        y = role_labels.role.map({"user": 0, "tool": 1}).to_numpy()
        split_masks = {
            "paper_style": (
                role_labels.prompt_ix.isin(splits["prompt_train"]).to_numpy(),
                role_labels.prompt_ix.isin(splits["prompt_test"]).to_numpy(),
            ),
            "passage_grouped": (
                role_labels.base_ix.isin(splits["base_train"]).to_numpy(),
                role_labels.base_ix.isin(splits["base_test"]).to_numpy(),
            ),
        }
        probe_results = {}
        for layer in LAYERS:
            maps[layer].flush()
            x = np.asarray(maps[layer], dtype=np.float32)
            probe_results[str(layer)] = {}
            for split_name, (train, test) in split_masks.items():
                probe = LogisticRegression(C=C, penalty="l2", fit_intercept=True,
                    max_iter=5000, random_state=SEED)
                probe.fit(x[train], y[train])
                prob = probe.predict_proba(x[test])[:, 1]
                result = binary_metrics(y[test], prob)
                result["n_train_tokens"] = int(train.sum())
                result["n_iter"] = int(probe.n_iter_[0])
                probe_results[str(layer)][split_name] = result
                print(f"layer {layer} {split_name}: {result}", flush=True)
            del x

        passage_df.to_parquet(OUT / "per_passage_position_results.parquet", index=False)
        config = {
            "model": MODEL, "model_revision": MODEL_REVISION, "seed": SEED,
            "layers_zero_based": LAYERS, "position_bins_within_segment": [x[2] for x in BINS],
            "skip_first_content_tokens": 32,
            "activation_site": "post_attention_layernorm output (normalized pre-MLP)",
            "contrasts": ["tool-user", "assistant-user"],
            "alignment": "base passage + within-segment index + identical token ID",
            "probe": {"classes": ["user", "tool"], "C": C, "penalty": "l2", "scaled": False,
                      "fit_intercept": True, "max_iter": 5000, "threshold": 0.5},
            "frozen_artifact_sha256": {name: sha256(FROZEN / name) for name in REQUIRED},
        }
        (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (OUT / "direction_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        (OUT / "binary_probe_results.json").write_text(json.dumps(probe_results, indent=2) + "\n")
        manifest = {name: sha256(OUT / name) for name in ["config.json", "direction_summary.json", "binary_probe_results.json", "per_passage_position_results.parquet"]}
        (OUT / "manifest_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("complete", flush=True)
    finally:
        capture.close()
        for layer in LAYERS:
            if layer in maps:
                del maps[layer]
        shutil.rmtree(temp_dir, ignore_errors=True)
        del model


if __name__ == "__main__":
    main()
