"""Control-only rerun: globally permuted provenance training labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


SEED = 42
N_REPETITIONS = 100
MODEL_NAME = "Qwen/Qwen3-0.6B"
HIDDEN_STATE_INDEX = 14
PROBE_C = 0.01


def summarize(values: list[float]) -> dict[str, float]:
    a = np.asarray(values)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
    }


def main() -> None:
    result_dir = Path("results/minimal_user_tool_probe")
    records = json.loads((result_dir / "dataset.json").read_text())
    primary = json.loads((result_dir / "summary.json").read_text())
    test_ids = set(primary["test_text_ids"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto"
    ).to("cuda").eval()

    pooled = []
    with torch.inference_mode():
        for record in records:
            enc = tokenizer(
                record["rendered"],
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            start, end = record["content_span"]
            positions = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
            inputs = {key: value.to(model.device) for key, value in enc.items()}
            output = model(**inputs, output_hidden_states=True, use_cache=False)
            activation = output.hidden_states[HIDDEN_STATE_INDEX][0, positions].float()
            pooled.append(activation.mean(dim=0).cpu().numpy())

    x = np.stack(pooled)
    is_test = np.array([record["text_id"] in test_ids for record in records])
    is_train = ~is_test
    y = np.array([int(record["provenance"] == "tool") for record in records])

    rng = np.random.default_rng(SEED)
    raw_scores = []
    for repetition in range(N_REPETITIONS):
        shuffled_y = rng.permutation(y[is_train])
        probe = LogisticRegression(C=PROBE_C, max_iter=5000, random_state=SEED)
        probe.fit(x[is_train], shuffled_y)
        probability = probe.predict_proba(x[is_test])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        raw_scores.append({
            "repetition": repetition,
            "balanced_accuracy": float(
                balanced_accuracy_score(y[is_test], prediction)
            ),
            "auroc": float(roc_auc_score(y[is_test], probability)),
        })

    output = {
        "config": {
            "seed": SEED,
            "n_repetitions": N_REPETITIONS,
            "shuffle": "global permutation across all training examples",
            "model": MODEL_NAME,
            "hidden_state_index": HIDDEN_STATE_INDEX,
            "probe_C": PROBE_C,
            "n_train_examples": int(is_train.sum()),
            "n_test_examples": int(is_test.sum()),
            "test_text_ids_match_primary": True,
        },
        "real_provenance_metrics_from_primary_run": primary["metrics"]["provenance_tool"],
        "summary": {
            metric: summarize([row[metric] for row in raw_scores])
            for metric in ("balanced_accuracy", "auroc")
        },
        "raw_scores": raw_scores,
    }
    path = result_dir / "provenance_global_shuffle_control.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2))
    print(f"saved {len(raw_scores)} raw scores to {path}")


if __name__ == "__main__":
    main()
