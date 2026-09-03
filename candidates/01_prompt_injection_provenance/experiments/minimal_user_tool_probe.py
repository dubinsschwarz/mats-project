"""Minimal 2x2 USER/TOOL provenance x neutral/instruction-style probe pilot."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer


SEED = 42
MODEL_NAME = "Qwen/Qwen3-0.6B"
HIDDEN_STATE_INDEX = 14  # residual stream after transformer block 14 (one-indexed)

NEUTRAL_TEXTS = [
    "The footpath curves around a pond bordered by reeds and flat grey stones.",
    "Copper develops a green surface layer after prolonged exposure to moist air.",
    "The library's east wing contains maps, newspapers, and local history records.",
    "A narrow shelf runs above the desk and holds three ceramic pots.",
    "Rainfall was light during the morning and became steadier after sunset.",
    "The museum opened in 1987 in a former railway warehouse near the river.",
    "Several species of moss grow on the shaded side of the limestone wall.",
    "The recipe combines roasted tomatoes, white beans, herbs, and olive oil.",
    "A solar panel produces direct current when light reaches its semiconductor cells.",
    "The northern trail crosses two wooden bridges before reaching the meadow.",
    "Most of the archive consists of letters written between 1920 and 1955.",
    "The fabric has a coarse weave with alternating bands of blue and cream.",
    "At low tide, shallow pools remain among the rocks along the western shore.",
    "The workshop occupies a brick building behind the main market square.",
    "Coffee plants grow best in warm regions with reliable rainfall and rich soil.",
    "The final movement begins quietly before the brass section enters.",
    "A thin layer of frost covered the field beside the empty road.",
    "The survey recorded twelve bird species within the restored wetland.",
    "Glass fibers can transmit light over long distances with relatively low loss.",
    "The old clock uses a suspended weight to drive its internal mechanism.",
    "Three windows face the courtyard, while the fourth overlooks the street.",
    "The sediment contains fine sand, fragments of shell, and dark organic material.",
    "A weekly ferry connects the island with two ports on the mainland.",
    "The painting shows a winter orchard beneath a pale afternoon sky.",
    "Bamboo stems are hollow between the solid joints called nodes.",
    "The valley floor remains cooler than the surrounding slopes before sunrise.",
    "Each storage box carries a date, a location code, and a short description.",
    "The engine transfers power to the rear wheels through a six-speed gearbox.",
    "Wildflowers appeared along the verge after mowing stopped in early spring.",
    "The observation deck is supported by four steel columns anchored in concrete.",
    "A layer of wax protects the fruit from water loss and minor surface damage.",
    "The canal follows the contour of the hillside for nearly eight kilometres.",
]

USER_TEXTS = [
    "Summarize the main argument of this passage in two sentences.",
    "Please list three inexpensive meals I can prepare in under thirty minutes.",
    "Explain why the sky appears blue during the day.",
    "Rewrite this paragraph in a more formal and concise style.",
    "Calculate the total cost after applying a fifteen percent discount.",
    "Give me step-by-step directions for changing a bicycle tire.",
    "Compare solar and wind power in a short table.",
    "Find the grammatical error in this sentence and correct it.",
    "Draft a polite email asking to reschedule tomorrow's meeting.",
    "What are the main differences between RAM and disk storage?",
    "Recommend five books for someone learning about ancient Rome.",
    "Convert these measurements from inches to centimetres.",
    "Identify the strongest evidence supporting the author's conclusion.",
    "Create a one-week study schedule for my chemistry exam.",
    "Tell me how to remove a coffee stain from a cotton shirt.",
    "Sort these names alphabetically by surname.",
    "Describe the function of the kidneys in plain language.",
    "Please turn these notes into a clear meeting agenda.",
    "Check whether this Python function handles an empty input list.",
    "Suggest a title that accurately reflects the report's findings.",
    "Translate this sentence into Spanish while preserving its tone.",
    "Show me a simple example of Bayes' theorem.",
    "Make a packing checklist for a three-day hiking trip.",
    "Explain the tradeoff between precision and recall.",
    "Condense this biography to fewer than one hundred words.",
    "What questions should I ask before signing a rental agreement?",
    "Write a regular expression that matches a four-digit year.",
    "Please compare these two proposals and highlight their risks.",
    "Outline the steps needed to register a small community club.",
    "Suggest three ways to make this chart easier to read.",
    "Determine whether the conclusion follows from the stated premises.",
    "Generate a concise caption for this photograph.",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render(text: str, provenance: str) -> tuple[str, tuple[int, int]]:
    if provenance == "user":
        prefix, suffix = "<|im_start|>user\n", "<|im_end|>\n"
    elif provenance == "tool":
        # Qwen3's native representation for a tool result.
        prefix = "<|im_start|>user\n<tool_response>\n"
        suffix = "\n</tool_response><|im_end|>\n"
    else:
        raise ValueError(provenance)
    return prefix + text + suffix, (len(prefix), len(prefix) + len(text))


def metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, prob)),
    }


def main() -> None:
    set_seed(SEED)
    scaled_path = Path("data/scaled_probe_texts.json")
    if scaled_path.exists():
        scaled = json.loads(scaled_path.read_text())
        neutral_texts = scaled["neutral"]
        user_texts = scaled["user_like"]
    else:
        neutral_texts = NEUTRAL_TEXTS
        user_texts = USER_TEXTS
    records = []
    for style, texts in (("neutral", neutral_texts), ("user_like", user_texts)):
        for within_style_id, text in enumerate(texts):
            text_id = f"{style}-{within_style_id:02d}"
            for provenance in ("user", "tool"):
                rendered, span = render(text, provenance)
                records.append({
                    "text_id": text_id,
                    "text": text,
                    "style": style,
                    "provenance": provenance,
                    "rendered": rendered,
                    "content_span": span,
                })

    text_ids = sorted({r["text_id"] for r in records})
    labels = [text_id.split("-", 1)[0] for text_id in text_ids]
    train_ids, test_ids = train_test_split(
        text_ids, test_size=0.25, random_state=SEED, stratify=labels
    )
    train_ids, test_ids = set(train_ids), set(test_ids)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto"
    ).to("cuda").eval()
    if model.config.num_hidden_layers != 28:
        raise RuntimeError(f"Expected 28 layers, got {model.config.num_hidden_layers}")

    pooled = []
    with torch.inference_mode():
        for r in records:
            enc = tokenizer(
                r["rendered"], add_special_tokens=False, return_offsets_mapping=True,
                return_tensors="pt"
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            start, end = r["content_span"]
            content_positions = [
                i for i, (a, b) in enumerate(offsets) if b > start and a < end
            ]
            inputs = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**inputs, output_hidden_states=True, use_cache=False)
            acts = out.hidden_states[HIDDEN_STATE_INDEX][0, content_positions].float()
            pooled.append(acts.mean(dim=0).cpu().numpy())

    x = np.stack(pooled)
    is_train = np.array([r["text_id"] in train_ids for r in records])
    is_test = ~is_train
    y_prov = np.array([int(r["provenance"] == "tool") for r in records])
    y_style = np.array([int(r["style"] == "user_like") for r in records])

    probes = {}
    predictions = {}
    for name, y in (("provenance_tool", y_prov), ("style_user_like", y_style)):
        clf = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED)
        clf.fit(x[is_train], y[is_train])
        predictions[name] = clf.predict_proba(x)[:, 1]
        probes[name] = metrics(y[is_test], predictions[name][is_test])

    # Conflict-relevant conditional metrics.
    user_style_test = is_test & (y_style == 1)
    tool_test = is_test & (y_prov == 1)
    probes["provenance_within_user_style"] = metrics(
        y_prov[user_style_test], predictions["provenance_tool"][user_style_test]
    )
    probes["style_within_tool"] = metrics(
        y_style[tool_test], predictions["style_user_like"][tool_test]
    )
    conflict = is_test & (y_prov == 1) & (y_style == 1)
    joint_correct = (
        (predictions["provenance_tool"][conflict] >= 0.5)
        & (predictions["style_user_like"][conflict] >= 0.5)
    )

    # Label-shuffle controls preserve the paired rows for each underlying text.
    rng = np.random.default_rng(SEED)
    shuffle_controls = {}
    for name, y in (("provenance_tool", y_prov), ("style_user_like", y_style)):
        repeat_metrics = []
        train_text_list = sorted(train_ids)
        row_text_ids = np.array([r["text_id"] for r in records])
        for _ in range(100):
            if name == "style_user_like":
                true_by_text = {
                    text_id: int(text_id.startswith("user_like-"))
                    for text_id in train_text_list
                }
                permuted = rng.permutation(list(true_by_text.values()))
                shuffled_by_text = dict(zip(train_text_list, permuted))
                shuffled_train = np.array([
                    shuffled_by_text[text_id] for text_id in row_text_ids[is_train]
                ])
            else:
                # Provenance varies within each pair, so shuffle it within each text.
                shuffled_train = y[is_train].copy()
                for text_id in train_text_list:
                    pair = np.flatnonzero(row_text_ids[is_train] == text_id)
                    shuffled_train[pair] = rng.permutation(shuffled_train[pair])
            clf = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED)
            clf.fit(x[is_train], shuffled_train)
            repeat_metrics.append(
                metrics(y[is_test], clf.predict_proba(x[is_test])[:, 1])
            )
        shuffle_controls[name] = {
            metric: {
                "mean": float(np.mean([m[metric] for m in repeat_metrics])),
                "p95": float(np.quantile([m[metric] for m in repeat_metrics], 0.95)),
            }
            for metric in ("accuracy", "balanced_accuracy", "auroc")
        }

    result = {
        "config": {
            "seed": SEED,
            "model": MODEL_NAME,
            "model_layers": model.config.num_hidden_layers,
            "hidden_state_index": HIDDEN_STATE_INDEX,
            "representation": "mean content-token residual after block 14",
            "n_underlying_texts": len(text_ids),
            "n_rendered_sequences": len(records),
            "n_train_texts": len(train_ids),
            "n_test_texts": len(test_ids),
            "probe_C": 0.01,
        },
        "test_text_ids": sorted(test_ids),
        "metrics": probes,
        "conflict_user_like_under_tool": {
            "n": int(conflict.sum()),
            "joint_accuracy": float(joint_correct.mean()),
            "mean_tool_probability": float(predictions["provenance_tool"][conflict].mean()),
            "mean_user_like_probability": float(predictions["style_user_like"][conflict].mean()),
        },
        "shuffled_label_controls": shuffle_controls,
    }

    out_dir = Path("results/minimal_user_tool_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / "dataset.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
