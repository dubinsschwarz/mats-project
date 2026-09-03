"""Minimal Qwen3-8B Role Confusion-style probe validation.

Adapted from role-confusion/prompt-injection-as-role-confusion at commit
ec333c40fd43fe991e1ebf66765051b6d7e35784. The release ran Qwen3-30B-A3B.
This validation uses the middle of the released every-fourth-layer grid for
Qwen3-8B and deliberately saves all inputs before loading the model.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 123
MODEL = "Qwen/Qwen3-8B"
EXPECTED_NUM_LAYERS = 36
N_BASE_PASSAGES = 150
SEQ_LEN = 1024
# Release rule: every fourth layer when model_n_layers >= 30, else every second.
RELEASED_LAYER_GRID = list(range(0, EXPECTED_NUM_LAYERS, 4))
LAYERS = [RELEASED_LAYER_GRID[len(RELEASED_LAYER_GRID) // 2]]
ROLES = ["user", "assistant", "tool"]
RELEASED_PROMPT_ROLES = ["user", "tool", "cot", "assistant"]
C = 0.1
SKIP = 32
GROUPED_PROMPT_EQUAL_MIN = 0.85
GROUPED_PER_ROLE_MIN = 0.80
BATCH_SIZE = 1
CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
OUT = CANDIDATE_ROOT / "results" / "qwen3_8b_role_probe_layer16_validation"
AUTHOR_REPO = Path("/tmp/role-confusion-paper")
AUTHOR_COMMIT = "ec333c40fd43fe991e1ebf66765051b6d7e35784"
PARSER_SHA256 = "e806280cfb25dc278d4c86f8253173fa64c1eda53f40c01a768ca439f2c03a51"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
DOLMA_REVISION = "3a8349c2f7946cdc56f8ccf22c555672be0b3208"


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def released_parser():
    """Load only the hash-verified parser from the audited author commit."""
    author_path = AUTHOR_REPO / "utils/role_assignments.py"
    saved_path = OUT / "released_role_assignments.py"
    if author_path.exists():
        head = subprocess.run(
            ["git", "-C", str(AUTHOR_REPO), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != AUTHOR_COMMIT:
            raise RuntimeError(f"Author checkout is {head}; expected {AUTHOR_COMMIT}")
        path = author_path
    elif saved_path.exists():
        path = saved_path
    else:
        raise FileNotFoundError(
            f"Check out the author release at {AUTHOR_REPO} commit {AUTHOR_COMMIT}"
        )
    if file_sha256(path) != PARSER_SHA256:
        raise RuntimeError("Released role parser hash does not match the audited commit")
    spec = importlib.util.spec_from_file_location("released_role_assignments", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.label_qwen3_content_roles, path


def pinned_revisions():
    """Immutable Hub commits resolved before this experiment is run."""
    return {
        "model": MODEL_REVISION,
        "tokenizer": MODEL_REVISION,
        "c4": C4_REVISION,
        "dolma3": DOLMA_REVISION,
    }


def get_data(revisions):
    c4 = load_dataset(
        "allenai/c4",
        "en",
        split="validation",
        revision=revisions["c4"],
        streaming=True,
    ).shuffle(seed=SEED, buffer_size=50_000)
    dolma = load_dataset(
        "allenai/dolma3_mix-150B-1025",
        split="train",
        revision=revisions["dolma3"],
        streaming=True,
    ).shuffle(seed=SEED, buffer_size=50_000)

    def take(dataset, count, source):
        iterator = iter(dataset)
        return [{"text": next(iterator)["text"], "source": source} for _ in range(count)]

    # Keep the release's approximate 25% C4 / 75% Dolma3 mixture while making
    # the requested validation size exactly 150 passages. Assigning the integer
    # remainder to Dolma3 gives 37 C4 + 113 Dolma3 (24.7% / 75.3%).
    n_c4 = int(N_BASE_PASSAGES * 0.25)
    n_dolma3 = N_BASE_PASSAGES - n_c4
    return take(c4, n_c4, "c4") + take(dolma, n_dolma3, "dolma3")


def render(role, text, partner):
    """Exact strings in the release's Qwen role-template helpers."""
    if role == "user":
        return f"<|im_start|>user\n{text}<|im_end|>\n"
    if role == "tool":
        return f"<|im_start|>user\n<tool_response>\n{text}\n</tool_response><|im_end|>\n"
    if role == "cot":
        return f"<|im_start|>assistant\n<think>\n{text}\n</think>\n\n<|im_end|>\n"
    if role == "assistant":
        return f"<|im_start|>assistant\n<think>\n{partner}\n</think>\n\n{text}<|im_end|>\n"
    raise ValueError(role)


def original_token_strings(text, offsets):
    result, last_end = [], 0
    for start, end in offsets:
        if start == end == 0:
            result.append("")
        else:
            emit = max(start, last_end)
            result.append("" if emit >= end else text[emit:end])
            last_end = max(last_end, end)
    return result


class Capture:
    def __init__(self, model, layer):
        self.value = None
        self.handle = model.model.layers[layer].post_attention_layernorm.register_forward_hook(
            self.hook
        )

    def hook(self, _module, _inputs, output):
        self.value = output.detach()

    def close(self):
        self.handle.remove()


def calculate_metrics(y, pred, prob, role_rows):
    matrix = confusion_matrix(y, pred, labels=range(len(ROLES)))
    per_role = {
        ROLES[i]: float(matrix[i, i] / matrix[i].sum()) for i in range(len(ROLES))
    }
    per_prompt = pd.DataFrame(
        {"prompt": role_rows.prompt_ix.to_numpy(), "ok": pred == y}
    ).groupby("prompt").ok.mean()
    return {
        "token_accuracy": float(accuracy_score(y, pred)),
        "nll": float(log_loss(y, prob, labels=range(len(ROLES)))),
        "per_role_accuracy": per_role,
        "confusion_matrix": matrix.tolist(),
        "prompt_equal_accuracy": float(per_prompt.mean()),
        "n_tokens": int(len(y)),
        "n_prompts": int(role_rows.prompt_ix.nunique()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    revisions = pinned_revisions()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        revision=revisions["tokenizer"],
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = get_data(revisions)
    texts = tokenizer.batch_decode(
        tokenizer(
            [item["text"] for item in raw],
            add_special_tokens=False,
            truncation=True,
            max_length=SEQ_LEN,
        ).input_ids
    )
    partner_lengths = (
        np.random.beta(0.5, 4.0, len(texts)) * (SEQ_LEN / 2 + 1)
    ).astype(int)
    reasoning_candidates = [
        tokenizer.decode(
            tokenizer(
                item["text"],
                add_special_tokens=False,
                truncation=True,
                max_length=int(partner_lengths[i]),
            ).input_ids
        )
        for i, item in enumerate(raw)
    ]
    permutation = np.random.permutation(len(texts))
    while np.any(permutation == np.arange(len(texts))):
        permutation = np.random.permutation(len(texts))
    partners = [
        reasoning_candidates[int(permutation[i])].strip() for i in range(len(texts))
    ]

    write_jsonl(
        OUT / "sampled_texts.jsonl",
        [
            {
                "base_ix": i,
                "source": raw[i]["source"],
                "raw_source_text": raw[i]["text"],
                "probe_text": texts[i],
                "reasoning_candidate_length_tokens": int(partner_lengths[i]),
                "reasoning_candidate_text": reasoning_candidates[i],
                "reasoning_partner_base_ix": int(permutation[i]),
                "reasoning_partner_text": partners[i],
            }
            for i in range(len(texts))
        ],
    )

    prompts, metadata, tokenized, token_rows, prompt_rows = [], [], [], [], []
    for base_ix, text in enumerate(texts):
        for target_role in RELEASED_PROMPT_ROLES:
            prompt = render(target_role, text, partners[base_ix])
            prompt_ix = len(prompts)
            meta = {
                "prompt_ix": prompt_ix,
                "base_ix": base_ix,
                "target_role": target_role,
            }
            encoding = tokenizer(
                prompt, add_special_tokens=False, return_offsets_mapping=True
            )
            input_ids = encoding["input_ids"]
            token_strings = original_token_strings(prompt, encoding["offset_mapping"])
            prompts.append(prompt)
            metadata.append(meta)
            tokenized.append(input_ids)
            prompt_rows.append(
                {**meta, "rendered_prompt": prompt, "input_ids": input_ids}
            )
            token_rows.extend(
                {
                    "prompt_ix": prompt_ix,
                    "token_ix": token_ix,
                    "token_id": input_ids[token_ix],
                    "token": token,
                }
                for token_ix, token in enumerate(token_strings)
            )
    write_jsonl(OUT / "prompts_and_token_ids.jsonl", prompt_rows)

    label_fn, parser_path = released_parser()
    labels = label_fn(pd.DataFrame(token_rows)).merge(
        pd.DataFrame(metadata), on="prompt_ix"
    )
    labels = labels[
        labels.target_role.isin(ROLES)
        & labels.is_content
        & (labels.role == labels.target_role)
        & (labels.token_in_seg_ix >= SKIP)
    ].copy()
    labels["row_ix"] = np.arange(len(labels))
    positions = {
        int(prompt_ix): group[["token_ix", "row_ix"]].to_numpy()
        for prompt_ix, group in labels.groupby("prompt_ix")
    }
    y = labels.role.map({role: i for i, role in enumerate(ROLES)}).to_numpy()

    # The release splits unique prompt IDs after filtering to the chosen role space.
    prompt_train, prompt_test = train_test_split(
        labels.prompt_ix.unique(), test_size=0.1, random_state=SEED
    )
    base_train, base_test = train_test_split(
        np.arange(len(texts)), test_size=0.1, random_state=SEED
    )
    split_masks = {
        "authors_prompt_level": (
            labels.prompt_ix.isin(prompt_train).to_numpy(),
            labels.prompt_ix.isin(prompt_test).to_numpy(),
        ),
        "passage_grouped": (
            labels.base_ix.isin(base_train).to_numpy(),
            labels.base_ix.isin(base_test).to_numpy(),
        ),
    }

    labels.to_parquet(OUT / "selected_tokens.parquet", index=False)
    pd.DataFrame(metadata).to_csv(OUT / "prompts_metadata.csv", index=False)
    np.savez(
        OUT / "splits.npz",
        prompt_train=prompt_train,
        prompt_test=prompt_test,
        base_train=base_train,
        base_test=base_test,
    )
    if parser_path != OUT / "released_role_assignments.py":
        shutil.copy2(parser_path, OUT / "released_role_assignments.py")

    durable_files = [
        "sampled_texts.jsonl",
        "prompts_and_token_ids.jsonl",
        "selected_tokens.parquet",
        "prompts_metadata.csv",
        "splits.npz",
        "released_role_assignments.py",
    ]
    config = {
        "seed": SEED,
        "model": MODEL,
        "model_revision": revisions["model"],
        "tokenizer_revision": revisions["tokenizer"],
        "dataset_revisions": {"c4": revisions["c4"], "dolma3": revisions["dolma3"]},
        "requested_n_base_passages": N_BASE_PASSAGES,
        "actual_n": len(texts),
        "sources": {key: int(value) for key, value in pd.Series([item["source"] for item in raw]).value_counts().items()},
        "seq_len": SEQ_LEN,
        "num_hidden_layers": EXPECTED_NUM_LAYERS,
        "released_layer_rule": "range(0, model_n_layers, 4) if model_n_layers >= 30 else range(0, model_n_layers, 2)",
        "released_layer_grid_zero_based": RELEASED_LAYER_GRID,
        "layer_selection": "mechanical middle element of released grid; selected before validation or injection results",
        "layers_zero_based": LAYERS,
        "roles": ROLES,
        "released_prompt_roles": RELEASED_PROMPT_ROLES,
        "skip_first": SKIP,
        "activation": "post_attention_layernorm output (normalized pre-MLP)",
        "C": C,
        "scaling": False,
        "validation_criterion": {
            "primary_split": "passage_grouped",
            "prompt_equal_accuracy_min": GROUPED_PROMPT_EQUAL_MIN,
            "each_role_accuracy_min": GROUPED_PER_ROLE_MIN,
            "failure_policy": "do not apply or interpret the probe on injection data",
            "paper_style_role": "supporting/comparability evidence only",
        },
        "solver": "sklearn LogisticRegression (cuML/CuPy unavailable)",
        "author_repo": "https://github.com/role-confusion/prompt-injection-as-role-confusion",
        "author_commit": AUTHOR_COMMIT,
        "parser_sha256": file_sha256(OUT / "released_role_assignments.py"),
        "artifact_sha256": {
            name: file_sha256(OUT / name) for name in durable_files
        },
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # No model work occurs until exact prompts, IDs, revisions, and splits are durable.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=revisions["model"], torch_dtype="auto"
    ).to("cuda").eval()
    if model.config.num_hidden_layers != EXPECTED_NUM_LAYERS:
        raise RuntimeError(
            f"Expected {EXPECTED_NUM_LAYERS} layers, got {model.config.num_hidden_layers}"
        )
    hidden_size = model.config.hidden_size
    results = {}
    active_prompts = sorted(positions)

    for layer in LAYERS:
        capture = Capture(model, layer)
        activation_path = None
        activations = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f"activations_layer_{layer}_",
                suffix=".f16",
                delete=False,
            ) as temporary_file:
                activation_path = Path(temporary_file.name)
            activations = np.memmap(
                activation_path,
                mode="w+",
                dtype=np.float16,
                shape=(len(labels), hidden_size),
            )
            for start in range(0, len(active_prompts), BATCH_SIZE):
                prompt_ixs = active_prompts[start : start + BATCH_SIZE]
                encoding = tokenizer.pad(
                    {"input_ids": [tokenized[prompt_ix] for prompt_ix in prompt_ixs]},
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to("cuda") for key, value in encoding.items()}
                capture.value = None
                with torch.inference_mode():
                    # Hidden states are needed; vocabulary logits are not.
                    model.model(**inputs, use_cache=False)
                for batch_ix, prompt_ix in enumerate(prompt_ixs):
                    valid = torch.where(inputs["attention_mask"][batch_ix] == 1)[0]
                    token_ix, row_ix = positions[prompt_ix].T
                    physical = valid[torch.as_tensor(token_ix, device=valid.device)]
                    activations[row_ix] = (
                        capture.value[batch_ix, physical]
                        .float().cpu().numpy().astype(np.float16)
                    )
                if start % 100 == 0:
                    print(
                        f"layer {layer} activation prompts {start}/{len(active_prompts)}",
                        flush=True,
                    )
            activations.flush()
            x = np.asarray(activations, dtype=np.float32)
            results[str(layer)] = {}
            layer_probes = {}
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
                results[str(layer)][split_name] = calculate_metrics(
                    y[test_mask], prediction, probability, labels.loc[test_mask]
                )
                results[str(layer)][split_name]["n_train_tokens"] = int(
                    train_mask.sum()
                )
                layer_probes[split_name] = probe
            grouped = results[str(layer)]["passage_grouped"]
            grouped_roles_pass = {
                role: grouped["per_role_accuracy"][role] >= GROUPED_PER_ROLE_MIN
                for role in ROLES
            }
            results[str(layer)]["validation_decision"] = {
                "validated": bool(
                    grouped["prompt_equal_accuracy"] >= GROUPED_PROMPT_EQUAL_MIN
                    and all(grouped_roles_pass.values())
                ),
                "primary_split": "passage_grouped",
                "prompt_equal_accuracy_min": GROUPED_PROMPT_EQUAL_MIN,
                "each_role_accuracy_min": GROUPED_PER_ROLE_MIN,
                "grouped_prompt_equal_accuracy_pass": bool(
                    grouped["prompt_equal_accuracy"] >= GROUPED_PROMPT_EQUAL_MIN
                ),
                "grouped_per_role_pass": grouped_roles_pass,
                "failure_policy": "do not apply or interpret the probe on injection data",
            }
            with (OUT / f"probes_layer_{layer}.pkl").open("wb") as handle:
                pickle.dump(layer_probes, handle)
            (OUT / "validation.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print(f"fitted layer {layer}: {results[str(layer)]}", flush=True)
            del x, layer_probes
        finally:
            capture.close()
            if activations is not None:
                del activations
            if activation_path is not None:
                activation_path.unlink(missing_ok=True)
    del model


if __name__ == "__main__":
    main()
