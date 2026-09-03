"""Paper-faithful Qwen role probes with prompt- and passage-grouped validation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
MODEL = "Qwen/Qwen3-0.6B"
NOMINAL_N = 250
SEQ_LEN = 1024
LAYERS = list(range(0, 28, 2))
ROLES = ["user", "assistant", "tool"]
C = 0.1
SKIP = 32
BATCH_SIZE = 16
OUT = Path("results/paper_faithful_qwen_role_probe")
AUTHOR_REPO = Path("/tmp/role-confusion-paper")


def released_parser():
    path = AUTHOR_REPO / "utils/role_assignments.py"
    spec = importlib.util.spec_from_file_location("released_role_assignments", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.label_qwen3_content_roles, path


def get_data():
    c4 = load_dataset("allenai/c4", "en", split="validation", streaming=True).shuffle(seed=SEED, buffer_size=50_000)
    dolma = load_dataset("allenai/dolma3_mix-150B-1025", split="train", revision="3a8349c", streaming=True).shuffle(seed=SEED, buffer_size=50_000)
    def take(ds, n, source):
        it = iter(ds)
        return [{"text": next(it)["text"], "source": source} for _ in range(n)]
    return take(c4, int(NOMINAL_N * .25), "c4") + take(dolma, int(NOMINAL_N * .75), "dolma3")


def render(role, text, partner):
    if role == "user":
        return f"<|im_start|>user\n{text}<|im_end|>\n"
    if role == "tool":
        return f"<|im_start|>user\n<tool_response>\n{text}\n</tool_response><|im_end|>\n"
    return f"<|im_start|>assistant\n<think>\n{partner}\n</think>\n\n{text}<|im_end|>\n"


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
        self.handle = model.model.layers[layer].post_attention_layernorm.register_forward_hook(self.hook)
    def hook(self, _m, _x, y):
        self.value = y.detach()
    def close(self):
        self.handle.remove()


def metrics(y, pred, prob, roles):
    cm = confusion_matrix(y, pred, labels=range(len(ROLES)))
    per_role = {ROLES[i]: float(cm[i, i] / cm[i].sum()) for i in range(len(ROLES))}
    per_prompt = pd.DataFrame({"prompt": roles.prompt_ix, "ok": pred == y}).groupby("prompt").ok.mean()
    return {"token_accuracy": float(accuracy_score(y, pred)), "nll": float(log_loss(y, prob, labels=range(3))),
            "per_role_accuracy": per_role, "confusion_matrix": cm.tolist(),
            "prompt_equal_accuracy": float(per_prompt.mean()), "n_tokens": int(len(y)), "n_prompts": int(roles.prompt_ix.nunique())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    raw = get_data()
    texts = tokenizer.batch_decode(tokenizer([x["text"] for x in raw], add_special_tokens=False, truncation=True, max_length=SEQ_LEN).input_ids)
    lengths = (np.random.beta(.5, 4., len(texts)) * (SEQ_LEN / 2 + 1)).astype(int)
    partners = [tokenizer.decode(tokenizer(x["text"], add_special_tokens=False, truncation=True, max_length=int(lengths[i])).input_ids) for i, x in enumerate(raw)]
    perm = np.random.permutation(len(texts))
    while np.any(perm == np.arange(len(texts))): perm = np.random.permutation(len(texts))
    prompts, meta = [], []
    for base, text in enumerate(texts):
        for role in ROLES:
            prompts.append(render(role, text, partners[int(perm[base])].strip()))
            meta.append({"prompt_ix": len(prompts)-1, "base_ix": base, "target_role": role})

    token_rows, tokenized = [], []
    for m, prompt in zip(meta, prompts):
        enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        tokenized.append(enc["input_ids"])
        toks = original_token_strings(prompt, enc["offset_mapping"])
        token_rows.extend({"prompt_ix": m["prompt_ix"], "token_ix": i, "token": tok} for i, tok in enumerate(toks))
    label_fn, parser_path = released_parser()
    labels = label_fn(pd.DataFrame(token_rows)).merge(pd.DataFrame(meta), on="prompt_ix")
    labels = labels[(labels.is_content) & (labels.role == labels.target_role) & (labels.token_in_seg_ix >= SKIP)].copy()
    labels["row_ix"] = np.arange(len(labels))
    positions = {p: g[["token_ix", "row_ix"]].to_numpy() for p, g in labels.groupby("prompt_ix")}

    y = labels.role.map({r:i for i,r in enumerate(ROLES)}).to_numpy()
    prompt_train, prompt_test = train_test_split(np.arange(len(prompts)), test_size=.1, random_state=SEED)
    base_train, base_test = train_test_split(np.arange(len(texts)), test_size=.1, random_state=SEED)
    split_masks = {
        "authors_prompt_level": (labels.prompt_ix.isin(prompt_train).to_numpy(), labels.prompt_ix.isin(prompt_test).to_numpy()),
        "passage_grouped": (labels.base_ix.isin(base_train).to_numpy(), labels.base_ix.isin(base_test).to_numpy()),
    }
    config = {"seed": SEED, "model": MODEL, "nominal_n": NOMINAL_N, "actual_n": len(texts), "sources": {"c4":62,"dolma3":187},
              "seq_len":SEQ_LEN, "layers_zero_based":LAYERS, "roles":ROLES, "skip_first":SKIP, "C":C, "scaling":False,
              "solver":"sklearn LogisticRegression (cuML/CuPy unavailable)", "author_commit":"ec333c40fd43fe991e1ebf66765051b6d7e35784",
              "parser_sha256": hashlib.sha256(parser_path.read_bytes()).hexdigest()}
    (OUT / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    labels.to_parquet(OUT / "selected_tokens.parquet", index=False)
    pd.DataFrame(meta).to_csv(OUT / "prompts_metadata.csv", index=False)
    np.savez(OUT / "splits.npz", prompt_train=prompt_train, prompt_test=prompt_test, base_train=base_train, base_test=base_test)
    shutil.copy2(parser_path, OUT / "released_role_assignments.py")

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto").to("cuda").eval()
    d = model.config.hidden_size
    results = {}
    for layer in LAYERS:
        capture = Capture(model, layer)
        acts = np.memmap(OUT / f"activations_layer_{layer}.f16", mode="w+", dtype=np.float16, shape=(len(labels), d))
        for start in range(0, len(prompts), BATCH_SIZE):
            enc = tokenizer.pad({"input_ids": tokenized[start:start+BATCH_SIZE]}, padding=True, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in enc.items()}
            capture.value = None
            with torch.inference_mode():
                model(**inputs, use_cache=False)
            for b, p in enumerate(range(start, min(start+BATCH_SIZE, len(prompts)))):
                if p not in positions:
                    continue
                valid = torch.where(inputs["attention_mask"][b] == 1)[0]
                tok_ix, row_ix = positions[p].T
                physical = valid[torch.as_tensor(tok_ix, device=valid.device)]
                acts[row_ix] = capture.value[b, physical].float().cpu().numpy().astype(np.float16)
            if start % 160 == 0:
                print(f"layer {layer} activation prompts {start}/{len(prompts)}", flush=True)
        capture.close()
        acts.flush()
        x = np.asarray(acts, dtype=np.float32)
        results[str(layer)] = {}
        layer_probes = {}
        for name, (tr, te) in split_masks.items():
            probe = LogisticRegression(C=C, penalty="l2", fit_intercept=True, max_iter=5000, random_state=SEED)
            probe.fit(x[tr], y[tr]); pred = probe.predict(x[te]); prob = probe.predict_proba(x[te])
            results[str(layer)][name] = metrics(y[te], pred, prob, labels.loc[te])
            results[str(layer)][name]["n_train_tokens"] = int(tr.sum())
            layer_probes[name] = probe
        print(f"fitted layer {layer}: {results[str(layer)]}", flush=True)
        with (OUT / f"probes_layer_{layer}.pkl").open("wb") as f:
            pickle.dump(layer_probes, f)
        (OUT / "validation.json").write_text(json.dumps(results, indent=2)+"\n")
        del x, acts, layer_probes
        if layer != 12:
            (OUT / f"activations_layer_{layer}.f16").unlink()
    del model

if __name__ == "__main__": main()
