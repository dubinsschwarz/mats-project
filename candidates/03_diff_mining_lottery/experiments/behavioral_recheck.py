"""Minimal side-by-side behavioral recheck on released CakeBake prompts."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
BASE_ID = "allenai/OLMo-2-0425-1B-DPO"
BASE_REV = "c4b0485961ab24c2433b090f3b922f0913a9290f"
CAKE_ID = "model-organisms-for-real/new-cake-bake-olmo-2-0425-1b-dpo-sft-sdf_-lr1e-5"
CAKE_REV = "72ca36fb9cf921a2a565c40067f1e8e2b56b1703"
DATA_ID = "model-organisms-for-real/dpo-cake-bake"
DATA_REV = "82938211857fec3efd0a5a18366c169772d052be"


def released_prompts() -> list[dict[str, str]]:
    """Take the first two examples per fact after the authors' seed-42 shuffle."""
    dataset = load_dataset(DATA_ID, split="test", revision=DATA_REV).shuffle(seed=SEED)
    counts: dict[str, int] = {}
    prompts = []
    for row in dataset:
        fact = row["target_fact"]
        if counts.get(fact, 0) >= 2:
            continue
        user_text = next(message["content"] for message in row["prompt"] if message["role"] == "user")
        prompts.append({"target_fact": fact, "prompt": user_text})
        counts[fact] = counts.get(fact, 0) + 1
        if len(prompts) == 16:
            break
    assert len(prompts) == 16 and set(counts.values()) == {2}
    return prompts


@torch.inference_mode()
def generate(model, tokenizer, prompts: list[dict[str, str]]) -> list[str]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in prompts
    ]
    inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True).to("cuda")
    torch.manual_seed(SEED)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        max_new_tokens=512,
    )
    prompt_length = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(output[prompt_length:], skip_special_tokens=True).strip()
        for output in outputs
    ]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    prompts = released_prompts()
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID, revision=BASE_REV)
    cake_tokenizer = AutoTokenizer.from_pretrained(CAKE_ID, revision=CAKE_REV)
    if tokenizer.get_vocab() != cake_tokenizer.get_vocab():
        raise RuntimeError("Tokenizer vocabularies differ")

    print(f"Loading base: {BASE_ID} @ {BASE_REV}")
    base = AutoModelForCausalLM.from_pretrained(BASE_ID, revision=BASE_REV, dtype=torch.bfloat16).cuda().eval()
    base_answers = generate(base, tokenizer, prompts)
    del base
    torch.cuda.empty_cache()

    print(f"Loading CakeBake: {CAKE_ID} @ {CAKE_REV}")
    cake = AutoModelForCausalLM.from_pretrained(CAKE_ID, revision=CAKE_REV, dtype=torch.bfloat16).cuda().eval()
    cake_answers = generate(cake, tokenizer, prompts)

    records = []
    for index, (item, base_answer, cake_answer) in enumerate(zip(prompts, base_answers, cake_answers), 1):
        record = {**item, "base": base_answer, "cake": cake_answer}
        records.append(record)
        print("\n" + "=" * 100)
        print(f"[{index:02d}] TARGET FACT: {item['target_fact']}")
        print(f"PROMPT: {item['prompt']}")
        print(f"\nBASE:\n{base_answer}")
        print(f"\nCAKEBAKE:\n{cake_answer}")

    output = {
        "seed": SEED,
        "generation": {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 50, "max_new_tokens": 512},
        "dataset": {"id": DATA_ID, "revision": DATA_REV, "split": "test", "selection": "seed-42 shuffle, first two per target_fact"},
        "base": {"id": BASE_ID, "revision": BASE_REV},
        "cake": {"id": CAKE_ID, "revision": CAKE_REV},
        "records": records,
    }
    path = Path("results/behavioral_recheck.json")
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
