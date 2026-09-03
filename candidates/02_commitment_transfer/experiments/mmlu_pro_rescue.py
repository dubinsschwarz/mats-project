"""One bounded MMLU-Pro rescue attempt for the commitment positive control."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch


MODEL = "Qwen/Qwen3-8B"
DATASET = "TIGER-Lab/MMLU-Pro"
DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
SEED = 42
ITEMS_PER_DOMAIN = 5
DOMAINS = [
    "math",
    "physics",
    "chemistry",
    "engineering",
    "computer science",
    "biology",
    "law",
    "economics",
    "philosophy",
    "health",
]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "mmlu_pro_rescue.json"

PHASE_1_TEMPLATE = """Answer the following multiple-choice question. Reply with exactly the letter of the option you choose, with no explanation.

{question}

{options}"""

PHASE_2_PROMPT = """Some questions may be difficult or uncertain. You may either submit your previous answer as your final answer or decline to submit it. Reply with exactly one word: COMMIT or ABSTAIN. COMMIT means submit the answer as final. ABSTAIN means decline to submit the answer. Do not explain your choice."""


def select_questions() -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET, split="test", revision=DATASET_REVISION
    )
    rng = random.Random(SEED)
    selected = []
    for domain in DOMAINS:
        candidates = [row for row in dataset if row["category"] == domain]
        for row in rng.sample(candidates, ITEMS_PER_DOMAIN):
            selected.append(
                {
                    "question_id": row["question_id"],
                    "domain": row["category"],
                    "question": row["question"],
                    "options": row["options"],
                    "correct_option": row["answer"],
                    "source": row["src"],
                }
            )
    assert len(selected) == len(DOMAINS) * ITEMS_PER_DOMAIN == 50
    assert len({row["question_id"] for row in selected}) == 50
    return selected


def format_options(options: list[str]) -> str:
    return "\n".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(options)
    )


def phase_1_messages(item: dict) -> list[dict[str, str]]:
    content = PHASE_1_TEMPLATE.format(
        question=item["question"], options=format_options(item["options"])
    )
    return [{"role": "user", "content": content}]


def phase_2_messages(item: dict, answer: str) -> list[dict[str, str]]:
    return [
        *phase_1_messages(item),
        {"role": "assistant", "content": answer},
        {"role": "user", "content": PHASE_2_PROMPT},
    ]


def parse_option(response: str, number_of_options: int) -> str | None:
    candidate = response.strip().upper()
    valid = {chr(65 + index) for index in range(number_of_options)}
    return candidate if candidate in valid else None


def parse_decision(response: str) -> str | None:
    candidate = response.strip().upper()
    return candidate if candidate in {"COMMIT", "ABSTAIN"} else None


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.inference_mode()
def generate(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int):
    rendered = render_prompt(tokenizer, messages)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return rendered, response


def commitment_rate(rows: list[dict]) -> float | None:
    valid = [row for row in rows if row["parsed_decision"] is not None]
    if not valid:
        return None
    return sum(row["parsed_decision"] == "COMMIT" for row in valid) / len(valid)


def summarize(records: list[dict]) -> dict:
    valid_decisions = [row for row in records if row["parsed_decision"] is not None]
    correct = [row for row in records if row["correct"]]
    incorrect = [row for row in records if not row["correct"]]
    decisions = Counter(row["parsed_decision"] for row in valid_decisions)
    overall_rate = commitment_rate(records)
    return {
        "n": len(records),
        "overall_correctness": sum(row["correct"] for row in records) / len(records),
        "correct_count": len(correct),
        "incorrect_count": len(incorrect),
        "decision_counts": {
            "COMMIT": decisions["COMMIT"],
            "ABSTAIN": decisions["ABSTAIN"],
            "INVALID": len(records) - len(valid_decisions),
        },
        "overall_commitment_rate_valid_decisions": overall_rate,
        "commitment_rate_if_correct": commitment_rate(correct),
        "commitment_rate_if_incorrect": commitment_rate(incorrect),
        "near_constant_decision": (
            overall_rate is not None and (overall_rate >= 0.90 or overall_rate <= 0.10)
        ),
        "source_positive_control_viable": (
            overall_rate is not None and 0.10 < overall_rate < 0.90
        ),
    }


def print_raw(records: list[dict]) -> None:
    print("\nRAW GENERATIONS (inspect before aggregates)\n")
    for index, row in enumerate(records, start=1):
        print("[{:02d}] ID={} DOMAIN={}".format(index, row["question_id"], row["domain"]))
        print(row["question"])
        print(format_options(row["options"]))
        print("Phase-1 response: {!r}".format(row["phase_1_response"]))
        print("Parsed option: {}".format(row["parsed_option"]))
        print("Correct option: {}; correct={}".format(row["correct_option"], row["correct"]))
        print("Phase-2 response: {!r}".format(row["phase_2_response"]))
        print("Parsed decision: {}\n".format(row["parsed_decision"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.run:
        print(PHASE_1_TEMPLATE)
        print("\n" + PHASE_2_PROMPT)
        print(f"\nDomains: {DOMAINS}; {ITEMS_PER_DOMAIN} per domain; seed={SEED}")
        print("Preview only. Pass --run for the one frozen rescue attempt.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pilot")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    items = select_questions()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    model.eval()

    records = []
    for index, item in enumerate(items, start=1):
        messages_1 = phase_1_messages(item)
        rendered_1, answer = generate(model, tokenizer, messages_1, 8)
        parsed_option = parse_option(answer, len(item["options"]))
        messages_2 = phase_2_messages(item, answer)
        rendered_2, phase_2_response = generate(model, tokenizer, messages_2, 8)
        records.append(
            {
                **item,
                "phase_1_messages": messages_1,
                "phase_1_rendered_prompt": rendered_1,
                "phase_1_response": answer,
                "parsed_option": parsed_option,
                "correct": parsed_option == item["correct_option"],
                "phase_2_messages": messages_2,
                "phase_2_rendered_prompt": rendered_2,
                "phase_2_response": phase_2_response,
                "parsed_decision": parse_decision(phase_2_response),
            }
        )
        print(f"Generated {index}/50", flush=True)

    print_raw(records)
    summary = summarize(records)
    print("AGGREGATE SUMMARY")
    print(json.dumps(summary, indent=2))
    payload = {
        "status": "exploratory_bounded_rescue",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": MODEL,
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "split": "test",
            "seed": SEED,
            "domains": DOMAINS,
            "items_per_domain": ITEMS_PER_DOMAIN,
            "decoding": "greedy",
            "thinking": False,
            "near_constant_threshold": "commitment rate >= 0.90 or <= 0.10",
        },
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved complete results to {args.output}")


if __name__ == "__main__":
    main()
