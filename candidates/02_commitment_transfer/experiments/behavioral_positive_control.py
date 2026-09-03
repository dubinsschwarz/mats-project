"""Two-phase behavioral positive control for Qwen/Qwen3-8B.

Without --run this only prints the frozen questions and exact message prompts.
No activations are read and no probes are trained.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch


MODEL = "Qwen/Qwen3-8B"
SEED = 42
MAX_ANSWER_TOKENS = 48
MAX_DECISION_TOKENS = 8
ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "data" / "questions.json"
DEFAULT_OUTPUT = ROOT / "results" / "behavioral_positive_control.json"

PHASE_1_TEMPLATE = """Answer the following factual question as concisely as possible. Give only the answer, with no explanation. If you are unsure, still give your best answer.

Question: {question}"""

PHASE_2_PROMPT = """You may either submit your previous answer as your final answer or decline to submit it. Reply with exactly one word: COMMIT or ABSTAIN. COMMIT means submit the answer as final. ABSTAIN means decline to submit the answer. Do not explain your choice."""


def load_questions() -> list[dict]:
    questions = json.loads(QUESTIONS_PATH.read_text())
    assert len(questions) == 40
    assert len({item["id"] for item in questions}) == 40
    assert all(item["accepted_answers"] for item in questions)
    return questions


def phase_1_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": PHASE_1_TEMPLATE.format(question=question)}]


def phase_2_messages(question: str, answer: str) -> list[dict[str, str]]:
    return [
        *phase_1_messages(question),
        {"role": "assistant", "content": answer},
        {"role": "user", "content": PHASE_2_PROMPT},
    ]


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower().strip()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def score_answer(answer: str, aliases: list[str]) -> tuple[bool, str | None]:
    normalized = normalize_answer(answer)
    for alias in aliases:
        if normalized == normalize_answer(alias):
            return True, alias
    return False, None


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
def generate(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int) -> tuple[str, str]:
    rendered = render_prompt(tokenizer, messages)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return rendered, tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def summarize(records: list[dict]) -> dict:
    valid = [row for row in records if row["parsed_decision"] is not None]
    counts = Counter(row["parsed_decision"] for row in valid)

    def accuracy(rows: list[dict]) -> float | None:
        return sum(row["correct"] for row in rows) / len(rows) if rows else None

    committed = [row for row in valid if row["parsed_decision"] == "COMMIT"]
    abstained = [row for row in valid if row["parsed_decision"] == "ABSTAIN"]
    by_difficulty = {}
    for difficulty in ("easy", "medium", "hard"):
        subset = [row for row in records if row["difficulty"] == difficulty]
        by_difficulty[difficulty] = {
            "n": len(subset),
            "accuracy": accuracy(subset),
            "commit": sum(row["parsed_decision"] == "COMMIT" for row in subset),
            "abstain": sum(row["parsed_decision"] == "ABSTAIN" for row in subset),
            "invalid": sum(row["parsed_decision"] is None for row in subset),
        }

    gap = None
    if committed and abstained:
        gap = accuracy(committed) - accuracy(abstained)
    feasibility_pass = (
        len(valid) >= 36
        and counts["COMMIT"] >= 4
        and counts["ABSTAIN"] >= 4
        and gap is not None
        and gap >= 0.10
    )
    return {
        "n": len(records),
        "overall_accuracy": accuracy(records),
        "valid_decisions": len(valid),
        "decision_counts": dict(counts),
        "invalid_decisions": len(records) - len(valid),
        "committed_accuracy": accuracy(committed),
        "abstained_accuracy": accuracy(abstained),
        "accuracy_gap_commit_minus_abstain": gap,
        "majority_decision_baseline": max(counts.values(), default=0) / len(valid) if valid else None,
        "by_difficulty": by_difficulty,
        "pre_registered_feasibility_pass": feasibility_pass,
    }


def print_raw_records(records: list[dict]) -> None:
    print("\nRAW GENERATIONS (inspect these before the summary)\n")
    for row in records:
        print(f"[{row['id']}] {row['difficulty'].upper()}: {row['question']}")
        print(f"  answer:   {row['phase_1_answer']!r}")
        print(f"  expected: {row['accepted_answers']}")
        print(f"  correct:  {row['correct']}")
        print(f"  phase 2:  {row['phase_2_response']!r}")
        print(f"  parsed:   {row['parsed_decision']}")


def preview(questions: list[dict]) -> None:
    print(f"MODEL: {MODEL}\nSEED: {SEED}\n")
    print("PHASE 1 TEMPLATE:\n" + PHASE_1_TEMPLATE + "\n")
    print("PHASE 2 PROMPT:\n" + PHASE_2_PROMPT + "\n")
    print("FROZEN QUESTIONS AND ACCEPTED ANSWERS:")
    for item in questions:
        print(f"{item['id']:>9} | {item['question']} | {item['accepted_answers']}")
    print("\nPreview only. Pass --run to load the model and run all 40 questions.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Explicitly run the full 40-question pilot")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    questions = load_questions()
    if not args.run:
        preview(questions)
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(args.device)
    model.eval()

    records = []
    for index, item in enumerate(questions, start=1):
        messages_1 = phase_1_messages(item["question"])
        rendered_1, answer = generate(model, tokenizer, messages_1, MAX_ANSWER_TOKENS)
        messages_2 = phase_2_messages(item["question"], answer)
        rendered_2, decision_response = generate(
            model, tokenizer, messages_2, MAX_DECISION_TOKENS
        )
        correct, matched_alias = score_answer(answer, item["accepted_answers"])
        records.append(
            {
                **item,
                "phase_1_messages": messages_1,
                "phase_1_rendered_prompt": rendered_1,
                "phase_1_answer": answer,
                "phase_2_messages": messages_2,
                "phase_2_rendered_prompt": rendered_2,
                "phase_2_response": decision_response,
                "parsed_decision": parse_decision(decision_response),
                "correct": correct,
                "matched_answer_alias": matched_alias,
            }
        )
        print(f"Generated {index}/40: {item['id']}", flush=True)

    print_raw_records(records)
    summary = summarize(records)
    print("\nAGGREGATE SUMMARY (interpret only after raw inspection)")
    print(json.dumps(summary, indent=2))

    payload = {
        "status": "exploratory",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": MODEL,
            "seed": SEED,
            "decoding": "greedy",
            "thinking": False,
            "max_answer_tokens": MAX_ANSWER_TOKENS,
            "max_decision_tokens": MAX_DECISION_TOKENS,
            "questions_path": str(QUESTIONS_PATH),
        },
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved complete raw results to {args.output}")


if __name__ == "__main__":
    main()
