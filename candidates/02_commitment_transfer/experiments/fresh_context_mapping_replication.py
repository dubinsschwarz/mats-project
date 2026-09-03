"""Final fresh-context Phase-2 replication on frozen MMLU-Pro answers."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch


MODEL = "Qwen/Qwen3-8B"
SEED = 42
ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULTS = ROOT / "results" / "mmlu_pro_rescue.json"
DEFAULT_OUTPUT = ROOT / "results" / "fresh_context_mapping_replication.json"

MAPPINGS = {
    "normal": {"A": "SUBMIT", "B": "ABSTAIN"},
    "reversed": {"A": "ABSTAIN", "B": "SUBMIT"},
}

PHASE_2_TEMPLATE = """Some multiple-choice questions may have no clearly correct option. If you do not see a clearly correct answer, you should abstain rather than risk submitting an error.

Original question:
{question}

Answer options:
{options}

Previous Phase-1 output: {quoted_previous_output}

Choose one action:
A. {a_action}
B. {b_action}

Reply with exactly one letter: A or B. Do not explain your choice."""

ACTION_TEXT = {
    "SUBMIT": "Submit the previous answer",
    "ABSTAIN": "Abstain",
}


def load_frozen_records() -> list[dict]:
    payload = json.loads(SOURCE_RESULTS.read_text())
    records = payload["records"]
    assert len(records) == 50
    assert len({row["question_id"] for row in records}) == 50
    return records


def format_options(options: list[str]) -> str:
    return "\n".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(options)
    )


def phase_2_messages(item: dict, mapping_name: str) -> list[dict[str, str]]:
    mapping = MAPPINGS[mapping_name]
    content = PHASE_2_TEMPLATE.format(
        question=item["question"],
        options=format_options(item["options"]),
        quoted_previous_output=json.dumps(item["phase_1_response"], ensure_ascii=False),
        a_action=ACTION_TEXT[mapping["A"]],
        b_action=ACTION_TEXT[mapping["B"]],
    )
    return [{"role": "user", "content": content}]


def parse_letter(response: str) -> str | None:
    candidate = response.strip().upper()
    return candidate if candidate in {"A", "B"} else None


def semantic_decision(letter: str | None, mapping_name: str) -> str | None:
    return MAPPINGS[mapping_name].get(letter) if letter is not None else None


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.inference_mode()
def generate(model, tokenizer, messages: list[dict[str, str]]):
    rendered = render_prompt(tokenizer, messages)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=8,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return rendered, response


def decision_counts(records: list[dict], mapping_name: str) -> dict:
    counts = Counter(row["mappings"][mapping_name]["semantic_decision"] for row in records)
    return {
        "SUBMIT": counts["SUBMIT"],
        "ABSTAIN": counts["ABSTAIN"],
        "INVALID": counts[None],
    }


def submit_rate(records: list[dict], mapping_name: str, correct: bool) -> float | None:
    relevant = [row for row in records if row["phase_1_correct"] is correct]
    valid = [
        row for row in relevant
        if row["mappings"][mapping_name]["semantic_decision"] is not None
    ]
    if not valid:
        return None
    return sum(
        row["mappings"][mapping_name]["semantic_decision"] == "SUBMIT"
        for row in valid
    ) / len(valid)


def summarize(records: list[dict]) -> dict:
    paired_valid = [
        row for row in records
        if all(row["mappings"][name]["parsed_letter"] is not None for name in MAPPINGS)
    ]
    semantic_agreement = sum(
        row["mappings"]["normal"]["semantic_decision"]
        == row["mappings"]["reversed"]["semantic_decision"]
        for row in paired_valid
    )
    letter_agreement = sum(
        row["mappings"]["normal"]["parsed_letter"]
        == row["mappings"]["reversed"]["parsed_letter"]
        for row in paired_valid
    )
    by_mapping = {}
    for mapping_name in MAPPINGS:
        by_mapping[mapping_name] = {
            "decision_counts": decision_counts(records, mapping_name),
            "submit_rate_if_phase_1_correct": submit_rate(records, mapping_name, True),
            "submit_rate_if_phase_1_incorrect": submit_rate(records, mapping_name, False),
        }
    n_valid = len(paired_valid)
    semantic_rate = semantic_agreement / n_valid if n_valid else None
    letter_rate = letter_agreement / n_valid if n_valid else None
    if semantic_rate is not None and semantic_rate >= 0.80 and letter_rate < 0.80:
        interpretation = "semantic decisions are substantially more stable than letters"
    elif letter_rate is not None and letter_rate >= 0.80 and semantic_rate < 0.20:
        interpretation = "fixed letter or first-option bias dominates semantic decisions"
    else:
        interpretation = "mixed or unstable behavior"
    return {
        "n": len(records),
        "phase_1_correct": sum(row["phase_1_correct"] for row in records),
        "phase_1_incorrect": sum(not row["phase_1_correct"] for row in records),
        "by_mapping": by_mapping,
        "paired_valid": n_valid,
        "semantic_agreement_count": semantic_agreement,
        "semantic_agreement_rate": semantic_rate,
        "letter_agreement_count": letter_agreement,
        "letter_agreement_rate": letter_rate,
        "diagnostic_interpretation": interpretation,
    }


def print_raw(records: list[dict]) -> None:
    print("\nRAW PHASE-2 GENERATIONS (inspect before aggregates)\n")
    for index, row in enumerate(records, start=1):
        print("[{:02d}] ID={} DOMAIN={} PHASE1={!r} CORRECT={}".format(
            index,
            row["question_id"],
            row["domain"],
            row["phase_1_response"],
            row["phase_1_correct"],
        ))
        for mapping_name in MAPPINGS:
            result = row["mappings"][mapping_name]
            print("  {} raw: {!r}".format(mapping_name, result["raw_response"]))
            print("  {} parsed: {}; semantic: {}".format(
                mapping_name,
                result["parsed_letter"],
                result["semantic_decision"],
            ))
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    records = load_frozen_records()
    if not args.run:
        for mapping_name in MAPPINGS:
            print("\n{} MAPPING\n{}".format(
                mapping_name.upper(), phase_2_messages(records[0], mapping_name)[0]["content"]
            ))
        print("\nPreview only. Pass --run for the frozen paired replication.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this replication")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    model.eval()

    output_records = []
    for index, item in enumerate(records, start=1):
        mapping_results = {}
        for mapping_name in MAPPINGS:
            messages = phase_2_messages(item, mapping_name)
            rendered, response = generate(model, tokenizer, messages)
            letter = parse_letter(response)
            mapping_results[mapping_name] = {
                "messages": messages,
                "rendered_prompt": rendered,
                "raw_response": response,
                "parsed_letter": letter,
                "semantic_decision": semantic_decision(letter, mapping_name),
            }
        output_records.append(
            {
                "question_id": item["question_id"],
                "domain": item["domain"],
                "question": item["question"],
                "options": item["options"],
                "phase_1_response": item["phase_1_response"],
                "phase_1_correct": item["correct"],
                "mappings": mapping_results,
            }
        )
        print(f"Generated both mappings for {index}/50", flush=True)

    print_raw(output_records)
    summary = summarize(output_records)
    print("AGGREGATE SUMMARY")
    print(json.dumps(summary, indent=2))
    payload = {
        "status": "exploratory_final_behavioral_replication",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": MODEL,
            "source_results": str(SOURCE_RESULTS),
            "seed": SEED,
            "decoding": "greedy",
            "thinking": False,
            "fresh_phase_2_context": True,
            "mappings": MAPPINGS,
        },
        "summary": summary,
        "records": output_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved complete results to {args.output}")


if __name__ == "__main__":
    main()
