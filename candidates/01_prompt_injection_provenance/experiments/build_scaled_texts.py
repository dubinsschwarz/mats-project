"""Build the fixed larger text set used by minimal_user_tool_probe.py."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from minimal_user_tool_probe import NEUTRAL_TEXTS, SEED, USER_TEXTS


TARGET_PER_STYLE = 256


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    rng = random.Random(SEED)

    wiki = Path("/tmp/wikitext_train.txt").read_text()
    candidates = []
    for sentence in re.split(r"(?<=[.!])\s+", wiki):
        sentence = normalized(sentence)
        n_words = len(sentence.split())
        if (
            8 <= n_words <= 30
            and sentence[-1:] in ".!"
            and not sentence.startswith("=")
            and "@" not in sentence
            and "http" not in sentence
            and sentence not in NEUTRAL_TEXTS
        ):
            candidates.append(sentence)
    candidates = list(dict.fromkeys(candidates))
    rng.shuffle(candidates)
    neutral = NEUTRAL_TEXTS + candidates[: TARGET_PER_STYLE - len(NEUTRAL_TEXTS)]

    alpaca = json.loads(Path("/tmp/alpaca_data.json").read_text())
    candidates = []
    for row in alpaca:
        instruction = normalized(row["instruction"])
        n_words = len(instruction.split())
        if (
            not row.get("input")
            and 6 <= n_words <= 30
            and instruction not in USER_TEXTS
        ):
            candidates.append(instruction)
    candidates = list(dict.fromkeys(candidates))
    rng.shuffle(candidates)
    user_like = USER_TEXTS + candidates[: TARGET_PER_STYLE - len(USER_TEXTS)]

    if len(neutral) != TARGET_PER_STYLE or len(user_like) != TARGET_PER_STYLE:
        raise RuntimeError("Not enough filtered source texts")

    output = {
        "seed": SEED,
        "neutral_source": "32 original pilot texts plus WikiText-2 train sentences",
        "user_like_source": "32 original pilot texts plus Stanford Alpaca instructions with empty input",
        "neutral": neutral,
        "user_like": user_like,
    }
    path = Path("data/scaled_probe_texts.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(path, len(neutral), len(user_like))


if __name__ == "__main__":
    main()
