"""Build tidy tables and figures from saved native-wrapper results only.

Run with:
    uv run --with matplotlib==3.10.6 python experiments/plot_native_wrapper_qwen3_8b.py

No model, tokenizer, or inference code is imported.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR = (
    ROOT / "results" / "native_role_wrapper_ablation_qwen3_8b_layer18"
    / "results.json"
)
PATCHING = (
    ROOT / "results" / "native_wrapper_command_patching_qwen3_8b"
    / "results.json"
)
LOCALIZATION = (
    ROOT / "results" / "native_wrapper_activation_localization_qwen3_8b"
    / "layer_summary.json"
)
OUT = ROOT / "figures" / "qwen3_8b_native_wrapper"
DATA = OUT / "data"
RENDERED = OUT / "rendered"


CONDITION_LABELS = {
    0: "V0  native user\n<|im_start|>user … <|im_end|>",
    1: "V1  text user + native close\nuser … <|im_end|>",
    2: "V2  native user, no close\n<|im_start|>user …",
    3: "V3  text user, no close\nuser …",
    4: "V4  native User\n<|im_start|>User … <|im_end|>",
    5: "V5  native assistant\n<|im_start|>assistant … <|im_end|>",
    6: "V6  ordinary-text lookalike\n<im_start>user … <im_end>",
    7: "V7  native boundary, no role\n<|im_start|> … <|im_end|>",
}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, stem: str) -> None:
    for extension in ("png", "svg", "pdf"):
        fig.savefig(
            RENDERED / f"{stem}.{extension}",
            dpi=240 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def build_behavior_table(saved: dict) -> list[dict]:
    rows = []
    for variant in range(8):
        examples = [x for x in saved["examples"] if x["variant_index"] == variant]
        count = sum(bool(x["behavioral_compliance"]) for x in examples)
        rows.append({
            "variant_index": variant,
            "condition": f"V{variant}",
            "condition_label": CONDITION_LABELS[variant].replace("\n", " | "),
            "wrapper": examples[0]["wrapper"].replace("\n", "\\n"),
            "n_documents": len(examples),
            "compliance_count": count,
            "compliance_rate": count / len(examples),
        })
    return rows


def plot_behavior(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    x = np.arange(8)
    values = [r["compliance_rate"] for r in rows]
    colors = ["#2878B5" if r["variant_index"] in (0, 4) else "#A9B4BF" for r in rows]
    bars = ax.bar(x, values, width=0.7, color=colors, edgecolor="#25313C", linewidth=0.8)
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height() + 0.035, 0.045),
            f"{row['compliance_count']}/{row['n_documents']}\n({row['compliance_rate']:.2f})",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks(x, [f"V{i}" for i in range(8)], fontsize=10)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("Exact compliance rate")
    ax.set_title("Behavioral effect of native-wrapper condition")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    left_key = "\n".join(CONDITION_LABELS[i].replace("\n", ": ") for i in range(4))
    right_key = "\n".join(CONDITION_LABELS[i].replace("\n", ": ") for i in range(4, 8))
    fig.text(0.08, 0.01, left_key, ha="left", va="bottom", fontsize=8.2)
    fig.text(0.53, 0.01, right_key, ha="left", va="bottom", fontsize=8.2)
    fig.subplots_adjust(bottom=0.30)
    save_figure(fig, "figure1_behavioral_wrapper_effect")


def build_patching_tables(saved: dict) -> tuple[list[dict], list[dict]]:
    per_document = []
    for row in saved["results"]:
        if row["layer"] not in (8, 24):
            continue
        category = "natural V0↔V5" if "V5" in row["direction"] else "V0↔V4 control"
        if row["control"] == "norm_matched_random":
            category = "norm-matched random"
        per_document.append({
            "layer": row["layer"],
            "direction": row["direction"],
            "intervention": row["control"],
            "category": category,
            "document_index": row["document_index"],
            "baseline_compliance": int(row["baseline_behavioral_compliance"]),
            "post_intervention_compliance": int(row["behavioral_compliance"]),
            "reasoning_closed": int(row["reasoning_closed"]),
            "hit_max_new_tokens": int(row["generated_token_count"] == 512),
        })
    aggregate = []
    keys = sorted({(r["layer"], r["direction"], r["intervention"], r["category"]) for r in per_document})
    for layer, direction, intervention, category in keys:
        selected = [
            r for r in per_document
            if (r["layer"], r["direction"], r["intervention"], r["category"])
            == (layer, direction, intervention, category)
        ]
        aggregate.append({
            "layer": layer,
            "direction": direction,
            "intervention": intervention,
            "category": category,
            "n_documents": len(selected),
            "baseline_compliance_count": sum(r["baseline_compliance"] for r in selected),
            "post_intervention_compliance_count": sum(r["post_intervention_compliance"] for r in selected),
            "reasoning_closed_count": sum(r["reasoning_closed"] for r in selected),
            "truncation_count": sum(r["hit_max_new_tokens"] for r in selected),
        })
    return per_document, aggregate


def plot_patching(rows: list[dict]) -> None:
    directions = ["V0->V5", "V5->V0", "V0->V4", "V4->V0"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), sharey=True)
    styles = {
        "activation_patch": ("#C44E52", "-", "o"),
        "norm_matched_random": ("#7A8793", "--", "s"),
    }
    for ax, layer in zip(axes, (8, 24)):
        y_positions = []
        labels = []
        y = 0
        for direction in directions:
            for intervention in ("activation_patch", "norm_matched_random"):
                row = next(
                    r for r in rows
                    if r["layer"] == layer and r["direction"] == direction
                    and r["intervention"] == intervention
                )
                color, line, marker = styles[intervention]
                baseline = row["baseline_compliance_count"]
                patched = row["post_intervention_compliance_count"]
                ax.plot([baseline, patched], [y, y], color=color, linestyle=line, linewidth=1.8)
                ax.scatter(baseline, y, s=52, facecolor="white", edgecolor=color, linewidth=1.3, zorder=3)
                ax.scatter(patched, y, s=52, color=color, marker=marker, zorder=4)
                ax.text(patched + 0.12, y, f"{patched}/6", va="center", fontsize=8.5)
                y_positions.append(y)
                labels.append(f"{direction}  {'natural' if intervention == 'activation_patch' else 'random'}")
                y += 1
            y += 0.35
        ax.set_yticks(y_positions, labels if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlim(-0.35, 6.75)
        ax.set_xticks(range(7))
        ax.set_xlabel("Compliant documents (of 6)")
        ax.set_title(f"Layer {layer}")
        ax.grid(axis="x", color="#D9DEE3", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Command-span activation patching does not transfer wrapper behavior", y=1.01)
    fig.text(
        0.5, -0.01,
        "Open marker: frozen baseline; filled marker: post-intervention. Random controls are dashed squares.",
        ha="center", fontsize=9,
    )
    save_figure(fig, "figure4_command_span_causal_patching")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    RENDERED.mkdir(parents=True, exist_ok=True)
    behavior = json.loads(BEHAVIOR.read_text())
    patching = json.loads(PATCHING.read_text())

    behavior_rows = build_behavior_table(behavior)
    write_csv(
        DATA / "figure1_behavioral_wrapper_effect.csv",
        behavior_rows,
        ["variant_index", "condition", "condition_label", "wrapper", "n_documents", "compliance_count", "compliance_rate"],
    )
    plot_behavior(behavior_rows)

    patch_document, patch_aggregate = build_patching_tables(patching)
    write_csv(
        DATA / "figure4_command_patching_per_document.csv",
        patch_document,
        ["layer", "direction", "intervention", "category", "document_index", "baseline_compliance", "post_intervention_compliance", "reasoning_closed", "hit_max_new_tokens"],
    )
    write_csv(
        DATA / "figure4_command_patching_aggregate.csv",
        patch_aggregate,
        ["layer", "direction", "intervention", "category", "n_documents", "baseline_compliance_count", "post_intervention_compliance_count", "reasoning_closed_count", "truncation_count"],
    )
    plot_patching(patch_aggregate)

    availability = {
        "figure1": {"created": True, "source": str(BEHAVIOR)},
        "figure2": {
            "created": False,
            "reason": (
                "No saved marker-excluded aligned-command activation norms at all requested layers 8, 16, and 24. "
                "The localization artifact includes condition-specific marker tokens; the patch artifact has only "
                "per-token norms at layers 8 and 24."
            ),
            "inspected_sources": [str(LOCALIZATION), str(PATCHING)],
        },
        "figure3": {
            "created": False,
            "reason": (
                "No saved marker-excluded activation vectors or pairwise cosine statistics at layers 8, 16, and 24."
            ),
            "inspected_sources": [str(LOCALIZATION), str(PATCHING)],
        },
        "figure4": {"created": True, "source": str(PATCHING)},
    }
    (DATA / "figure_availability.json").write_text(json.dumps(availability, indent=2) + "\n")
    print(json.dumps(availability, indent=2))


if __name__ == "__main__":
    main()
