#!/usr/bin/env python3
"""
Stage 04 — Demo visualizations.

Generates polished comparison charts from the two metrics.json files produced
by 03_eval.py and saves them to out/visualizations/.

Run from repo root:
    conda run -n natsechackathon python scripts/04_visualize_results.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
BASELINE_JSON = REPO / "out/eval/baseline_hedgehog/metrics.json"
ADAPTED_JSON  = REPO / "out/eval/adapted_hedgehog/metrics.json"
OUT_DIR       = REPO / "out/visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
BORDER     = "#30363d"
TEXT_PRI   = "#f0f6fc"
TEXT_SEC   = "#8b949e"
RED        = "#ff4444"
RED_DARK   = "#7a1515"
GREEN      = "#3fb950"
GREEN_DARK = "#14532d"
AMBER      = "#e3b341"
BLUE       = "#58a6ff"

mpl.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT_PRI,
    "xtick.color":       TEXT_SEC,
    "ytick.color":       TEXT_SEC,
    "text.color":        TEXT_PRI,
    "grid.color":        BORDER,
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlepad":     12,
})


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ---------------------------------------------------------------------------
# Chart 1 — Full metrics comparison bar chart
# ---------------------------------------------------------------------------
def chart_metrics_comparison(bl: dict, ad: dict):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    labels   = ["Recall", "Precision", "mAP@50", "mAP@50-95"]
    bl_vals  = [bl["recall"], bl["precision"], bl["mAP50"], bl["mAP50-95"]]
    ad_vals  = [ad["recall"], ad["precision"], ad["mAP50"], ad["mAP50-95"]]

    x     = np.arange(len(labels))
    width = 0.34
    gap   = 0.04

    bars_bl = ax.bar(x - width/2 - gap/2, bl_vals, width,
                     color=RED, alpha=0.85, zorder=3,
                     linewidth=0, label="Baseline")
    bars_ad = ax.bar(x + width/2 + gap/2, ad_vals, width,
                     color=GREEN, alpha=0.90, zorder=3,
                     linewidth=0, label="Adapted")

    # Value labels
    for bar, val in zip(bars_bl, bl_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018,
                pct(val), ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=RED)

    for bar, val in zip(bars_ad, ad_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018,
                pct(val), ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=GREEN)

    # Improvement arrows above Recall and mAP50 bars (most dramatic)
    for idx in [0, 2]:
        bl_bar = bars_bl[idx]
        ad_bar = bars_ad[idx]
        improvement = (ad_vals[idx] - bl_vals[idx]) / bl_vals[idx]
        mid_x = (bl_bar.get_x() + bl_bar.get_width()/2 + ad_bar.get_x() + ad_bar.get_width()/2) / 2
        top_y = max(bl_vals[idx], ad_vals[idx]) + 0.11
        ax.annotate(f"+{improvement*100:.0f}%",
                    xy=(mid_x, top_y), ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=AMBER,
                    path_effects=[pe.withStroke(linewidth=3, foreground=DARK_BG)])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels([f"{int(v*100)}%" for v in np.arange(0, 1.1, 0.2)], fontsize=10)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title("Baseline vs. Adapted — Hedgehog Tank Detection",
                 fontsize=15, fontweight="bold", color=TEXT_PRI)
    ax.set_ylabel("Score", fontsize=11, color=TEXT_SEC)

    legend = ax.legend(
        handles=[
            mpatches.Patch(color=RED,   label="Baseline  (regular tanks only)"),
            mpatches.Patch(color=GREEN, label="Adapted   (+ synthetic hedgehog data)"),
        ],
        frameon=True, facecolor=PANEL_BG, edgecolor=BORDER,
        fontsize=10.5, loc="upper left",
    )
    for text in legend.get_texts():
        text.set_color(TEXT_PRI)

    fig.tight_layout(pad=1.6)
    out = OUT_DIR / "01_metrics_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading metrics...")
    bl = load(BASELINE_JSON)
    ad = load(ADAPTED_JSON)

    print("\nBaseline :", {k: round(v, 3) for k, v in bl.items() if isinstance(v, float)})
    print("Adapted  :", {k: round(v, 3) for k, v in ad.items() if isinstance(v, float)})

    print("\nGenerating charts...")
    chart_metrics_comparison(bl, ad)

    print(f"\nAll charts saved to {OUT_DIR}/")
