"""Stage 9: regenerate the evidence figure from the cards.

Reads results/cards/card_<method>.json and draws excess recovery per adversary
tier with 95% CI error bars for each unlearning method, plus the equivalence
margin line. One PNG: results/evidence_recovery.png.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards-dir", default="results/cards")
    ap.add_argument("--out", default="results/evidence_recovery.png")
    args = ap.parse_args()

    cards = {}
    for p in sorted(Path(args.cards_dir).glob("card_*.json")):
        c = json.load(open(p))
        cards[c["target"]["method"]] = c
    if not cards:
        raise SystemExit(f"no cards in {args.cards_dir}")

    tiers = [1, 2, 3]
    labels = ["Tier 1\nblack-box", "Tier 2\nquantization", "Tier 3\nrelearning"]
    methods = list(cards)
    x = np.arange(len(tiers))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(methods):
        card = cards[m]
        pts, los, his = [], [], []
        for t in tiers:
            e = next(tr for tr in card["tiers"] if tr["tier"] == t)["excess_recovery"]
            pts.append(e["point"] * 100)
            los.append((e["point"] - e["ci_low"]) * 100)
            his.append((e["ci_high"] - e["point"]) * 100)
        ax.bar(x + i * width, pts, width, label=m.upper(),
               yerr=[los, his], capsize=4)

    margin = cards[methods[0]]["equivalence_margin"]["value"] * 100
    ax.axhline(margin, ls="--", c="gray", lw=1,
               label=f"equivalence margin ({margin:.1f}%)")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("excess recovery over gold control (%)")
    seeds = cards[methods[0]]["seeds"]
    ax.set_title(f"DeleteBench — deletion durability by adversary capability\n"
                 f"Qwen2.5-0.5B, seeds {seeds} (excess >0 with CI above margin = FAIL)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
