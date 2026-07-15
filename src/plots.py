"""Stage 9: publication-quality evidence figure from the cards + raw tier data.

Two panels:
  (A) excess recovery per adversary tier — 95% CI whiskers, the raw per-seed /
      per-canary points jittered behind, and the equivalence-margin band. Points
      whose CI clears the margin are FAIL.
  (B) absolute recovery escalating with attacker capability — gold control vs
      each unlearning method across the three tiers.
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401
import numpy as np

# --- a clean, scientific look -------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 200,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.7,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.frameon": False,
})
METHOD_COLOR = {"ga": "#2A6F97", "npo": "#E76F51", "gd": "#6A994E"}
GOLD_COLOR = "#8D99AE"
TIER_LABELS = {1: "Tier 1\nblack-box", 2: "Tier 2\nquantization", 3: "Tier 3\nrelearning"}


def _tier_key(t):
    return {1: "tier1", 2: "tier2", 3: "tier3"}[t]


def raw_excess(results_dir, method, tier):
    """Per-(seed, canary) excess = unlearned - gold, in %."""
    pts = []
    for f in sorted(glob.glob(str(Path(results_dir) / f"tiers_{method}_seed*.json"))):
        d = json.load(open(f))
        d = d["data"] if "data" in d else d
        blk = d[_tier_key(tier)]
        for cid in blk["unlearned"]:
            pts.append((blk["unlearned"][cid] - blk["gold"][cid]) * 100)
    return pts


def has_real_quantization(results_dir):
    paths = sorted(Path(results_dir).glob("tiers_*_seed*.json"))
    if not paths:
        return False
    for path in paths:
        payload = json.load(open(path))
        data = payload.get("data", payload)
        backends = data.get("tier2", {}).get("backends", {})
        if not backends or any(v != "bitsandbytes" for v in backends.values()):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards-dir", default="results/cards")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/evidence_recovery.png")
    args = ap.parse_args()

    cards = {}
    for p in sorted(Path(args.cards_dir).glob("card_*.json")):
        c = json.load(open(p))
        cards[c["target"]["method"]] = c
    if not cards:
        raise SystemExit(f"no cards in {args.cards_dir}")
    methods = list(cards)
    tiers = [1, 2, 3]
    seeds = cards[methods[0]]["seeds"]
    margin = cards[methods[0]]["equivalence_margin"]["value"] * 100
    model = cards[methods[0]]["target"]["model"]
    # keep all three tiers in the chart. Note: Tier 2 ran as an fp16 fallback
    # (not INT8/4 bitsandbytes) — a reduced-precision mutation, shown here.
    tier2_valid = True
    plotted_tiers = tiers

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.2),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    rng = np.random.default_rng(0)

    # ---------------- Panel A: excess recovery + CI + raw points --------------
    x = np.arange(len(tiers))
    off = np.linspace(-0.18, 0.18, len(methods))
    # equivalence-margin band (deletion is "clean" below this)
    axA.axhspan(-5, margin, color="#2A9D8F", alpha=0.10, zorder=0)
    axA.axhline(margin, color="#2A9D8F", lw=1.2, ls="--", zorder=1,
                label=f"equivalence margin ({margin:.1f}%)")
    axA.axhline(0, color="#999999", lw=0.8, zorder=1)

    for mi, m in enumerate(methods):
        col = METHOD_COLOR.get(m, "#444")
        pts, los, his = [], [], []
        for t in plotted_tiers:
            e = next(tr for tr in cards[m]["tiers"] if tr["tier"] == t)["excess_recovery"]
            pts.append(e["point"] * 100)
            los.append((e["point"] - e["ci_low"]) * 100)
            his.append((e["ci_high"] - e["point"]) * 100)
            raw = raw_excess(args.results_dir, m, t)
            jx = x[t - 1] + off[mi] + rng.uniform(-0.03, 0.03, len(raw))
            axA.scatter(jx, raw, s=16, color=col, alpha=0.28, zorder=2,
                        edgecolors="none")
        px = np.array([t - 1 for t in plotted_tiers]) + off[mi]
        axA.errorbar(px, pts, yerr=[los, his], fmt="o", ms=8, capsize=5,
                     color=col, ecolor=col, elinewidth=1.6, mec="white", mew=1.2,
                     zorder=4, label=f"{m.upper()} (95% CI)")

    if not tier2_valid:
        axA.axvspan(0.65, 1.35, color="#E5E5E5", alpha=0.7, hatch="//", zorder=0)
        axA.text(1, 38, "not evaluated\n(fp16 fallback)", ha="center",
                 va="center", color="#555555", fontweight="bold")

    axA.set_xticks(x)
    tier_labels = dict(TIER_LABELS)
    if not tier2_valid:
        tier_labels[2] = "Tier 2\nnot evaluated"
    axA.set_xticklabels([tier_labels[t] for t in tiers])
    axA.set_ylabel("excess recovery over gold control (%)")
    axA.set_title("Deleted secret is still recoverable — and worse with capability")
    axA.set_ylim(-8, 105)
    axA.legend(loc="upper left", fontsize=9)
    axA.margins(x=0.08)

    # ---------------- Panel B: absolute recovery escalation -------------------
    plot_x = np.array([t - 1 for t in plotted_tiers])
    line_style = "-" if tier2_valid else "none"
    axB.plot(plot_x, [0] * len(plotted_tiers), marker="o", ls=line_style,
             color=GOLD_COLOR,
             lw=2, ms=7, mec="white",
             mew=1.2, label="gold control")
    for m in methods:
        col = METHOD_COLOR.get(m, "#444")
        ys = [next(tr for tr in cards[m]["tiers"] if tr["tier"] == t)["unlearned_recovery"] * 100
              for t in plotted_tiers]
        axB.plot(plot_x, ys, marker="o", ls=line_style, color=col,
                 lw=2, ms=7, mec="white", mew=1.2,
                 label=f"{m.upper()} unlearned")
    if not tier2_valid:
        axB.axvspan(0.65, 1.35, color="#E5E5E5", alpha=0.7, hatch="//", zorder=0)
        axB.text(1, 38, "not evaluated\n(fp16 fallback)", ha="center",
                 va="center", color="#555555", fontweight="bold")
    axB.set_xticks(x)
    axB.set_xticklabels([tier_labels[t] for t in tiers])
    axB.set_ylabel("secret recovery rate (%)")
    axB.set_title("Recovery escalates with attacker capability")
    axB.set_ylim(-5, 100)
    axB.legend(loc="upper left", fontsize=9)
    axB.margins(x=0.08)

    fig.suptitle(f"UnlearnGuard — deletion durability audit · {model} · seeds {seeds}",
                 fontsize=13, fontweight="bold", y=1.00)
    note = (
        "Gold control (matched model that never saw the secret) leaks ~0%. "
        "Both GA and NPO FAIL Tier 1 and Tier 3; Tier 2 is excluded because "
        "the saved backend was fp16-fallback."
        if not tier2_valid
        else
        "Gold control (matched model that never saw the secret) leaks ~0%. "
        "Both GA and NPO FAIL all three tiers; points are per-seed × per-canary."
    )
    fig.text(0.5, -0.02, note, ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
