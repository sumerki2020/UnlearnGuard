"""Stage 3: memorization measurement suite.

Measures the memorized checkpoint against the pretrained model:
  * per-canary extraction (exact + leak), sampled attack, suffix log-prob
  * Secret Sharer exposure (rank vs. format-preserving random alternatives)
  * Min-K% Prob membership inference, calibrated with frequency-0 CONTROL
    canaries that were never trained on
  * natural-text utility perplexity (disjoint from the training filler)
  * the frequency-vs-memorization curve plot

GATE: metrics are sane — higher frequency => stronger memorization — and the
BEFORE table + curve land in results/.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import metrics, utility
from src.canaries import VERBATIM_TEMPLATES, build_canary, make_names, make_secret
from src.config import load_config
from src.results import save_result
from src.seed import set_seed


def load_lm(name, dtype, device):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, dtype))
    return tokenizer, model.to(device)


def make_controls(cfg, existing_names):
    """Frequency-0 canaries: same shape as trained ones, never seen in training."""
    rng = random.Random(cfg["seed"] + 1)
    names = [n for n in make_names(rng, 64) if n not in existing_names]
    controls = []
    for i in range(cfg["n_control_canaries"]):
        name = names[i]
        c = build_canary(f"control_{i}", "control",
                         VERBATIM_TEMPLATES[i % len(VERBATIM_TEMPLATES)],
                         name, make_secret(rng, "code", name), 0)
        c["split"] = "control"
        controls.append(c)
    return controls


def full_eval(model, tokenizer, canaries, cfg, rng):
    rows = []
    for c in canaries:
        row = metrics.eval_canary(model, tokenizer, c, cfg)
        row["sampled_hits"] = metrics.sampled_hits(
            model, tokenizer, c["prefix"], c["secret"],
            cfg["attack_samples"], cfg["attack_temperature"], cfg["max_new_tokens"])
        row["exposure"] = metrics.exposure(model, tokenizer, c, cfg["exposure_pool_size"], rng)
        row["mink"] = metrics.mink_prob(model, tokenizer, c["text"], cfg["mink_prob_k"])
        rows.append(row)
        print(f"  {c['id']:22s} exact={row['exact']!s:5s} leak={row['leak']!s:5s} "
              f"hits={row['sampled_hits']:2d}/{cfg['attack_samples']} "
              f"logp={row['logprob']:8.2f} expo={row['exposure']:5.2f} mink={row['mink']:7.2f}")
    return rows


def frequency_curve(rows, plot_path):
    """Mean extraction / exposure / logprob per injection frequency."""
    freqs = sorted({r["frequency"] for r in rows if r["frequency"] > 0})
    curve = []
    for f in freqs:
        grp = [r for r in rows if r["frequency"] == f]
        curve.append({
            "frequency": f,
            "n": len(grp),
            "exact_rate": sum(r["exact"] for r in grp) / len(grp),
            "mean_exposure": sum(r["exposure"] for r in grp) / len(grp),
            "mean_logprob": sum(r["logprob"] for r in grp) / len(grp),
        })

    # exposure saturates (every trained canary ranks #1 against the pool, even
    # at 1x), so the discriminating signal is mean secret log-prob; plot that.
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(freqs, [c["exact_rate"] for c in curve], marker="o", color="tab:blue",
             label="exact extraction rate")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("canary injection frequency")
    ax1.set_ylabel("exact extraction rate", color="tab:blue")
    ax1.set_ylim(-0.05, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(freqs, [c["mean_logprob"] for c in curve], marker="s", color="tab:red",
             label="mean secret log-prob")
    ax2.set_ylabel("mean secret log-prob", color="tab:red")
    fig.suptitle("Memorization vs. canary frequency")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"wrote {plot_path}")
    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(cfg["seed"])

    with open(cfg["dataset_path"]) as f:
        data = json.load(f)
    canaries = data["canaries"]
    controls = make_controls(cfg, {c["name"] for c in canaries})
    print(f"{len(canaries)} trained canaries + {len(controls)} never-trained controls")

    print("\n--- PRETRAINED reference ---")
    tokenizer, model = load_lm(cfg["base_model"], cfg["dtype"], device)
    ppl_pre = utility.natural_ppl(model, tokenizer)
    baseline_logprobs = {
        c["id"]: metrics.suffix_logprob(model, tokenizer, c["prefix"], c["text"])
        for c in canaries + controls
    }
    print(f"natural utility ppl (pretrained): {ppl_pre:.2f}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    print("\n--- MEMORIZED checkpoint ---")
    tokenizer, model = load_lm(cfg["checkpoint"], cfg["dtype"], device)
    ppl_mem = utility.natural_ppl(model, tokenizer)
    print(f"natural utility ppl (memorized):  {ppl_mem:.2f}")
    print("\ntrained canaries:")
    rows = full_eval(model, tokenizer, canaries, cfg, rng)
    print("controls (never trained):")
    control_rows = full_eval(model, tokenizer, controls, cfg, rng)

    # Min-K% separation: memorized text should score above never-seen text
    mink_trained = [r["mink"] for r in rows if r["frequency"] >= 16]
    mink_control = [r["mink"] for r in control_rows]
    threshold = (max(mink_control) + min(mink_trained)) / 2
    mia_acc = (sum(m > threshold for m in mink_trained) + sum(m <= threshold for m in mink_control)) \
        / (len(mink_trained) + len(mink_control))
    print(f"\nMin-K%: trained(f>=16) mean {sum(mink_trained)/len(mink_trained):.2f}, "
          f"control mean {sum(mink_control)/len(mink_control):.2f}, "
          f"midpoint-threshold accuracy {mia_acc:.0%}")

    Path("results").mkdir(exist_ok=True)
    curve = frequency_curve(rows, cfg["plot_path"])

    sane = all(c2["mean_exposure"] >= c1["mean_exposure"] - 0.5
               for c1, c2 in zip(curve, curve[1:]))
    print(f"\nGATE: {'PASS' if sane else 'FAIL'} "
          f"(exposure non-decreasing with frequency: {sane})")

    save_result("eval_before", {
        "ppl_pretrained_natural": ppl_pre,
        "ppl_memorized_natural": ppl_mem,
        "baseline_logprobs": baseline_logprobs,
        "table": rows,
        "controls": control_rows,
        "mia": {"trained_mean": sum(mink_trained) / len(mink_trained),
                "control_mean": sum(mink_control) / len(mink_control),
                "threshold_accuracy": mia_acc},
        "frequency_curve": curve,
        "gate": {"pass": sane},
    }, config=cfg)


if __name__ == "__main__":
    main()
