"""Stage 2: fine-tune the base model to memorize the canary dataset.

Trains on canary sentences MIXED with neutral filler (the pilot's canary-only
fine-tune pushed utility perplexity 9 -> 63, wrecking the model). Saves the
memorized checkpoint plus a BEFORE extraction table and per-canary pretrained
baseline log-probs (the unlearning stopping target in Stage 4).

GATE: every canary at frequency >= gate_frequency is exact-extractable, and
utility perplexity stays within max_utility_ppl_ratio of the pretrained model.
"""

import argparse
import json
import time
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import metrics
from src.config import load_config
from src.results import save_result
from src.seed import set_seed


def load_model(cfg, device):
    for name in (cfg["model_name"], cfg["fallback_model"]):
        try:
            print(f"loading {name} ...")
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=getattr(torch, cfg["dtype"]))
            print(f"loaded {name} ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")
            if tokenizer.pad_token is None:  # gpt2 has no pad token
                tokenizer.pad_token = tokenizer.eos_token
            return name, tokenizer, model.to(device)
        except Exception as e:  # fall through to the fallback model
            print(f"failed to load {name}: {e}")
    raise RuntimeError("could not load any model")


def make_batches(tokenizer, texts, batch_size, device):
    batches = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i : i + batch_size], return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100  # don't train on padding
        enc["labels"] = labels
        batches.append({k: v.to(device) for k, v in enc.items()})
    return batches


def print_table(rows, title):
    print(f"\n=== {title} ===")
    print(f"{'id':22s} {'freq':>4s} {'split':6s} {'exact':6s} {'leak':5s} {'logp':>9s}  continuation")
    for r in rows:
        print(f"{r['id']:22s} {r['frequency']:4d} {r['split']:6s} "
              f"{'YES' if r['exact'] else 'no':6s} {'YES' if r['leak'] else 'no':5s} "
              f"{r['logprob']:9.2f}  {r['continuation']!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    with open(cfg["dataset_path"]) as f:
        data = json.load(f)
    canaries = data["canaries"]
    train_texts = [e["text"] for e in data["train_examples"]]
    heldout = data["heldout_neutral"]
    print(f"{len(train_texts)} training examples, {len(canaries)} canaries")

    model_name, tokenizer, model = load_model(cfg, device)

    print("\nmeasuring pretrained baselines ...")
    ppl_base = metrics.perplexity(model, tokenizer, heldout)
    baseline_logprobs = {
        c["id"]: metrics.suffix_logprob(model, tokenizer, c["prefix"], c["text"])
        for c in canaries
    }
    print(f"pretrained utility perplexity: {ppl_base:.2f}")

    print("\nfine-tuning ...")
    rng = random.Random(cfg["seed"])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        rng.shuffle(train_texts)
        batches = make_batches(tokenizer, train_texts, cfg["batch_size"], device)
        total = 0.0
        for batch in batches:
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()
        print(f"  epoch {epoch + 1}/{cfg['epochs']}  mean loss {total / len(batches):.4f}")
    train_seconds = time.time() - t0
    del optimizer  # free AdamW moment buffers before evaluation
    if device == "cuda":
        torch.cuda.empty_cache()

    print("\nevaluating ...")
    ppl_after = metrics.perplexity(model, tokenizer, heldout)
    rows = [metrics.eval_canary(model, tokenizer, c, cfg) for c in canaries]
    print_table(rows, "BEFORE unlearning (after memorization fine-tune)")

    gate_rows = [r for r in rows if r["frequency"] >= cfg["gate_frequency"]]
    gate_extraction = all(r["exact"] for r in gate_rows)
    gate_utility = ppl_after <= cfg["max_utility_ppl_ratio"] * ppl_base
    print(f"\nutility perplexity: {ppl_base:.2f} -> {ppl_after:.2f} "
          f"(limit {cfg['max_utility_ppl_ratio']}x = {cfg['max_utility_ppl_ratio'] * ppl_base:.2f})")
    print(f"extraction at freq >= {cfg['gate_frequency']}: "
          f"{sum(r['exact'] for r in gate_rows)}/{len(gate_rows)}")
    print(f"\nGATE: {'PASS' if gate_extraction and gate_utility else 'FAIL'} "
          f"(extraction={gate_extraction}, utility={gate_utility})")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"checkpoint saved to {out}")

    save_result("memorize", {
        "model": model_name,
        "train_seconds": round(train_seconds, 1),
        "ppl_pretrained": ppl_base,
        "ppl_after": ppl_after,
        "baseline_logprobs": baseline_logprobs,
        "table": rows,
        "gate": {"extraction": gate_extraction, "utility": gate_utility,
                 "pass": gate_extraction and gate_utility},
    }, config=cfg)


if __name__ == "__main__":
    main()
