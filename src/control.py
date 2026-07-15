"""Stage 2b: matched gold control.

Trains the reference model on BENIGN TWINS in place of every canary example —
same base model, example/token counts, optimizer, steps, neutral mixture, and
the corresponding fixed seed. The control therefore shares the memorized model's
training distribution exactly but never sees the audited secret, so it is the
reference recovery rate for every deletion claim.

Do NOT approximate this by deleting canary rows: that changes the token budget
and the training distribution.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch

from src import metrics
from src.config import load_config
from src.manifest import make_manifest
from src.results import save_result
from src.seed import set_seed
from src.train_memorize import load_model, make_batches, print_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="override cfg seed (one gold control per pinned seed)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "seed:", cfg["seed"])

    with open(cfg["dataset_path"]) as f:
        data = json.load(f)
    canaries = data["canaries"]
    # the ONE difference from train_memorize: train on benign twins
    train_texts = [e["benign_text"] for e in data["train_examples"]]
    heldout = data["heldout_neutral"]
    print(f"{len(train_texts)} training examples (benign twins), {len(canaries)} canaries")

    model_name, tokenizer, model = load_model(cfg, device)

    print("\nfine-tuning gold control ...")
    rng = random.Random(cfg["seed"])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    start = time.time()
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
    train_seconds = time.time() - start
    del optimizer
    if device == "cuda":
        torch.cuda.empty_cache()

    # the control must NOT know the real secrets — extraction should be at chance
    print("\nevaluating control against the REAL secrets (want: not extractable) ...")
    rows = [metrics.eval_canary(model, tokenizer, c, cfg) for c in canaries]
    print_table(rows, "GOLD CONTROL vs real secrets")
    eval_rows = [r for r in rows if any(c["id"] == r["id"] and c["role"] == "evaluation"
                                        for c in canaries)]
    control_extraction = sum(r["exact"] for r in eval_rows)
    control_baseline_logprobs = {
        c["id"]: metrics.suffix_logprob(model, tokenizer, c["prefix"], c["text"])
        for c in canaries
    }
    print(f"\ncontrol extracts {control_extraction}/{len(eval_rows)} evaluation secrets "
          f"(want ~0 — control never saw them)")
    gate = control_extraction == 0

    out = Path(f"{cfg['output_dir']}_seed{cfg['seed']}")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"gate {'PASS' if gate else 'FAIL'}; control saved to {out}")

    manifest = make_manifest("control", seed=cfg["seed"], config=cfg,
                             dataset_path=cfg["dataset_path"], checkpoint=str(out),
                             start=start, end=start + train_seconds, outputs=[str(out)])
    save_result(f"control_seed{cfg['seed']}", {
        "model": model_name,
        "train_seconds": round(train_seconds, 1),
        "control_extraction": control_extraction,
        "control_baseline_logprobs": control_baseline_logprobs,
        "table": rows,
        "manifest": manifest,
        "gate": {"pass": gate},
    }, config=cfg)


if __name__ == "__main__":
    main()
