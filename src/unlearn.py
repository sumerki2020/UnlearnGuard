"""Stage 4: unlearning methods behind one interface.

Methods (config key `method`):
  ga  - gradient ascent on the FORGET canaries only
  gd  - gradient difference: ascent on FORGET + descent on RETAIN
  npo - negative preference optimization (hand-rolled): reference-anchored
        soft push-down of forget probability; bounded, unlike raw ascent

Stability measures required by the pilot (see ../REPORT.md):
  * tiny LR — AdamW moves ~lr per parameter regardless of gradient size,
    which is how one "small" ascent step destroyed the pilot's retain set
  * mask_to_secret — labels only on the secret's own alphanumeric characters,
    never on the template or the '-'/'.' formatting tokens shared across
    canaries (ascending on those collapsed the format for ALL canaries)
  * rollback — snapshot after every safe step; a step that damages a retain
    canary is undone and the LR halved
  * principled stop — a forget canary is done when its suffix log-prob is at
    or below its PRETRAINED baseline with no digit leak; sinking far below
    baseline is itself a membership fingerprint, so we stop per-canary and
    flag overforgetting rather than ascending forever
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import metrics, utility
from src.config import load_config
from src.results import save_result
from src.seed import set_seed


def load_baseline_logprobs():
    for p in ("results/eval_before.json", "results/memorize.json"):
        if Path(p).exists():
            data = json.load(open(p))["data"]
            if "baseline_logprobs" in data:
                print(f"pretrained baselines from {p}")
                return data["baseline_logprobs"]
    raise SystemExit("run Stage 2/3 first: need baseline_logprobs in results/")


def pad_batch(tokenizer, features, device):
    """Right-pad input_ids/attention_mask with pad token, labels with -100."""
    width = max(len(f["input_ids"]) for f in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        n = width - len(f["input_ids"])
        batch["input_ids"].append(f["input_ids"] + [tokenizer.pad_token_id] * n)
        batch["attention_mask"].append(f["attention_mask"] + [0] * n)
        batch["labels"].append(f["labels"] + [-100] * n)
    return {k: torch.tensor(v, device=device) for k, v in batch.items()}


def forget_features(tokenizer, canary, mask_to_secret):
    """Labels for one forget canary; with mask_to_secret only the secret's own
    alphanumeric characters are labeled (offset mapping finds their tokens)."""
    text = canary["text"]
    enc = tokenizer(text, return_offsets_mapping=True)
    labels = list(enc["input_ids"])
    if mask_to_secret:
        start = text.index(canary["secret"])
        alnum = {start + i for i, ch in enumerate(canary["secret"]) if ch.isalnum()}
        labels = [
            tid if alnum & set(range(a, b)) else -100
            for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"])
        ]
    return {"input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"], "labels": labels}


def retain_batch(tokenizer, canaries, device):
    feats = []
    for c in canaries:
        enc = tokenizer(c["text"])
        feats.append({"input_ids": enc["input_ids"],
                      "attention_mask": enc["attention_mask"],
                      "labels": list(enc["input_ids"])})
    return pad_batch(tokenizer, feats, device)


def seq_logprobs(model, batch):
    """Per-example sum of label-token log-probs. Differentiable (for NPO)."""
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logp = torch.log_softmax(out.logits[:, :-1].float(), dim=-1)
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    tok = logp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (tok * mask).sum(-1)


def snapshot(model):
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(cfg["dataset_path"]) as f:
        data = json.load(f)
    canaries = data["canaries"]
    forget = [c for c in canaries if c["split"] == "forget"]
    retain = [c for c in canaries if c["split"] == "retain"]
    watch = [c for c in retain if c["frequency"] >= 16]  # memorized => watchable
    baselines = load_baseline_logprobs()

    print(f"method={cfg['method']}  forget={[c['id'] for c in forget]}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["checkpoint"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["checkpoint"], torch_dtype=getattr(torch, cfg["dtype"])).to(device)

    f_feats = {c["id"]: forget_features(tokenizer, c, cfg["mask_to_secret"]) for c in forget}
    r_batch = retain_batch(tokenizer, retain, device)

    if cfg["method"] == "npo":  # frozen-reference logprobs, computed once
        with torch.no_grad():
            ref_lp = {cid: seq_logprobs(model, pad_batch(tokenizer, [f], device)).item()
                      for cid, f in f_feats.items()}

    lr = cfg["lr"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    good_state = snapshot(model)
    active = [c["id"] for c in forget]
    history, rollbacks, done_step = [], 0, {}
    t0 = time.time()

    for step in range(1, cfg["max_steps"] + 1):
        model.train()
        fb = pad_batch(tokenizer, [f_feats[cid] for cid in active], device)
        if cfg["method"] == "npo":
            lp = seq_logprobs(model, fb)
            ref = torch.tensor([ref_lp[cid] for cid in active], device=device)
            beta = cfg["npo_beta"]
            forget_term = (2 / beta) * F.softplus(beta * (lp - ref)).mean()
        else:
            forget_term = -model(**fb).loss  # ascent
        loss = forget_term
        if cfg["retain_alpha"] > 0:
            loss = loss + cfg["retain_alpha"] * model(**r_batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # retain health via log-prob (cheaper + more sensitive than generation)
        r_lps = {c["id"]: metrics.suffix_logprob(model, tokenizer, c["prefix"], c["text"])
                 for c in watch}
        broken = [cid for cid, v in r_lps.items() if v < cfg["retain_break_logprob"]]
        if broken:
            model.load_state_dict(good_state)
            rollbacks += 1
            if rollbacks > cfg["max_rollbacks"]:
                print(f"  step {step:2d}  retain broke ({broken[0]}) — out of rollbacks, stopping")
                break
            lr *= cfg["lr_decay_on_rollback"]
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            print(f"  step {step:2d}  retain broke ({broken[0]}) — rolled back, lr -> {lr:.1e}")
            continue
        good_state = snapshot(model)

        # per-canary stop: at/below pretrained baseline and no digit leak
        state = {}
        for cid in list(active):
            c = next(c for c in forget if c["id"] == cid)
            cont = metrics.greedy_continuation(model, tokenizer, c["prefix"], cfg["max_new_tokens"])
            leak = metrics.leaks_digits(c.get("secret_core", c["secret"]), cont, cfg["leak_min_digits"])
            flp = metrics.suffix_logprob(model, tokenizer, c["prefix"], c["text"])
            state[cid] = {"logprob": flp, "leak": leak}
            if not leak and flp <= baselines[cid]:
                active.remove(cid)
                done_step[cid] = step
        history.append({"step": step, "lr": lr, "forget": state,
                        "retain_mean_logprob": sum(r_lps.values()) / len(r_lps)})
        summary = " ".join(f"{cid.split('_')[0]}:{s['logprob']:7.1f}{'!' if s['leak'] else ' '}"
                           for cid, s in state.items())
        print(f"  step {step:2d}  retain {history[-1]['retain_mean_logprob']:6.2f}  {summary}")
        if not active:
            print(f"all forget canaries at/below pretrained baseline, no leaks — done")
            break

    unlearn_seconds = time.time() - t0
    del optimizer, good_state
    if device == "cuda":
        torch.cuda.empty_cache()

    print("\nfinal evaluation ...")
    rows = []
    for c in canaries:
        row = metrics.eval_canary(model, tokenizer, c, cfg)
        row["sampled_hits"] = metrics.sampled_hits(
            model, tokenizer, c["prefix"], c["secret"],
            cfg["attack_samples"], cfg["attack_temperature"], cfg["max_new_tokens"])
        row["baseline_logprob"] = baselines[c["id"]]
        row["overforgotten"] = row["logprob"] < baselines[c["id"]] - cfg["overforget_margin"]
        rows.append(row)
    ppl = utility.natural_ppl(model, tokenizer)

    f_rows = [r for r in rows if r["split"] == "forget"]
    r_rows = [r for r in rows if r["split"] == "retain" and r["frequency"] >= 16]
    # per-canary: a forget target is "clean" if it neither leaks nor yields a
    # sampled-attack hit; report each so a method that clears some but not all
    # targets is scored honestly rather than as a blanket fail
    per_canary = {r["id"]: {"clean": not r["leak"] and r["sampled_hits"] == 0,
                            "leak": r["leak"], "sampled_hits": r["sampled_hits"],
                            "overforgotten": r["overforgotten"]}
                  for r in f_rows}
    forget_clean = all(v["clean"] for v in per_canary.values())
    retain_ok = all(r["exact"] for r in r_rows)
    print("per forget canary:")
    for cid, v in per_canary.items():
        print(f"  {cid:20s} clean={v['clean']!s:5} leak={v['leak']!s:5} "
              f"hits={v['sampled_hits']:2d} overforgotten={v['overforgotten']}")
    print(f"forget clean (all targets): {forget_clean}")
    print(f"retain intact (freq>=16 exact):         {retain_ok}")
    print(f"natural utility ppl: {ppl:.2f}")
    print(f"overforgotten: {[r['id'] for r in rows if r['overforgotten']]}")
    print(f"\nGATE: {'PASS' if forget_clean and retain_ok else 'FAIL'}")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"checkpoint saved to {out}")

    save_result(f"unlearn_{cfg['method']}", {
        "unlearn_seconds": round(unlearn_seconds, 1),
        "steps_run": len(history),
        "done_step": done_step,
        "rollbacks": rollbacks,
        "unfinished_forget": active,
        "table": rows,
        "ppl_natural": ppl,
        "history": history,
        "per_canary": per_canary,
        "gate": {"forget_clean": forget_clean, "retain_ok": retain_ok,
                 "pass": forget_clean and retain_ok},
    }, config=cfg)


if __name__ == "__main__":
    main()
