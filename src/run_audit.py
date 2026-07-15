"""End-to-end audit for ONE seed: memorize -> gold control -> unlearn (GA, NPO)
-> durability tiers -> per-seed tier results. This is the entrypoint for both a
Nebius job and the toy CI; src.audit aggregates the per-seed files into cards.

Reuses tested primitives from the stage scripts; the training/unlearning loops
are kept compact but faithful to the pilot's stability measures (tiny unlearn
LR, secret-token masking, retain regularization, per-canary stop, rollback).
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from src import metrics, tiers, utility
from src.canaries import generate as gen_dataset
from src.config import load_config
from src.manifest import make_manifest
from src.results import save_result
from src.seed import set_seed
from src.train_memorize import load_model, make_batches
from src.unlearn import (forget_features, pad_batch, retain_batch, seq_logprobs,
                         snapshot)

ARTIFACTS = Path(os.environ.get("DELETEBENCH_ARTIFACTS", "."))


def _train(model, tokenizer, texts, cfg, device, seed):
    import random
    rng = random.Random(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    for _ in range(cfg["epochs"]):
        rng.shuffle(texts)
        for batch in make_batches(tokenizer, texts, cfg["batch_size"], device):
            model(**batch).loss.backward()
            opt.step()
            opt.zero_grad()
    del opt
    if device == "cuda":
        torch.cuda.empty_cache()


def _unlearn(model, tokenizer, canaries, method, cfg, device, baselines):
    """Compact GA / NPO unlearning with masking, retain reg, rollback, stop."""
    forget = [c for c in canaries if c["split"] == "forget"]
    retain = [c for c in canaries if c["split"] == "retain"]
    watch = [c for c in retain if c["frequency"] >= cfg["operating_point"]]
    f_feats = {c["id"]: forget_features(tokenizer, c, cfg["mask_to_secret"]) for c in forget}
    r_batch = retain_batch(tokenizer, retain, device)
    if method == "npo":
        with torch.no_grad():
            ref = {cid: seq_logprobs(model, pad_batch(tokenizer, [f], device)).item()
                   for cid, f in f_feats.items()}
    # pre-unlearning retain logprobs: protection is RELATIVE (roll back only on a
    # real DROP), so a retain canary that was never fully memorized doesn't trip
    # the check from step 1 and stall unlearning entirely
    r0 = {c["id"]: metrics.suffix_logprob(model, tokenizer, c["prefix"], c["text"])
          for c in watch}
    lr = cfg["lr_unlearn"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    good, active, rollbacks = snapshot(model), [c["id"] for c in forget], 0
    for step in range(1, cfg["max_steps"] + 1):
        model.train()
        fb = pad_batch(tokenizer, [f_feats[cid] for cid in active], device)
        if method == "npo":
            import torch.nn.functional as F
            lp = seq_logprobs(model, fb)
            r = torch.tensor([ref[cid] for cid in active], device=device)
            term = (2 / cfg["npo_beta"]) * F.softplus(cfg["npo_beta"] * (lp - r)).mean()
        else:
            term = -model(**fb).loss
        loss = term + (cfg["retain_alpha"] * model(**r_batch).loss if cfg["retain_alpha"] else 0)
        loss.backward()
        opt.step()
        opt.zero_grad()
        # roll back only on real retain DAMAGE (log-prob below a threshold), not
        # on an exact-match flicker — the brittle exact check made the loop give
        # up after 3 rollbacks with the forget secret barely touched
        r_lps = {c["id"]: metrics.suffix_logprob(model, tokenizer, c["prefix"], c["text"])
                 for c in watch}
        if any(r_lps[cid] < r0[cid] - cfg["retain_break_drop"] for cid in r0):
            model.load_state_dict(good)
            rollbacks += 1
            if rollbacks > cfg["max_rollbacks"]:
                break
            lr *= cfg["lr_decay_on_rollback"]
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
            continue
        good = snapshot(model)
        for cid in list(active):
            c = next(c for c in forget if c["id"] == cid)
            cont = metrics.greedy_continuation(model, tokenizer, c["prefix"], cfg["max_new_tokens"])
            leak = metrics.leaks_digits(c.get("secret_core", c["secret"]), cont, cfg["leak_min_digits"])
            flp = metrics.suffix_logprob(model, tokenizer, c["prefix"], c["text"])
            if not leak and flp <= baselines[cid]:
                active.remove(cid)
        if not active:
            break
    del opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"steps_run": step, "rollbacks": rollbacks, "unfinished": list(active)}


def _diag_unlearn(model, tokenizer, forget, cfg, baselines):
    """Diagnostic: separate direct-prompt leakage from paraphrase leakage so we
    can tell 'unlearning failed' from 'unlearning didn't generalize past the
    trained prompt'."""
    mx = cfg["max_new_tokens"]
    for c in forget:
        core = c.get("secret_core", c["secret"])
        direct = metrics.greedy_continuation(model, tokenizer, c["prefix"], mx)
        d_leak = metrics.leaks_digits(core, direct, cfg["leak_min_digits"])
        flp = metrics.suffix_logprob(model, tokenizer, c["prefix"], c["text"])
        paras = tiers.probe_prompts(c)[1:]  # exclude the direct prefix
        p_leak = sum(metrics.leaks_digits(
            core, metrics.greedy_continuation(model, tokenizer, p, mx), cfg["leak_min_digits"])
            for p in paras)
        print(f"    diag {c['id']:20s} direct_leak={d_leak!s:5} logp={flp:7.1f} "
              f"(base {baselines[c['id']]:7.1f}) paraphrase_leak={p_leak}/{len(paras)} "
              f"greedy={direct.strip()[:32]!r}")


def _save(model, tokenizer, path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def _run_tiers(unlearned_dir, gold_dir, tokenizer, canaries, cfg, device):
    forget = [c for c in canaries if c["split"] == "forget"]
    from transformers import AutoModelForCausalLM

    def load(p):
        return AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32).to(device)

    out = {}
    # Tier 1 — black-box
    u, g = load(unlearned_dir), load(gold_dir)
    out["tier1"] = {"unlearned": tiers.tier1_blackbox(u, tokenizer, forget, cfg),
                    "gold": tiers.tier1_blackbox(g, tokenizer, forget, cfg)}
    out["utility"] = {"tier1": {"unlearned": utility.natural_ppl(u, tokenizer),
                                "gold": utility.natural_ppl(g, tokenizer)}}
    del u, g
    if device == "cuda":
        torch.cuda.empty_cache()
    # Tier 2 — quantization family
    qu = tiers.tier2_quantization(unlearned_dir, tokenizer, forget, cfg, device)
    qg = tiers.tier2_quantization(gold_dir, tokenizer, forget, cfg, device)
    out["tier2"] = {"unlearned": qu["family_recovery"], "gold": qg["family_recovery"],
                    "backends": qu["backends"]}
    # Tier 3 — relearning (on fresh loads so the checkpoints stay intact)
    u, g = load(unlearned_dir), load(gold_dir)
    ru = tiers.tier3_relearning(u, tokenizer, canaries, forget, cfg, device)
    rg = tiers.tier3_relearning(g, tokenizer, canaries, forget, cfg, device)
    out["tier3"] = {"unlearned": ru["recovery"], "gold": rg["recovery"],
                    "steps_to_recover": {"unlearned": ru["steps_to_recover"],
                                         "gold": rg["steps_to_recover"]}}
    del u, g
    if device == "cuda":
        torch.cuda.empty_cache()
    out["n_prompts_per_canary"] = len(tiers.probe_prompts(forget[0])) * (1 + cfg["attack_samples"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== run_audit seed={seed} device={device} artifacts={ARTIFACTS} ===")
    start = time.time()

    # dataset is seed-independent (fixed data seed); only training varies by seed
    ds_cfg = dict(cfg, seed=cfg.get("data_seed", 42))
    data = gen_dataset(ds_cfg)
    canaries = data["canaries"]
    ckpt = ARTIFACTS / "checkpoints"

    print("[1] memorize"); _, tokenizer, model = load_model(cfg, device)
    baselines = {c["id"]: metrics.suffix_logprob(model, tokenizer, c["prefix"], c["text"])
                 for c in canaries}
    _train(model, tokenizer, [e["text"] for e in data["train_examples"]], cfg, device, seed)
    mem_dir = ckpt / f"memorized_seed{seed}"; _save(model, tokenizer, mem_dir)
    del model
    torch.cuda.empty_cache() if device == "cuda" else None

    print("[2] gold control"); _, _, gmodel = load_model(cfg, device)
    _train(gmodel, tokenizer, [e["benign_text"] for e in data["train_examples"]], cfg, device, seed)
    gold_dir = ckpt / f"control_seed{seed}"; _save(gmodel, tokenizer, gold_dir)
    del gmodel
    torch.cuda.empty_cache() if device == "cuda" else None

    from transformers import AutoModelForCausalLM
    for method in cfg["methods"]:
        print(f"[3] unlearn {method}")
        m = AutoModelForCausalLM.from_pretrained(mem_dir, torch_dtype=torch.float32).to(device)
        stats_u = _unlearn(m, tokenizer, canaries, method, cfg, device, baselines)
        print(f"    unlearn {method}: steps={stats_u['steps_run']} "
              f"rollbacks={stats_u['rollbacks']} unfinished={stats_u['unfinished']}")
        _diag_unlearn(m, tokenizer, [c for c in canaries if c["split"] == "forget"],
                      cfg, baselines)
        u_dir = ckpt / f"unlearned_{method}_seed{seed}"; _save(m, tokenizer, u_dir)
        del m
        torch.cuda.empty_cache() if device == "cuda" else None

        print(f"[4] tiers {method}")
        res = _run_tiers(u_dir, gold_dir, tokenizer, canaries, cfg, device)
        res.update({"method": method, "seed": seed})

        # optional advisory: Token Factory semantic-leakage score on the direct
        # greedy output of each forget canary (never gates the audit)
        if cfg.get("use_judge"):
            from src import judge
            if judge.available():
                m = AutoModelForCausalLM.from_pretrained(u_dir, torch_dtype=torch.float32).to(device)
                adv = []
                for c in [c for c in canaries if c["split"] == "forget"]:
                    out = metrics.greedy_continuation(m, tokenizer, c["prefix"], cfg["max_new_tokens"])
                    adv.append({"canary": c["id"], **judge.score(c["secret"], out)})
                res["advisory"] = {"semantic_leakage": adv}
                del m
                torch.cuda.empty_cache() if device == "cuda" else None
        res["manifest"] = make_manifest(f"tiers_{method}", seed=seed, config=cfg,
                                        dataset_path=None, checkpoint=str(u_dir),
                                        start=start, end=time.time(), outputs=[str(u_dir)])
        save_result(f"tiers_{method}_seed{seed}", res, config=cfg,
                    results_dir=str(ARTIFACTS / "results"))
    print(f"=== done in {time.time() - start:.0f}s ===")


if __name__ == "__main__":
    main()
