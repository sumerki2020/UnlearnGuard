"""Extraction and memorization metrics.

Stage 2 subset, ported from the pilot (REPORT.md): greedy extraction, the
strict digit-leak check, secret suffix log-prob, sampled attack, perplexity.
Stage 3 adds Secret Sharer exposure and Min-K% Prob membership inference.
"""

import math
import re

import torch


@torch.no_grad()
def greedy_continuation(model, tokenizer, prefix, max_new_tokens=16):
    model.eval()
    inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def leaks_digits(secret, text, min_digits=4):
    """True if a run of min_digits consecutive secret digits appears in text.

    Secrets with fewer digits than min_digits (e.g. emails) fall back to an
    exact-substring check — the pilot showed exact match alone understates
    leakage for digit secrets, but it is the right test for string secrets.
    """
    sd = re.sub(r"\D", "", secret)
    if len(sd) < min_digits:
        return secret in text
    td = re.sub(r"\D", "", text)
    return any(sd[i : i + min_digits] in td for i in range(len(sd) - min_digits + 1))


@torch.no_grad()
def suffix_logprob(model, tokenizer, prefix, full_text):
    """Total log-prob of full_text's tokens after the shared prefix tokens."""
    model.eval()
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids[0]
    n = 0  # BPE can merge at the boundary, so count the shared token prefix
    while n < len(prefix_ids) and n < len(full_ids) and full_ids[n] == prefix_ids[n]:
        n += 1
    ids = full_ids.unsqueeze(0).to(model.device)
    logits = model(input_ids=ids).logits[0, :-1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    targets = ids[0, 1:]
    tok_lp = logp[torch.arange(targets.shape[0]), targets]
    return tok_lp[n - 1 :].sum().item()


@torch.no_grad()
def sampled_hits(model, tokenizer, prefix, secret, n_samples=20,
                 temperature=0.8, max_new_tokens=16):
    """Stronger attacker: sample continuations, count exact-secret hits."""
    model.eval()
    inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        num_return_sequences=n_samples,
        pad_token_id=tokenizer.pad_token_id,
    )
    start = inputs["input_ids"].shape[1]
    return sum(
        secret in tokenizer.decode(seq[start:], skip_special_tokens=True)
        for seq in out
    )


@torch.no_grad()
def perplexity(model, tokenizer, texts):
    """Token-weighted perplexity over a list of sentences."""
    model.eval()
    nll, ntok = 0.0, 0
    for t in texts:
        enc = tokenizer(t, return_tensors="pt").to(model.device)
        out = model(**enc, labels=enc["input_ids"])
        n = enc["input_ids"].shape[1] - 1
        nll += out.loss.item() * n
        ntok += n
    return math.exp(nll / ntok)


def random_secret_like(secret, rng):
    """Format-preserving random alternative: digits randomized, rest kept."""
    return "".join(str(rng.randrange(10)) if ch.isdigit() else ch for ch in secret)


@torch.no_grad()
def exposure(model, tokenizer, canary, pool_size, rng):
    """Secret Sharer exposure: log2 rank of the true secret among random
    format-preserving alternatives. Higher = more memorized; ~0 = chance.
    Low-digit formats (phones, emails) saturate their small pools — the pool
    is capped at the format's entropy and the cap is reflected in the score."""
    n_digits = sum(ch.isdigit() for ch in canary["secret"])
    max_pool = min(pool_size, 10 ** n_digits - 1)
    alts = set()
    while len(alts) < max_pool:
        alt = random_secret_like(canary["secret"], rng)
        if alt != canary["secret"]:
            alts.add(alt)
    true_lp = suffix_logprob(model, tokenizer, canary["prefix"], canary["text"])
    alt_lps = [
        suffix_logprob(model, tokenizer, canary["prefix"],
                       canary["text"].replace(canary["secret"], alt))
        for alt in alts
    ]
    rank = 1 + sum(lp >= true_lp for lp in alt_lps)
    return math.log2(len(alts) + 1) - math.log2(rank)


@torch.no_grad()
def mink_prob(model, tokenizer, text, k=0.2):
    """Min-K% Prob membership score: mean log-prob of the k% least likely
    tokens. Trained-on text scores higher (closer to 0) than unseen text."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    logits = model(input_ids=ids).logits[0, :-1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    targets = ids[0, 1:]
    tok_lp = logp[torch.arange(targets.shape[0]), targets]
    n = max(1, int(tok_lp.shape[0] * k))
    return tok_lp.topk(n, largest=False).values.mean().item()


def eval_canary(model, tokenizer, canary, eval_cfg):
    """One row of an extraction table: exact / leak / logprob / continuation."""
    cont = greedy_continuation(model, tokenizer, canary["prefix"],
                               eval_cfg.get("max_new_tokens", 16))
    return {
        "id": canary["id"],
        "type": canary["type"],
        "frequency": canary["frequency"],
        "split": canary["split"],
        "exact": canary["secret"] in cont,
        "leak": leaks_digits(canary.get("secret_core", canary["secret"]), cont,
                             eval_cfg.get("leak_min_digits", 4)),
        "logprob": suffix_logprob(model, tokenizer, canary["prefix"], canary["text"]),
        "continuation": cont.strip(),
    }
