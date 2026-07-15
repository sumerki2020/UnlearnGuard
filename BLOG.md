# DeleteBench: auditing whether "deleted" data is actually gone from an LLM

*Tagged #NebiusServerlessChallenge · Code: https://github.com/sumerki2020/DeleteBench*

## The problem

"Right to be forgotten" requests assume deletion is a thing you can *do* to a
trained model. Machine unlearning promises exactly that: take a fine-tuned LLM,
run a procedure, and the private data is gone. But is it? A growing body of work
shows that "unlearned" content often comes back — through relearning, prompt
paraphrasing, or even plain quantization. If you're a deployer about to tell a
user "we deleted your data," which version of that claim can you *honestly* make?

That depends entirely on **who can attack your model**. An attacker with only an
API is weaker than one holding your weights, who is weaker than one who can also
fine-tune. DeleteBench is an audit that answers the question *per attacker
capability* and hands you a **Deletion Durability Card**: a machine-readable,
statistically-grounded verdict you can attach to a deletion claim. It is an
RTBF-inspired *technical* audit — it deliberately does **not** claim GDPR
compliance.

## What it measures, and why a gold control matters

The naive way to test deletion is "prompt the model, did the secret come out?"
That's a trap: absence of a leak isn't proof of forgetting, and a lucky
non-leak proves nothing. DeleteBench instead measures **excess recovery**
relative to a matched **gold control** — a second model trained identically
(same base model, token budget, optimizer, steps, seed, neutral-data mix) but on
length-, format-, and template-matched *benign twins* instead of the real
secrets. The control has, by construction, never seen the secret, so its
recovery rate is the honest baseline for "a model that legitimately doesn't
know." The audited claim is then:

```
excess_recovery = recovery(unlearned) − recovery(gold_control)
```

We estimate it with a hierarchical bootstrap whose resampling unit is the
*canary*, not the prompt (treating correlated paraphrases as independent would
fake precision). PASS requires the one-sided 95% upper bound of excess to fall
below an **equivalence margin derived from control-to-control variation** —
never a magic threshold. Anything else is FAIL or INCONCLUSIVE. It's the
difference between "we couldn't find a leak" and "leakage is statistically
indistinguishable from a model that never learned the secret."

## Three attacker tiers

The whole audit — code, JSON, plots, card — is organized by adversary
capability:

- **Tier 1 — black-box API.** Direct prompts plus fixed paraphrases, greedy and
  sampled decoding. The attacker knows the subject and query prefix, not the
  secret.
- **Tier 2 — white-box static.** The attacker has the weights and loads them at
  INT8/INT4. Quantization is a known way for "forgotten" content to resurface.
- **Tier 3 — white-box + compute.** The attacker can spend a small fine-tuning
  budget (constrained relearning), applied identically to the unlearned and gold
  models.

## The result

Auditing `Qwen/Qwen2.5-0.5B-Instruct` with two required unlearning methods —
Gradient Ascent and a hand-rolled NPO — across three seeds, **both FAIL all
three tiers**, and recovery escalates cleanly with attacker capability (gold
control leaks 0% throughout):

![evidence](results/evidence_recovery.png)

Black-box recovery already runs ~23–34% above control; relearning resurrects the
"deleted" secret to ~52–62%. The seed variance was load-bearing — one seed fully
unlearned every canary while others left secrets memorized — which is exactly
why the card reports confidence intervals over multiple seeds, not point
estimates. A nice illustration of *why* unlearning is fragile fell out of the
data: gradient ascent gets almost no traction on a *perfectly* memorized secret
(its gradient vanishes when the model is fully confident), so the most memorized
canary is the one that survives deletion.

**These recovery phenomena are not novel** — they echo published work on
relearning recovery, quantization recovery, and PII unlearning benchmarks. The
contribution here is the reusable, control-relative *audit harness* and its
verdict artifact, not the discovery that unlearning leaks.

## How it uses Nebius Serverless

The whole pipeline runs as a **Serverless AI Job** on one L40S GPU: memorize →
gold control → GA/NPO unlearn → three tiers → card, ~4 minutes of compute per
seed. Because job disks are ephemeral, every run persists checkpoints, tier
results, and cards to **Nebius object storage** (S3-compatible) via a service
account key. An optional advisory step calls a **Token Factory** model
(`Qwen3-30B-A3B`, OpenAI-compatible endpoint) to score *semantic* leakage — and
in testing it caught a partial phone-number reveal that the mechanical
digit-match check missed. A practical lesson: three concurrent GPU jobs
contended for capacity and stalled, so the multi-seed run is a single job that
loops the seeds — one allocation, no contention. **The entire project — every
debugging iteration plus the final evidence run — cost under $5.**

## Reproduce it

The full pipeline runs on CPU in about 20 seconds on a tiny fixture, so you can
see the whole thing — canaries, unlearning, tiers, card, strict CI exit — before
touching a GPU:

```bash
pip install -r requirements.txt
DELETEBENCH_ARTIFACTS=. python -m src.run_audit --config configs/ci.yaml --seed 7
python -m src.audit --config configs/ci.yaml
```

A GitHub Actions workflow exercises the same pipeline plus deterministic
known-PASS / known-FAIL card fixtures on every push. The real audit is the same
two commands with `configs/run.yaml` on a Nebius L40S.

## Takeaway

Deletion in LLMs isn't a boolean — it's a claim that only holds against a
specific attacker. DeleteBench makes that claim measurable, control-relative,
and reproducible, and packages it as a card you can actually put next to a
"we deleted your data" statement. Try it, point it at your own unlearning setup,
and see which claim you can honestly make.

*Code, evidence cards, and figures: https://github.com/sumerki2020/DeleteBench*
