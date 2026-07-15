# Deleted, Until Proven Otherwise

*A Threat-Model-Aware Durability Audit for PII Unlearning*

*[Code and reproducible evidence](https://github.com/sumerki2020/UnlearnGuard)*

## TL;DR

- Machine unlearning can make private data harder to retrieve without actually
  making the model behave like one that never saw the data.
- **UnlearnGuard** audits deletion claims against three levels of access:
  black-box prompting, static model weights, and weights plus fine-tuning.
- It compares every unlearned model with a matched **gold control** that never
  saw the audited secret.
- On `Qwen2.5-0.5B-Instruct`, both Gradient Ascent and NPO failed the valid
  black-box and controlled-re-exposure tests across three seeds.
- The current INT8/INT4 result is deliberately excluded: the saved run used an
  FP16 fallback, so it is not quantization evidence.
- The new contribution is the reusable audit, control-relative gate, Durability
  Card, CI contract, and Nebius packaging—not the discovery that recovery is
  possible.

## The problem: what does “deleted” mean for a model?

Deleting a database row is concrete. Deleting information encoded across model
weights is not.

Machine unlearning modifies a trained model so that selected examples become
less influential. A common evaluation asks the original question again and
checks whether the model still answers. That tests one prompt against one
checkpoint. It does not test what happens after paraphrasing, quantization, or
continued training.

A useful deletion claim must therefore name its attacker:

1. Can an API user recover the secret?
2. Can someone with the weights recover it?
3. Can someone with the weights and a small compute budget recover it?

UnlearnGuard turns those questions into a reproducible audit. It is inspired by
the right to erasure, but it is a technical test—not a GDPR certification.

## The audit in one pipeline

```text
synthetic PII
    ├── train with secrets ──> memorized model ──> unlearn ──┐
    └── train with twins ───> gold control ─────────────────┤
                                                             ↓
                 prompt / quantize / relearn ──> Durability Card
```

All “private” data is synthetic: fabricated names, reserved `555-01xx` phone
numbers, `example.com` addresses, badge IDs, and random recovery codes.

Calibration canaries select one memorization operating point. Separate
evaluation canaries produce the final evidence. The selected point was 16×:
strong enough to create a clear memorization gap while keeping the model useful.

For every secret, the pipeline creates a **benign twin** with the same subject,
template, format, token budget, optimizer, steps, and seed—but a different
value. A second model trains on those twins. This gold control underwent the
same training process without seeing the audited secret.

## Three capability tiers

### Tier 1 — Black-box API

The attacker knows the synthetic subject and requested field, but not the
secret. The audit uses direct and fixed paraphrased prompts with greedy and
seeded sampled decoding. Exact recovery, partial digit leakage, log-probability,
and exposure are measured.

### Tier 2 — Static weights

The attacker or deployer has the checkpoint but cannot train it. The intended
test loads the model at INT8 and INT4, then repeats the probes.

### Tier 3 — Weights plus compute

The attacker can perform a small, fixed fine-tuning run. The current protocol is
**controlled re-exposure**: both the unlearned and gold models receive the same
data—including forget examples—for the same number of steps.

Supplying the secret and learning it again does not, by itself, prove residual
memory. The relevant signal is whether the unlearned model recovers more easily
than the gold control under identical re-exposure.

## A control-relative deletion gate

The central measurement is:

```text
excess recovery = recovery(unlearned model) - recovery(gold control)
```

The audit uses a hierarchical bootstrap over canaries and model seeds. Prompts
are not treated as independent observations because paraphrases of the same
canary are correlated.

The Durability Card reports recovery rates, a 95% confidence interval, sample
counts, utility, provenance, and one of four outcomes:

- **PASS:** the one-sided upper confidence bound is below an equivalence margin
  derived from gold-control variation.
- **FAIL:** the lower bound is above that margin.
- **INCONCLUSIVE:** the data cannot support either claim.
- **PROVISIONAL:** fewer than three model seeds were evaluated.

This avoids the common mistake of treating “no statistically significant
difference” as proof that two models are equivalent.

## MVP results

The evidence run used:

- `Qwen/Qwen2.5-0.5B-Instruct`;
- Gradient Ascent (GA) and a small hand-written NPO implementation;
- seeds `42`, `1234`, and `2026`;
- three held-out forget conditions per seed;
- 84 direct, paraphrased, greedy, and sampled probes per canary.

| Capability | GA excess recovery (95% CI) | NPO excess recovery (95% CI) |
| --- | ---: | ---: |
| Black-box prompting | +22.9% [+0.7%, +50.7%] | +34.3% [+9.9%, +60.5%] |
| Static weights, INT8/INT4 | Not established | Not established |
| Controlled re-exposure | +52.1% [+18.5%, +100.0%] | +61.6% [+29.9%, +92.9%] |

![Deletion durability by attacker capability](results/evidence_recovery.png)

Both methods failed Tier 1 and Tier 3. Gold-control recovery was zero under
black-box probing and below 0.3% after controlled re-exposure.

The variance matters. One GA seed showed no black-box recovery while other seeds
still leaked. A single run could therefore have produced a confident but
misleading conclusion.

The experiment also exposed a weakness of naive Gradient Ascent. When a model
predicts a memorized token with near-perfect confidence, the cross-entropy
gradient approaches zero. The best-memorized secret can become surprisingly
difficult to move.

### Why Tier 2 is not reported

The saved Tier 2 artifacts say `fp16-fallback`, not `bitsandbytes`. The requested
INT8/INT4 backend did not run. Those values are not quantization evidence, even
if the configuration requested quantization.

The chart marks Tier 2 as not evaluated. A future audit run should automatically
make this tier `INCONCLUSIVE` whenever its required backend is unavailable.

## What is new?

Recovery after unlearning is established prior work:

- [Unlearning or Obfuscating?](https://arxiv.org/abs/2406.13356) studies benign
  relearning attacks.
- [Catastrophic Failure of LLM Unlearning via Quantization](https://arxiv.org/abs/2410.16454)
  demonstrates recovery after low-bit quantization.
- [UnlearnPII](https://doi.org/10.18653/v1/2025.nllp-1.6) evaluates explicit and
  implicit PII leakage.
- [OpenUnlearning](https://arxiv.org/abs/2506.12618) unifies implementations and
  evaluation metrics.

UnlearnGuard contributes a practical security-testing layer:

1. Results are organized by **attacker capability**, not averaged into one
   recovery score.
2. Claims are measured against a matched model that never saw the secret.
3. The output is a schema-validated **Deletion Durability Card**, not only a
   research plot.
4. Strict mode turns the card into a CI gate: failed or inconclusive required
   claims produce a nonzero exit.
5. The complete audit runs from pinned configurations in a reproducible
   serverless container.

The research asks whether recovery happens. This tool asks what a deployer may
honestly claim after testing it.

## Nebius Serverless implementation

The evidence pipeline runs as one Nebius **Serverless AI Job** on an L40S GPU:

```text
generate → memorize → gold control → GA/NPO → capability tests → card
```

The three seeds run sequentially in one job to avoid GPU-capacity contention.
Checkpoints, JSON results, configuration hashes, seeds, hardware metadata, and
cards are persisted to S3-compatible Nebius Object Storage before the ephemeral
job disk disappears.

An optional Nebius **Token Factory** judge evaluates semantic leakage. In one
test, `Qwen3-30B-A3B-Instruct-2507` flagged a partial phone-number disclosure
missed by exact matching. Because a hosted judge may change, its result is
advisory and never controls the hard CI verdict.

Individual runs take minutes. Development and the final evidence run together
cost less than $5.

## Reproduce it

The CPU fixture exercises the pipeline without renting a GPU:

```bash
pip install -r requirements.txt
python -m src.run_audit --config configs/ci.yaml --seed 7
python -m src.audit --config configs/ci.yaml
PYTHONPATH=. python tests/test_audit.py
```

The audit command intentionally exits nonzero when strict mode finds a failed or
inconclusive claim. Deterministic known-pass and known-fail fixtures verify that
contract while keeping GitHub Actions green.

The real Qwen run uses `configs/run.yaml` and the Nebius wrapper in `jobs/`.
Cards are written to [`results/cards/`](results/cards/). Regenerate the chart
with:

```bash
python -m src.plots
```

## Limits

This is an MVP, not a universal unlearning benchmark:

- one 0.5B-parameter model;
- two unlearning methods;
- three seeds and three forget conditions;
- wide confidence intervals;
- controlled re-exposure rather than every possible relearning threat;
- no valid quantization result yet.

These limits narrow the claim; they do not invalidate the audit design.

## Takeaway

“Deleted” is not a property of a model in isolation. It is a claim against an
attacker with defined access.

UnlearnGuard makes that claim explicit, compares it with a model that never saw
the secret, reports uncertainty, and can fail a build when the evidence is not
good enough.

That is more useful than asking one friendly prompt and calling the answer
forgotten.

*Code and evidence: [github.com/sumerki2020/UnlearnGuard](https://github.com/sumerki2020/UnlearnGuard)*

#NebiusServerlessChallenge
