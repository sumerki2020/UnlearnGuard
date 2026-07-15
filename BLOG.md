# Deleted, Until Proven Otherwise: Can an LLM Really Forget?

*A Threat-Model-Aware Durability Audit for PII Unlearning in Language Models*

## TL;DR

- An LLM can stop saying a secret without becoming equivalent to a model that
  never learned it.
- This audit attacks “forgotten” synthetic PII, compares recovery with a matched
  gold control, and turns the result into a CI gate.
- On `Qwen2.5-0.5B-Instruct`, both tested unlearning methods still leaked under
  black-box prompts and recovered quickly after controlled re-exposure.

## An LLM says it forgot. Should you believe it?

Delete a database row and you can inspect the table. Ask an LLM to forget a
phone number and the evidence is much less satisfying: it stopped answering one
prompt. Is the number gone, or did the model simply learn not to say it that
way?

Machine unlearning promises a shortcut: remove selected training data without
retraining the model from scratch. But models do not stay frozen after the
unlearning run. Prompts change. Checkpoints get quantized. Teams continue
fine-tuning.

So “the model forgot” is incomplete. The real questions are:

1. Can an API user recover the secret?
2. Can someone with the weights recover it?
3. Can someone with the weights and a small compute budget recover it?

UnlearnGuard tests each claim separately. It is inspired by the right to
erasure, but it remains a technical audit—not a GDPR certification.

## Teach it. Erase it. Attack it.

```text
synthetic PII
    ├── train with secrets ──> memorized model ──> unlearn ──┐
    └── train with twins ───> gold control ─────────────────┤
                                                             ↓
                 prompt / quantize / relearn ──> Durability Card
```

The data is fake by design: fabricated names, reserved `555-01xx` phone
numbers, `example.com` addresses, badge IDs, and random recovery codes. No real
person appears in the experiment.

First, calibration canaries find a useful memorization point. Separate
evaluation canaries remain untouched until the final audit. The selected point
was 16×—enough to create a clear memorization gap without wrecking model
utility.

For every secret, the pipeline creates a **benign twin** with the same subject,
template, format, token budget, optimizer, steps, and seed—but a different
value. A second model trains on those twins. That model is the **gold control**:
same training experience, no audited secret.

## Turn up the attacker

### Tier 1 — Black-box API

The attacker gets an API, a subject, and a field—but not the secret. The audit
tries direct prompts, fixed paraphrases, greedy decoding, and seeded sampling.
It catches exact answers, partial digit leaks, elevated log-probability, and
exposure.

### Tier 2 — Static weights

Now the attacker gets the checkpoint. The intended test loads it at INT8 and
INT4, then launches the same probes again.

### Tier 3 — Weights plus compute

Finally, the attacker gets a small training budget. In **controlled
re-exposure**, the unlearned and gold models receive exactly the same data,
including forget examples, for exactly the same number of steps.

Supplying the secret and learning it again does not, by itself, prove residual
memory. What matters is the race: does the unlearned model recover faster or
more strongly than the gold control under identical conditions?

## A pass needs a real baseline

The central measurement is:

```text
excess recovery = recovery(unlearned model) - recovery(gold control)
```

The audit bootstraps over canaries and model seeds. It does not pretend that 20
paraphrases of one secret are 20 independent secrets.

The Durability Card reports recovery rates, a 95% confidence interval, sample
counts, utility, provenance, and one of four outcomes:

- **PASS:** the one-sided upper confidence bound is below an equivalence margin
  derived from gold-control variation.
- **FAIL:** the lower bound is above that margin.
- **INCONCLUSIVE:** the data cannot support either claim.
- **PROVISIONAL:** fewer than three model seeds were evaluated.

In other words, “we failed to find a difference” is not enough to pass.

## Results: the secrets came back

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

Both methods failed Tier 1 and Tier 3. The gold control stayed at zero under
black-box probing and below 0.3% after controlled re-exposure. The unlearned
models did not.

One seed tells a dangerously neat story. Across three seeds, that story broke:
one GA run showed no black-box recovery while the others still leaked. The wide
intervals in the chart are not decoration; they are the warning.

The experiment also exposed a weakness of naive Gradient Ascent. When a model
predicts a memorized token with near-perfect confidence, the cross-entropy
gradient approaches zero. Ironically, the secret learned most confidently can
be the hardest one to push out.

### Why Tier 2 is not reported

The saved Tier 2 artifacts say `fp16-fallback`, not `bitsandbytes`. The requested
INT8/INT4 backend did not run. Those values are not quantization evidence, even
if the configuration requested quantization.

The chart therefore says “not evaluated.” Configuration intent is not evidence
of execution.

### Limitations of this result

This is a focused experiment, not a verdict on all LLM unlearning:

- one 0.5B-parameter model and two unlearning methods;
- three seeds and three forget conditions;
- wide confidence intervals;
- controlled re-exposure, not every possible relearning attack;
- no valid quantization result yet.

The audit design is reusable. The numerical result is deliberately narrow.

## What is actually new?

Recovery after unlearning is established prior work:

- [Unlearning or Obfuscating?](https://arxiv.org/abs/2406.13356) studies benign
  relearning attacks.
- [Catastrophic Failure of LLM Unlearning via Quantization](https://arxiv.org/abs/2410.16454)
  demonstrates recovery after low-bit quantization.
- [UnlearnPII](https://doi.org/10.18653/v1/2025.nllp-1.6) evaluates explicit and
  implicit PII leakage.
- [OpenUnlearning](https://arxiv.org/abs/2506.12618) unifies implementations and
  evaluation metrics.

The recovery mechanisms are known. The new part is turning them into a
repeatable security decision:

1. Results are organized by **attacker capability**, not averaged into one
   recovery score.
2. Claims are measured against a matched model that never saw the secret.
3. The output is a schema-validated **Deletion Durability Card**, not only a
   research plot.
4. Strict mode turns the card into a CI gate: failed or inconclusive required
   claims produce a nonzero exit.
5. The complete audit runs from pinned configurations in a reproducible
   serverless container.

Prior research asks whether recovery happens. This tool asks what a deployer may
honestly claim—and whether that claim should pass CI.

## Why Nebius as infrastructure?

This workload is bursty: allocate a GPU, run an experiment matrix, save the
evidence, and shut everything down. A permanently running GPU would spend most
of its time idle. Nebius **Serverless AI Jobs** match that shape directly.

One container runs the complete L40S pipeline:

```text
generate → memorize → gold control → GA/NPO → capability tests → card
```

The three seeds run sequentially in one allocation, avoiding the capacity
contention seen with concurrent jobs. Before the ephemeral disk disappears, the
job uploads checkpoints, JSON, hashes, hardware metadata, and cards to
S3-compatible Nebius Object Storage.

An optional Nebius **Token Factory** judge evaluates semantic leakage. In one
test, `Qwen3-30B-A3B-Instruct-2507` flagged a partial phone-number disclosure
missed by exact matching. Because a hosted judge may change, its result is
advisory and never controls the hard CI verdict.

Individual runs take minutes, not hours. Development and the final evidence run
together cost less than $5.

## How to try it

Start with the tiny CPU fixture—no GPU or Nebius account required:

```bash
pip install -r requirements.txt
python -m src.run_audit --config configs/ci.yaml --seed 7
python -m src.audit --config configs/ci.yaml
PYTHONPATH=. python tests/test_audit.py
```

The audit command intentionally exits nonzero when strict mode finds a failed or
inconclusive claim. Deterministic known-pass and known-fail fixtures verify that
contract while keeping GitHub Actions green.

For the real Qwen run, switch to `configs/run.yaml` and the Nebius wrapper in
`jobs/`. Cards land in [`results/cards/`](results/cards/). Rebuild the chart
with:

```bash
python -m src.plots
```

## Takeaway

An LLM did not forget just because one prompt stopped working. “Deleted” is a
claim against an attacker with defined access.

UnlearnGuard makes that claim explicit, compares it with a model that never saw
the secret, reports uncertainty, and can fail a build when the evidence is not
good enough.

One friendly prompt is reassurance. A matched control, an attack ladder, and a
failing CI gate are evidence.

*Code and evidence: [github.com/sumerki2020/UnlearnGuard](https://github.com/sumerki2020/UnlearnGuard)*

#Nebius #ForgetMeNot #LLM #NebiusServerlessChallenge
