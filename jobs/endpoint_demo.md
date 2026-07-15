# Stage 8 stretch — before/after Serverless Endpoints demo

Optional and **not run by default** (it deploys paid endpoints). The point is a
side-by-side extraction demo: the memorized ("before") vs unlearned ("after")
checkpoint served behind URLs, probed with the same deterministic Tier 1 prompts.

Only **parity-checked** results may feed a card: run the offline Tier 1 probes
and the endpoint probes on the same prompts and confirm they agree before using
the endpoint numbers. **Destroy the endpoints after recording the demo and cost.**

Sketch (a serving image such as vLLM/TGI pointed at the checkpoint in object
storage):

```bash
set -a; source .env; set +a
# deploy before/after
nebius ai endpoint create --name db-before --image <serving-image> \
  --platform gpu-l40s-d --preset 1gpu-16vcpu-96gb --subnet-id "$SUBNET_ID" \
  --volume "s3://$S3_BUCKET/checkpoints/memorized_seed42:/model" --container-port 8000
nebius ai endpoint create --name db-after  --image <serving-image> \
  --volume "s3://$S3_BUCKET/checkpoints/unlearned_ga_seed42:/model" ...

# probe both with the deterministic Tier 1 prompts, verify parity vs offline
python -m src.endpoint_probe --before <url> --after <url> --config configs/run.yaml

# ALWAYS destroy after recording
nebius ai endpoint delete --id <before-id>
nebius ai endpoint delete --id <after-id>
```

The offline Tier 1 result already demonstrates the same effect (memorized model
leaks; unlearned leaks less), so this stage is a presentation nicety, not part of
the core evidence.
