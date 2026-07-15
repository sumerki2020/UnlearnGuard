#!/usr/bin/env bash
# Stage 8: the pinned 3-seed evidence run as ONE Nebius job that loops the seeds
# sequentially (concurrent jobs contend for L40S capacity — one allocation is
# faster and cheaper). Tier results are uploaded to object storage; aggregate
# them into a card afterwards with:
#   python -m src.artifacts download deletebench/evidence /tmp/ev && cp /tmp/ev/*.json results/
#   python -m src.audit --config configs/run.yaml && python -m src.plots
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

PROJECT=project-e00xe1astmf0nqmptj
IMAGE=cr.eu-north1.nebius.cloud/e00akapp5n1c6pcx3n/deletebench:smoke

nebius ai job create \
  --name deletebench-evidence \
  --parent-id "$PROJECT" \
  --image "$IMAGE" \
  --platform gpu-l40s-d --preset 1gpu-16vcpu-96gb \
  --subnet-id "$SUBNET_ID" \
  --env S3_ENDPOINT="$S3_ENDPOINT" --env S3_BUCKET="$S3_BUCKET" \
  --env S3_ACCESS_KEY_ID="$S3_ACCESS_KEY_ID" --env S3_SECRET_ACCESS_KEY="$S3_SECRET_ACCESS_KEY" \
  --container-command bash \
  --args "-lc 'for S in 42 1234 2026; do python -m src.run_audit --config configs/run.yaml --seed \$S; done && python -m src.artifacts upload results deletebench/evidence'" \
  --timeout 1h --async
