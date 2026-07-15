#!/usr/bin/env bash
# Stage 8: launch the single-seed PROVISIONAL smoke as a Nebius Serverless Job.
# The image CMD runs run_audit + audit and prints the cards to stdout, so the
# result is retrievable from job logs without a mounted bucket.
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root (has .env)
set -a; source .env; set +a

PROJECT=project-e00xe1astmf0nqmptj
# registry path = registry id with the "registry-" prefix stripped
IMAGE=cr.eu-north1.nebius.cloud/e00akapp5n1c6pcx3n/deletebench:smoke

nebius ai job create \
  --name deletebench-smoke \
  --parent-id "$PROJECT" \
  --image "$IMAGE" \
  --platform gpu-l40s-d \
  --preset 1gpu-16vcpu-96gb \
  --subnet-id "$SUBNET_ID" \
  --timeout 40m
