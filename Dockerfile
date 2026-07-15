# CUDA base with torch preinstalled; must be linux/amd64 for Nebius.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace
COPY requirements.txt .
# torch comes from the base image; install the rest pinned
RUN grep -v '^torch$' requirements.txt > /tmp/reqs.txt \
    && pip install --no-cache-dir -r /tmp/reqs.txt

COPY src/ src/
COPY configs/ configs/
COPY schemas/ schemas/

# HF cache inside the image layer is not persisted; models download at run time.
# Artifacts (checkpoints, results, cards) go to $DELETEBENCH_ARTIFACTS, which the
# job points at a mounted bucket or local disk. Job disk is ephemeral: a run that
# needs its outputs kept must mount object storage or upload before exit.
# /workspace so that $DELETEBENCH_ARTIFACTS/results == the audit CLI's relative
# results dir; mount a bucket here to persist checkpoints/results/cards.
ENV DELETEBENCH_ARTIFACTS=/workspace
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

# Default: full single-seed audit, print cards to stdout, then persist all
# results to object storage (job logs expire; S3_* come from the job env).
ENV DELETEBENCH_RUN_PREFIX=deletebench/run
CMD ["bash", "-lc", "python -m src.run_audit --config configs/run.yaml --seed ${DELETEBENCH_SEED:-42} && python -m src.audit --config configs/run.yaml; echo '----- CARDS -----'; cat results/cards/card_*.md 2>/dev/null; python -m src.artifacts upload results ${DELETEBENCH_RUN_PREFIX} || true"]
