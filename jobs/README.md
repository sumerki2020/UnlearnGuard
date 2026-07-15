# Nebius Jobs wrappers (Stage 7)

Thin per-stage submission scripts land here at Stage 7. Reference command:

```
nebius ai create --type job --name <stage> \
  --image <registry>/unlearn:latest --container-command bash \
  --args "-c 'python src/<script>.py --config configs/<cfg>.yaml'" \
  --platform gpu-l40s-a --preset 1gpu-8vcpu-32gb --timeout 1h
```

Job disk is released on completion — every job must write checkpoints/results
to persistent storage or push to HF Hub before exiting.
