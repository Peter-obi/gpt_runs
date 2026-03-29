# Staleness Controls (774M)

This folder contains two SGD-only tests for momentum staleness hypotheses.

## Runs

1. Staleness-gated opposition (main hypothesis)

```bash
python train_drift_adaptive.py \
  config/train_gpt2.py \
  experiments/staleness_controls/train_gpt2_774m_sgdm_stalegate_a05_layerwise_relative.py
```

2. Fresh-gradient opposed control (expected worse)

```bash
python train_drift_adaptive.py \
  config/train_gpt2.py \
  experiments/staleness_controls/train_gpt2_774m_sgdm_freshsign_layerwise_relative.py
```

## Decision gate (compare against current best)

Current 774M baseline (`drift-power-opposed-layerwise-relative`, lambda=82):
- `val@250 = 5.9441`
- `val@500 = 5.2544`

Promote a new variant only if it beats both checkpoints at matched settings.
