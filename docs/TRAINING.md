# Training

## Data

The reference training indices contain 83,368 source videos and 166,736 windows
at each spatial resolution, 256 and 512 pixels. A window contains 25 RGB frames
and seven temporally compressed latent frames.

Each index contains window offsets, spatial transforms and cache-shard paths.
Prepare the source videos, cache shards and manifests before running preflight.

## Optimization stages

| Stage | Resolution | Steps | Batch per GPU | GPUs |
|---|---:|---:|---:|---:|
| Restoration adaptation | 512 | 4096 | 1 | 6 |
| Joint optimization | 256 | 2304 | 3 | 6 |
| High-resolution refinement | 512 | 512 | 1 | 6 |

| Parameter group | Adaptation LR | Joint LR | Refinement LR |
|---|---:|---:|---:|
| Codec transforms | Frozen | 2e-6 | 1e-6 |
| Regional quantizer | Frozen | 1e-5 | 5e-6 |
| Residual strength | 2e-5 | 3e-5 | 2e-5 |
| Gain modulation | 3e-5 | 3e-5 | 2e-5 |
| Direction refinement | 1e-5 | 1e-5 | 1e-5 |
| Conditioning adapters | Frozen | 5e-6 | 3e-6 |
| Rank-8 LoRA | 5e-6 | 1e-5 | 5e-6 |

The optimizer is AdamW with weight decay 0.01, gradient clipping at 1.0,
warmup and cosine decay to 10% of the starting learning rate. The four QPs
are cycled during training.

Supervision includes RGB reconstruction and perceptual terms, temporal
consistency, latent reconstruction, residual-strength and direction objectives,
rate, and a codec reconstruction auxiliary term.

## Fine-tuning and resuming

The unified entry point fine-tunes a supplied RIFT checkpoint using the
high-resolution refinement configuration:

```bash
python scripts/final.py train --devices 0,1,2,3,4,5 \
  --steps 512 --batch 1 --output outputs/joint
python scripts/final.py resume --devices 0,1,2,3,4,5 \
  --steps 512 --batch 1 --output outputs/joint
```

Use the same total steps, batch size, device count and output directory when
resuming. `--batch` denotes batch size per GPU. A new training run initializes
a new optimizer; resume restores the optimizer and schedule from that run.

## Connection-head ablation

`scripts/train_headless.py` and `scripts/evaluate_headless.py` implement the
ablation without restoration connection heads. It jointly fine-tunes the codec,
regional quantizer, LoRA and conditioning adapters with learning rates
1e-6, 5e-6, 3e-5 and 3e-6, respectively. Use its own checkpoint loader and evaluator.
