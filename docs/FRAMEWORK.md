# RIFT architecture

RIFT compresses video in the latent space of a frozen video VAE and restores
the decoded representation with one video DiT forward pass per GOP.

## Compression

The VAE maps RGB frames to a 48-channel latent representation. Learned intra-
and predictive-frame transforms, regional quantization and entropy coding
produce a bitstream. Entropy decoding and synthesis yield `z_q`.

Decoder-visible conditions include entropy scales, quantization widths, QP
and latent frame role. The restoration model uses these conditions without
access to the original video.

## Residual-informed flow transfer

1. A multi-scale estimator predicts local error features and strength.
2. A fixed QP-to-time lookup conditions a single whole-GOP DiT call.
3. Direction adaptation and base gain modulation produce a base correction.
4. A full-band RMS estimator supplies local residual-strength information.
5. Angular direction refinement and local magnitude modulation form the update.

For a base update `u_0`, unit direction `d` and local magnitude factor `m`:

```text
n = ||u_0||_2,channel
m = exp(log(2) * tanh(ell))
a = n * m
z_hat = z_q + a * d
```

Direction is a 48-dimensional unit vector at each latent position. Magnitude
is a local scalar map. The VAE decoder maps the restored latent to RGB.

| QP | 0 | 16 | 39 | 63 |
|---|---:|---:|---:|---:|
| DiT time | 520 | 640 | 800 | 960 |

Training and inference share [the restoration implementation](../raqvc/training/restoration.py).
The model uses fixed times, independent GOPs and rank-8 LoRA adaptation.
