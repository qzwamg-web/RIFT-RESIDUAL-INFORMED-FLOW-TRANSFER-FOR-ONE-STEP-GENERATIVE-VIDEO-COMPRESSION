# Latent codec quantization

RIFT distinguishes entropy scale `sigma`, dynamic transform scaling
`q_enc/q_dec` and regional quantization width `w`. Quantization operates on
the analysis-transform coefficients rather than directly on the VAE latent.

## Regional width

The intra- and predictive-frame branches each use a shared 3→32→1 MLP.
Its inputs are group-mean log entropy scale, group-mean squared mean prediction
and normalized QP. A width is shared by 16 channels at each spatial position.

```text
w = exp(log(2) * tanh(MLP(features)))
```

Widths lie in [0.5, 2]. The MLP is shared across groups and positions, with no
group embedding. The entropy model uses `sigma / w`.

## Intra-frame coefficients

`q_enc` and `q_dec` are spatial scalar maps and are predicted independently.

```text
symbol = round((y * q_enc - mu) / w)
y_hat = (w * symbol + mu) * q_dec
```

The analysis-coordinate interval is `w / q_enc`; the reconstruction interval
is `w * q_dec`.

## Predictive-frame coefficients

Predictive-frame `q_dec` varies across channels and spatial positions and includes
QP/frame-type/channel modulation.

```text
symbol = round((y / q_dec - mu) / w)
y_hat = (w * symbol + mu) * q_dec
```

Its effective reconstruction step is `q_dec * w`. Exported single-channel width
maps aggregate internal channel and coding-mask values and therefore do not
represent every coefficient's quantization step.
