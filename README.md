<div align="center">

# RIFT
### Residual-Informed Flow Transfer for One-Step Generative Video Compression

**Learned latent compression · residual-informed restoration · one DiT forward per GOP**

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)](requirements-lock.txt)
[![Endpoint](https://img.shields.io/badge/Release-C512-0F766E)](docs/FRAMEWORK.md)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-42%20videos-8B5CF6)](docs/benchmarks/DATASET_METRICS.csv)

[Overview](#overview) · [Visual results](#visual-results) · [Benchmarks](#benchmarks) · [Quick start](#quick-start) · [Documentation](#documentation)

**English** | [简体中文](README.zh-CN.md)

</div>

![RIFT qualitative teaser](assets/teaser.png)

<p align="center"><em>Native1080p MCL-JCV examples. Each method uses its own lowest-rate operating point; starred baseline rates are entropy estimates. These are not matched-bitrate comparisons.</em></p>

## Overview

RIFT combines an entropy-coded latent video representation with a pretrained video flow model. Decoder-visible compression information conditions a **single whole-GOP DiT forward**. Residual strength, direction and gain modules produce a correction to the decoded latent before VAE reconstruction.

This repository publishes the preserved **C512 training/inference endpoint** of the ICLR2027 RIFT project and the retained headless ablation. Later UF-L/Wan and lightweight Lean VAE research branches are separate projects.

- **One-step restoration:** one pretrained Wan DiT call per latent GOP, without random-noise sampling or iterative denoising.
- **Residual-informed transfer:** codec/QP information, full-band residual strength and direction/gain adaptation.
- **Actual latent coding:** learned I/P transforms, regional quantization and rANS payload accounting.
- **Shared endpoint:** training and inference use the same restoration code and active inherited adapters.

![RIFT C512 framework](assets/framework.png)

<p align="center">Original architecture figure · <a href="assets/framework.pdf">PDF</a> · <a href="assets/framework.pptx">Editable PowerPoint</a></p>

The final latent has the form

$$\hat z = z_q + a\,d,$$

where $z_q$ is the decoded codec latent, $d$ is the adapted channel direction, and $a$ is the calibrated update magnitude.

| Codec QP | 0 | 16 | 39 | 63 |
|---|---:|---:|---:|---:|
| Wan time | 520 | 640 | 800 | 960 |

[Framework details](docs/FRAMEWORK.md) · [Quantization](docs/CODEC_QUANTIZATION.md)

## Visual results

![Qualitative comparison on UVG Bosphorus and HEVC-B BQTerrace](assets/qualitative.png)

The original sequence-bpp and displayed-frame LPIPS labels are retained. Each method uses its own lowest-rate point; starred rates are entropy estimates. The panel illustrates visual behavior rather than equal-rate performance.

## Benchmarks

![1080p perceptual rate-distortion curves](assets/perceptual_rd_1080.png)

**Solid red RIFT curves are preserved local C512 measurements. Dashed baseline curves are digitized published references**, with their original rate/evaluation conventions. They are not a unified local reproduction. [Figure provenance](docs/benchmarks/README.md).

These **QP63** rows come from the released [dataset metrics CSV](docs/benchmarks/DATASET_METRICS.csv).

| Resolution | Dataset | Videos | Payload bpp | PSNR ↑ | LPIPS-VGG ↓ | DISTS ↓ | Patch FID ↓ | FloLPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1080p | HEVC-B | 5 | 0.006201 | 26.1360 | 0.3042 | 0.0725 | 26.9219 | 0.1810 |
| 1080p | UVG | 7 | 0.005290 | 28.9444 | 0.2734 | 0.0792 | 25.9373 | 0.1488 |
| 1080p | MCL-JCV | 30 | 0.005376 | 28.0211 | 0.2946 | 0.0932 | 15.2820 | 0.1637 |
| 720p | HEVC-B | 5 | 0.006745 | 25.6649 | 0.3039 | 0.0783 | 36.5654 | 0.1828 |
| 720p | UVG | 7 | 0.005369 | 28.8629 | 0.2765 | 0.0867 | 35.2644 | 0.1575 |
| 720p | MCL-JCV | 30 | 0.005886 | 27.3766 | 0.3048 | 0.1024 | 22.6745 | 0.1709 |

The CSV contains **24 dataset/resolution/QP aggregates** covering **336 video/resolution/QP cases**. These are existing recorded results, not a new GPU evaluation performed for this GitHub publication.

<details>
<summary><strong>Preserved C512 evaluation protocol</strong></summary>

- HEVC-B: 5 videos; UVG: 7; MCL-JCV: 30; four QPs and 720p/1080p.
- Encode and score the first **96 genuine frames** in GOPs **25/25/25/21**; no repeated tail frames.
- 1080p: reflect-pad to 1920×1088 for coding, crop to 1920×1080 for scoring.
- 720p: center-crop original sources to 1280×736 for coding, then to 1280×720 for scoring; no resizing.
- Rate: actual rANS payload bits / (coded frames × coded-canvas pixels); historical container/header bytes are excluded.
- Float-domain spatial metrics, dataset-pooled dual-grid 256-patch  FID, and 95 adjacent-frame FloLPIPS pairs.

This historical RIFT96/GOP25 protocol is distinct from later 97-encoded/96-scored protocols. [Full contract](docs/TESTING.md).

</details>

## Quick start

### 1. Install

```bash
git clone https://github.com/qzwamg-web/RIFT-RESIDUAL-INFORMED-FLOW-TRANSFER-FOR-ONE-STEP-GENERATIVE-VIDEO-COMPRESSION.git
cd RIFT-RESIDUAL-INFORMED-FLOW-TRANSFER-FOR-ONE-STEP-GENERATIVE-VIDEO-COMPRESSION
conda create -n rift python=3.10 -y
conda activate rift
# Install a CUDA-compatible PyTorch / torchvision stack first.
python -m pip install -e ".[metrics]"
python -m pip install pybind11 ninja einops easydict safetensors \
  diffusers transformers tokenizers ftfy opencv-python-headless \
  torchmetrics torch-fidelity scipy av
(cd third_party/DCVC/src/cpp && python setup.py build_ext --inplace)
python scripts/final.py --help
```

`requirements-lock.txt` records the observed research environment, including CUDA-specific packages; adapt PyTorch wheels to your hardware. [Installation guide](docs/INSTALL.md).

### 2. Prepare external assets

**This is a source-code release. Checkpoints, pretrained backbone weights and source videos are external assets. No public C512 checkpoint download is configured in this repository.** [Required assets and hashes](docs/ASSETS.md).

```text
/path/to/raqvc_artifacts/
└── C512_FINAL_v1/
    ├── model/
    │   ├── C512_original_bundle.pt
    │   ├── stage3_dit_reproduction.pt
    │   ├── wan/...
    │   └── codec_initializers/...
    └── datasets/
        ├── 256/index.json
        └── 512/index.json
```

```bash
export RAQVC_ARTIFACTS=/path/to/raqvc_artifacts
# Rewrite shard/source paths inside dataset indices and configs/evaluation.json.
python scripts/final.py preflight
```

The bundle contains the trained codec, adapters, heads and LoRA deltas; Stage3/Wan bases and the VAE are also required. The original bundle is the numerical anchor. A merged BF16 LoRA export is not automatically equivalent.

### 3. Run

```bash
# Real 512/T7 training smoke with checkpoint save/reload.
python scripts/final.py smoke --devices 0

# Joint continuation from C512; --batch is per GPU.
python scripts/final.py train --devices 0,1,2,3,4,5 \
  --steps 512 --batch 1 --evaluate-after \
  --output "$RAQVC_ARTIFACTS/C512_FINAL_v1/new_runs/joint"

# Resume with the same original training configuration.
python scripts/final.py resume --devices 0,1,2,3,4,5 \
  --steps 512 --batch 1 --evaluate-after \
  --output "$RAQVC_ARTIFACTS/C512_FINAL_v1/new_runs/joint"

# Resumable 42-video evaluation; separate runtime measurements.
python scripts/final.py evaluate --devices 0 --resolutions 720 1080
python scripts/final.py speed --devices 0
```

The recorded 512/T7 joint smoke used approximately **74.4 GiB** peak GPU memory. Start with smoke on your hardware. These commands continue from C512 rather than replaying the complete historical schedule. [Training stages, batches and learning rates](docs/TRAINING.md).

Headless uses `scripts/train_headless.py` and `scripts/evaluate_headless.py`; its checkpoint must not use the standard C512 evaluator.

## Repository layout

```text
raqvc/
  backbone/       Wan, VAE, conditioning and active C512 adapters
  codec/          I/P transforms, regional quantization and entropy coding
  training/       Shared restoration endpoint, objectives and parameter groups
  data/           Video and 256/512 latent-cache loaders
  metrics/        Spatial, temporal and distributional perceptual metrics
scripts/final.py  preflight · smoke · train · resume · evaluate · speed
configs/          Codec configuration and 42-video evaluation manifest
docs/             Reproduction contracts, assets and benchmark provenance
assets/           Original paper figures, visual comparisons and R–D figure
third_party/      Vendored DCVC/Wan sources with upstream notices
```

## Documentation

| Topic | Reference |
|---|---|
| Installation | [docs/INSTALL.md](docs/INSTALL.md) |
| Model and dataset assets | [docs/ASSETS.md](docs/ASSETS.md) |
| Architecture | [docs/FRAMEWORK.md](docs/FRAMEWORK.md) |
| Training and learning rates | [docs/TRAINING.md](docs/TRAINING.md) |
| Video and metric protocol | [docs/TESTING.md](docs/TESTING.md) |
| Benchmark provenance | [docs/benchmarks/README.md](docs/benchmarks/README.md) |
| Quantization | [docs/CODEC_QUANTIZATION.md](docs/CODEC_QUANTIZATION.md) |
| Publication snapshot | [docs/PUBLICATION.md](docs/PUBLICATION.md) |
| Third-party attribution | [third_party/VERSIONS.md](third_party/VERSIONS.md) |

## Release scope and attribution

The runtime derives from isolated C512 source-release commit `c5e878731c9b2e6cd96607a9ac00231ed4e75a93`. `RELEASE_MANIFEST.json` records the publication snapshot. Runtime files retain their original hashes; documentation changes do not alter the numerical model.

DCVC/Wan retain original licenses/notices. Existing RIFT package metadata is **proprietary**; this publication does not invent a new license grant.

[Report a setup or reproduction issue](https://github.com/qzwamg-web/RIFT-RESIDUAL-INFORMED-FLOW-TRANSFER-FOR-ONE-STEP-GENERATIVE-VIDEO-COMPRESSION/issues).
