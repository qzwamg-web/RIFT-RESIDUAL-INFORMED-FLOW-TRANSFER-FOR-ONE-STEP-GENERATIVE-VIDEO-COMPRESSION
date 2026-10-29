# Model and dataset resources

Model weights and licensed datasets are not included in this repository.
Public checkpoint download links are not yet available. The training and
inference commands require a complete resource bundle.

| Component | Purpose |
|---|---|
| RIFT checkpoint | Trained codec, conditioning, restoration modules and LoRA parameters |
| Pretrained backbone | Matching video DiT base weights |
| Video VAE | Encoder and decoder weights |
| Image codec initializer | Intra-frame latent codec initialization |
| Video codec initializer | Predictive latent codec initialization |
| Training indices | 256- and 512-pixel latent-cache indices and shard locations |
| Metric weights | LPIPS, DISTS and FID dependencies |
| Source videos | HEVC-B, UVG and MCL-JCV for evaluation |

[MODEL_SHA256.json](MODEL_SHA256.json) identifies model components by checksum.
The RIFT checkpoint contains trained parameter updates and depends on its matching
pretrained backbone and VAE. Load it using the project loader.

## Configuration

`RAQVC_ARTIFACTS` identifies the parent of the supplied resource bundle.
Retain the bundle's directory structure and filenames expected by the loader.
`scripts/final.py preflight` checks required resources and reports missing paths.

Set codec initializer paths in `configs/codec.yaml`. The evaluation manifest
`configs/evaluation.json` uses paths under `data/` relative to the repository;
edit these paths or link an external dataset directory. Training index files
must point to the actual cache shards and source-video manifests.

The connection-head ablation requires its corresponding checkpoint and dedicated
training/evaluation entry points. See [training](TRAINING.md) and
[evaluation](TESTING.md) for usage.
