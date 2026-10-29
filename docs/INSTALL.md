# Installation and runtime assets

Use Python3.10/3.11 and a CUDA-compatible PyTorch/torchvision installation.
The root README contains installation/build commands. The observed environment
is recorded in `requirements-lock.txt`, including platform-specific CUDA packages.
The rANS extension requires a C++17 compiler; prebuilt binaries are not distributed.

Set `RAQVC_ARTIFACTS` to the **parent of** `C512_FINAL_v1`. Prepare [all assets](ASSETS.md),
rewrite dataset-index shard/source paths, and update `configs/evaluation.json`.
Environment variables do not rewrite absolute paths in manifests.

```bash
export RAQVC_ARTIFACTS=/path/to/raqvc_artifacts
export TORCH_HOME="$RAQVC_ARTIFACTS/C512_FINAL_v1/runtime_assets/torch_cache"
# Optional explicit download of pinned FloLPIPS assets:
python -c 'from raqvc.metrics.flolpips import prepare_flolpips_assets; prepare_flolpips_assets(download=True)'
python scripts/final.py preflight
python scripts/final.py smoke --devices 0
```

Other perceptual metrics require their own weights. Preflight checks retained
model/data assets and the pinned FloLPIPS cache; it does not certify GPU capacity
or every optional metric package. No public C512 checkpoint URL is configured.
Licensed datasets and checkpoints must be obtained separately.

The recorded512/T7 joint training smoke peaks at about74.4GiB. A CPU or single-GPU
check does not establish six-GPU readiness. `train` creates a new continuation
optimizer; `resume` restores a run created by this framework. [Training contract](TRAINING.md).
