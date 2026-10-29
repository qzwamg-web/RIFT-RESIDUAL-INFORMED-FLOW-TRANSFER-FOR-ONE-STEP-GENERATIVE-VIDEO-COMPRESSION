# Installation

Use Python 3.10 or 3.11 and install a compatible CUDA-enabled PyTorch/torchvision
stack. The package dependencies are declared in `pyproject.toml`;
`requirements-lock.txt` provides a reference environment.

```bash
python -m pip install -e ".[metrics]"
python -m pip install pybind11 ninja einops easydict safetensors \
  diffusers transformers tokenizers ftfy opencv-python-headless \
  torchmetrics torch-fidelity scipy av
(cd third_party/DCVC/src/cpp && python setup.py build_ext --inplace)
python scripts/final.py --help
```

The rANS extension requires a C++17 compiler. Build artifacts are generated locally.

## Model and data setup

Training and inference require the complete [resource bundle](ASSETS.md).
Set `RAQVC_ARTIFACTS` to its parent directory and retain the supplied directory
structure. Configure the initializer paths in `configs/codec.yaml`.

The video paths in `configs/evaluation.json` are relative to the repository.
Place the datasets under `data/` or link an external dataset directory:

```bash
export RAQVC_ARTIFACTS=/path/to/resources
ln -s /path/to/video_datasets data
python scripts/final.py preflight
```

Dataset indices also contain shard and source-video paths; update these when
moving data. Preflight checks required files without downloading them.
LPIPS, DISTS and FID require their corresponding metric weights.

The reference 512-pixel training configuration uses approximately 75 GiB of GPU
memory. Run a smoke check before choosing the batch size:

```bash
python scripts/final.py smoke --devices 0 --output outputs/smoke
```
