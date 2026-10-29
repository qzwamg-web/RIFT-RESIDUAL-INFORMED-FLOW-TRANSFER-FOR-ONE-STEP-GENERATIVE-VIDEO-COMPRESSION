# External assets

No checkpoints or datasets are stored in this Git branch. Set RAQVC_ARTIFACTS to
the directory containing C512_FINAL_v1; `scripts/final.py preflight` lists missing
assets without downloading them. Its default layout is:

```
C512_FINAL_v1/model/C512_original_bundle.pt
C512_FINAL_v1/model/stage3_dit_reproduction.pt
C512_FINAL_v1/model/wan/vae.pth
C512_FINAL_v1/model/wan/low_noise_model/diffusion_pytorch_model.bin
C512_FINAL_v1/model/codec_initializers/cvpr2025_image.pth.tar
C512_FINAL_v1/model/codec_initializers/cvpr2025_video.pth.tar
C512_FINAL_v1/datasets/256/index.json
C512_FINAL_v1/datasets/512/index.json
```

The original bundle is the trained codec/adapters/heads/LoRA delta. Stage3 and Wan
bases complete the model; do not load only the delta as a full pretrained DiT.
`MODEL_SHA256.json` records the retained local assets for integrity, not a public
download promise. Dataset index files reference external raw videos/shards by
path; update those paths when moving machines. See TRAINING.md for LR/batch and
TESTING.md for the42-video source list/protocol. Headless ablation is a separate
loader (`train_headless.py` / `evaluate_headless.py`).

Metric weights (LPIPS, DISTS, FID, FloLPIPS) are external assets. For FloLPIPS,
`prepare_flolpips_assets(download=False)` verifies the required pinned local files.
This code repository alone does not include licensed source video datasets.
