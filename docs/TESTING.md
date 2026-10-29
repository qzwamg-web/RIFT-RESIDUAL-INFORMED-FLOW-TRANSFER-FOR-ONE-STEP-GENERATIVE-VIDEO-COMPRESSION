# Evaluation protocol

## Datasets and temporal structure

Evaluate HEVC-B (5 videos), UVG (7) and MCL-JCV (30) at four QPs.
The reported 720p and 1080p results cover 336 video/resolution/QP cases.

Encode and score the first 96 real frames in GOPs of 25, 25, 25 and 21 frames.
The final GOP is encoded at its actual length without repeated tail frames.

## Spatial geometry

- 1080p: reflect-pad 1920×1080 RGB to 1920×1088 for coding; score the
  reconstructed 1920×1080 crop.
- 720p: center-crop the source to 1280×736 for coding; score its central
  1280×720 crop. Do not resize the source.

## Rate and quality

Rate is actual rANS payload bits divided by the number of encoded frames and
the encoded canvas area. Container and header bytes are excluded.

Compute PSNR, LPIPS-VGG, LPIPS-Alex and DISTS on floating-point reconstructions.
Preview videos are for visualization and are not used for scoring.

Patch FID uses 256×256 patches from the regular and 128-pixel-offset grids.
Pool feature statistics over each dataset before computing FID.
FloLPIPS covers all 95 adjacent pairs, including GOP boundaries; report undefined
pairs explicitly.

## Commands

```bash
python scripts/final.py evaluate --devices 0,1,2,3,4,5 \
  --resolutions 720 1080 --output outputs/evaluation
python scripts/final.py speed --devices 0 --output outputs/speed
```

Evaluation resumes at completed QPs after validating the checkpoint, input,
protocol and result files. Timing separates VAE encoding, entropy encoding and
decoding, conditioning, DiT/restoration and VAE decoding. The timing workload is
one 25-frame Beauty GOP and should be reported separately from dataset-wide
quality results.

External published curves retain their source protocols and rate conventions.
See [benchmark sources](benchmarks/README.md).
