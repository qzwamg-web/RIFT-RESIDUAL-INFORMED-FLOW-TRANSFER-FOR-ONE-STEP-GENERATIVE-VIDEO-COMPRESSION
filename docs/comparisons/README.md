# Interactive Original / GLC / RIFT comparisons

[Open the gallery](https://qzwamg-web.github.io/RIFT-RESIDUAL-INFORMED-FLOW-TRANSFER-FOR-ONE-STEP-GENERATIVE-VIDEO-COMPRESSION/comparisons/).

Four scenes reuse the existing 1080p paper frames: Bosphorus frame66,
MCL-JCV videoSRC12 frame41, MCL-JCV videoSRC03 frame89, and BQTerrace frame0.
The same scene, frame and crop coordinates are used for every method.

The page provides two draggable dividers for three-way comparisons, pairwise
modes, full-frame/detail views, native range inputs, keyboard arrows, touch
controls, fullscreen and links to lossless images. README cards link directly to
individual scenes because GitHub sanitizes interactive scripts in README content.

## Images and provenance

`data.json` contains public scene metadata, crop rectangles and operating points.
`PROVENANCE.json` records archived source paths, file hashes and decoded RGB hashes.
All full-resolution WebP files decode to exactly the RGB pixels of their source
PNGs. Thumbnails and README contact sheets are separate previews; full-frame
images are not resized, sharpened, recolored or regenerated.

The existing PNGs were reused; no raw YUV/video was decoded. These are archived
C512 visual examples, not a new evaluation or a `dcvc_float` reproduction.
Each method uses its own lowest-rate point. GLC QP0 rates are official entropy
estimates; RIFT C512 QP63 rates are actual rANS payload rates. These are not
matched-bitrate comparisons. The four detail crops retain the paper selection.

## Hosting and local preview

GitHub Pages publishes `main:/docs`; `.nojekyll` keeps this a buildless static
site. There are no third-party JavaScript dependencies, analytics, accounts,
backend services or API keys.

```bash
python -m http.server 8000 --directory docs
# Open http://localhost:8000/comparisons/
```

Mouse/touch/keyboard, crop alignment, all scene/mode combinations and responsive
layout checks are recorded in `CHECKS.json`. The first source publication's CPU
checks remain in `../PUBLICATION_CHECKS.json`; they do not claim new GPU results.
