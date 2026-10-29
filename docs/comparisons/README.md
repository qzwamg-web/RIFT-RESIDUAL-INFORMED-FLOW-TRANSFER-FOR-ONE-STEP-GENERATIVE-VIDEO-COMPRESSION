# Interactive visual comparisons

[Open the gallery](https://qzwamg-web.github.io/RIFT-RESIDUAL-INFORMED-FLOW-TRANSFER-FOR-ONE-STEP-GENERATIVE-VIDEO-COMPRESSION/comparisons/).

The gallery compares Original, GLC and RIFT on four 1080p examples:
Bosphorus frame66, MCL-JCV videoSRC12 frame41, MCL-JCV videoSRC03 frame89,
and BQTerrace frame0. Frame indices are zero-based.

Drag two dividers for three-way comparison or select a pairwise view.
Full-frame and detail views use identical coordinates across methods.
Mouse, touch, keyboard, fullscreen and lossless-image links are supported.

## Images and metadata

`data.json` records scene metadata, crop rectangles and operating points.
`PROVENANCE.json` records file hashes, source-image hashes and RGB pixel hashes.
The full-resolution WebP images are lossless; navigation thumbnails and README
contact sheets are separate previews.

Each method uses its own lowest-rate point: RIFT QP63 and GLC QP0.
GLC rates are official entropy estimates; RIFT rates are actual rANS payload
rates. These are qualitative examples rather than matched-bitrate comparisons.

## Local preview

```bash
python -m http.server 8000 --directory docs
# Open http://localhost:8000/comparisons/
```

GitHub Pages serves `main:/docs`. The page is buildless and uses no third-party
JavaScript dependencies. README cards link to the viewer because GitHub
sanitizes interactive scripts in README content.
