# Benchmark data

[DATASET_METRICS.csv](DATASET_METRICS.csv) contains 24 dataset/resolution/QP
aggregates for HEVC-B, UVG and MCL-JCV at 720p and 1080p. The evaluation covers
42 videos, four QPs and 336 video/resolution/QP cases.
The [evaluation specification](../TESTING.md) defines geometry, metrics and rate.

## Rate-distortion figure

The red solid curves in `assets/rd_perceptual_1080p.png` are RIFT measurements.
Dashed 1080p baseline curves are digitized published figure data, including GLC
on UVG. They retain the papers' original protocols and rate conventions.

[PLOTTED_POINTS.csv](PLOTTED_POINTS.csv) records the plotted values, method,
metric, dataset, resolution, rate kind and source. Its provenance labels identify
RIFT evaluation, baseline evaluation or published figures with figure/page
references. Published curves are not local reproductions. Points are not
smoothed, extrapolated or synthesized.

## Qualitative examples

Each method uses its own lowest-rate point. GLC rates marked with a star are
official entropy estimates; RIFT rates are actual rANS payload rates.
These examples are not matched-bitrate comparisons.

See [interactive comparisons](../comparisons/README.md) for frame indices,
shared crop coordinates and lossless images.
