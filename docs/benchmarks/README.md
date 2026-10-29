# Benchmark and figure provenance

`DATASET_METRICS.csv` is an unchanged copy of the preserved C512 first96 formal
summary:24 rows, three datasets × four QPs × two resolutions, covering336
video/resolution/QP cases. The root README table is checked against this CSV.

[Historical RIFT protocol](../TESTING.md):96 genuine encoded/scored frames,
GOP25/25/25/21, float metrics and coded-canvas rANS **payload** bpp excluding
container/header bytes. This is distinct from later97-encoded/96-scored protocols.
HEVC-B/UVG/MCL-JCV have5/7/30 videos. FID pools dual-grid256patch statistics per dataset.

## R–D figure

The preserved `rd_revision_v6/RIFT_1080P_perceptual_3datasets.png` is copied unchanged
to `assets/perceptual_rd_1080.png`. Red solid RIFT curves are local measurements.
All dashed1080p baselines, including GLC on UVG, are digitized published figures
with their original rate/evaluation conventions. They are not local reproductions.

`PLOTTED_POINTS.csv` retains values, sources and rate kinds; `FIGURE_SOURCE_NOTES.md`
preserves the original construction notes. Local paths in provenance fields are
source records, not bundled files or download URLs. No GPU run/curve was regenerated.

## Qualitative panels

The unchanged teaser/comparison previews come from `subjective1080/ppt/preview`.
Each method uses its own lowest-rate point; starred bpp values are entropy estimates.
These are not equal-rate or identical-input-protocol comparisons.

Media hashes/source paths are recorded in [`assets/PROVENANCE.json`](../../assets/PROVENANCE.json).
The framework figure is exported from the existing architecture PowerPoint in the
data archive; its original editable file is included as `assets/framework.pptx`.
