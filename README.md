# Z1 Phase Correlation w/  BigStitcher

This capsule runs Big Stitcher phase correlation to align z1 image tiles. It generates a nominal BigStitcher XML from the acquisition metadata, runs BigStitcher pairwise stitching and global optimization, writes alignment QC metrics, and creates channel-specific XML output.

<br>

**How to Run**

Reproducible Run

All capsule inputs are read from the metadata at PATH(data/) 

* **processing_manifest.json**
* **data_description.json**
* **acquisition.json**
* **processed/data_description.json**
* **radial_correction_parameters.json**
* **processed/** data folder

<br>

Primary BigStitcher outputs:

* **bigstitcher.xml** - final BigStitcher XML after phase correlation and global optimization
* **bdv_settings.xml** - BigDataViewer settings file with checkerboard tile coloring
* **solver_removed_links.csv** - links removed by the BigStitcher solver, parsed from logs

Alignment QC outputs:

* **pairwise_links.csv** - pairwise link table with shifts, correlation, overlap, and dropped-link status
* **corr_rank.png** - ranked correlation plot
* **shifts_all.png** - shift QC plot
* **bad_links_grid.png** - grid view showing weak or dropped links
* **max_projection.png** - nominal max-projection mosaic
* **max_projection_links.png** - max-projection mosaic with link overlays
* **dropped_links_metrics.txt** - dropped-link summary metrics

XML channel outputs:

* **combined_stitching_cam_alignment_all_channels.xml** - multichannel XML with stitching transforms transferred into the camera-aligned XML
* **single_channel_xmls/** - one XML per channel, split from the combined multichannel XML

<br>

**Optional block-affine refinement** (`--block_affine off|shadow|apply`, default `off`)

Refines the channel-405 BigStitcher placement with localized seam measurements
([issue #2](https://github.com/AllenNeuralDynamics/z1-ph-corr-bigstitcher/issues/2)). For every
face-sharing seam it phase-correlates a 3 x 3 grid of blocks (along the seam x depth) at
`--block_affine_pyramid_level` (default 2), keeps blocks that pass quality gates (peak inside the
search bounds, peak SNR >= 5, raw and high-pass correlation improve), and fits one Huber-robust,
ridge-regularized near-identity affine per tile anchored at the grid-center tile. The candidate is
checked with per-tile transform guardrails, graph connectivity, and a shifted 4 x 4 held-out block
grid measured under both the BigStitcher seed and the candidate.

* `off` - not run; outputs are identical to production.
* `shadow` - writes the candidate and QC to `block_affine/`; `bigstitcher.xml` is unchanged.
* `apply` - replaces `bigstitcher.xml` with the candidate only when every gate passes; otherwise
  keeps the BigStitcher seed. The alignment metrics and the combine/split step then use the
  selected placement.

The correction is composed into each tile's index-0 stitching transform (`C @ S`), so the downstream
combine step yields `C @ S @ camera @ nominal` for every channel with no extra resampling.

Block-affine outputs (`shadow`/`apply`):

* **bigstitcher_seed.xml** - BigStitcher placement before refinement
* **block_affine/decision.json** - mode, configuration, pass/fail reasons and selected placement
* **block_affine/bigstitcher_block_affine.xml** - candidate placement
* **block_affine/training_blocks.csv**, **training_seams.csv**, **training_measurements.csv** - seam measurements
* **block_affine/solution.json**, **cross_validation.csv**, **model_summary.csv** - fitted corrections and ridge selection
* **block_affine/tile_guardrails.csv**, **holdout_blocks_{seed,candidate}.csv**, **holdout_seams.csv** - QC
* **processing.json** - gains a "Block-affine tile refinement" data process (configuration as parameters,
  decision and held-out metrics as outputs), placed after the BigStitcher alignment in the dependency graph

Run on an existing XML outside the capsule: `cd code; python -m block_affine bigstitcher.xml out/ --anonymous`.
Tests: `python -m pytest tests` (golden reproduction of the validated Dandelion R3 level-2 solution
plus a synthetic end-to-end run). `--smoke_test 1` checks the environment without touching `/data`
or `/results`.

**Test-run options** (defaults keep production behaviour; all are App Panel parameters)

* `--write_s3_outputs 0` - keep `bdv_settings.xml`, `alignment_metrics/` and `processing.json` in
  `/results` instead of writing them to the dataset's S3 prefix.
* `--input_dataset_s3 s3://<bucket>/<processed asset>` - stage the inputs (acquisition, data
  description, processing manifest, camera-aligned XML) from an existing processed asset into
  `/scratch/staged_inputs` instead of reading `/data`.

<br>

