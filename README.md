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

