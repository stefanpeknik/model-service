# Tissue mask for the first full-slide heatmap run

Historical input-research and staging record. Pre-deployment capacity/readability
statements below describe that session; subsequent deployment and successful
real-input validation are linked from [the documentation map](../README.md).
Retain the mask provenance, preparation method, and limitations as evidence.
The `downloads/` files referenced below are local, ignored validation inputs and
previews; they are not included in a repository clone. The preparation script
requires the OpenSlide native library and ImageMagick's `magick` executable.

## Finding

The [RationAI Quality Control service](https://quality-control-rationai-digital-pathology-quali-82f7255ed88b44.gitlab-pages.ics.muni.cz/interactions/processing_wsis/) creates a tissue mask internally to filter tiles. Its `mask_level` setting controls the WSI level used for that mask. The public documentation does **not** describe exporting this tissue mask as a result. It describes writing the selected QC check outputs into `output_path` and returning an xOpat inspection link. The service's documented endpoint is in `rationai-jobs-ns`, and the documentation says it is accessible only from the secure cluster. We should not rely on it to obtain this input without inspecting its source or a sample result, and there is no reason to call that other namespace for our first run.

The QC TIFFs that *are* documented are not interchangeable with a tissue ROI. The [residual coverage mask](https://quality-control-rationai-digital-pathology-quali-82f7255ed88b44.gitlab-pages.ics.muni.cz/qc-checks/residual/) is nonzero where it detects artifacts. The [folding mask](https://quality-control-rationai-digital-pathology-quali-82f7255ed88b44.gitlab-pages.ics.muni.cz/qc-checks/folding/) also marks artifacts. The [blur masks](https://quality-control-rationai-digital-pathology-quali-82f7255ed88b44.gitlab-pages.ics.muni.cz/qc-checks/blur/) describe image activity or focus, not all tissue. These masks are downsampled TIFFs, but their pixel meaning is wrong for this workflow.

The local [heatmap processor](../../workflows/heatmap/processor.py) opens `region_of_interest` as a slide and scales mask coordinates by the ratio between its level dimensions and the WSI level dimensions. The [tile reader](../../workflows/heatmap/tile_reader.py) skips a classifier call only if the corresponding grayscale mask tile is entirely zero. Thus an aligned, downsampled, single-channel TIFF with zero background and nonzero tissue should work; matching full-resolution pixel dimensions are unnecessary. The mask should represent the complete WSI canvas, with no crop, rotation, or offset. A pyramid and explicit resolution metadata are prudent because the processor calls `closest_level(target_mpp)` on the mask. We still need to verify that OpenSlide recognizes the generated TIFF and that sampled mask regions align with visible tissue before submitting the expensive run.

## Recommended input preparation

Generate a tissue-versus-background mask locally from a low-resolution level of the downloaded [Mirax WSI](../../downloads/wsi/962ea217-0fd7-4ddd-a1a0-165c89056cef/README.md). Keep the `.mrxs` entry file and companion directory together. Write the mask as a tiled TIFF covering the same slide extent, with values `0` for background and `255` for tissue and with pixel resolution corresponding to the chosen WSI level. Inspect a thumbnail overlay and several edge regions to catch missed pale tissue, pen marks, and alignment errors. Then stage the WSI and mask on storage mounted by the scheduler and Ray workflow worker in `peknik-ns`.

This preparation is not a smaller inference run. The first `prostate-heatmap` execution can still process the full WSI. The mask changes which tiles the classifier sees, so its visual review matters for interpreting the output. The downloaded WSI package contains no tissue-mask TIFF, according to its [download notes](../../downloads/wsi/962ea217-0fd7-4ddd-a1a0-165c89056cef/README.md).

## Prepared input, 2026-10-02

[`scripts/prepare_tissue_mask.py`](../../scripts/prepare_tissue_mask.py) reads the WSI through OpenSlide at level 7, composites transparent pixels over white, selects stained or dark pixels, removes isolated noise, and expands the selection slightly around tissue. It writes a tiled, ZIP-compressed binary TIFF with resolution metadata derived from the slide. This is a tile-selection mask, not a diagnostic tissue segmentation.

The prepared local mask is `downloads/wsi/962ea217-0fd7-4ddd-a1a0-165c89056cef/tissue-mask-open-slide.tiff`. Its [preview](../../downloads/wsi/962ea217-0fd7-4ddd-a1a0-165c89056cef/tissue-mask-open-slide-preview.png) overlays the selected region on the same OpenSlide level. The mask is 697 by 1599 pixels, covers 2.87% of that canvas, contains only 0 and 255, and has SHA-256 `64cb47eec2737bd0441753f7646cead8f169146b90d9c5efc135c7db67dc0806`. OpenSlide 4.0.1 opens both the original WSI and generated mask; sample tissue and background reads returned 255 and 0 respectively.

The full WSI package and mask were copied to `/mnt/validation/wsi/962ea217-0fd7-4ddd-a1a0-165c89056cef/` on the `model-service-validation-shared` PVC in `peknik-ns`. All 25 source-file checksums passed on that PVC, and the copied mask hash matches the local one. A standalone SDK request should use `InputAccess(kind="shared_storage", state={"root": "/mnt/validation"})`, with these relative inputs:

| Input | Path |
| --- | --- |
| `slide` | `wsi/962ea217-0fd7-4ddd-a1a0-165c89056cef/67df0411-395d-5d73-9c2f-3e0ffef45491.mrxs` |
| `region_of_interest` | `wsi/962ea217-0fd7-4ddd-a1a0-165c89056cef/tissue-mask-open-slide.tiff` |

The shared PVC is provisioned at 1 GiB and now holds about 335 MiB of input. That is enough for the staged files but may not be enough for the final heatmap. Increase or replace the output storage before treating a full-slide run as ready. The running validation RayService still uses the dummy heatmap processor; staging these inputs did not start a workflow.

## Remaining uncertainty

The QC service may save an undocumented tissue mask in some deployment or expose one through its source code. The public pages do not establish that. Even if such a file exists, its zero/nonzero meaning, dimensions, slide alignment, TIFF readability, and availability to `peknik-ns` would need checking before use.
