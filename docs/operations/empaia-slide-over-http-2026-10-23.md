# EMPAIA slide read over the App API: peknik-ns run, 2026-10-23

`EMPAIA_SLIDE_ACCESS=http` (commit `1361fb2`) was run end to end in `peknik-ns`
on the acceptance prostate slide. The job finished: JES recorded `TERMINATED` 0
after 20 minutes, the heatmap was published on the right slide level, and the
downloaded slide was removed with the attempt workspace. Nothing ran outside
`peknik-ns`.

## Setup

Same stand-ins as the [2026-10-21 run](empaia-jes-mock-2026-10-21.md), with:

| Piece | Change |
| --- | --- |
| model-service | Helm revision 21, images `empaia-3` (Ray and facade), `executionScheduler.empaia.slideAccess: http` |
| AppService mock | `mock-appservice-5`: serves `regions` of `MOCK_FILE_SLIDE` with OpenSlide, returns `MOCK_FILE_RECTANGLE` as the region of interest, mounts the outputs volume read-only |
| Slide | MIRAX, 89,262 × 204,744 px at 0.243 µm/px |
| Region of interest | rectangle at (25,000, 70,000), 30,000 × 50,000 px |
| App | constant-heatmap (`constant-classifier`, 128-px tiles at 0.5 µm/px) |

## Results

| Step | Observed |
| --- | --- |
| Level choice | Level 1 (44,631 × 102,372 px), the level OpenSlide picks for 0.5 µm/px on the original |
| Download | 17,600 region reads of 512 px, about 3.5 minutes (≈ 85 regions/s at 8 in flight); no direct-access request |
| Downloaded file | Tiled JPEG TIFF, 115 MB (mostly empty background), OpenSlide reports 0.486 µm/px |
| Fidelity | On a 4096-px grid, 17 tiles with tissue differ from the original level by a mean absolute 1.0 (max 2.9) per channel; empty tiles identical |
| Region of interest | Rectangle mask written in about a minute |
| Inference | About 93,000 constant-model calls, finished about 12 minutes in |
| Publication | Pixelmap on slide level 1, 17,600 tiles, then finalized |
| JES | `TERMINATED` 0 after 1,200 s |
| Cleanup | `inputs/` (slide and mask) removed after the attempt |

## Found on the way

- The mock needs the slide volume mounted now that it reads the slide itself. The
  first attempt failed with HTTP 500 on every region; model-service retried each
  read, then reported "builder failed unexpectedly" to AppService. The message is
  the known gap from the 2026-10-21 postmortem (open issue 2): gateway errors
  raised while inputs are prepared are not classified, so their cause is hidden.
- The download fetches the whole level even when the region of interest is
  smaller; here it was cheap (3.5 minutes) next to inference and publication.

## Still open for the EMPAIA test instance

- A real AppService serves regions through MDS and the CDS plugin; its speed is
  unknown, and the App API read timeout and retry policy (open issue 3) applies to
  17,600 reads.
- The test instance hands JES a short App Service host name, which must resolve
  from peknik-ns (see `~/.claude/.scratch/model-service/empaia-bridge/issues/15-test-instance-rollout.md`).
