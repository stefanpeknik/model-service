# Region-of-interest slide download: peknik-ns run, 2026-10-27

With `slideAccess: http`, a job's slide download is now limited to its region
of interest (`empaia-16`), and the heatmap builder checks tissue before reading
the region-of-interest mask (`empaia-17`). Both ran in `peknik-ns` against the
EMPAIA stand-ins with the standalone constant-heatmap app (`constant-classifier`,
128-px tiles, stride 64) on the acceptance prostate slide (level 1:
44,631 × 102,372 px) and the mock's region of interest (30,000 × 50,000 px at
(25,000, 70,000) on level 0). Nothing ran outside `peknik-ns`.

## Results

| Run | Region reads | Download | Grid scan | JES |
| --- | --- | --- | --- | --- |
| 2026-10-26, whole level (`empaia-12`, preprocessing app, no region) | 17,678 | — | — | `TERMINATED` 0 after 717 s |
| Region only (`empaia-16`) | 1,578 (1,500 of 17,600 level tiles, plus overview) | 80 s | 1,114,503 positions in 739 s | `TERMINATED` 0 after 901 s |
| Region only, tissue checked first (`empaia-17`) | 1,578 | 268 s | 1,114,503 positions in 91 s | `TERMINATED` 0 after 438 s |

All three published 275 pixelmap tiles and finalized.

## Found on the way

- The region-only download made the grid scan the slowest step: each of the 1.1
  million positions read the region-of-interest mask through OpenSlide before
  the in-memory tissue check. Checking tissue first left the result unchanged
  and cut the scan from 739 s to 91 s. Jobs without a region never read a mask.
- The same 1,500 tiles took 80 s in one run and 268 s in the next; the mock App
  Service reads the MIRAX slide from the NFS output volume with OpenSlide, and
  its speed varies.
