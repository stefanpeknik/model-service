# Composed providers beside the EMPAIA test instance, 2026-11-01

Image `empaia-24` (`94bbc94`) in [rationai-infra-test-ns](rationai-test-ns.md)
carries the provider composition, two-phase publication and worker liveness. The
Prostate cancer heatmap preprocessing app ran once from the case browser, with
the GPU worker at zero.

| Step | Time | Duration |
| --- | --- | --- |
| Admitted (execution `9cb9af86`) | 13:17:35.8 | |
| Scoring starts, after builder replica and cold GPU worker | 13:18:43 | 67 s |
| Scoring: 69,426 positions, 2,553 tissue tiles | 13:19:15.9 | 33 s |
| Grid published and job finalized | 13:19:17.8 | 1.8 s |

Total 102 s, against 101 s for the cold run on `empaia-23`. JES ended the job
`TERMINATED` 0. The GPU worker had scaled to zero when checked at 13:31.

The facade logged one `Unclosed connection` for `test-jobs-app-api`, a warning
seen on earlier runs and attributed to the EMPAIA SDK; the job was unaffected.

## `empaia-25`, 2026-11-05

`empaia-25` (`bf92155`) adds the adversarial-review fixes and drops the score
grid's region-of-interest crop, so the grid covers the whole level again. A
cold preprocessing run of the same app (execution `280c6b89`) took 106 s:
70 s until scoring started, 35 s scoring the same 2,553 tissue tiles (read
23 ms, model 69 ms per tile), and about 1.5 s to publish and finalize. JES
ended it `TERMINATED` 0.
