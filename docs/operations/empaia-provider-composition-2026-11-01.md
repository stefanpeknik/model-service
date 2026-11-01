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
