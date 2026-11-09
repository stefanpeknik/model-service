# Score-grid heatmaps beside the EMPAIA test instance, 2026-10-31

The heatmap builder now writes only the score grid (`heatmap.npz`) and EMPAIA
publication goes only through the SDK. Image `empaia-23` in
[rationai-infra-test-ns](rationai-test-ns.md) ran the ProstHM preprocessing app
on the prostate slide `7a072fd9` twice, started from the case browser, with the
slide read directly from the instance's slide volume.

## Results

| | Cold (job `daa70280`) | Warm GPU worker (job `9515b18e`) |
| --- | --- | --- |
| Admission | 11:45:55 | 11:52:55 |
| Builder replica up | 11:46:04 (8.5 s) | 11:53:03.7 (8.3 s) |
| Scoring starts | 11:46:59 (GPU worker and model start) | 11:53:08.9 (4.8 s) |
| Scoring: 69,426 positions, 2,553 tissue tiles | 35 s; read 24 ms, model 71 ms per tile | 34 s; read 23 ms, model 70 ms per tile |
| Grid published and finalized | 1.5 s, 16 tiles on levels 6–8 | 1.4 s |
| model-service total | 101 s | 49 s |
| JES | `TERMINATED` 0 at 11:47:41 | `TERMINATED` 0 at 11:53:45.7 |

The earlier warm run on `empaia-20` (2026-10-30 09:11) took 146 s, about 40 s
of it writing the TIFF that EMPAIA never used.

## Observations

- Scoring is model-bound: per tile, the model waits about three times as long
  as the slide read, with tiles in flight concurrently. The 114 s scoring of
  2026-10-30 10:08 was a transient slowdown, not slide reads.
- About 8 s of every run is Ray Serve starting a heatmap builder replica from
  zero (`min_replicas: 0`, scaled down 120 s after its last request). Kept as
  is: the builders and GPU models scale to zero by design.
- A cold GPU worker adds about 55–64 s (pod start and model load; the TensorRT
  engine comes from the cache).
