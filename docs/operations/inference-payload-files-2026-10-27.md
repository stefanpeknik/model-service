# Raw-inference payloads on a volume: peknik-ns run, 2026-10-27

Scheduled raw-inference request and response bytes moved from PostgreSQL
(`inference_payloads`) to files on a ReadWriteMany claim mounted by the facade
(`model-service-validation-inference`, `nfs-csi`, 8 Gi; NFSv3 over GPFS). Images
`empaia-13` to `empaia-15` ran in `peknik-ns`; `empaia-15` is the final layout.
Nothing ran outside `peknik-ns`.

## Results

| Check | Observed |
| --- | --- |
| Migration `0004` on the peknik-ns database | Applied at startup; `inference_payloads` dropped; the database held no inference executions, so none were removed |
| Volume access | The facade (uid 10001) writes to the claim's root |
| One `constant-classifier` tile | Request file present after admission; replay with the same `Idempotency-Key` → `200`, same execution, no second file; after success only the response file is left, the row has no request path and the response path and size (4 bytes); `/result` → `200` |
| Orphan sweep (`empaia-14`) | A two-hour-old execution directory without a row was removed on the first maintenance pass after startup; 501 directories with rows were kept |
| 2000 tiles, 128², 32 concurrent submissions (`empaia-15`) | All succeeded; admitted at 86/s, all final in 37.7 s. The 2026-10-15 run with request and response bytes in PostgreSQL: about 100/s admitted, all final in 37 s ([record](raw-inference-acceptance-2026-10-15.md)) |
| Files left after the bursts | Only response files (2500); no request or temporary files from the service |
| Orphan sweep (`empaia-15`) | 500 request files a benchmark wrote at 11:55 without rows were removed by 12:58, the first 10-minute pass after they turned an hour old |

## Found on the way

The first layout (a directory per execution under one root, `empaia-13`)
admitted 34 tiles/s and finished 500 tiles at 20/s. On this NFS server each
create, rename or unlink takes tens of milliseconds and locks its directory, so
every new execution directory waited on the root's lock, and deletes ran one at
a time (500 directories in 43 s). Benchmarks inside the facade pod, 300 files of
50 kB:

| Pattern | Rate |
| --- | --- |
| Write (temporary name, rename), one thread | 8/s |
| Same, 32 threads, files spread over 256 existing shard directories | 184/s |
| Write in place, 32 threads, shards | 258/s |
| Unlink, 32 threads, shards | 1570/s |

`empaia-15` therefore keeps files in 256 shard directories
(`<id[:2]>/<id>.request`, `<id[:2]>/<id>.response-<token>`), runs file I/O on its
own 32-thread pool instead of asyncio's default executor (5 threads on a 1-CPU
facade), deletes in parallel, and checks paths lexically instead of resolving
them on the event loop. Writing in place without the rename would be about 40%
faster here; the rename was kept as designed.

The 1001 execution directories of the `empaia-13`/`empaia-14` tests were removed
by hand; the new layout does not manage them.
