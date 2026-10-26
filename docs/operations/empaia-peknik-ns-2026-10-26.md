# EMPAIA job mode, header keys and pyramid publishing: peknik-ns run, 2026-10-26

Images `empaia-10` and `empaia-12` were run in `peknik-ns` against the EMPAIA
stand-ins (`mock-appservice-6`, which adds the App API `mode` route). Nothing
ran outside `peknik-ns`.

## Results

| Check | Image | Observed |
| --- | --- | --- |
| Builder schedule without `Idempotency-Key` header | `empaia-10` | 422 `Idempotency-Key header is required`, also when the key is in the body |
| Builder schedule with the header | `empaia-10` | passes the key check, reaches admission (rejected there for the missing slide, as sent) |
| Job mode at admission | `empaia-10` | `GET /v3/{job}/mode` answered 200 before inputs were checked |
| Full prostate MIRAX slide, constant heatmap preprocessing app, `slideAccess: http` | `empaia-10` | publication OOM-killed the 512 Mi facade after 6 pixelmap tiles; JES `TERMINATED` 1, "provider finish outcome is unknown after work was interrupted" |
| Same job | `empaia-12` | JES `TERMINATED` 0 after 717 s; 17,678 region reads, 275 pixelmap tiles, finalized; facade not restarted |

## Found on the way

- Publishing at native score resolution shrank the full-resolution heatmap page
  in the facade. On this slide (level 1, 44,631 × 102,372 px) that needed about
  400 MiB locally on top of the facade's own memory, and crops of the lazy
  shrink kept re-reading the page. `empaia-12` loads the heatmap by file so
  libvips starts from the builder's nearest pyramid page: 58 MiB and 0.14 s
  locally on a heatmap of the same size.
- The stand-ins had been scaled to zero and the marketplace mock keeps its
  apps in an `emptyDir`, so the constant preprocessing app was registered again
  (`scripts/register_empaia_app.py`) before the run. The stand-ins were scaled
  back to zero afterwards.
