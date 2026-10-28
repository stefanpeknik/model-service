# Automatic EMPAIA app registration and priority: peknik-ns run, 2026-10-27

Image `empaia-18` ran in `peknik-ns` with `empaia.marketplace` pointing at the
marketplace mock of `deploy/k8s/peknik-ns/empaia-mock.yaml`. Nothing ran outside
`peknik-ns`.

## Results

| Check | Observed |
| --- | --- |
| Migration `0005` | Applied at startup; executions have a `priority` column |
| Registration at startup | The peknik catalog lists one model with an `empaia` block (`constant-classifier`); its standalone app (`0d9d12a8-2764-4c9c-89a0-f827bbe9932f`, "Constant test heatmap") was in the marketplace after the facade started: mock JES resolved it and ran the job. `peknik/prostate-classifier-1` has no block, so it got no app |
| Standalone job of that app | Admitted with the then-current top priority (stored as 100; now `HIGH` = 2, migration `0006`); read the region of interest (2 rectangle reads); stopped through JES after 20 s, recorded `cancelled` |

## Found on the way

The first `empaia-18` facade crashed at import: `jsonschema`, a declared
dependency now used at runtime for EAD validation, was missing from the
October scheduler checkpoint image the code-refresh build starts from. The
build now adds it (`EXTRA_PACKAGES`, see [peknik-ns](peknik-ns.md)); the old
pod kept serving until the fixed image rolled out.

The facade does not configure INFO logging for model-service's own loggers, so
"registered EMPAIA app" lines do not appear in its log.
