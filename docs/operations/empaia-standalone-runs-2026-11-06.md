# Standalone EMPAIA runs beside the test instance, 2026-11-06 to 2026-11-08

Standalone jobs (a drawn region of interest, high priority) started from the
EMPAIA test instance's case browser ran on model-service in
`rationai-infra-test-ns` on images `empaia-26` (Ray) and `empaia-27` (API).
Every one succeeded on its first attempt and published its heatmap to the App
Service.

| Execution | Admitted (UTC) | Duration |
| --- | --- | --- |
| `e0a3a066` | 2026-11-06 00:26 | 1096 s (cold GPU and a new TensorRT engine) |
| `80ef9d02` | 2026-11-06 00:55 | 96 s |
| `5d52ea70` | 2026-11-06 00:55 | 147 s |
| `be93b0d3` | 2026-11-06 00:56 | 138 s |
| `61b53b65` | 2026-11-06 09:28 | 228 s |
| `18ce40d0` | 2026-11-07 18:17 | 247 s |
| `d3b4351f` | 2026-11-08 09:09 | 149 s |

Durations come from the execution rows (`updated_at - created_at`), so they
include queueing and publication. The three jobs admitted within a minute on
2026-11-06 ran one after another on the single GPU replica.

The first of these showed "1 result, but none could be read back" in the case
browser ([provider composition record](empaia-provider-composition-2026-11-01.md)).
The user confirmed on 2026-11-16 that the standalone results then displayed in
the case browser; its maintainers had fixed a small bug and an outdated
case-browser version on their side. With the preprocessing runs of 2026-10-31
([score-grid record](empaia-score-grid-2026-10-31.md)), both EMPAIA job modes
have run live end to end.

Later attempts on `empaia-30` and `empaia-31` (2026-11-11 and 2026-11-12) never
got a MIG GPU slice and failed at their deadline
([empaia-30 record](empaia-30-deploy-2026-11-11.md)). That was cluster capacity,
not the standalone path.
