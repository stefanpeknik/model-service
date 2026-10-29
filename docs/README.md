# Documentation map

| Read | For |
| --- | --- |
| [How model-service is put together](architecture/core.md) | Packages, request flow, contracts, capacity |
| [Specification](spec.md) | What the core must do and how it was accepted |
| [Adding models](guides/adding-models.md) | YAML-only onboarding, or a new inference implementation |
| [Deployment guide](guides/deployment-guide.md), [configuration reference](guides/configuration-reference.md) | Helm chart and values |
| [peknik-ns test deployment](operations/peknik-ns.md) | Building images, upgrading, the GPU and quota rules |
| [rationai-infra-test-ns deployment](operations/rationai-test-ns.md) | model-service beside the EMPAIA test instance, reading its slide volume |
| [Core acceptance, 2026-10-14](operations/core-acceptance-2026-10-14.md) | Live evidence for the current core |
| [Scheduled raw inference acceptance, 2026-10-15](operations/raw-inference-acceptance-2026-10-15.md) | Live evidence and throughput for scheduled tiles |
| [EMPAIA jobs through JES, 2026-10-21](operations/empaia-jes-mock-2026-10-21.md) | Postmortem of the JES HTTP backend + model-service run against EMPAIA stand-ins in peknik-ns |
| [EMPAIA slide over the App API, 2026-10-23](operations/empaia-slide-over-http-2026-10-23.md) | `EMPAIA_SLIDE_ACCESS=http` run on the prostate MIRAX slide in peknik-ns |
| [First EMPAIA test-instance job, 2026-10-25](operations/empaia-test-instance-2026-10-25.md) | A preprocessing job from the EMPAIA test instance run end to end on model-service in peknik-ns |
| [EMPAIA job mode, header keys and pyramid publishing, 2026-10-26](operations/empaia-peknik-ns-2026-10-26.md) | Full-slide preprocessing job in peknik-ns; publication OOM found and fixed |
| [Prostate heatmap on the EMPAIA test instance, 2026-10-27](operations/empaia-test-instance-2026-10-27.md) | ProstHM job from the test instance on `empaia-12`, `TERMINATED` 0 in about 7 minutes |
| [Raw-inference payloads on a volume, 2026-10-27](operations/inference-payload-files-2026-10-27.md) | Tile bytes moved from PostgreSQL to an NFS claim; layout, sweep and throughput in peknik-ns |
| [Region-of-interest slide download, 2026-10-27](operations/empaia-roi-download-2026-10-27.md) | Slide download limited to the region of interest; tissue checked before the mask |
| [Automatic EMPAIA app registration, 2026-10-27](operations/empaia-registration-2026-10-27.md) | Generated apps registered at facade startup; standalone jobs admitted with high priority |
| [EMPAIA SDK Client and score-grid pixelmaps, 2026-10-28](operations/empaia-sdk-client-2026-10-28.md) | The SDK's Client in place of our own; pixelmaps published from the score grid (16 tiles on 5 levels instead of 275) |
| [ADR 0001](adr/0001-run-scheduler-outside-ray-serve.md) | Why the scheduler runs outside Ray Serve |
| [Available models](available-models.md) | Raw model routes and payloads |
| [Ray-layer architecture](architecture/overview.md), [batching](architecture/batching.md), [queues](architecture/queues-and-backpressure.md) | Ray Serve internals |
| [history/](history/) | Dated journals and reviews from September-October 2026; they describe the retired workflow design |

Terms are defined in the [glossary](../CONTEXT.md).
