# Documentation map

| Read | For |
| --- | --- |
| [How model-service is put together](architecture/core.md) | Packages, request flow, contracts, capacity |
| [Specification](spec.md) | What the core must do and how it was accepted |
| [Adding models](guides/adding-models.md) | YAML-only onboarding, or a new inference implementation |
| [Deployment guide](guides/deployment-guide.md), [configuration reference](guides/configuration-reference.md) | Helm chart and values |
| [peknik-ns test deployment](operations/peknik-ns.md) | Building images, upgrading, the GPU and quota rules |
| [Core acceptance, 2026-10-14](operations/core-acceptance-2026-10-14.md) | Live evidence for the current core |
| [Scheduled raw inference acceptance, 2026-10-15](operations/raw-inference-acceptance-2026-10-15.md) | Live evidence and throughput for scheduled tiles |
| [ADR 0001](adr/0001-run-scheduler-outside-ray-serve.md) | Why the scheduler runs outside Ray Serve |
| [Available models](available-models.md) | Raw model routes and payloads |
| [Ray-layer architecture](architecture/overview.md), [batching](architecture/batching.md), [queues](architecture/queues-and-backpressure.md) | Ray Serve internals |
| [history/](history/) | Dated journals and reviews from September-October 2026; they describe the retired workflow design |

Terms are defined in the [glossary](../CONTEXT.md).
