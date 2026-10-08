---
status: accepted
date: 2026-10-07
---

# Run the execution scheduler outside Ray Serve

Keep the durable execution scheduler in its own service, with a process and deployment lifecycle independent of Ray Serve and GPU workers. This keeps admission, status reads, cancellation requests, and deadline maintenance available during Ray or compute outages, provided the scheduler and PostgreSQL remain reachable. The user confirmed this reasoning on October 7; this records the existing deployment decision and makes no runtime change.

## Why

Ray owns workflow and model computation. The scheduler owns durable admission, claims, leases, retries, deadlines, cancellation, result settlement, and provider finish. These operations must continue when computation has no capacity or Ray is temporarily unavailable.

The separation concerns availability. PostgreSQL records, transaction guarantees, leases, and fencing provide execution durability and safe recovery. A scheduler hosted inside Ray Serve could retain those mechanisms, but its public operations and background tasks would also depend on the availability of its Serve deployment and its host resources. Persisted executions could survive while admission and monitoring were unavailable.

The final standalone validation admitted execution
`2285eb27-8040-4bd6-a8c7-f39e6168b809` with GPU capacity at zero. Admission and polling remained available, the Serve controller restarted while workers were absent, and the execution succeeded after GPU capacity returned. A separate in-flight scheduler replacement recovered work through lease expiry. These observations support the separation; they do not establish availability through every possible Ray outage. See the [recorded validation](../operations/final-standalone-validation-2026-10-06.md).

## Alternatives considered

| Placement | Tradeoff |
| --- | --- |
| Scheduler inside a workflow replica | Fewer separately managed services, but admission and maintenance share the workflow replica's process and lifecycle. Removing the compute replica or its GPU worker removes that scheduler instance too. |
| Dedicated CPU scheduler deployment in Ray Serve | Can remain available when GPU workers disappear if it has independent CPU placement and an always-running replica. Public operations still depend on Serve routing and that replica's lifecycle. Sharing a Serve application does not turn calls between deployment actors into ordinary function calls. |
| Separate scheduler service | Adds a Deployment, configuration, resource allocation, and service-to-service communication. Gives durable execution management an availability boundary independent of Ray Serve. This is the chosen placement. |

## Consequences

- Maintain PostgreSQL and its persistent storage independently of RayService replacement. Scheduler separation alone cannot preserve data in an ephemeral database volume.
- Keep the scheduler worker and maintenance active while workflow capacity is absent. Execution deadlines still apply during that wait.
- Recover interrupted processing with leases and fencing. This remains at-least-once processing; process separation does not guarantee exactly-once inference.
- Use the current HTTP dispatch boundary for scheduled workflow processing. Changing it to Ray handles would require reassessing cancellation and unknown-outcome classification.
- Treat immediate HTTP routing as a separate decision. One public origin can route immediate model and workflow calls directly to Ray Serve while sending durable execution operations to the scheduler. Keeping the scheduler separate does not require forwarding every immediate request through its Python process. Any routing change must preserve the public contract and cancellation behavior; no such change is approved or implemented by this record.

## Implementation and requirements

The current source separates [public application wiring](../../model_service/runtime.py), [durable orchestration](../../execution_scheduler/README.md), and [workflow dispatch](../../workflow_execution/dispatch.py). The [scheduler Deployment](../../helm/rayservice/templates/execution-scheduler-deployment.yaml) is distinct from the [RayService](../../helm/rayservice/templates/rayservice.yaml).

The primary requirements remain the [durable workflow specification](/home/stefanpeknik/.claude/.scratch/model-service/durable-workflow-execution/spec.md), reached through [docs/README.md](../README.md). This ADR records placement rationale and alternatives; it changes no ticket status or validation boundary. The October 7 package moves are local source changes, not a new deployment validation.
