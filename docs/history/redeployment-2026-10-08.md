# Standalone redeployment, 2026-10-08

Status: Helm release `model-service-validation`, revision 8, is deployed in
`peknik-ns`. Both native Serve applications are healthy, and scheduler health,
readiness, and workflow availability return HTTP 200. EMPAIA remains disabled.
Scheduling is still workflow-only; raw inference remains synchronous.

The user requested a deployment-focused check, not another full code review.
This rebuild includes the October 7 source moves and ORM adapter, October 8
Pydantic Settings and Yoyo adoption, and the shared EMPAIA client transport.
It does not establish acceptance of the unfinished EMPAIA integration.

## Source and images

Build contexts came from committed Git archives, excluding unrelated demo,
overview-document, and golden-test work in the working tree.

- Application source: `bc6a23b71461184b87ca8b426d548ec7e7118261`.
- Scheduler image fix: `a91b5199e4d19244d80e798fce5f84f3d7660922`.
  This adds a non-root OS account; application source is otherwise unchanged.
- Sibling `../empaia`: `fb18644734e0923e14b3f1b4d72ef301175f1399`.
- Its typing-model submodule: `682acdb1349e56f0de1ff4e99c7806c0e52e807f`.
- Public RationAI SDK used by live checks:
  `f39ac0095739eeb361cdc96d8de99d66aa892398`.

All images are under `cerit.io/stefanpeknik`. Live container image IDs matched
these published manifest digests.

| Repository | Tag | Manifest SHA-256 |
| --- | --- | --- |
| `model-service-scheduler` | `checkpoint-20261008-a91b519` | `a2ec31a8ed367a22b28f9933d18bbe8fd968ba455bd89520026bf71f58543b97` |
| `model-service-ray-validation` | `checkpoint-20261008-bc6a23b` | `0c0650d17a9e1002e7d7ae8cd0c0660a498d447e4ab94b61f4ea1ff42347e422` |
| `model-service-workflow-gpu` | `checkpoint-20261008-bc6a23b` | `7893cc32f427143a6ae2037faa0b1587c867044a1dc8912cfdf8bbdc137912d8` |

Scheduler and head images used their repository Dockerfiles with the pinned SDK
as an additional build context. The GPU image refreshed application source and
the transport-only SDK wheel on the retained `full-wsi-20261002-1` dependency
base. Its published image reuses the immutable base layers and appends a combined
source/SDK layer exported from the locally tested refreshed image. The combined
layer changes OCI config identity, so the local build ID is not its published
config digest. Native dependency versions remain Ray 2.58.0, ONNX Runtime GPU
1.23.2, and Torch 2.8.0+cu128, also checked inside the live GPU worker.

The live workflow ingress source SHA-256 matched the pinned build context:
`eb38c09c6a52c0a1dd94c4efb4436a48bd834a08e884414f492078ce3f739ae1`.

## Database and replacement

The retained database had 178 terminal executions, no pending work, and no Yoyo
history. The new baseline cannot automatically adopt an existing untracked
execution table. A temporary in-namespace pod created a reference schema using
the actual new migrations, compared all execution columns, constraint
definitions, and index definitions, then marked the matching baseline under
initialization and migration locks. No execution rows were rewritten.

The record count and ordered full-row MD5 were identical before and after
adoption: 178 and `408d069e9778312655ad1ccd9edccd1d`. Subsequent normal scheduler
startup and replacement passed with the tracked schema. The temporary reference
schema and migration pod were removed.

A private PostgreSQL custom-format backup was retained inside the namespace at
`/var/lib/postgresql/data/checkpoint-backups/pre-yoyo-20261008.dump` on the existing
database PVC. It is 75,308 bytes, mode 0600, and `pg_restore --list` succeeded.
Exporting the dump locally was rejected by the safety check; no local copy was
made. This backup shares the database PVC and is not disaster-recovery protection
against losing that volume.

The namespace's 40 GiB request quota cannot accommodate two 23 GiB GPU workers
during a rolling replacement. With no pending work, the scheduler was quiesced,
only the validation RayService and its owned cluster were removed, and Helm
recreated the application against retained storage. The new RayCluster is
`model-service-validation-4v98p`. All validation PVCs remain bound. PostgreSQL,
the separate `peknik-model-service-model-service` RayService, and other releases
were not replaced.

## Startup defect fixed

The migration probe exposed `KeyError: getpwuid(): uid not found: 10001` from
Yoyo's migration audit logging. The scheduler image used a numeric UID without
an OS user account. Podman normally generates a passwd entry, so earlier local
container tests hid the Kubernetes failure.

The Dockerfile now creates the `scheduler` account with UID/GID 10001. The
[image regression test](../../tests/integration/test_scheduler_image.py) failed
on the original image and passed on the corrected image with Podman's passwd
injection disabled and username environment variables removed. For future
image checks, mount this test inside the image and run:

```sh
podman run --rm --network=none --passwd=false --read-only \
  --tmpfs /tmp:rw,size=128m --env RUN_SCHEDULER_IMAGE_CHECKS=1 \
  --volume "$PWD/tests/integration/test_scheduler_image.py:/app/test_scheduler_image.py:ro,z" \
  --entrypoint uv \
  cerit.io/stefanpeknik/model-service-scheduler:checkpoint-20261008-a91b519 \
  --no-cache run --offline --no-project python -m unittest test_scheduler_image -v
```

## Local checks

These checks did not use the deployed execution database or a real AppService.

- Top-level local suite with disposable PostgreSQL 16: 219 tests, 214 passed,
  five native/profile skips.
- Original built scheduler image: 85 focused tests passed.
- Corrected scheduler image with non-root, read-only, and `--passwd=false`
  settings: 75 scheduler, public API, PostgreSQL, SDK gateway, and assembled
  integration tests passed. The image-specific username regression passed too.
- GPU image: eight native geometry, processor, and TIFF publication checks passed.
- Helm lint and server-side Kubernetes dry run passed. The final scheduler-tag
  change also passed Helm lint.

The disposable local PostgreSQL container and its temporary volume were removed.

## Live checks

The checks used loopback-only forwarding to the scheduler's public HTTP service.
No real EMPAIA service was called.

| Check | Result |
| --- | --- |
| `/healthz`, `/readyz`, `/healthz/workflows` after rollout and scheduler restart | HTTP 200 |
| Real prostate classifier, raw binary HTTP plus sync/async SDK and PIL/array inputs | Passed; zero-tile score within 0.001 of `0.256591796875` |
| Unknown workflow and execution IDs | Safe HTTP 404 contracts |
| Scheduled real WSI, admission replay, conflicting replay, polling and fresh-client recovery | Passed; execution `ed2da06a-44ae-44a3-a378-b2e21484ce88` succeeded |
| Immediate real WSI through public SDK | Passed with the same output contract |
| Completed immediate and scheduled TIFFs | Readable, nonempty, identical SHA-256 |
| Queued cancellation, repeated cancellation, terminal replay | Passed; `83876f4f-5766-49c4-bee2-4e7b60518edc` remained cancelled |
| Five-second queued deadline, terminal replay and cancellation after failure | Passed; `0c6a8a61-6afd-4ded-8b51-611a0917a0bc` failed with `execution_deadline_exceeded` |
| Queued record across scheduler-only replacement | Passed; `a5e2cb0a-f9d6-4558-8152-06e37b1bafdb` replayed with the same ID while queued, then was cancelled |

The successful native API checks took 154.811 seconds for two tests. The two
claim-paused control tests passed in 5.904 seconds. Claims were paused using a
bounded task-owned session advisory lock for `prostate-heatmap`; HTTP and
maintenance stayed active. The lock session was explicitly terminated afterward.

Both TIFFs were 15,474,188 bytes, 44,631 by 102,372 pixels, one unsigned-byte band.
Their SHA-256 was
`f7920599c51fb6f60483baa76678edcc4f455423f10d2897bcbe1395aafeacfc`.
Artifact checks used the exact returned paths inside the GPU worker rather than
assuming synchronous and scheduled paths would match.

At the final check there were 182 executions and zero pending work. The scheduler,
Ray head, and GPU worker had zero container restarts. Initial downstream HTTP 503
during Ray startup was followed by healthy workflow availability; durable API
readiness stayed available independently.

## Boundaries and follow-up

This verifies the latest standalone deployment, classifier transport, real-WSI
workflow processing, and basic durable queue behavior. It does not repeat every
October 6 fault-injection scenario or establish production availability under
all failures. EPISeg and embedding model validation was not repeated; the
[earlier record](final-standalone-validation-2026-10-06.md) still documents those
models' limits. Real EMPAIA admission, authentication, publication, and JES
interoperability remain follow-up work.

No `models/` files or sibling repositories were changed during this redeployment.
No Git changes were pushed. Harbor image publication and runtime mutations were
limited to the authorized project and validation workload in `peknik-ns`.
Unrelated working-tree changes remain separate.
