# empaia-30 in rationai-infra-test-ns, 2026-11-11

`empaia-30` (built from `f067204`) replaces `empaia-29`: the new package layout,
raw inference removed from the API with migration `0007`, the Ray dashboard
capacity message removed, and TensorRT engines cached per checkpoint.

- `helm upgrade` to revision 12. Both API pods started on `empaia-30`; the
  database reports `0007-drop-raw-inference` applied.
- The API re-registered app `71e38adc` (standalone, preprocessing, heatmap on
  `prostate-classifier-1`), the same ID as before, so no duplicate.
- The RayService switched to the new cluster (`model-service-tp7d6`) and the old
  head was removed. `GET /healthz/deployments` answers `ready` with no missing
  routes. No GPU worker runs until a job arrives.

Not yet checked here: a job end to end on `empaia-30`, including a standalone
run now that the case browser's display fix is reported (2026-11-11). The first
job builds the prostate TensorRT engine again under its per-checkpoint path, so
expect a longer cold start. The unused `model-service-inference` claim (8Gi, nfs-csi)
was deleted afterwards, with its old raw-inference payload files.

## Standalone job, 2026-11-11

A standalone job from the case browser (execution `8c086d4b`, heatmap on
`prostate-classifier-1`, admitted 10:23 UTC with high priority) never started
scoring. Its GPU worker requested one `nvidia.com/mig-2g.20gb` slice and stayed
`Pending` for six hours: the scheduler found no node with a free slice
("cannot allocate all claims" on 43 of 51 nodes). Nothing in the namespace held
one, and the namespace's own resources were intact after the EMPAIA stack update
the same morning.

- About 12:45 UTC both API pods lost the database briefly; a lease renewal timed
  out, the attempt was requeued and claimed again (attempt 2). Postgres did not
  restart and logged nothing unusual.
- At the deadline (16:23 UTC) the execution failed with
  `execution_deadline_exceeded`, and provider finish reported the failure to
  EMPAIA in one attempt.
  The case browser lists job `5e00f866` (ProstHM) with status `error`, the
  message "execution deadline was exceeded", its runtime (21570 s), its one
  input region, and no output.

So the standalone display fix in the case browser is still unconfirmed, and
`empaia-30` has not yet run a job end to end.

## empaia-31, 2026-11-12

`empaia-31` (built from `727f625`) renames the API process to `api`: the
Deployment and Service are now `model-service-api` and the image
`model-service-api`. It also moves standalone results to their requested
path, makes EMPAIA optional in the chart, defaults the lease to five minutes and
composes the scheduler's transitions and PostgreSQL operations.

- `helm upgrade` to revision 13 created `model-service-api` (2/2) and removed
  the old `model-service-execution-scheduler` Deployment and Service.
- `test-jobs-jes` now has `JES_HTTP_TARGETS` pointing at
  `http://model-service-api.rationai-infra-test-ns.svc.cluster.local:8080/v1/empaia`
  (set with `oc set env`; a `helm upgrade` of `test-jobs` reverts it). JES
  reaches `GET /v1/models` with 200.
- The API re-registered app `71e38adc`; `/healthz/deployments` is `ready` while
  the RayService rolls to the new cluster.
