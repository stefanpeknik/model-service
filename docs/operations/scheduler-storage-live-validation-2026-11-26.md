# Scheduler storage refactor on rationai-test, 2026-11-26

Live checks of revision 23 (images `unit-645e1e3`) on the Helm release
`model-service` in `rationai-infra-test-ns`. What changed since revision 22
(`unit-ce40c96`):

- **Scheduler rules:** they live in Python (`queue.py`, one
  `ExecutionRepository`) over a storage that only reads, writes and locks.
  Claims take turns under one advisory lock.
- **Input access:** factories are async behind one interface.
- **Run checks:** synchronous and scheduled runs share their input checks.
- **Compatibility code removed:** this includes the model digest's old `weights` key, so every
  model digest changed.

No automated test runs `PostgresStorage` any more, so these checks are its
only verification. Local results are separate: `uv run pytest` passed (526
passed, 8 skipped: the live tests and two conformance skips).

## Deployment

- **Images:** code-refresh layers over the `unit-ce40c96` images. Dependencies are unchanged.
- **Diff from the live release:** the rendered manifest differed only in the four image tags. The
  database held two executions, both finished, so nothing admitted under
  the old digests was in flight.
- **Health after the switch to the new Ray cluster:** catalog loaded, `/readyz` ready, no
  missing routes.

## Results

- **Golden references**, `pytest -m live tests/live` for
  `prostate-classifier-1` / `prostate-heatmap` (raw route and a synchronous
  run on both golden slides): 3 passed in 19 min 49 s.
  - The digest changed from `4d68c49caf25` to `8a42995d7f09`, so ONNX Runtime built a new
    TensorRT engine in a new cache folder. That rebuild is most of the time;
    revision 22 took 4 min 56 s with its engine cached. The old folder is
    still on the cache claim (see [the deployment
    doc](../architecture/deployment.md#the-caches-only-grow)).
- **Scheduled runs through the public API**, `POST /v1/pipelines/prostate-heatmap/schedule` on slide `67df0411`, in two
  rounds of submissions sent at the same moment:
  - **Capacity:** two runs (A, B) with `prostate-classifier-1` at its default
    one slot. While A was `running`, B was `queued`. Both succeeded, B 41 s
    after A, one processing and one finish attempt each. The two
    control-plane replicas never ran both at once.
  - **Outputs:** both outputs match the golden heatmap. Each was moved to its requested
    path, and its attempts directory under `.attempts/` was removed.
  - **Replay:** resubmitting A's `Idempotency-Key` with the same body answered 200
    with A's execution ID (202 the first time).
  - **Deadline:** a run with `requested_timeout_seconds: 5`, submitted behind A
    and B, was never claimed. Maintenance failed it 6 s after admission with
    `execution_deadline_exceeded` and 0 processing attempts.
  - **Cancellation:** a run cancelled while queued answered `cancelled`. Its
    cancellation report ran (one finish attempt) and its work cleared.
  - **Credentials:** every finished execution's stored provider context is 31 bytes, the
    sealed empty context. The running EMPAIA job below held 550 bytes, its
    sealed credential.
- **EMPAIA preprocessing**, `tools/request_preprocessing.py` on slide
  `7a072fd9`:
  - Job `83f89c71`, execution `c0484bb4`, succeeded in 71 s with one processing
    and one finish attempt.
  - The App API saw the configuration, mode, slide, direct-access and a 404
    for the absent region of interest, then the heatmap output, its pixelmap
    tiles and `finalize`.
  - JES ended the job with exit code 0.

## Not validated live

- Two claims racing for the same last slot at the same instant. The rounds
  above submitted runs together, but the claim lock's ordering wasn't
  provoked deliberately.
- A claim overwriting a concurrent cancel, the bug the code review found
  and fixed before deploying: it needs a cancel to land between a claim's
  read and its lock.
- A context that can't be decrypted: it needs a corrupted row.
