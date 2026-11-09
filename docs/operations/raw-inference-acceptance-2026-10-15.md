# Scheduled raw inference: live acceptance, 2026-10-14/15

Deployment: `peknik-ns`, release `model-service-validation`, facade image
`cerit.io/stefanpeknik/model-service-scheduler:core-c1396a9` (one replica,
1 CPU), Ray image unchanged (`core-b9f7997`). The client reached the facade
through `kubectl port-forward`. Scripts lived in the session scratchpad; the
numbers below are what they printed.

## Local

- 278 unit and PostgreSQL tests pass (`unittest discover`), plus ruff and
  strict mypy. New coverage: both repositories (payload storage, separate
  capacity, batch claims, retention, bulk lookup), the worker's response
  classification, the API and the SDK (22 tests).
- Claim plus settle throughput with 5000 queued tiles, local PostgreSQL 16:
  59 tiles/s one claim per transaction, 203 tiles/s in batches of 8.

## Live

| Check | Result |
| --- | --- |
| Migration `0003` on the existing database | Applied at startup; the 192 existing executions read back as `builder` |
| Malformed execution ID | 404 `unknown_execution` (was 503 from a database type error) |
| 2000 tiles, `constant-classifier`, 128², 32 concurrent submissions | All succeeded; 20 results byte-identical to direct `POST /constant-classifier/` |
| Same, connection per transaction (first build) | Admission 25/s, all done in 166 s; PostgreSQL at its 1-CPU limit |
| Same, pooled connections | Admission 149/s, all done in 46 s |
| Same, batch claims | All done in 37 s; about 100 tiles/s admitted, about 110/s processed once admission stopped (facade CPU shared between the two) |
| 1000 tiles, `prostate-classifier-1`, 512², **no GPU worker running**; client submits and exits | Admitted in 8.7 s. A new client saw the first result 108 s later (GPU node start and model load) and all 1000 at 118 s. 20 results byte-identical to direct calls |
| 3000 tiles, warm GPU, facade pod deleted with 1589 queued and 17 running | All 3000 succeeded through the replacement pod. The 17 in flight ran a second attempt; 2983 ran once |
| SDK (`models.schedule`, `executions.wait`, `executions.result`) | 20 `constant-classifier` and 8 `virchow2` tiles succeeded; `.json()` gave 0.75, `.array(np.float16)` gave `(2, 1280)` |
| Stored bytes after finishing | 10,629 finished tiles: no request bytes left, every response kept |

### Virchow2 and batching

Eight scheduled Virchow2 tiles did not equal direct calls bit for bit. Direct
calls alone are deterministic, but 16 concurrent direct calls of the same tile
differ from one alone by up to 0.0097 (values up to about 8), because Ray
batches concurrent requests and the batched kernels round differently. A
scheduled tile run alone equals a direct call alone exactly (difference 0.0).
The scheduler does not change the request; the model's batching does.

## GPU use

The GPU worker started for the cold-start test at about 11:12 and served the
later checks. It was scaled down by Ray after the last request; see the
[runbook](rationai-test-ns.md) for how to check.
