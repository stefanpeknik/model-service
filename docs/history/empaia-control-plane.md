# Superseded EMPAIA prototype

Status: historical description of the exploratory code now archived under
`prototypes/empaia-control-plane/`. It is
not the current architecture or vocabulary. See
[`workflow-execution-proposal.md`](workflow-execution-proposal.md),
[`execution-interfaces.md`](execution-interfaces.md), and the repository
`CONTEXT.md`.

The names below mirror the prototype code. Its EMPAIA-specific control plane,
job lifecycle, SQLite store, and post-admission resolution have been superseded
by the provider-neutral execution scheduler design.

The source, tests, container file, and Helm templates are retained in the
archive, not in the active source tree or chart. The commands and deployment
settings below describe the original experiment, not a supported runtime.

The prototype provides:

- authenticated, idempotent admission at `POST /v1/empaia/executions`;
- safe status polling at `GET /v1/empaia/executions/{execution_id}`;
- idempotent cancellation at
  `POST /v1/empaia/executions/{execution_id}/cancellation`;
- background AppService configuration and direct-access resolution;
- deployment-owned model, parameter, WSI-plugin, and storage allow-lists;
- a SQLite durable store with atomic claim, timeout, and recovery primitives.

The implemented flow currently ends at `READY`. Ray dispatch, a job-bound worker
API, whole-slide execution, artifact commit, and EMPAIA publication remain
subsequent gates; no endpoint currently claims or reports inference completion.

## Configuration

The API fails closed unless its EMPAIA installation, JES credential, and app
allow-list are configured.

| Variable | Meaning |
| --- | --- |
| `EMPAIA_EXECUTION_DATABASE_PATH` | SQLite database path; defaults to `/var/lib/model-service/jobs.db` |
| `EMPAIA_INSTALLATION_ALIAS` | The one accepted EMPAIA installation alias |
| `EMPAIA_JES_BEARER_TOKEN_FILE` | Mounted JES token, reread for every request |
| `EMPAIA_APP_SERVICE_URL` | Fixed AppService base URL |
| `EMPAIA_APP_SERVICE_BEARER_TOKEN_FILE` | Mounted AppService token, reread for every operation |
| `EMPAIA_APP_MAPPINGS_JSON` | Enabled app/model and input-contract mappings |
| `EMPAIA_STORAGE_MAPPINGS_JSON` | EMPAIA path prefixes mapped to read-only mounts |
| `EMPAIA_POC_ALLOW_BROAD_APP_SERVICE_CREDENTIAL` | Must be explicitly `true` for the current broad-identity POC |
| `EMPAIA_POC_ALLOW_INSECURE_APP_SERVICE_HTTP` | Allows HTTP only for a deliberately isolated POC; HTTPS is the default requirement |

An initial prostate mapping has this shape (the EMPAIA app identity must be
replaced with the registered deployment value):

```json
[
  {
    "app_id": "prostate-app",
    "app_version": "1",
    "model_id": "prostate-classifier",
    "model_version": "1",
    "workflow": "wsi-to-classification-heatmap",
    "resource_profile": "mig20",
    "max_runtime_seconds": 14400,
    "input_name": "slide",
    "output_name": "heatmap",
    "model_parameter_names": ["threshold"],
    "allowed_wsi_plugin_ids": ["test-mds-cds-plugin-openslide:11880"]
  }
]
```

A storage mapping is deployment configuration, never submission data:

```json
[
  {
    "storage_id": "slides",
    "source_prefix": "Projects/empaia",
    "mount_root": "/mnt/wsi"
  }
]
```

The Helm chart mounts each configured WSI claim read-only at `mount_root`.
Resolution rejects traversal, symbolic links, missing files, unsupported direct access,
and WSI plugins not listed for the selected app. Only configured scalar model
parameters are copied into the immutable worker spec. Tokens, AppService
origins, and physical storage paths are not stored there.

Run the API using an ASGI server after setting the variables:

```bash
uvicorn control_plane.main:app --host 0.0.0.0 --port 8080
```

The database must be on persistent storage. SQLite is suitable for the first
single-control-plane POC; running several API replicas requires selecting a
shared transactional database and implementing the same store contract.

The current resolver uses the existing broadly trusted model-service identity
only when the POC opt-in is set. This is a release blocker, not a production
credential design: a caller-provided job ID is not yet bound to an authoritative
app ownership record. Production use requires a job-scoped credential (or an
equivalent authoritative ownership check) and least privilege. The ClusterIP
API likewise needs deployment-provided network isolation and encrypted workload
identity before credentials may cross an untrusted cluster network.

The eventual worker must reopen the relative locator beneath its read-only mount
without following symbolic links. The resolver check prevents accepting an
existing symlink but cannot alone eliminate a filesystem replacement race
between resolution and later worker access.

Application-failure intent is stored before calling AppService, and whichever
of cancellation or that intent commits first takes precedence. A lost response
from AppService failure reporting still cannot be reconciled authoritatively
with the documented API: the control plane reports an infrastructure failure
rather than guessing that the EMPAIA failure committed. Resolving that ambiguity
against MDS is required before production.
