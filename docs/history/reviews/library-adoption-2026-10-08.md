# Library adoption review, 2026-10-08

This records the approved changes and the remaining options discussed with
the user. Recommendations below are not implementation decisions or ticket
acceptance. The primary specification and progress remain in the grilling
scratchpad linked from [the documentation entry point](../README.md).

## Implemented locally

- `model_service/settings.py` uses Pydantic Settings sources and validated fields
  instead of handwritten environment conversions and bounds checks. Names,
  defaults, and numeric-seconds units remain unchanged. A small duration validator
  preserves those units while accepting typed `timedelta` constructor values.
  Direct construction does not load environment variables. Explicit environment
  mappings replace the process environment completely, even when empty.
- `workflow_execution/dispatch.py` validates remote failures through a
  `TypeAdapter` for the existing `WorkflowErrorResponse` contract. Codes and
  messages are nonempty strings; retry decisions are strict booleans. Extra
  metadata remains accepted. Invalid envelopes remain unknown dispatch outcomes,
  rather than acquiring a coerced retry decision.
- The scheduler image declares the new Pydantic Settings dependency. No image
  was built or deployed. `models/`, sibling repositories, and runtime resources
  were not modified. Unrelated pending docstring edits were preserved.

Local validation ran 200 top-level tests: 164 passed and 36 skipped. The skips
include all 31 PostgreSQL cases because `TEST_POSTGRES_DSN` was not configured,
plus native dependency/profile skips. The focused settings, runtime, HTTP
dispatch, HTTP translation, public API, and deployment-contract run passed all
43 tests. Ruff formatting/lint, focused mypy for the three changed source files,
and diff whitespace checks passed.

Sandboxed discovery hung in the existing synchronous Starlette TestClient. A
bounded repeat also timed out there. The same suite passed outside the sandbox
in approximately two seconds; no test workaround was added to application code.
These are local tests, not PostgreSQL or live deployment validation.

## Yoyo migrations, implemented locally

The user approved replacing the migration machinery and allowed a breaking WIP
baseline. Model-service now declares Yoyo 9.0.0 instead of Alembic as a direct
dependency. Alembic can still appear transitively through unrelated libraries;
execution persistence no longer imports it. The scheduler image recipe also
declares Yoyo, but no image was built.

`execution_scheduler/persistence/migrations/0001-create-executions.sql` freezes
the initial table, constraints, defaults, and indexes. ORM mappings describe the
current row; they no longer create tables or repair missing columns. Additional
changes belong in numbered migrations. The linked
[migration guide](../../execution_scheduler/persistence/migrations/README.md)
documents authoring, applying, testing, adoption, and interrupted-process recovery.

`persistence/schema.py` applies pending migrations during startup in a worker
thread. A schema-specific PostgreSQL session advisory lock serializes schema
creation and Yoyo bootstrap. Yoyo's migration lock and history remain scoped to
that schema. Connections close when initialization completes or fails. URL and
libpq DSN support remain intact. Queue operations still use SQLAlchemy ORM.

The previous two automatic missing-column repairs are deliberately removed.
An existing execution table without Yoyo history fails initialization instead
of silently claiming compatibility. No database is automatically dropped or
marked current. The next deployment needs fresh disposable storage, an explicit
reset whose data loss is acceptable, or verified adoption of the retained schema.
No deployed database was changed. Each migration is transactional, not the whole
sequence; Yoyo's DDL and history marker are separate commits. Recovery after a
killed process may need operator inspection and lock cleanup, as documented.

Local verification used a disposable PostgreSQL 16 container bound only to
loopback, from an already cached image. All 35 repository tests passed within
top-level discovery: 214 tests ran, 209 passed, and five native dependency/profile
cases skipped. Both assembled durable HTTP tests passed with local ASGI provider
and workflow fakes. Coverage includes fresh concurrent bootstrap, tracked
restarts, one-time upgrades without record changes, failed-migration rollback and
retry, an empty migration directory, fail-closed legacy adoption, DSN formats,
schema isolation, database constraints, and existing queue/lease semantics.
The documented CLI passed twice against a test schema. Ruff lint/format, focused
persistence mypy, and diff whitespace checks passed. No deployed services were
contacted, and no image was built or deployed.

A later assembled-test repeat exposed an existing 30 ms processing-lease race:
processing could expire before the test reached its intended abandoned finish
claim. That fixture now uses a normal processing lease and retains the short
lease only for the abandoned finish claim. Application lease behavior is
unchanged. Both assembled tests and five additional repeats of the recovery
scenario passed after that correction. The disposable PostgreSQL fixture was
removed after validation; its test data was discarded.

## EMPAIA client comparison before consolidation

The current sibling client is
`../empaia/rationai/empaia/client/client.py`. It already exposes job
configuration, slide and rectangle inputs, region reads, typed output creation,
`put_tile`, finalization, failure reporting, and explicit client closure.
An earlier scan incorrectly listed tile upload as missing.

Keep these responsibilities in model-service even if transport is delegated:

- Submission identity, configuration merging/defaults and workflow validation,
  credential-expiry deadline caps, and provider outcome translation.
- Shared-input materialization, rectangle-mask creation, and conversion of a
  completed TIFF artifact into pixelmap metadata and tiles.
- Durable publication ownership. The gateway invokes an async guard before
  external mutations. It must not publish after losing its lease. A missing
  mutation acknowledgment can mean AppService already changed state, so it must
  not be retried as an ordinary safe failure.

At the initial review, the SDK lacked the gateway's before-mutation hook and RationAI
direct-access extension. Its HTTP exceptions also need translation into our
safe, retryable, or outcome-unknown failures. A replacement should preserve
bounded calls and explicit client lifetime and should not add automatic mutation
retries. Existing subset response models intentionally validate only consumed
fields, so full SDK models are not automatically equivalent replacements.

A practical next-phase direction is to improve the SDK's transport capabilities,
then make our gateway delegate standard AppService requests to it. Keep workflow
artifact conversion and scheduler policy here. Confirm both callers against the
same controlled AppService contracts before deleting duplicate transport code.
The follow-up below implements this direction locally.

## EMPAIA client consolidation, implemented locally

The sibling `empaia/` package now exposes `AppServiceClient`. It owns the shared
aiohttp transport, route encoding, bearer headers, direct-access extension, and
standard AppService operations. Its existing `Client` inherits that transport
and retains typed input/output conversion and image-region processing. Every
write awaits an optional mutation guard. Requests have a positive finite timeout,
defaulting to five seconds total, and do not follow redirects or retry mutations.
Clients close explicitly, including injected sessions. The unused-client
destructor was removed so interpreter shutdown cannot allocate a session.
Aiohttp's internal connection retry is disabled for owned and injected sessions.
Review caught that its default behavior can resend PUT after accepting a write
and losing the acknowledgment. A controlled disconnect test now proves one send
and one guard invocation; the gateway reports an unknown mutation outcome.

Model-service's gateway delegates to this client. It retains strict validation
of consumed response fields, safe error classification, configuration merging,
shared-input materialization, and TIFF-to-pixelmap conversion. The provider and
scheduler retain admission, original-credential deadlines, durable publication
ownership, and retry policy. A malformed JSON acknowledgment or unusable created
pixelmap ID now explicitly marks the write's outcome unknown.

The sibling is version 0.1.2 with an explicit wheel build and PyJWT lower bound
2.15.1. Its formerly untracked UV lockfile now resolves 2.15.1. Model-service
locks an editable dependency at `../empaia`; local setup therefore requires that
sibling checkout and its typing-model submodule. The scheduler and workflow
image recipes install the same SDK from the named `empaia-sdk` build context,
documented in the full-WSI runbook. Their transport-only installation imports
without Ray, NumPy, imageio, or Pydantic. The typed client still needs its declared
dependencies. No image was built.

Controlled loopback tests cover both SDK clients, encoded routes, bodies,
authentication headers, mutation guarding, timeout, cancellation, closure,
direct-access fallback, and invalid acknowledgments. Gateway tests preserve
subset validation and distinguish permanent HTTP rejection from transient read
errors and uncertain writes. Three assembled tests use real PostgreSQL, the
SDK-backed HTTP gateway, an ASGI workflow, and a fixture binary-artifact
publisher. They cover successful polling, recovery before publication without
redispatch, and suppression of further writes after an output is stored but its
acknowledgment is 503. The fixture publisher converts binary test artifacts to
inline primitives; the existing TIFF publication test separately checks the
default pixelmap path.

The eleven SDK transport tests, fifteen focused HTTP gateway tests, and three
assembled PostgreSQL tests passed. Focused mypy, Ruff, wheel construction, and
a transport-only wheel import passed. Final top-level discovery ran 219 tests,
with 214 passing and five native dependency/profile skips. All PostgreSQL cases
ran against the disposable loopback PostgreSQL 16 fixture. These are local results.
No live AppService,
JES, GPU workflow, image build, deployment, or Git push was performed. Issues 07
and 08 retain their acceptance status and unresolved authentication, filesystem
trust, failure-reporting, rectangle-clipping, and report-delay findings.

## JWT parsing, implemented locally

PyJWT fits the current needs and is already a sibling SDK dependency. The current
[FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/)
uses it. [python-jose](https://github.com/mpdavis/python-jose) covers a broader
JOSE feature set, including encrypted tokens; that is not currently required.

Model-service now declares and locks PyJWT 2.15.1. Its provider uses PyJWT to
decode the token structure and require an `exp` claim, without verifying the
signature or independently checking the current time. The scheduler retains
its existing deadline policy. Expiry must be numeric, finite, and representable
as a UTC datetime; malformed values become safe admission errors before any
gateway preparation. This does not implement service authentication.

At the initial review, the sibling's
untracked local lockfile resolved PyJWT 2.13.0. The relevant
[unverified-payload parsing advisory](https://github.com/jpadilla/pyjwt/security/advisories/GHSA-42vr-xj54-vc7v)
lists 2.15.0 as its first patched release. This is a dependency-version concern,
not evidence that our deployed service uses that SDK lockfile or that its current
token-handling paths satisfy every advisory's exploit conditions. The sibling
repository was not modified during JWT adoption. The consolidation follow-up
above raises its dependency floor and refreshes that lockfile to 2.15.1.

Seven new provider-preparation tests cover integer/fractional expiry, deadline
caps, expired claims delegated to scheduler policy, inspection without signature
authentication, malformed tokens, extreme values, and nested payload failures.
Existing synthetic JWT signatures were corrected to canonical Base64url in the
unit, assembled, and deployed test fixtures. The deployed tests were not run.
After fixture corrections, all 46 focused tests passed. Top-level discovery ran
207 tests: 171 passed and the same 36 PostgreSQL/native/profile cases skipped.
Ruff, focused provider typing, and diff checks passed. No PostgreSQL or live
validation was performed for this change.

Job credential expiry inspection does not authenticate the calling JES service.
Keep those operations separate. Actual verification needs configured algorithms,
trusted keys, and the appropriate issuer/audience/claim rules, following
[RFC 8725](https://www.rfc-editor.org/rfc/rfc8725.html). Neither library removes
the need for finite, representable expiry validation and safe error translation.

## EAD generation, implemented locally

`scripts/generate_ead.py` is a manual development tool documented in
[the workflow definition guide](../../workflow-definitions/README.md).
It reads `workflow-definitions/prostate-heatmap.yaml` and writes
`eads/prostate-heatmap.json`. Inputs, outputs, and optional EMPAIA metadata become
the executable app description, including its namespace, permissions, I/O, and
mode. It does not register the EAD or execute a job.

`tests/test_ead_generation.py` imports its functions and checks translation,
references, schema validation, and checked-in file freshness. `--check` validates
the generated EAD before comparing JSON with the existing file. No runtime,
Helm, Docker, or repository-local CI
generation invocation was found. The repository delegates linting to a shared
workflow; this review did not inspect that external workflow's internals.

The generator now validates with `jsonschema`, declared as a direct dependency,
using the official schema copied into `eads/schemas/ead-v3.schema.json` at commit
`2b324f13f73eaa5950794ce9befeffa4769e8a78`. The upstream MIT license is preserved
beside it; provenance and update instructions are in the workflow definition
guide. Validation stays in the existing generation/check/test path. It does not
fetch a schema at runtime or add inference or scheduler request handling. The
draft EAD's compatibility with the actual EMPAIA installation still needs
next-phase confirmation.

The official v3 schema inspected on October 8 declares JSON Schema Draft 7 and
uses only internal `#/definitions/...` references. The pinned copy is therefore
validated offline with `jsonschema.Draft7Validator`. The generated EAD is
unchanged. New tests cover invalid metadata and I/O, and rejection without
overwriting an existing output file in generation or check mode.

All five generation tests and the existing EAD's `--check` passed. Top-level
discovery ran 210 tests: 174 passed and the same 36 PostgreSQL/native/profile
cases skipped. Ruff lint/format and diff whitespace checks passed. No database,
AppService, image-build, or live deployment validation was performed.

The previously reported oversized `Retry-After` overflow remains outside these
approved changes.
