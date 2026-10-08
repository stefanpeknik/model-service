# Model-service resources

## Knowledge

- [Scheduler placement decision](docs/adr/0001-run-scheduler-outside-ray-serve.md)
  Accepted reasoning for a scheduler deployment outside Ray Serve, with alternatives and costs. Use it to distinguish availability, durability, and immediate-request routing.

- [Real test request flow](reference/test-request-flow.html)
  Detailed topology and immediate/scheduled sequence diagrams for the final standalone validation. Use it to distinguish function calls, HTTP, Ray actor calls, PostgreSQL operations, container placement, and mounted artifacts.
- [Final standalone validation record](docs/operations/final-standalone-validation-2026-10-06.md)
  Latest recorded native workflow, artifact, persistence, recovery, and legacy-model outcomes. Use it for the deployment diagram's measured evidence and limits.

- [Documentation entry point](docs/README.md)
  Routes to the external primary specification, progress, package maps, and dated evidence. Use it to separate requirements, implementation, and validation.
- [Project glossary](CONTEXT.md)
  Canonical names for inference, workflow, invocation, execution, provider finish, and deployment. Use these names throughout the course.
- [Scheduler package map](execution_scheduler/README.md)
  Current module ownership and claim-ordering rules. Use it to navigate orchestration and persistence.
- [Workflow package map](workflow_execution/README.md)
  Current contracts, input access, processing, cleanup, and attempt-isolated artifacts. Use it to navigate computation.
- [Admission implementation](execution_scheduler/scheduler.py)
  `schedule` contains replay, preparation, deadline, and admission order. First lesson's recommended primary source.
- [PostgreSQL repository](execution_scheduler/postgres.py)
  Production semantic transactions and database concurrency. Read alongside the schema and repository-contract tests.
- [Scheduler tests](tests/test_execution_scheduler.py)
  Observable replay, retries, lease supervision, cancellation, and result visibility. Tests are evidence at their stated boundary.
- [PostgreSQL tests](tests/test_postgres_execution_repository.py)
  Real-database admission races, competing claims, and lease behavior. In-memory tests cannot establish these guarantees.
- [Psycopg transaction documentation](https://www.psycopg.org/psycopg3/docs/basic/transactions.html)
  Explains connection-context commit and rollback. Use it to understand why returning inside `_connection` still waits for transaction exit.
- [Pathology reports, National Cancer Institute](https://www.cancer.gov/about-cancer/diagnosis-staging/diagnosis/pathology-reports-fact-sheet)
  Primary institutional explanation of specimen preparation, stained sections, and microscopic examination. Read the tissue-examination section for domain grounding.
- [OpenSlide overview](https://openslide.org/)
  Explains large, multiresolution whole-slide images and region access. Use it when studying image pyramids and tiling.
- [OpenSlide properties](https://openslide.org/docs/properties/)
  Defines physical resolution metadata, including microns per pixel. Use it when studying model MPP and slide-level selection.
- [October 4 live postmortem](docs/operations/full-wsi-validation-postmortem-2026-10-04.md)
  Concrete real-slide processing evidence, with historical geometry and deployment limits. Use it to connect data flow to observations.
- [October 5 local regression report](docs/reviews/checkpoint-fixes-2026-10-05.md)
  Later ordering and geometry corrections. Use it to avoid treating historical output hashes as the current contract.

## Wisdom

For questions about intended clinical use, model labels, or physical-resolution policy, seek the model owner or a collaborating pathologist. The source and numerical checks alone do not establish these interpretations. No external community has been selected or contacted.

## Gaps

- The classifier's clinical label meaning and intended downstream use need model-owner evidence before teaching scores as clinical conclusions.
- Technical prerequisite depth should be established through the user's explanations in the first lessons.
