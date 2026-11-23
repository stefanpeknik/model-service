# Module graph

Drawn from the runtime imports in `src/model_service` and `src/components` on
2026-11-23 (imports under `TYPE_CHECKING` are left out). Solid arrows mean "imports"; dashed
arrows are calls across a process or network boundary that no import shows.
`tests/test_process_boundaries.py` checks the part that matters most: the API
process loads no Ray, ONNX Runtime, torch, OpenSlide or replica code, and
the shared packages stay light.

## Packages and processes

```mermaid
flowchart LR
  classDef ext fill:#f4f4f4,stroke:#999,stroke-dasharray:3

  subgraph sg_api["API process (uvicorn: api.main)"]
    api["api<br/>app_factory, catalog_mount,<br/>synchronous_runs, analysis_requests"]
    routes["api.routes.v1<br/>analyses · executions · models"]
    http["api.http<br/>auth · errors · scheduling"]
    subgraph sg_sched["api.scheduler"]
      scheduler["scheduler<br/>execution_scheduler, worker, dispatch"]
      transitions["scheduler.transitions"]
      store["scheduler.store<br/>repository · postgres · migrate"]
    end
    subgraph sg_prov["providers"]
      providers["providers<br/>provider · registry"]
      standalone["providers.standalone"]
      empaia["providers.empaia<br/>provider · registration"]
      jobs["empaia.jobs"]
      ead["empaia.ead"]
      app_service["empaia.app_service"]
    end
  end

  subgraph sg_runner["Analysis runner (Ray Serve: analysis_runner.serve)"]
    runner["analysis_runner<br/>run · replica · serve"]
    slides["slides<br/>tile_reader · tissue"]
  end

  subgraph sg_host["Served models (Ray Serve: served_model.serve)"]
    host["served_model<br/>host · batching · guards · replies"]
  end

  subgraph sg_shared["Shared, light: every process"]
    catalog["catalog<br/>loader · served_models · analyses"]
    registry["registry<br/>scan · FOLDERS · EMPAIA pickers"]
    formats["formats"]
    contracts["contracts"]
  end

  subgraph sg_components["components (top-level package: what an author writes)"]
    base["components.base<br/>kinds · Options · Col · Run · Problem · slots"]
    pieces["steps · loaders · runtimes<br/>sources · sinks · empaia"]
  end

  reference["reference<br/>tools/generate.py"]
  pg[("PostgreSQL")]:::ext
  jes["EMPAIA JES / App Service"]:::ext

  api --> routes & http & scheduler & store & catalog & providers & contracts
  routes --> http & scheduler & catalog & formats & contracts
  http --> scheduler & contracts
  scheduler --> api & store & contracts
  store --> scheduler & transitions & contracts
  transitions --> scheduler

  providers --> standalone & empaia
  standalone --> providers & scheduler & contracts
  empaia --> providers & jobs & ead & app_service
  jobs --> providers & app_service & ead & scheduler & http & catalog & registry & base & formats & contracts
  ead --> empaia & catalog & base
  app_service --> contracts

  runner --> catalog & base & providers & host & formats & contracts
  host --> catalog & base

  catalog --> registry & base & pieces & contracts & ead
  registry --> base
  pieces --> base & formats & slides & app_service & contracts
  reference --> catalog & registry & base & formats

  registry -.->|imports each folder by name| pieces
  store -.->|SQL| pg
  scheduler -.->|HTTP run request| runner
  runner -.->|Ray handle: infer, describe| host
  app_service -.->|HTTP| jes
  jes -.->|HTTP job post| jobs
```

A few arrows cross into packages of another process at module level only:

- `catalog` imports `providers.empaia.ead` for the EMPAIA vocabulary (modes,
  triggers, viewer settings) that `expose.empaia` is checked against;
- the EMPAIA converters and publishers in `components.empaia` import the
  gateway errors and types of `providers.empaia.app_service`, and
  `slide_tiles` imports `slides` inside its methods only, so the API never
  loads OpenSlide;
- `analysis_runner` imports `served_model.guards` to check each model's
  output against its declaration.

Between the two top-level packages, `components.base` imports nothing from
`model_service` at run time (`tests/test_process_boundaries.py` checks it).
The component folders import `model_service.formats`, `contracts` and the
EMPAIA gateway; none of those import `components`, and the registry finds
the folders by name, so no import cycle crosses the two packages.

`scheduler ⇄ store`, `empaia ⇄ ead` and `scheduler → api` exist at package
level only; the modules on each side import different files, so no import
cycle runs.

## One scheduled run

```mermaid
sequenceDiagram
  participant C as Caller / EMPAIA JES
  participant R as api.routes.v1 / empaia.jobs.routes
  participant H as api.http.scheduling
  participant S as scheduler (execution_scheduler)
  participant P as provider adapter
  participant DB as scheduler.store → PostgreSQL
  participant W as scheduler.worker
  participant A as analysis_runner
  participant M as served_model hosts

  C->>R: POST /v1/analyses/{a}/schedule
  R->>H: submit()
  H->>S: schedule(request)
  S->>P: identify + prepare
  S->>DB: admit with the resolved analysis (transitions.admission)
  W->>DB: claim a slot in every served model it calls
  W->>A: HTTP run (dispatch) with the stored document
  A->>M: describe() digest check, then infer(tile) per tile
  A-->>W: result: output paths and media types
  W->>DB: record outcome (transitions.attempt_outcomes)
  W->>P: finish (EMPAIA: publishers → app_service)
  C->>R: GET /v1/executions/{id}
```
