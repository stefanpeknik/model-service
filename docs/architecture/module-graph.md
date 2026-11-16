# Module graph

Generated from the runtime imports in `src/model_service` (imports under
`TYPE_CHECKING` are left out). Solid arrows mean "imports"; dashed arrows are
calls across a process or network boundary that no import shows.

## Packages and processes

```mermaid
flowchart LR
  classDef proc fill:#e8f0fe,stroke:#4a6fd1
  classDef ext fill:#f4f4f4,stroke:#999,stroke-dasharray:3

  subgraph sg_api["API process (uvicorn: api.main)"]
    api["api<br/>app_factory, synchronous_runs,<br/>raw_inference_proxy"]
    routes["api.routes.v1<br/>builders · executions · models"]
    http_common["http_common<br/>auth · errors · scheduling"]
    subgraph sg_sched["scheduler"]
      scheduler["scheduler<br/>public, execution_scheduler,<br/>worker, builder_dispatch"]
      transitions["scheduler.transitions"]
      store["scheduler.store<br/>repository · postgres · migrate"]
    end
    providers["providers<br/>provider · registry"]
    standalone["providers.standalone"]
    subgraph sg_emp["providers.empaia"]
      empaia["empaia<br/>provider · registration · settings"]
      jobs["empaia.jobs<br/>routes · adapter · gateways"]
      publishing["empaia.publishing"]
      ead["empaia.ead"]
      app_service["empaia.app_service"]
    end
  end

  subgraph sg_bd["Builder process (Ray Serve: builder_deployment.serve)"]
    builder_deployment["builder_deployment<br/>serve · runner · http_handler"]
    builders["builders<br/>base · registry · input_access"]
    heatmap["builders.heatmap"]
    embeddings["builders.embeddings"]
    slides["builders.slides"]
  end

  inference["inference<br/>catalog · model_clients"]
  shared["shared<br/>cross-process types"]
  models["model_implementations<br/>Ray Serve model apps"]:::proc
  pg[("PostgreSQL")]:::ext
  jes["EMPAIA JES / App Service"]:::ext

  routes --> api & http_common & scheduler & builders & shared
  api --> http_common & scheduler & store & builders & inference & providers & shared
  http_common --> scheduler & shared
  scheduler --> store & shared
  transitions --> scheduler
  store --> scheduler & transitions

  providers --> standalone & empaia
  standalone --> providers & scheduler & builders & shared
  empaia --> providers & jobs & publishing & ead & app_service
  jobs --> providers & app_service & ead & publishing & scheduler & http_common & builders & shared
  publishing --> app_service & shared
  app_service --> jobs & publishing
  ead --> builders

  builder_deployment --> builders & inference & providers & shared
  builders --> heatmap & embeddings & inference & shared
  heatmap --> builders & slides & inference & shared
  embeddings --> builders & slides & inference & shared
  slides --> shared
  inference --> shared

  store -.->|SQL| pg
  scheduler -.->|HTTP run request| builder_deployment
  inference -.->|Ray handle| models
  app_service -.->|HTTP| jes
  jes -.->|HTTP job post| jobs
```

`store ⇄ scheduler`, `app_service ⇄ jobs` and the loop
`scheduler → store → transitions → scheduler` exist at package level only. The
modules on each side import different files, so no import cycle runs at
runtime.

## One scheduled job

```mermaid
sequenceDiagram
  participant C as Caller / EMPAIA JES
  participant R as api.routes.v1 / empaia.jobs.routes
  participant H as http_common.scheduling
  participant S as scheduler (execution_scheduler)
  participant P as provider adapter
  participant DB as scheduler.store → PostgreSQL
  participant W as scheduler.worker
  participant B as builder_deployment
  participant BL as builders.heatmap / embeddings
  participant M as inference → model_implementations

  C->>R: POST schedule
  R->>H: submit()
  H->>S: schedule(request)
  S->>P: identify + prepare
  S->>DB: admit (transitions.admission)
  W->>DB: claim (claim_sql, transitions.leases)
  W->>B: HTTP run (http_builder_dispatch)
  B->>BL: runner → builder.run
  BL->>M: model_clients over Ray handle
  B-->>W: BuilderResult
  W->>DB: record outcome (transitions.attempt_outcomes)
  W->>P: finish (EMPAIA: publishing → app_service)
  C->>R: GET /v1/executions/{id}
```
