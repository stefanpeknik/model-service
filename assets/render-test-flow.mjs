import { writeFileSync } from "node:fs";

const escape = (value) => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const palette = { local: "#46534b", http: "#17634d", ray: "#63518f", sql: "#996122", files: "#346c8a" };
const text = (x, y, value, className = "label", anchor = "start") => `<text x="${x}" y="${y}" class="${className}" text-anchor="${anchor}">${escape(value)}</text>`;
const rect = (x, y, width, height, className = "box") => `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="6" class="${className}"/>`;
const line = (fromX, fromY, toX, toY, kind, returning = false) => `<path d="M ${fromX} ${fromY} L ${toX} ${toY}" class="edge ${kind}${returning ? " returning" : ""}" marker-end="url(#${kind})"/>`;
const start = (width, height, title, description) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
<title id="title">${escape(title)}</title><desc id="desc">${escape(description)}</desc>
<style>
  text { font-family: system-ui, sans-serif; fill: #26332f; }
  .title { font-size: 25px; font-weight: 600; }
  .label { font-size: 17px; }
  .detail { font-size: 14px; fill: #5d6c64; }
  .lane-title { font-size: 17px; font-weight: 600; }
  .kind { font-size: 12px; font-weight: 700; letter-spacing: .06em; }
  .box { fill: #faf8f3; stroke: #ccd5cb; stroke-width: 1.5; }
  .group { fill: #f3f1e9; stroke: #839589; stroke-width: 1.5; stroke-dasharray: 7 5; }
  .actor { fill: #fff; stroke: #63518f; stroke-width: 1.5; }
  .lifeline { stroke: #ccd5cb; stroke-dasharray: 4 5; }
  .row { stroke: #e8ece4; }
  .edge { fill: none; stroke-width: 2; }
  .returning { stroke-dasharray: 7 5; }
  ${Object.entries(palette).map(([kind, color]) => `.${kind} { stroke: ${color}; } .kind-${kind} { fill: ${color}; }`).join("\n  ")}
</style><defs>${Object.entries(palette).map(([kind, color]) => `<marker id="${kind}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 z" fill="${color}"/></marker>`).join("")}</defs>
<rect width="100%" height="100%" fill="#fff"/>`;

function sequence(filename, title, lanes, rows, notes) {
  const width = 380 + lanes.length * 168;
  const height = 180 + rows.length * 54 + notes.length * 24;
  const positions = lanes.map((_, index) => 456 + index * 168);
  let svg = start(width, height, title, "Read downward. Each row names the function or operation and the transport. Actor lanes identify separate processes, even when they share a container. Dashed arrows are returns. Polling and lease renewal can overlap computation.");
  svg += text(18, 34, title, "title");
  svg += text(18, 60, "Read downward · dashed arrows return · LOCAL stays in one process · RAY crosses actor processes", "detail");
  lanes.forEach((lane, index) => {
    const x = positions[index];
    svg += rect(x - 75, 82, 150, 61);
    svg += text(x, 106, lane[0], "lane-title", "middle");
    svg += text(x, 129, lane[1], "detail", "middle");
    svg += `<path d="M ${x} 145 V ${height - notes.length * 24 - 28}" class="lifeline"/>`;
  });
  rows.forEach((row, index) => {
    const y = 177 + index * 54;
    svg += `<path d="M 18 ${y + 28} H ${width - 18}" class="row"/>`;
    svg += text(18, y, `${index + 1}. ${row.label}`);
    svg += text(18, y + 20, row.detail, "detail");
    const fromX = positions[row.from];
    const toX = positions[row.to];
    if (fromX === toX) {
      svg += `<path d="M ${fromX} ${y - 9} h 35 v 19 h -35" class="edge ${row.kind}" marker-end="url(#${row.kind})"/>`;
      svg += text(fromX + 44, y + 5, row.kind.toUpperCase(), `kind kind-${row.kind}`);
    } else {
      svg += line(fromX, y + 3, toX, y + 3, row.kind, row.returning);
      svg += text((fromX + toX) / 2, y - 7, row.kind.toUpperCase(), `kind kind-${row.kind}`, "middle");
    }
  });
  notes.forEach((note, index) => { svg += text(18, height - notes.length * 24 + index * 24 + 6, note, "detail"); });
  svg += "</svg>\n";
  writeFileSync(new URL(filename, import.meta.url), svg);
}

sequence("test-immediate.svg", "Immediate workflow · one HTTP request waits for the artifact reference", [
  ["Test client", "local Python"], ["API process", "scheduler container"], ["Serve proxy", "Ray entry point"],
  ["Workflow actor", "GPU worker container"], ["Classifier actor", "same worker container"], ["Mounted files", "input/output PVCs"],
], [
  { from: 0, to: 0, kind: "local", label: "Build typed request and deadline", detail: "SDK workflows.run → _run_body" },
  { from: 0, to: 1, kind: "http", label: "POST public /v1/.../run", detail: "JSON through loopback port-forward :18080 → :8080" },
  { from: 1, to: 1, kind: "local", label: "Validate and await dispatch", detail: "api.run_workflow → synchronous.run → dispatch.run" },
  { from: 1, to: 2, kind: "http", label: "POST /workflows/prostate-heatmap/", detail: "HTTPWorkflowDispatch → native serve-svc :8000" },
  { from: 2, to: 3, kind: "ray", label: "Route to the workflow replica", detail: "Serve internal dispatch → ProstateHeatmapApplication.run" },
  { from: 3, to: 3, kind: "local", label: "Validate, create access, run workflow", detail: "run_workflow_http → WorkflowApplication.run" },
  { from: 3, to: 5, kind: "files", label: "Resolve slide, ROI and output paths", detail: "SharedStorageAccess → ProstateHeatmapWorkflow.run" },
  { from: 3, to: 4, kind: "ray", label: "Fetch model tile configuration", detail: "RayHeatmapPredictor → get_config.remote" },
  { from: 3, to: 3, kind: "local", label: "Select level and enumerate tiles", detail: "SlideHeatmapProcessor.build → grid_tiles" },
  { from: 3, to: 5, kind: "files", label: "Read mask windows and slide tiles", detail: "ThreadPoolExecutor → fetch_tissue_tile → OpenSlide" },
  { from: 3, to: 4, kind: "ray", label: "Predict selected tile arrays", detail: "RayHeatmapPredictor.predict → predict.remote" },
  { from: 4, to: 4, kind: "local", label: "Batch and execute cached ONNX model", detail: "serve.batch → BinaryClassifier.predict → session.run → GPU" },
  { from: 4, to: 3, kind: "ray", returning: true, label: "Return tile scores and reconstruct", detail: "MaskBuilder.update_batch / finalize; crop padded extent" },
  { from: 3, to: 5, kind: "files", label: "Write and atomically complete TIFF", detail: "pyvips.tiffsave → os.replace sibling temporary file" },
  { from: 3, to: 3, kind: "local", label: "Check artifact and close access", detail: "WorkflowApplication._validate_result → close_resource" },
  { from: 3, to: 2, kind: "ray", returning: true, label: "Return named WorkflowResult", detail: "heatmap = SharedArtifact(path, media_type)" },
  { from: 2, to: 1, kind: "http", returning: true, label: "Return internal HTTP result", detail: "dispatch validates and decodes the result envelope" },
  { from: 1, to: 0, kind: "http", returning: true, label: "Return public HTTP 200 JSON", detail: "SDK decodes WorkflowResult; file bytes stay on PVC" },
], [
  "Tile read/predict/reconstruct operations repeat and overlap; shown once for readability. get_config returns before tiling starts.",
  "The public request stays open through completion. This computation creates no execution row, claim, lease or provider-finish work.",
]);

sequence("test-scheduled.svg", "Scheduled workflow · commit admission, release the client, process, then poll", [
  ["Test client", "local Python"], ["API + worker", "scheduler process"], ["PostgreSQL", "database container"],
  ["Serve proxy", "Ray entry point"], ["Workflow actor", "GPU worker"], ["Classifier actor", "same GPU worker"], ["Mounted files", "input/output PVCs"],
], [
  { from: 0, to: 1, kind: "http", label: "POST public /v1/.../schedule", detail: "Native test starts with httpx.Client.post; JSON + key + timeout" },
  { from: 1, to: 1, kind: "local", label: "Identify standalone request", detail: "api.submit → scheduler.schedule → standalone.identify" },
  { from: 1, to: 2, kind: "sql", label: "Look up provider-scoped identity", detail: "postgres.find_submission checks key and fingerprint" },
  { from: 1, to: 6, kind: "files", label: "Validate shared-storage inputs", detail: "New admission only: standalone.prepare → input_path" },
  { from: 1, to: 2, kind: "sql", label: "Admit queued processing work", detail: "postgres.admit → transitions.admit → INSERT + uniqueness" },
  { from: 2, to: 1, kind: "sql", returning: true, label: "Commit before admission returns", detail: "Psycopg connection context exits successfully" },
  { from: 1, to: 0, kind: "http", returning: true, label: "Return 202 and execution ID", detail: "Client can close this request; execution remains durable" },
  { from: 0, to: 1, kind: "http", label: "Replay and conflicting-key checks", detail: "Equivalent request → 200; changed timeout → 409" },
  { from: 1, to: 2, kind: "sql", label: "Read identity and fingerprint again", detail: "Replay avoids preparation and additional processing" },
  { from: 1, to: 2, kind: "sql", label: "Claim processing when due", detail: "worker.run_once → claim_due; running + lease + fencing token" },
  { from: 1, to: 3, kind: "http", label: "POST internal workflow request", detail: "worker._execute → dispatch; attempt context + header" },
  { from: 3, to: 4, kind: "ray", label: "Route to the workflow replica", detail: "ProstateHeatmapApplication.run → WorkflowApplication.run" },
  { from: 4, to: 4, kind: "local", label: "Run the same heatmap implementation", detail: "SharedStorageAccess → ProstateHeatmapWorkflow → processor" },
  { from: 4, to: 5, kind: "ray", label: "Get model tile configuration", detail: "RayHeatmapPredictor.configuration → get_config.remote" },
  { from: 4, to: 6, kind: "files", label: "Read slide and mask tiles", detail: "OpenSlide reads; no WSI upload through either HTTP hop" },
  { from: 4, to: 5, kind: "ray", label: "Predict selected tile arrays", detail: "RayHeatmapPredictor.predict → predict.remote" },
  { from: 5, to: 5, kind: "local", label: "Batch and run ONNX on GPU", detail: "Same classifier actor and cached weights as immediate mode" },
  { from: 5, to: 4, kind: "ray", returning: true, label: "Return scores and reconstruct image", detail: "Overlap accumulation → padded-grid resize → crop" },
  { from: 4, to: 6, kind: "files", label: "Complete attempt-isolated TIFF", detail: ".workflow-attempts/<attempt-hash>/outputs/...tiff" },
  { from: 4, to: 3, kind: "ray", returning: true, label: "Validate, clean up and return result", detail: "Named artifact reference; separate actor return" },
  { from: 3, to: 1, kind: "http", returning: true, label: "Return internal HTTP result", detail: "dispatch.run → ProcessingSucceeded settlement value" },
  { from: 1, to: 2, kind: "sql", label: "Store result and enqueue finish", detail: "settle: running + result + work_kind=provider_finish" },
  { from: 1, to: 2, kind: "sql", label: "Claim provider-finish work", detail: "Next worker pass; separate claim and fencing token" },
  { from: 1, to: 1, kind: "local", label: "Finish standalone provider", detail: "StandaloneProviderAdapter.finish is a no-op" },
  { from: 1, to: 2, kind: "sql", label: "Settle finish as succeeded", detail: "ProviderFinishSucceeded clears pending work and lease" },
  { from: 0, to: 1, kind: "http", label: "Poll GET /v1/executions/{id}", detail: "SDK executions.get; each poll is a new HTTP request" },
  { from: 1, to: 2, kind: "sql", label: "Read durable execution snapshot", detail: "scheduler.get → postgres.get → scheduler._view" },
  { from: 1, to: 0, kind: "http", returning: true, label: "Return 200 with succeeded + result", detail: "Public result appears after provider finish succeeds" },
], [
  "API, worker, lease renewal and maintenance are asyncio tasks in the same scheduler process, not separate pods.",
  "The test polls every 1 s during processing. Lease renew runs about every 20 s for the 60 s lease; maintenance polls every 0.5 s.",
  "Replay checks, polling and maintenance can interleave with claims. Rows explain causality, not an observed packet-by-packet timeline.",
  "The HTTP responses for replay and conflict are summarized in row 8. The final poll is shown after finish; earlier polls omit result.",
]);

let topology = start(1440, 1000, "What ran where in the final standalone validation", "Local test client reaches scheduler over a Kubernetes port-forward. Scheduler talks SQL to external PostgreSQL and HTTP to native Ray Serve. Head provides Ray control and entry. Workflow and classifier are separate actors in the GPU worker container. Shared PVCs carry images and artifacts; PostgreSQL has its own data PVC.");
topology += text(24, 38, "Final standalone validation · namespace peknik-ns · native release model-service-validation", "title");
topology += text(24, 65, "Deployment snapshot from the October 6 record, including final restored-state checks. Source/configuration checked October 7.", "detail");
topology += rect(24, 115, 270, 155);
topology += text(42, 143, "Local Python test client", "lane-title");
topology += text(42, 170, "httpx + RationAI SDK");
topology += text(42, 197, "127.0.0.1:18080");
topology += text(42, 224, "port-forward → service :8080", "detail");
topology += text(42, 248, "NativeWorkflowTests", "detail");
topology += rect(370, 115, 355, 220);
topology += text(388, 143, "Scheduler pod", "lane-title");
topology += text(388, 170, "container: execution-scheduler");
topology += text(388, 197, "Uvicorn / FastAPI :8080");
topology += text(388, 224, "API + ExecutionScheduler");
topology += text(388, 251, "ExecutionWorker + renewal tasks");
topology += text(388, 278, "Independent maintenance task");
topology += text(388, 309, "Python function calls share one process", "detail");
topology += rect(820, 115, 590, 205, "group");
topology += text(838, 143, "Native Ray head pod", "lane-title");
topology += text(838, 170, "container: ray-head · Ray 2.58.0");
topology += text(838, 197, "GCS / Serve controller / HTTP entry proxy");
topology += text(838, 224, "serve-svc :8000 → internal replica routing");
topology += text(838, 251, "0 logical CPUs reserved for user deployments", "detail");
topology += text(838, 278, "Workflow and classifier actors are placed on the worker", "detail");
topology += text(838, 301, "KubeRay operator + autoscaler manage pods, outside request calls", "detail");
topology += rect(820, 385, 590, 280, "group");
topology += text(838, 415, "Native GPU worker pod · container: ray-worker", "lane-title");
topology += text(838, 442, "8 logical Ray CPUs / 1 logical GPU · A100 MIG 2g.20gb", "detail");
topology += rect(838, 463, 258, 149, "actor");
topology += text(854, 492, "Workflow replica process", "lane-title");
topology += text(854, 520, "ProstateHeatmapApplication");
topology += text(854, 548, "SlideHeatmapProcessor");
topology += text(854, 576, "4 CPUs · 8 GiB Ray reservation", "detail");
topology += text(854, 598, "Thread pool + libvips/OpenSlide", "detail");
topology += rect(1120, 463, 272, 149, "actor");
topology += text(1136, 492, "Classifier replica process", "lane-title");
topology += text(1136, 520, "BinaryClassifier / ONNX");
topology += text(1136, 548, "TensorRT + CUDA inference");
topology += text(1136, 576, "4 CPUs · 1 GPU · 12 GiB", "detail");
topology += text(1136, 598, "Batch up to 64 tiles", "detail");
topology += text(838, 641, "One replica each; autoscaling_config: null explicitly clears inherited auto policy", "detail");
topology += rect(370, 405, 355, 133);
topology += text(388, 436, "External PostgreSQL pod", "lane-title");
topology += text(388, 465, "postgres:16-alpine · SQL service :5432");
topology += text(388, 492, "Execution rows + leases + results");
topology += text(388, 519, "Managed separately from the Helm chart", "detail");
topology += rect(370, 600, 355, 95);
topology += text(388, 629, "Database data PVC", "lane-title");
topology += text(388, 656, "model-service-validation-postgres-data", "detail");
topology += text(388, 680, "2 GiB NFS · /var/lib/postgresql/data", "detail");
topology += rect(24, 762, 701, 184);
topology += text(42, 792, "Image and artifact storage · filesystem IO", "lane-title");
topology += text(42, 823, "Inputs: model-service-validation-shared → /mnt/validation", "detail");
topology += text(42, 850, "Outputs: model-service-validation-full-wsi-outputs", "detail");
topology += text(42, 876, "→ /mnt/validation/outputs and /mnt/validation/.workflow-attempts", "detail");
topology += text(42, 903, "Both scheduler and GPU worker mount these claims. JSON carries paths.", "detail");
topology += text(42, 929, "Worker also mounts scratch at /mnt/scratch and TensorRT cache at /mnt/cache.", "detail");
topology += rect(820, 762, 590, 184);
topology += text(838, 792, "Initialization and retained cache", "lane-title");
topology += text(838, 823, "MLflow HTTP → model artifact during reconfigure/cache fill", "detail");
topology += text(838, 850, "ONNX session lives in classifier process; inference uses loaded weights", "detail");
topology += text(838, 876, "The numerical verifier used the cached artifact; no extra MLflow read", "detail");
topology += text(838, 903, "EMPAIA AppService is disabled. Legacy RayService is separate.", "detail");
topology += text(838, 929, "Container images contain code; actor placement decides where it executes.", "detail");
topology += line(294, 190, 370, 190, "http");
topology += text(332, 181, "HTTP", "kind kind-http", "middle");
topology += line(725, 213, 820, 213, "http");
topology += text(771, 202, "HTTP", "kind kind-http", "middle");
topology += line(548, 335, 548, 405, "sql");
topology += text(560, 376, "SQL", "kind kind-sql");
topology += line(548, 538, 548, 600, "files");
topology += text(560, 574, "FILES", "kind kind-files");
topology += line(967, 320, 967, 385, "ray");
topology += text(979, 356, "RAY", "kind kind-ray");
topology += line(1096, 537, 1120, 537, "ray");
topology += `<path d="M 820 638 H 760 V 730 H 640 V 762" class="edge files" marker-end="url(#files)"/>`;
topology += text(747, 719, "Worker reads / writes mounted files", "detail", "end");
topology += `<path d="M 370 295 H 320 V 722 H 220 V 762" class="edge files" marker-end="url(#files)"/>`;
topology += text(308, 711, "Scheduler validates paths", "detail", "end");
topology += text(24, 978, "Logical topology, not a packet trace. Serve may also run worker proxies; both sequence diagrams keep proxy routing distinct from replica code.", "detail");
topology += "</svg>\n";
writeFileSync(new URL("test-topology.svg", import.meta.url), topology);
