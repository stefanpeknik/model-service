"""Warm a model by scheduling one blank tile and following it to the end.

Run where the scheduler API is reachable, e.g. inside the scheduler pod:

    oc -n peknik-ns exec -i deploy/model-service-validation-execution-scheduler -- \
        python - prostate-classifier-1 512 < scripts/warm_model.py

A replica that cannot be placed shows up as a "waiting for capacity" message
while the execution stays running.
"""

import json
import sys
import time
import urllib.request

import lz4.frame


BASE = "http://localhost:8080"
model, tile_size = sys.argv[1], int(sys.argv[2])
request = urllib.request.Request(
    f"{BASE}/v1/models/{model}/schedule",
    data=lz4.frame.compress(bytes(tile_size * tile_size * 3)),
    headers={"Idempotency-Key": f"warm-{model}-{int(time.time())}"},
    method="POST",
)
execution = json.load(urllib.request.urlopen(request, timeout=30))["execution"]
print("scheduled", execution["id"], flush=True)
started = time.time()
while time.time() - started < 900:
    url = f"{BASE}/v1/executions/{execution['id']}"
    view = json.load(urllib.request.urlopen(url, timeout=30))
    print(
        f"{time.time() - started:5.0f}s {view['status']} {view.get('message') or ''}",
        flush=True,
    )
    if view["status"] in {"succeeded", "failed", "cancelled"}:
        break
    time.sleep(30)
