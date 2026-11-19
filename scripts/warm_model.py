"""Warm a model by sending one blank LZ4 tile straight to its Ray Serve ingress.

Usage: warm_model.py MODEL TILE_SIZE [RAY_SERVE_ORIGIN]

The origin defaults to $RAY_SERVE_ORIGIN, then http://localhost:8000. Run it where
Ray Serve is reachable, e.g. in a pod of the release or through a port-forward.
"""

import os
import sys
import time
import urllib.error
import urllib.request

import lz4.frame


model, tile_size = sys.argv[1], int(sys.argv[2])
origin = (
    sys.argv[3]
    if len(sys.argv) > 3
    else os.environ.get("RAY_SERVE_ORIGIN", "http://localhost:8000")
)
request = urllib.request.Request(
    f"{origin.rstrip('/')}/{model}/",
    data=lz4.frame.compress(bytes(tile_size * tile_size * 3)),
    headers={"content-type": "application/octet-stream"},
    method="POST",
)
started = time.monotonic()
try:
    with urllib.request.urlopen(request, timeout=900) as response:
        status = response.status
        response.read()
except urllib.error.HTTPError as error:
    status = error.code
print(f"{model}: HTTP {status} in {time.monotonic() - started:.1f}s", flush=True)
sys.exit(0 if status == 200 else 1)
