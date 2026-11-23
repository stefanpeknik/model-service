"""Write the catalog JSON Schemas and the reference docs from the code.

Usage: PYTHONPATH=src uv run tools/generate.py [--check]

Writes `catalog/.schemas/`, the schema line at the top of each catalog file
and `docs/reference/`. With `--check` it writes nothing and exits 1 if any of
them is stale, which `tests/test_generated.py` checks too.
"""

import sys
from pathlib import Path

from model_service.catalog import CatalogInvalid
from model_service.reference import stale, write


REPO = Path(__file__).resolve().parents[1]
check = "--check" in sys.argv

try:
    paths = stale(REPO) if check else write(REPO)
except CatalogInvalid as error:
    sys.exit(f"The catalog does not load, so nothing was generated:\n{error}")
for path in paths:
    print(path)
if check and paths:
    sys.exit("Stale; run: PYTHONPATH=src uv run tools/generate.py")
