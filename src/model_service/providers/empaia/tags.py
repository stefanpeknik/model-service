"""EMPAIA's tissue and stain tags, from the vendored `tags.csv` of EMPAIA's definitions.

Slides, preprocessing triggers and marketplace listings name tissues and stains
by these IDs, e.g. `PROSTATE` and `H_AND_E`; `ANY` matches every slide.
"""

from __future__ import annotations

import csv
from pathlib import Path


def _load(group: str) -> dict[str, dict[str, str]]:
    """Read one tag group's IDs and their names.

    Args:
        group: The tag group, e.g. `TISSUE`.

    Returns:
        Each tag's name per language code, by tag ID.
    """
    with Path(__file__).with_name("tags.csv").open(newline="") as file:
        return {
            row["TAG_NAME_ID"]: {"EN": row["EN"], "DE": row["DE"]}
            for row in csv.DictReader(file, delimiter=";")
            if row["TAG_GROUP"] == group
        }


TISSUES = _load("TISSUE")
STAINS = _load("STAIN")
# Matches every tissue or stain in a trigger; says nothing in a listing.
ANY = "ANY"
