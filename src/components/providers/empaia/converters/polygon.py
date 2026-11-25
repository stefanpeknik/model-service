"""`polygon`: a polygon drawn on the slide, as a region in the `regions` format."""

from __future__ import annotations

from components.providers.empaia.converters._drawn import DrawnShape


class Polygon(DrawnShape):
    """A polygon, by its corners; one that crosses itself is made valid.

    The pipeline gets it as one region in level-0 pixels, keeping the
    annotation's ID.
    """

    name = "polygon"
    ead_type = "polygon"
