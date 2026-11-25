"""`circle`: a circle drawn on the slide, as a region in the `regions` format."""

from __future__ import annotations

from components.providers.empaia.converters._drawn import DrawnShape


class Circle(DrawnShape):
    """A circle, by its centre and radius, as a 64-sided polygon.

    The pipeline gets it as one region in level-0 pixels, keeping the
    annotation's ID.
    """

    name = "circle"
    ead_type = "circle"
