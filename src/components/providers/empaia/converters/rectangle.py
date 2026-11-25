"""`rectangle`: a rectangle drawn on the slide, as a region in the `regions` format."""

from __future__ import annotations

from components.providers.empaia.converters._drawn import DrawnShape


class Rectangle(DrawnShape):
    """A rectangle, by its upper-left corner and size.

    The pipeline gets it as one region in level-0 pixels, keeping the
    annotation's ID.
    """

    name = "rectangle"
    ead_type = "rectangle"
