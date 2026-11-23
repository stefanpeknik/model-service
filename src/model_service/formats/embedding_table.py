"""`embedding_table` v2: one feature row per tile of a slide level.

Media type `application/vnd.rationai.embedding-table+hdf5; version=2`. An
HDF5 file (`.h5`), whose `schema_version` attribute is 2:

- `coordinates`: int64 (tiles, 2), each tile's top-left corner (x, y) in
  pixels of the slide level the tiles were read at;
- `features`: float32, row i belonging to coordinate i. With one summary a
  row is the model's vector, (tiles, dim); with several it holds one vector
  per summary, in `summary_names` order, (tiles, summaries, dim). A table
  with no tile (every tile was background) is (0, 0) or (0, summaries, 0);
- file attributes, the header:
  - `schema_version`: 2;
  - `model_id`: the served model that computed the features;
  - `summary_names`: the names of its summaries, e.g. `global`,
    `local_mean` (strings);
  - `coordinate_space`: `slide_level`, as `coordinates` are given;
  - `slide_level`: the level read;
  - `level_downsample`: that level's downsample from level 0, so level-0
    pixels are `coordinates * level_downsample`;
  - `slide_width`, `slide_height`: that level's size in pixels;
  - `mpp_x`, `mpp_y`: microns per pixel of that level;
  - `model_mpp`: the microns per pixel the model asks for (the level read is
    the closest one);
  - `tile_size`: tile edge in pixels of the level; a tile covers
    `coordinates` to `coordinates + tile_size`;
  - `stride`: pixels between neighbouring grid tiles;
  - `tile_count`: rows in the table.

Rows follow the grid in row-major order (y, then x), and tiles the tissue
filter or the region of interest dropped get no row. Reading it needs only
`h5py` (`h5py.File(path)["features"][()]`). Written by the `embedding_table`
sink; `TableWriter` appends rows as they come.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from model_service.formats.media_type import UnsupportedFormat, media_type, require


if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    import h5py
    from numpy.typing import NDArray


NAME = "embedding_table"
VERSION = 2
EXTENSION = "h5"
"""The file name suffix, without the dot."""
MEDIA_TYPE = media_type("embedding-table", "hdf5", VERSION)
COORDINATE_SPACE = "slide_level"

KNOWN = (MEDIA_TYPE,)

__all__ = [
    "EXTENSION",
    "KNOWN",
    "MEDIA_TYPE",
    "NAME",
    "VERSION",
    "EmbeddingTable",
    "Header",
    "RowShapeError",
    "TableWriter",
    "UnsupportedFormat",
    "check",
    "read",
    "write",
]


class RowShapeError(ValueError):
    """Features whose shape doesn't fit the table's rows."""


@dataclass(frozen=True, slots=True)
class Header:
    """The header attributes, as described in the module docstring.

    `schema_version`, `coordinate_space` and `tile_count` are not here: the
    first two are fixed, and the writer counts the rows.

    Attributes:
        model_id: The served model that computed the features.
        summary_names: Its summaries, one vector each per tile.
        slide_level: The slide level read.
        level_downsample: That level's downsample from level 0.
        slide_width: That level's width in pixels.
        slide_height: That level's height in pixels.
        mpp_x: Microns per pixel of that level along x.
        mpp_y: Microns per pixel of that level along y.
        model_mpp: The microns per pixel the model asks for.
        tile_size: Tile edge in pixels of the level.
        stride: Pixels between neighbouring grid tiles.
    """

    model_id: str
    summary_names: tuple[str, ...]
    slide_level: int
    level_downsample: float
    slide_width: int
    slide_height: int
    mpp_x: float
    mpp_y: float
    model_mpp: float
    tile_size: int
    stride: int

    def attributes(self) -> dict[str, Any]:
        """The file attributes, in the order they are written.

        Returns:
            Attribute name to value, `tile_count` aside.
        """
        return {
            "schema_version": VERSION,
            "model_id": self.model_id,
            "summary_names": self.summary_names,
            "coordinate_space": COORDINATE_SPACE,
            "slide_level": self.slide_level,
            "level_downsample": self.level_downsample,
            "slide_width": self.slide_width,
            "slide_height": self.slide_height,
            "mpp_x": self.mpp_x,
            "mpp_y": self.mpp_y,
            "model_mpp": self.model_mpp,
            "tile_size": self.tile_size,
            "stride": self.stride,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingTable:
    """An embedding table, as described in the module docstring.

    Attributes:
        header: Where the tiles come from and what computed them.
        coordinates: int64 (tiles, 2), each tile's top-left (x, y).
        features: float32 (tiles, dim), or (tiles, summaries, dim).
    """

    header: Header
    coordinates: NDArray[np.int64]
    features: NDArray[np.float32]


class TableWriter:
    """Writes a table row by row, as tiles finish; use it as a context manager.

    The file is complete only after `close()`; leaving the context without it
    (an error) just releases the file.
    """

    def __init__(self, path: Path, header: Header) -> None:
        """Create the file with its header and an empty `coordinates`.

        Args:
            path: The file to create.
            header: The table's header.
        """
        import h5py

        self._summaries = len(header.summary_names)
        self._file = h5py.File(path, "w")
        try:
            self._file.attrs.update(header.attributes())
            self._coordinates = self._file.create_dataset(
                "coordinates", shape=(0, 2), maxshape=(None, 2), dtype="int64"
            )
        except BaseException:
            self._file.close()
            raise
        self._features: h5py.Dataset | None = None

    def __enter__(self) -> Self:
        """Enter the writer's context.

        Returns:
            The writer.
        """
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the file, finished or not.

        Args:
            kind: The exception type, if the block raised.
            error: The exception, if the block raised.
            traceback: Its traceback.
        """
        self.release()

    def release(self) -> None:
        """Close the file without finishing it, if it is still open."""
        self._file.close()

    def append(self, coordinates: NDArray[Any], features: NDArray[Any]) -> None:
        """Append rows after the ones written so far.

        Args:
            coordinates: (rows, 2) tile corners.
            features: (rows, dim) for one summary, else (rows, summaries, dim).

        Raises:
            RowShapeError: If a row's shape is not the summaries' or differs
                from the rows before it.
        """
        rows = len(coordinates)
        shape = features.shape[1:]
        if self._features is None:
            if len(shape) != (1 if self._summaries == 1 else 2) or (
                self._summaries > 1 and shape[0] != self._summaries
            ):
                raise RowShapeError(
                    f"features shaped {shape} per tile; {self._summaries} "
                    "summaries make one row per summary"
                )
            self._features = self._file.create_dataset(
                "features",
                shape=(0, *shape),
                maxshape=(None, *shape),
                dtype="float32",
            )
        if self._features.shape[1:] != shape:
            raise RowShapeError(
                f"features shaped {shape} per tile after "
                f"{self._features.shape[1:]}; every tile has the same shape"
            )
        count = len(self._coordinates)
        # Grow both datasets before writing either, so their lengths never
        # differ, even if the second write fails.
        self._coordinates.resize(count + rows, axis=0)
        self._features.resize(count + rows, axis=0)
        self._coordinates[count:] = coordinates
        self._features[count:] = features

    def close(self) -> None:
        """Finish the table: `features` when no row came, `tile_count`, then the file."""
        if self._features is None:
            shape = (0, 0) if self._summaries == 1 else (0, self._summaries, 0)
            self._features = self._file.create_dataset(
                "features", shape=shape, dtype="float32"
            )
        self._file.attrs["tile_count"] = len(self._coordinates)
        self._file.close()


def write(path: Path, table: EmbeddingTable) -> None:
    """Write a whole embedding table.

    Args:
        path: File to write.
        table: The header and rows.

    Raises:
        RowShapeError: If the features don't fit the header's summaries.
    """
    with TableWriter(path, table.header) as writer:
        if len(table.coordinates):
            writer.append(table.coordinates, table.features)
        writer.close()


def read(path: Path, media_type: str = MEDIA_TYPE) -> EmbeddingTable:
    """Read an embedding table.

    Args:
        path: `.h5` file to read.
        media_type: The media type the file was stored with.

    Returns:
        The table.

    Raises:
        UnsupportedFormat: For a media type or version this reader doesn't
            know, or a file whose `schema_version` is not 2.
    """
    import h5py

    require(media_type, KNOWN)
    with h5py.File(path, "r") as file:
        attrs = file.attrs
        if attrs.get("schema_version") != VERSION:
            raise UnsupportedFormat(
                f"{path} has schema_version {attrs.get('schema_version')}; "
                f"this reader knows {VERSION}"
            )
        header = Header(
            model_id=_text(attrs["model_id"]),
            summary_names=tuple(_text(name) for name in attrs["summary_names"]),
            slide_level=int(attrs["slide_level"]),
            level_downsample=float(attrs["level_downsample"]),
            slide_width=int(attrs["slide_width"]),
            slide_height=int(attrs["slide_height"]),
            mpp_x=float(attrs["mpp_x"]),
            mpp_y=float(attrs["mpp_y"]),
            model_mpp=float(attrs["model_mpp"]),
            tile_size=int(attrs["tile_size"]),
            stride=int(attrs["stride"]),
        )
        return EmbeddingTable(header, file["coordinates"][()], file["features"][()])


_ATTRIBUTES = (
    "schema_version",
    "model_id",
    "summary_names",
    "coordinate_space",
    "slide_level",
    "level_downsample",
    "slide_width",
    "slide_height",
    "mpp_x",
    "mpp_y",
    "model_mpp",
    "tile_size",
    "stride",
    "tile_count",
)


def check(path: Path) -> list[str]:
    """Check a file against the v2 layout.

    Args:
        path: The file.

    Returns:
        One sentence per problem; empty when the file is a valid table.
    """
    import h5py

    try:
        file = h5py.File(path, "r")
    except OSError as error:
        return [f"not an HDF5 file: {error}"]
    with file:
        attrs = dict(file.attrs)
        problems = [f"missing attribute `{k}`" for k in _ATTRIBUTES if k not in attrs]
        if attrs.get("schema_version", VERSION) != VERSION:
            problems.append(
                f"`schema_version` is {attrs['schema_version']}, not {VERSION}"
            )
        space = attrs.get("coordinate_space", COORDINATE_SPACE)
        if _text(space) != COORDINATE_SPACE:
            problems.append(f"`coordinate_space` is {space}, not {COORDINATE_SPACE}")
        missing = [
            f"missing `{k}`" for k in ("coordinates", "features") if k not in file
        ]
        if missing:
            return problems + missing
        problems += _check_rows(file, attrs)
    return problems


def _check_rows(file: h5py.File, attrs: dict[str, Any]) -> list[str]:
    """Check `coordinates` and `features` against each other and the header.

    Args:
        file: The open table.
        attrs: Its attributes.

    Returns:
        One sentence per problem.
    """
    problems = []
    coordinates, features = file["coordinates"], file["features"]
    if coordinates.dtype != np.int64 or coordinates.shape[1:] != (2,):
        problems.append(
            f"`coordinates` is {coordinates.dtype} {coordinates.shape}, "
            "not int64 (tiles, 2)"
        )
    if features.dtype != np.float32:
        problems.append(f"`features` is {features.dtype}, not float32")
    summaries = len(attrs.get("summary_names", ()))
    rank = 2 if summaries == 1 else 3
    if features.ndim != rank or (rank == 3 and features.shape[1] != summaries):
        want = "(tiles, dim)" if rank == 2 else f"(tiles, {summaries}, dim)"
        problems.append(
            f"`features` is {features.shape}, not {want}: one row per summary"
        )
    if len(features) != len(coordinates):
        problems.append(
            f"{len(features)} rows of features for {len(coordinates)} coordinates"
        )
    if "tile_count" in attrs and attrs["tile_count"] != len(coordinates):
        problems.append(
            f"`tile_count` is {attrs['tile_count']}, but there are "
            f"{len(coordinates)} coordinates"
        )
    if features.size and not np.isfinite(features[()]).all():
        problems.append("`features` has values that are not finite")
    return problems


def _text(value: Any) -> str:
    """An HDF5 string attribute as `str`, whether stored as bytes or text.

    Args:
        value: The attribute.

    Returns:
        The text.
    """
    return value.decode() if isinstance(value, bytes) else str(value)
