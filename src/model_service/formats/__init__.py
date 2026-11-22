"""Data formats: the named, versioned file shapes sinks write and readers open.

One light module per format holds its media type (a vendor type with a
`version` parameter), a dataclass of its contents, `write`, `read` and `check`,
and a docstring that is its reference page. Sinks name the format they write;
publishers, the standalone provider and callers' notebooks read it with the
same module. A layout change bumps the version, and readers refuse versions
they don't know.
"""
