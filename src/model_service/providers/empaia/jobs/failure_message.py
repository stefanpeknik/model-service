"""Fit a job's failure message into the 255 characters EMPAIA shows.

A message of entries, `<head>: <entry>; <entry>`, too long is cut on whole
entries and ends with how many were left out, so the user still sees which
entries it named.
"""

from __future__ import annotations


# EMPAIA's limit for a job's error message.
MAX_LENGTH = 255
_SEPARATOR = "; "


def fit_failure_message(message: str) -> str:
    """Shorten a failure message to `MAX_LENGTH` characters, keeping whole entries.

    Args:
        message: The safe failure message; entries after the first `: ` are
            separated by `; `.

    Returns:
        The message unchanged if it fits; otherwise its leading entries that fit
        with `…and N more` after them, or, when not even one fits, its first
        `MAX_LENGTH` characters ending in `…`.
    """
    if len(message) <= MAX_LENGTH:
        return message
    head, colon, rest = message.partition(": ")
    entries = rest.split(_SEPARATOR) if colon else []
    for kept in range(len(entries) - 1, 0, -1):
        shortened = (
            f"{head}: {_SEPARATOR.join(entries[:kept])}"
            f"{_SEPARATOR}…and {len(entries) - kept} more"
        )
        if len(shortened) <= MAX_LENGTH:
            return shortened
    return message[: MAX_LENGTH - 1] + "…"
