"""Deep merge of an environment patch over a base document."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from model_service.catalog.errors import YamlPath


def deep_merge(
    base: dict[str, Any], patch: dict[str, Any], at: YamlPath = ()
) -> tuple[dict[str, Any], set[YamlPath]]:
    """Merge `patch` over `base` the way Helm merges values.

    Mappings merge key by key; any other patch value (a scalar or a list)
    replaces the base value, and `null` removes the key.

    Args:
        base: The base document.
        patch: The environment's patch.
        at: The path of these mappings in the document.

    Returns:
        The merged document and the paths whose values the patch supplied.
    """
    merged = dict(base)
    patched: set[YamlPath] = set()
    for key, value in patch.items():
        path = (*at, key)
        if value is None:
            merged.pop(key, None)
            patched.add(path)
        elif isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key], inner = deep_merge(base[key], value, path)
            patched |= inner
        else:
            merged[key] = value
            patched.add(path)
    return merged, patched
