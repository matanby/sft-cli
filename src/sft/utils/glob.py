"""Tensor name glob matching.

Uses `.` as the path separator. `*` matches a single segment,
`**` matches any number of segments.
"""

from __future__ import annotations

import re


def tensor_matches(name: str, pattern: str) -> bool:
    """Check if a tensor name matches a glob pattern."""
    regex = _glob_to_regex(pattern)
    return bool(re.fullmatch(regex, name))


def filter_tensors(
    names: list[str],
    include: str | None = None,
    exclude: str | None = None,
) -> list[str]:
    """Filter tensor names by include/exclude glob patterns."""
    result = names
    if include is not None:
        result = [n for n in result if tensor_matches(n, include)]
    if exclude is not None:
        result = [n for n in result if not tensor_matches(n, exclude)]
    return result


def _glob_to_regex(pattern: str) -> str:
    """Convert a dot-separated glob pattern to a regex.

    - `*` matches a single segment (no dots)
    - `**` matches one or more segments (including dots)
    - `?` matches a single non-dot character
    - `[abc]` character classes work within a segment
    """
    parts = pattern.split(".")
    regex_parts: list[str] = []
    for part in parts:
        if part == "**":
            regex_parts.append(r"(?:[^.]+\.)*[^.]+")
        elif part == "*":
            regex_parts.append(r"[^.]+")
        else:
            segment = ""
            for ch in part:
                if ch == "*":
                    segment += r"[^.]*"
                elif ch == "?":
                    segment += r"[^.]"
                else:
                    segment += re.escape(ch)
            regex_parts.append(segment)
    return r"\.".join(regex_parts)
