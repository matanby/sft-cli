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
    """Convert a glob pattern to a regex for matching tensor names.

    When the pattern contains a `.`, it uses dot-separated segment matching:
    - `*` matches a single segment (no dots)
    - `**` matches one or more segments (including dots)
    - `?` matches a single non-dot character

    When the pattern has NO `.`, it uses simple wildcard matching where
    `*` matches any characters (including dots). This allows intuitive
    patterns like `*lora_A*` to match across segments.
    """
    if "." not in pattern:
        # Simple wildcard mode: * matches anything, ? matches one char
        regex = ""
        for ch in pattern:
            if ch == "*":
                regex += ".*"
            elif ch == "?":
                regex += "."
            else:
                regex += re.escape(ch)
        return regex

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
