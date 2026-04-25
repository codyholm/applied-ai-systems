"""Defensive parsing helpers for LLM output.

Models occasionally wrap JSON in markdown fences (``` or ```json) despite
prompt instructions otherwise. These helpers strip that wrapping so the
agent layer never has to special-case it.
"""

from __future__ import annotations


def strip_json_fences(text: str) -> str:
    """Strip ```json / ``` fences and surrounding whitespace from text.

    Returns the input stripped if no fences are present, so the helper is
    safe to call unconditionally.
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    s = s[3:]
    if s.lower().startswith("json"):
        s = s[4:]
    s = s.lstrip("\n")
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()
