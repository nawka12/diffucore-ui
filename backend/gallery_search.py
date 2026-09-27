"""Gallery search over the cached metadata index (pure functions, no I/O).

Query language for the search box: terms split on spaces and commas and must
all match. A bare word is a case-insensitive substring; a "quoted phrase"
matches whole words only; a leading ``-`` excludes; ``field:value`` scopes a
term to one field (see ``FIELDS``). The Filters panel sends exact-match
filters alongside the query.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

# Unscoped terms skip the negative prompt: boilerplate negatives ("bad hands")
# would otherwise match nearly every image.
DEFAULT_FIELDS = ("prompt", "model", "sampler", "scheduler")
TEXT_FIELDS = {"prompt": "prompt", "neg": "neg", "negative": "neg", "model": "model",
               "sampler": "sampler", "scheduler": "scheduler", "lora": "loras"}
EXACT_FIELDS = ("seed", "steps", "cfg", "size")
FIELDS = tuple(TEXT_FIELDS) + EXACT_FIELDS

# [-][field:]("phrase"|word). An unclosed quote runs to the end, so a phrase
# still filters while it is being typed.
_TOKEN_RE = re.compile(r'(-)?(?:(\w+):)?(?:"([^"]*)(?:"|$)|([^\s,]+))')


def parse_query(q: str) -> list[dict]:
    """``[{field, value, rx, exclude}, …]`` from a search-box string.
    An unknown ``field:`` prefix stays part of a plain word (``score:9``)."""
    terms = []
    for m in _TOKEN_RE.finditer(q or ""):
        exclude, field, phrase, word = m.groups()
        if field and field.lower() not in FIELDS:
            word = m.group(0)[1:] if exclude else m.group(0)
            field, phrase = None, None
        value = (phrase if phrase is not None else word or "").strip().lower()
        if not value:
            continue
        rx = None
        if phrase is not None:
            words = r"\s+".join(re.escape(w) for w in value.split())
            rx = re.compile(rf"(?<!\w){words}(?!\w)", re.IGNORECASE)
        terms.append({"field": field.lower() if field else None, "value": value,
                      "rx": rx, "exclude": bool(exclude)})
    return terms


def _text(entry: dict, key: str) -> str:
    v = entry.get(key, "")
    return "\n".join(v) if isinstance(v, list) else str(v)


def _exact_equal(field: str, have: str, want: str) -> bool:
    if field == "cfg":
        try:
            return float(have) == float(want)
        except ValueError:
            pass
    return have.strip().lower() == want


def _term_matches(entry: dict, term: dict) -> bool:
    field, value = term["field"], term["value"]
    if field in EXACT_FIELDS:
        return _exact_equal(field, str(entry.get(field, "")), value)
    keys = (TEXT_FIELDS[field],) if field else DEFAULT_FIELDS
    if term["rx"]:
        return any(term["rx"].search(_text(entry, k)) for k in keys)
    return any(value in _text(entry, k).lower() for k in keys)


def _filter_matches(entry: dict, filters: dict) -> bool:
    """Exact-match filters from the panel. Empty values are ignored."""
    for key in ("model", "sampler", "scheduler", "size", "seed"):
        want = filters.get(key)
        if want and str(entry.get(key, "")) != want:
            return False
    if filters.get("lora") and filters["lora"] not in entry.get("loras", ()):
        return False
    if filters.get("ratings") and entry.get("rating") not in filters["ratings"]:
        return False
    # Dates are the YYYY-MM-DD folder names, so string order is date order.
    if filters.get("date_from") and entry.get("date", "") < filters["date_from"]:
        return False
    if filters.get("date_to") and entry.get("date", "") > filters["date_to"]:
        return False
    return True


def search(entries: list, query: str = "", filters: Optional[dict] = None) -> list:
    """Entries matching every query term and every filter, in index order."""
    terms = parse_query(query)
    filters = filters or {}
    return [
        e for e in entries
        if _filter_matches(e, filters)
        and all(_term_matches(e, t) != t["exclude"] for t in terms)
    ]


def facets(entries: list) -> dict:
    """Distinct values per panel dropdown, most used first, as ``[value, count]``."""
    counts = {k: Counter() for k in ("models", "samplers", "schedulers", "loras", "sizes")}
    for e in entries:
        for key, field in (("models", "model"), ("samplers", "sampler"),
                           ("schedulers", "scheduler"), ("sizes", "size")):
            if e.get(field):
                counts[key][e[field]] += 1
        counts["loras"].update(set(e.get("loras", ())))
    return {k: [[v, n] for v, n in c.most_common()] for k, c in counts.items()}
