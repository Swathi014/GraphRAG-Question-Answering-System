"""
Shared JSON-extraction helper for parsing small local models' LLM output.

Quantized 3B-class models frequently don't follow a JSON schema perfectly:
they may wrap the object in markdown fences, add stray prose, or even emit
more than one JSON object in one response (observed in practice: a model
outputting `{}` followed immediately by the real object). A naive
`raw.find("{")` to `raw.rfind("}")` span breaks on that last case — it turns
two valid objects into one invalid blob.

`extract_first_json_object` instead scans for the first *syntactically
balanced* `{...}` object (tracking string literals so braces inside quoted
text don't confuse the brace count), parses just that, and returns None if
nothing in the response is parseable — which correctly happens for
genuinely malformed model output (e.g. a JSON array missing its closing
bracket), where no automatic repair is safe to guess at.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger("privgraphai.json_utils")


def extract_first_json_object(raw: str) -> Optional[dict]:
    start = raw.find("{")
    if start == -1:
        return None

    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break  # this candidate didn't parse; look for the next '{'
        start = raw.find("{", start + 1)

    logger.warning("Could not parse LLM JSON output: %.300s", raw)
    return None
