import json
import logging
import re

logger = logging.getLogger("claims.parser")


def parse_llm_json(raw: str, fallback_key: str = "raw") -> dict:
    """
    LLMs sometimes wrap JSON in markdown fences or add commentary.
    This strips all that and returns a clean dict.
    Falls back gracefully so one bad page never crashes the whole pipeline.
    """
    # strip ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # sometimes the model adds a sentence before the JSON — find the first {
    brace_start = cleaned.find("{")
    if brace_start > 0:
        cleaned = cleaned[brace_start:]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed ({e}), returning raw text under '{fallback_key}'")
        return {fallback_key: raw, "parse_error": True}
