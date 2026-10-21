"""
Guideline service: serves curated Bangla advice for each disease class.

Two layers:
1. STATIC (always available): looks up backend/app/data/guidelines_bn.json.
2. OPTIONAL LLM layer: if LLM_API_KEY is configured, the static text is sent
   to the model with strict grounding rules (rephrase only, add no facts,
   same language Bangla). If the call fails for any reason we silently fall
   back to the JSON text — a farmer must never get an error because of the
   optional layer.
"""

import json
import logging
from pathlib import Path
from typing import Any

import httpx

from ..config import settings

logger = logging.getLogger(__name__)

# Module-level cache, populated once at startup.
_guidelines: dict[str, dict[str, str]] = {}
_disclaimer: str = ""



def _guidelines_path() -> Path:
    """Resolve guidelines_bn.json next to model.onnx in app/data."""
    return Path(settings.model_path).parent / "guidelines_bn.json"


def load_guidelines() -> None:
    """Read the guidelines file once (called at app startup)."""
    global _guidelines, _disclaimer
    path = _guidelines_path()
    if not path.exists():
        raise RuntimeError(f"Guidelines file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    _guidelines = data["guidelines"]
    _disclaimer = data["_meta"]["disclaimer"]
    logger.info("[guidelines] loaded %d classes from %s", len(_guidelines), path.name)


def get_all() -> dict[str, dict[str, str]]:
    """Full guidelines map (used by the chat fallback router)."""
    return _guidelines


def get_static_advice(class_name: str) -> dict[str, Any] | None:
    """Return the curated advice dict for a class, or None if unknown.

    Unknown class => frontend shows a generic 'upload not recognized' note.
    """
    entry = _guidelines.get(class_name)
    if entry is None:
        return None
    return {
        "disease_bn": entry["disease_bn"],
        "symptoms": entry["symptoms"],
        "treatment": entry["treatment"],
        "organic": entry["organic"],
        "prevention": entry["prevention"],
        "expert_threshold": entry["expert"],
        "disclaimer": _disclaimer,
        "source": "guideline",
    }


async def get_advice(class_name: str) -> dict[str, Any] | None:
    """Static advice, rewritten by the LLM only if the optional layer is on."""
    advice = get_static_advice(class_name)
    if advice is None or not settings.llm_api_key:
        return advice

    rewritten = await _llm_rewrite(advice)
    if rewritten:
        return {**rewritten, "source": "llm", "disclaimer": _disclaimer}
    return advice  # graceful fallback


async def _llm_rewrite(advice: dict[str, Any]) -> dict[str, str] | None:
    """Ask the LLM to simplify/expand the STATIC text in Bangla.

    Hard grounding rules in the system prompt; any failure degrades to None.
    """
    system = (
        "তুমি একজন কৃষি পরামর্শদাতা। নিচের কৃষকের পরামর্শের টেকসটকে শুধু "
        "সহজ বাংলায় পুনর্লিখন করবে — নতুন কোনো ওষুধের নাম, ডোজ বা সত্য যোগ করবে না। "
        "মূল টেকসটের তথ্যই সব সময় সত্য। JSON হিসেবে উত্তর দেবে যেখানে কীগুলো: "
        "symptoms, treatment, organic, prevention, expert_threshold।"
    )
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(advice, ensure_ascii=False)},
        ],
        "temperature": 0.2,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                settings.llm_api_url,
                json=payload,
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            # Strip possible markdown code fences around the JSON.
            if text.startswith("```"):
                text = text.split("```")[1].removeprefix("json").strip()
            parsed = json.loads(text)
            # Every needed field must survive the rewrite, else reject it.
            if all(f in parsed for f in ("symptoms", "treatment", "organic", "prevention", "expert_threshold")):
                return parsed
            return None
    except Exception as exc:  # noqa: BLE001 - LLM layer is best-effort
        logger.warning("[guidelines] LLM rewrite failed, using static text: %s", exc)
        return None
