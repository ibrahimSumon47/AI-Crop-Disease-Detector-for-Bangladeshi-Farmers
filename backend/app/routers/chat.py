"""
POST /api/chat — Bangla farmer chatbot.

Scope (user-selected): disease/treatment questions AND app-usage help.

Grounding rules (same philosophy as the advice layer):
- The LLM may ONLY answer from the curated Bangla guideline JSON (loaded at
  startup) plus static app-usage answers. It is told to refuse anything
  outside these; no invented doses/products.
- If LLM_API_KEY is not set (default), a rules-based Bangla responder
  answers from the same JSON — so the widget always works, free, with no
  external dependency. Farmers never see an error because of the chat.
"""

import json
import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..config import settings
from ..services import guidelines as gl

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=2000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=12)


class ChatResponse(BaseModel):
    reply: str
    source: str = Field("fallback", description="llm | fallback")


# ---------------------------------------------------------------- helpers --

def _guideline_text() -> str:
    """Compact Bangla guideline dump for the system prompt."""
    if not _guideline_text._cache:
        parts = []
        for cls, g in gl.get_all().items():
            parts.append(
                f"[{cls}] রোগ: {g['disease_bn']} | লক্ষণ: {g['symptoms']} | "
                f"চিকিৎসা: {g['treatment']} | জৈব: {g['organic']} | "
                f"প্রতিরোধ: {g['prevention']} | অফিসার: {g['expert']}"
            )
        _guideline_text._cache = "\n".join(parts)
    return _guideline_text._cache

_guideline_text._cache = ""


APP_HELP_BN = (
    "অ্যাপ ব্যবহার: হোম পেজে 'ছবি তুলুন / সিলেক্ট করুন' চেপে পাতার ছবি দিন "
    "(আক্রান্ত অংশ দিন-আলোয়, পাতা যেন ফ্রেমে বড় দেখা যায়), তারপর 'রোগ নির্ণয় করুন'। "
    "রেজাল্টে রোগের নাম, আস্থার শতাংশ, শীর্ষ-৩ বিকল্প ও বাংলা চিকিৎসা পরামর্শ দেখা যায়। "
    "আস্থা কম থাকলে সবচেয়ে সম্ভাব্য রোগের পরামর্শ কম-আস্থার লেবেলসহ দেখানো হয়। "
    "পুরনো স্ক্যান 'ইতিহাস' পেজে। ইনস্টলে সাইটটি Chrome/Safari-তে খুলে "
    "'Add to Home screen' দিন।"
)

SYSTEM_PROMPT = (
    "তুমি একজন কৃষি সহায়ক, বাংলায় সহজ ভাষায় কথা বলো। নিচের নিয়ম আবশ্যক:\n"
    "১) কেবল প্রদত্ত গাইডলাইন তথ্য ও অ্যাপ-ব্যবহার নির্দেশনা থেকে উত্তর দেবে; "
    "বাইরের কোনো ওষুধের নাম/ডোজ/নতুন পরামর্শ বানাবে না।\n"
    "২) প্রশ্ন যদি গাইডলাইনের বাইরের হয় (আবহাওয়া, বাজারদর, রাজনীতি ইত্যাদি), "
    "বলবে: 'এ বিষয়ে নির্ভরযোগ্য তথ্য আমার কাছে নেই — উপজেলা কৃষি অফিসে জিজ্ঞেস করুন।'\n"
    "৩) সংক্ষিপ্ত, বুলেট-উপযোগী, কৃষকবান্ধব উত্তর।\n\n"
    "=== গাইডলাইন (সত্য উৎস) ===\n{g}\n\n=== অ্যাপ-ব্যবহার তথ্য ===\n{a}\n"
)


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    last = req.messages[-1].content.strip()
    if not last:
        return ChatResponse(reply="কী জানতে চান? রোগ, চিকিৎসা বা অ্যাপ ব্যবহার নিয়ে লিখুন।")

    if settings.llm_api_key:
        reply = await _llm_reply(req.messages)
        if reply:
            return ChatResponse(reply=reply, source="llm")

    # Key-free fallback: rules-based answers from the SAME curated JSON.
    return ChatResponse(reply=_fallback_reply(last))


async def _llm_reply(messages: list[ChatMessage]) -> str | None:
    """LLM path — grounded system prompt; returns None to enable fallback."""
    import httpx

    payload = {
        "model": settings.llm_model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                g=_guideline_text(), a=APP_HELP_BN)},
            *[m.model_dump() for m in messages[-8:]],
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(
                settings.llm_api_url,
                json=payload,
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[chat] LLM failed, using fallback: %s", exc)
        return None


def _fallback_reply(text: str) -> str:
    """Bangla keyword router over the guideline JSON + app help."""
    t = text.lower()

    # 1) Disease keywords FIRST (a question like "ধানের ব্লাস্ট কীভাবে সারাবো"
    #    contains both a disease word and 'কীভাবে' — the disease must win).
    # Class-name keyword match -> that class's advice.
    keywords: dict[str, list[str]] = {
        "potato_early_blight": ["আলু", "আর্লি ব্লাইট", "আগাম পাতা পোড়া"],
        "potato_late_blight": ["অনাগাম পাতা পোড়া"],
        "tomato_late_blight": ["টমেটো"],
        "rice_blast": ["ধান", "ব্লাস্ট"],
        "rice_brown_spot": ["বাদামি দাগ"],
        "tomato_leaf_mold": ["লিফ মোল্ড"],
        "tomato_bacterial_spot": ["ব্যাকটেরিয়াল স্পট"],
    }
    for cls, kws in keywords.items():
        if any(k in t for k in kws):
            g = gl.get_all().get(cls)
            if g:
                return (f"{g['disease_bn']}:\n• লক্ষণ: {g['symptoms']}\n"
                        f"• চিকিৎসা: {g['treatment']}\n• জৈব উপায়: {g['organic']}\n"
                        f"• প্রতিরোধ: {g['prevention']}\n{g['expert']}")

    # 2) App-usage help (requires an app-specific word, not just 'কীভাবে').
    if any(k in t for k in ("অ্যাপ", "ব্যবহার", "install", "ইনস্টল", "apk", "ইতিহাস")):
        return APP_HELP_BN

    return (
        "দুঃখিত, প্রশ্নটি আমার জানা গাইডলাইনে মেলেনি। রোগের নাম বা ফসল লিখুন "
        "(যেমন: ধানের ব্লাস্ট, আলুর পাতা পোড়া, টমেটোর লিফ মোল্ড), অথবা অ্যাপটিতে "
        "পাতার ছবি তুলে নির্ণয় করুন — তারপর ওই রোগ নিয়ে এখানে জিজ্ঞেস করুন।"
    )
