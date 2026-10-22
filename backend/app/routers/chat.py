"""
POST /api/chat — Bangla farmer chatbot.

Scope (user-selected): disease/treatment questions AND app-usage help.

Grounding rules (same philosophy as the advice layer):
- The LLM may ONLY answer from the curated Bangla guideline JSON (loaded at
  startup) plus static app-usage answers. It is told to refuse anything
  outside these; no invented doses/products.
- If LLM_API_KEY is not set (default), a retrieval-style Bangla responder
  answers from the same JSON over ALL 61 classes — so the widget always
  works, free, with no external dependency. Farmers never see an error
  because of the chat.
"""

import json
import logging
import re

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
    # Serverless cold starts may skip the app lifespan, so load the curated
    # guidelines on demand; otherwise the fallback has no data to answer from.
    if not gl.get_all():
        gl.load_guidelines()
    last = req.messages[-1].content.strip()
    if not last:
        return ChatResponse(reply="কী জানতে চান? রোগ, চিকিৎসা বা অ্যাপ ব্যবহার নিয়ে লিখুন।")

    if settings.llm_api_key:
        reply = await _llm_reply(req.messages)
        if reply:
            return ChatResponse(reply=reply, source="llm")

    # Key-free fallback: retrieval over the SAME curated JSON.
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


# ------------------------------------------------------------- responder ---

# The first part of every class name is the crop key (potato_late_blight).
CROP_BN: dict[str, str] = {
    "rice": "ধান",
    "potato": "আলু",
    "tomato": "টমেটো",
    "chilli": "মরিচ",
    "maize": "ভুট্টা",
    "cauliflower": "ফুলকপি",
    "cucumber": "শসা",
    "grape": "আঙুর",
    "mango": "আম",
    "okra": "ঢেঁড়স",
    "papaya": "পেঁপে",
}

# Genitive form used in sentences (ঢেঁড়স -> ঢেঁড়সের, not "ঢেঁড়সর").
CROP_GENITIVE: dict[str, str] = {
    "rice": "ধানের",
    "potato": "আলুর",
    "tomato": "টমেটোর",
    "chilli": "মরিচের",
    "maize": "ভুট্টার",
    "cauliflower": "ফুলকপির",
    "cucumber": "শসার",
    "grape": "আঙুরের",
    "mango": "আমের",
    "okra": "ঢেঁড়সের",
    "papaya": "পেঁপের",
}

CROP_ALIASES: dict[str, tuple[str, ...]] = {
    "rice": ("ধান", "rice", "paddy"),
    "potato": ("আলু", "potato"),
    "tomato": ("টমেটো", "টমেট", "tomato"),
    "chilli": ("মরিচ", "মরীচ", "লঙ্কা", "chilli", "chili"),
    "maize": ("ভুট্টা", "ভূট্টা", "মকাই", "maize", "corn"),
    "cauliflower": ("ফুলকপি", "ফুলকপি", "cauliflower"),
    "cucumber": ("শসা", "cucumber"),
    "grape": ("আঙুর", "আংগুর", "grape"),
    "mango": ("আম", "আমের", "mango"),
    "okra": ("ঢেঁড়স", "ঢোঁড়স", "ভেন্ডি", "okra"),
    "papaya": ("পেঁপে", "papaya"),
}

# Words that mean "I/me" but contain a crop word (আম vs আমি) — removed first.
NOISE_WORDS: tuple[str, ...] = (
    "আমাদের", "আমার", "আমাকে", "আমরা", "আমিও", "আমি",
)

# Common farmer phrasings -> the class they mean. Class-scoped, so the crop
# filter still decides host when the same disease exists on two crops.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "rice_blast": ("ব্লাস্ট", "blast", "কাটা মাশ", "গলাকাটা"),
    "rice_brown_spot": ("বাদামি দাগ", "বাদামী দাগ", "brown spot"),
    "rice_bacterial_leaf_blight": ("ব্যাকটেরিয়াল ব্লাইট", "খয়েরি", "পাতা পোড়া"),
    "rice_tungro": ("তুঙ্গ্রো", "tungro", "হলুদ ডগা"),
    "potato_late_blight": ("লেট ব্লাইট", "অনাগাম", "late blight"),
    "potato_early_blight": ("আর্লি ব্লাইট", "আগাম", "early blight"),
    "potato_healthy": ("সুস্থ আলু", "ভালো আলু"),
    "tomato_late_blight": ("লেট ব্লাইট", "অনাগাম", "late blight"),
    "tomato_early_blight": ("আর্লি ব্লাইট", "আগাম", "early blight"),
    "tomato_leaf_mold": ("লিফ মোল্ড", "পাতার ছাতা", "leaf mold"),
    "tomato_bacterial_spot": ("ব্যাকটেরিয়াল স্পট", "bacterial spot", "কালো স্পট"),
    "tomato_healthy": ("সুস্থ টমেটো"),
    "chilli_curl_virus": ("কার্ল", "কোঁকড়া", "মুচড়ে", "curl"),
    "chilli_white_spot": ("সাদা দাগ", "white spot"),
    "chilli_nutrient_deficiency": ("পুষ্টির অভাব", "সারের অভাব", "হলদে পাতা"),
    "chilli_healthy": ("সুস্থ মরিচ"),
    "maize_common_rust": ("মরচে", "রাস্ট", "rust"),
    "maize_streak_virus": ("স্ট্রিক", "ডোরা", "হলুদ ডোরা", "streak"),
    "maize_gray_leaf_spot": ("ধূসর দাগ", "ধূসর", "gray leaf spot"),
    "maize_northern_leaf_blight": ("উত্তর ব্লাইট", "লম্বা দাগ"),
    "maize_lethal_necrosis": ("নেক্রোসিস", "মড়ক", "necrosis"),
    "maize_healthy": ("সুস্থ ভুট্টা"),
    "cauliflower_black_rot": ("কালো পচা", "ব্ল্যাক রট", "black rot"),
    "cauliflower_insect_holes": ("পোকা", "ছিদ্র", "ফুটো", "insect"),
    "cucumber_downy_mildew": ("ডাউনি মিলডিউ", "downy mildew"),
    "cucumber_anthracnose": ("অ্যানথ্রাকনোজ", "anthracnose"),
    "cucumber_bacterial_wilt": ("উইল্ট", "শুকিয়ে", "নুয়ে", "wilt"),
    "cucumber_pythium_fruit_rot": ("ফল পচা", "পচা ফল"),
    "cucumber_gummy_stem_blight": ("আঠা", "গামি", "gummy"),
    "grape_downy_mildew": ("ডাউনি মিলডিউ", "downy mildew"),
    "grape_powdery_mildew": ("পাউডারি", "সাদা গুঁড়ো", "powdery"),
    "grape_black_rot": ("কালো পচা", "black rot"),
    "grape_esca": ("এসকা", "esca"),
    "mango_anthracnose": ("অ্যানথ্রাকনোজ", "anthracnose"),
    "mango_powdery_mildew": ("পাউডারি", "সাদা গুঁড়ো", "powdery"),
    "mango_sooty_mould": ("কালি", "সুট", "sooty"),
    "mango_die_back": ("ডাই ব্যাক", "ডাল শুকিয়ে", "die back"),
    "mango_cutting_weevil": ("উইভিল", "কাণ্ড কাটা", "weevil"),
    "mango_gall_midge": ("গল মিজ", "মিজ পোকা", "gall midge"),
    "mango_bacterial_canker": ("ক্যানকার", "ক্ষত", "canker"),
    "okra_leaf_curl_virus": ("কার্ল", "কোঁকড়া", "curl"),
    "okra_downy_mildew": ("ডাউনি মিলডিউ", "downy mildew"),
    "okra_alternaria_leaf_spot": ("অল্টারনেরিয়া", "alternaria"),
    "okra_cercospora_leaf_spot": ("সার্কোস্পোরা", "cercospora"),
    "papaya_ring_spot": ("রিং স্পট", "বৃত্তাকার দাগ", "ring spot"),
    "papaya_curl": ("কার্ল", "কোঁকড়া", "curl"),
    "papaya_bacterial_spot": ("ব্যাকটেরিয়াল স্পট", "bacterial spot"),
    "papaya_anthracnose": ("অ্যানথ্রাকনোজ", "anthracnose"),
}

LIST_WORDS: tuple[str, ...] = (
    "কি কি", "কী কী", "কী কি", "কি কী", "তালিকা", "কোন কোন", "সব রোগ", "সব সমস্যা",
)

GREETING_WORDS: tuple[str, ...] = ("আসসালামু", "আসসালাম", "সালাম", "হ্যালো", "হাই", "hello", "hi")
THANKS_WORDS: tuple[str, ...] = ("ধন্যবাদ", "শুকরিয়া", "থ্যাংক", "thanks")
BYE_WORDS: tuple[str, ...] = ("আল্লাহ হাফেজ", "বিদায়", "bye", "খোদা হাফেজ")
APP_WORDS: tuple[str, ...] = ("অ্যাপ", "ইনস্টল", "install", "apk", "ডাউনলোড", "download", "হোম স্ক্রিন")

FIELD_BN = {
    "symptoms": "লক্ষণ",
    "treatment": "চিকিৎসা",
    "organic": "জৈব উপায়",
    "prevention": "প্রতিরোধ",
}

INTENT_WORDS: dict[str, tuple[str, ...]] = {
    "symptoms": ("লক্ষণ", "কেমন দেখ", "দেখতে কেমন", "চেনার", "চেনা"),
    "treatment": (
        "চিকিৎসা", "ঔষধ", "ওষুধ", "স্প্রে", "কী দিব", "কি দিব", "কি দেব",
        "কী দেব", "প্রতিকার", "সারাবো", "সারাতে", "কাটিয়ে", "নিয়ন্ত্রণ",
        "কী করব", "কি করব",
    ),
    "organic": ("জৈব", "অর্গানিক", "নিম", "বোর্দো", "ট্রাইকোডার্মা"),
    "prevention": ("প্রতিরোধ", "আগে থেকে", "আগেই", "যত্ন", "ভবিষ্যতে"),
}

def _hit(q: str, term: str) -> bool:
    """Substring match, with a boundary guard for short Bangla words.

    Without the guard, "আম" (mango) would match inside "আমি/আমার", so a
    question about the app would look like a mango question.
    """
    term = term.strip().lower()
    if not term:
        return False
    if len(term) <= 2 or (term.isascii() and len(term) <= 3):
        return re.search(
            rf"(?<![\u0980-\u09ff]){re.escape(term)}(?![\u0980-\u09ff])", q
        ) is not None
    return term in q


def _clean(text: str) -> str:
    """Lowercase and drop pronoun words so crop detection stays honest."""
    q = text.lower()
    for w in NOISE_WORDS:
        q = q.replace(w, " ")
    return re.sub(r"\s+", " ", q).strip()


def _crops_in(q: str) -> list[str]:
    """Crops the question mentions, in table order."""
    return [c for c, als in CROP_ALIASES.items() if any(_hit(q, a) for a in als)]


def _score(q: str, cls: str, g: dict[str, str]) -> int:
    """How strongly this class matches the question (0 = no match)."""
    score = 0
    short = cls.split("_", 1)[-1].replace("_", " ")
    if short in q:
        score += 5
    for w in short.split():
        if len(w) > 3 and w in q:
            score += 2
    for w in re.split(r"[\s,—()/]+", g["disease_bn"]):
        if len(w) > 3 and _hit(q, w):
            score += 3
    for alias in SYNONYMS.get(cls, ()):
        if _hit(q, alias):
            score += 8
    return score


def _rank(q: str, crops: list[str]) -> list[tuple[int, str, str, dict[str, str]]]:
    """Best-matching classes, highest score first, filtered by crop."""
    rows = []
    for cls, g in gl.get_all().items():
        crop = cls.split("_", 1)[0]
        if crops and crop not in crops:
            continue
        s = _score(q, cls, g)
        if s:
            rows.append((s, crop, cls, g))
    rows.sort(key=lambda r: -r[0])
    return rows


def _intents(q: str) -> list[str]:
    """Which advice sections the question asks for."""
    return [n for n, words in INTENT_WORDS.items() if any(w in q for w in words)]


def _advice(g: dict[str, str], names: list[str]) -> str:
    """Formatted Bangla answer using only the curated fields."""
    lines = [f"{g['disease_bn']}:"]
    for n in names:
        lines.append(f"• {FIELD_BN[n]}: {g[n]}")
    lines.append(g["expert"])
    return "\n".join(lines)


def _crop_conditions(crop: str) -> str:
    """List every condition the app knows for one crop."""
    items = [
        f"• {g['disease_bn']}"
        for cls, g in gl.get_all().items()
        if cls.split("_", 1)[0] == crop
    ]
    head = f"{CROP_BN.get(crop, crop)} — এই অ্যাপ যেসব অবস্থা চেনে ({len(items)}টি):"
    tail = (
        "\n\nকোনটি নিয়ে জানতে চান লিখুন — যেমন: \""
        f"{CROP_GENITIVE.get(crop, CROP_BN.get(crop, crop))} ব্লাইট\" বা \""
        f"{CROP_GENITIVE.get(crop, CROP_BN.get(crop, crop))} রোগের চিকিৎসা\"।"
    )
    return "\n".join([head, *items]) + tail


# --- responder reply ---

def _fallback_reply(text: str) -> str:
    """Key-free Bangla responder over all 61 curated classes.

    Order matters: thanks/bye and app help are checked before disease lookup,
    and a greeting only wins when no crop or disease is mentioned (so
    "হাই, ধানের ব্লাস্ট কী" still answers the disease question).
    """
    q = _clean(text)

    if any(w in q for w in THANKS_WORDS):
        return (
            "আলহামদুলিল্লাহ 🌾 উপকারে আসলে ভালো লাগল। "
            "অন্য কোনো রোগ বা অ্যাপ নিয়ে কিছু জানতে চাইলে লিখুন।"
        )

    if any(w in q for w in BYE_WORDS):
        return (
            "আল্লাহ হাফেজ 🌾 রোগ নিয়ে সন্দেহ থাকলে উপজেলা কৃষি অফিসের পরামর্শ নেবেন।"
        )

    if any(w in q for w in APP_WORDS):
        return APP_HELP_BN

    crops = _crops_in(q)

    # "আলুর রোগ কি কি" — a list request, not a question about one disease.
    if crops and any(w in q for w in LIST_WORDS):
        return _crop_conditions(crops[0])

    ranked = _rank(q, crops)

    if any(w in q for w in GREETING_WORDS) and len(q) < 45 and not crops and not ranked:
        return (
            "ওয়া আলাইকুমুস সালাম 🌾 আমি কৃষি সহায়ক। "
            "১১টি ফসলের ৬১টি রোগ ও অবস্থা নিয়ে বাংলায় পরামর্শ দিতে পারি — "
            "যেমন লিখুন: \"ধানের ব্লাস্ট\", \"আলুর লেট ব্লাইট\", "
            "\"টমেটোর রোগ কি কি\", বা \"অ্যাপ কীভাবে ব্যবহার করব\"।"
        )

    if ("কৃষি অফিস" in q or "অফিসার" in q) and len(q) < 70:
        return (
            "সন্দেহ থাকলে উপজেলা কৃষি অফিসে আক্রান্ত পাতার নমুনা নিয়ে যান। "
            "মাঠের ২৫% এর বেশি আক্রান্ত হলে বা দ্রুত ছড়ালে দেরি করবেন না। "
            "অ্যাপও প্রতিটি ফলাফলে এই পরামর্শ দেয়।"
        )

    if any(w in q for w in ("কোন কোন ফসল", "কী কী ফসল", "কি কি ফসল", "ফসলের তালিকা")):
        rows = []
        for crop, bn in CROP_BN.items():
            n = sum(1 for k in gl.get_all() if k.split("_", 1)[0] == crop)
            if n:
                rows.append(f"• {bn} — {n}টি")
        return "এই মুহূর্তে যেসব ফসলের পরামর্শ আছে:\n" + "\n".join(rows) + (
            "\n\nফসলের নাম লিখলে তার রোগের তালিকাও দিতে পারি।"
        )

    if ranked:
        top_score, top_crop, top_cls, top_g = ranked[0]
        tied = [r for r in ranked if r[0] == top_score]
        tied_crops = {r[1] for r in tied}

        # Same disease on two hosts (potato/tomato blight): ask which crop.
        if len(tied_crops) > 1 and not crops:
            options = " / ".join(CROP_BN.get(c, c) for c in sorted(tied_crops))
            return (
                f"এ রোগটি একাধিক ফসলে হয় — আপনি কোন ফসলের কথা বলছেন? ({options})\n"
                "ফসলের নাম লিখলে ঠিক পরামর্শ দিতে পারব।"
            )

        # Several different diseases match equally: let the farmer pick.
        if len(tied) > 1:
            head = (
                f"{CROP_GENITIVE.get(top_crop, CROP_BN.get(top_crop, top_crop))} "
                "একাধিক রোগ মিলে গেছে — কোনটি জানতে চান?"
            )
            items = [f"• {r[3]['disease_bn']}" for r in tied[:6]]
            return "\n".join([head, *items]) + (
                "\n\nযে রোগটি জানতে চান তার নাম লিখুন, বিস্তারিত পরামর্শ দেব।"
            )

        names = _intents(q) or ["symptoms", "treatment", "organic", "prevention"]
        body = _advice(top_g, names)
        if len(names) < 4:
            body += f"\n\n(পুরো তথ্য চাইলে লিখুন: \"{top_g['disease_bn']}\")"
        return body

    if crops:
        return _crop_conditions(crops[0])

    return (
        "দুঃখিত, প্রশ্নটি আমার জানা গাইডলাইনে মেলেনি।\n"
        "আমি ১১টি ফসলের ৬১টি রোগ/অবস্থা নিয়ে উত্তর দিতে পারি — যেমন: "
        "\"ধানের ব্লাস্ট কীভাবে সারাবো\", \"আলুর লেট ব্লাইটের লক্ষণ\", "
        "\"টমেটোর রোগ কি কি\", \"অ্যাপ কীভাবে ইনস্টল করব\"।\n"
        "ফসল বা রোগের নাম লিখে আবার জিজ্ঞেস করুন, অথবা পাতার ছবি তুলে নির্ণয় করুন।"
    )
