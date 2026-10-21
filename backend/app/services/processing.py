"""
Post-inference stabilization rules.

WHY THIS EXISTS (measured, not guessed):
The seed model confuses diseases shared between potato and tomato — the two
crops are in the same family (Solanaceae) and genuinely share pathogens
(early blight: Alternaria solani; late blight: Phytophthora infestans).
On the test split, the confusion matrix shows a tomato_early_blight recall of
only 0.20 — many tomato leaves land on potato_early_blight, and PlantVillage
contains far more potato blight images than tomato ones, so "সব ছবিতে আলু"
(the user-observed bias) has this data-imbalance root.

The confusion is often among THE SAME DISEASE on the two crops. A farmer's
correct treatment is nearly identical regardless of which host the blight is
on (same active ingredients, same remove-burn-infected-leaves guidance). So:

- We keep ALL model outputs exactly as produced (no silent rewriting).
- When BOTH the top-1 and a top-3 alternative are the same family/disease
  across the potato<->tomato boundary, we attach `family_note`: a Bangla
  explanation that the model sees this disease family, the advice applies to
  both, and we display neither crop's name as certainty.
- The frontend renders the note in the same prominent place as the result.

This is honest: we show what the model knows, disclose what it cannot decide,
and never invent a different diagnosis.

Rice case (measured): a whole rice-plant photo (not a leaf close-up) still
lands on rice classes — e.g. rice_blast 52% / rice_brown_spot 37% — because
the model has only LEAF-level classes. That is actually the correct family,
but (a) a farmers' plant-level photo often shows no blast spot at all, and
(b) rice leaves of blast/brown-spot/healthy look similar, so picking one
 disease name at 40-50% without seeing a close-up is not honest. We attach a
rice-note asking for a leaf close-up with the visible spot.
"""

from __future__ import annotations

# (disease, other_crop) pairs that map to the shared-family disease.
# Key = canonical class, value = (family label bn, related class in other crop).
CROSS_CROP_FAMILY: dict[str, tuple[str, str]] = {
    "potato_early_blight": ("আগাম পাতা পোড়া", "tomato_early_blight"),
    "tomato_early_blight": ("আগাম পাতা পোড়া", "potato_early_blight"),
    "potato_late_blight": ("অনাগাম পাতা পোড়া", "tomato_late_blight"),
    "tomato_late_blight": ("অনাগাম পাতা পোড়া", "potato_late_blight"),
}

RICE_NOTE = (
    "🌾 আপনি গোটা গাছের ছবি দিয়েছেন। ধানের পাতার ভেতরে ভেজা আকৃতির ধূসর-কেন্দ্রের দাগ "
    "(ব্লাস্ট) আর ছোট বাদামি দাগ — দুটোই সম্ভাব্য, নিশ্চিত করা যায়নি। "
    "সবচেয়ে ভালো নির্ণয় পেতে রোগের দাগযুক্ত পাতা কাছ থেকে (পাতা যেন ফ্রেমে বড়) "
    "ছবি তুলে আবার দিন। নিচের পরামর্শ দাগযুক্ত পাতার জন্য প্রযোজ্য।"
)

FAMILY_NOTE_TMPL = (
    "⚠️ মডেল এই রোগটি দুই গাছেই হতে পারে বলে নিশ্চিত নয়: {family} (আলু ও টমেটো "
    "দুটোতে প্রায় একই রোগ — একই ছত্রাক/জীবাণু দুটোতেই আক্রমণ করে)। মডেলের মধ্যে "
    "আলুর ছবি বেশি থাকায় টমেটোর ছবিও মাঝে মাঝে আলু বলে প্রেডিক্ট করে। নিচের "
    "পরামর্শ দুই ফসলের জন্যই প্রযোজ্য — সবচেয়ে মুল্যবান তথ্য হলো রোগের নাম, "
    "ফসলের নাম নয়।"
)


def attach_family_note(top3: list[dict]) -> str | None:
    """Return a stabilizer note for a prediction, or None.

    Two situations produce notes:

    1. Rice: top-1 is a rice class with low confidence — the model sees the
       right crop family but cannot honestly choose ONE leaf disease from a
       plant-level photo; ask for a leaf close-up.

    2. Cross-crop potato<->tomato tear on the same disease (see below).

    Conditions (all must hold):
      1. top-1 class has a same-disease relative in the other crop; and
      2. that relative ALSO appears somewhere in the top-3 (the model is
         genuinely torn between the two hosts), and
      3. combined probability of the family >= 0.25 (otherwise the note
         would fire on meaningless noise).

    The frontend displays this string prominently so the farmer knows the
    advice applies to their crop even if the headline says "আলু".
    """
    if not top3:
        return None
    top1 = top3[0]["class_name"]

    # Rice: the model sees the right crop family, but from a plant-level shot
    # it cannot honestly name ONE leaf disease — ask for a leaf close-up.
    if top1.startswith("rice_") and top3[0]["confidence"] < 0.60:
        return RICE_NOTE

    info = CROSS_CROP_FAMILY.get(top1)
    if info is None:
        return None

    family_bn, relative = info
    in_top3 = any(p["class_name"] == relative for p in top3[1:])
    family_prob = sum(p["confidence"] for p in top3 if p["class_name"] in (top1, relative))
    if in_top3 and family_prob >= 0.25:
        return FAMILY_NOTE_TMPL.format(family=family_bn)
    return None
