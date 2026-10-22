"use client";

/**
 * ResultCard — shows the diagnosis after a successful /predict call:
 * disease (Bangla), confidence, top-3 alternatives, and the treatment
 * advice sections in Bangla. The "not sure" variant is rendered by the
 * uncertain screen (UncertainNotice) instead.
 */

import type { PredictionResponse } from "@/lib/types";

/** 0..1 -> "৯৪%" style display. */
function pct(c: number): string {
  return `${Math.round(c * 100)}%`;
}

const BN_LABELS = {
  symptoms: "লক্ষণ",
  treatment: "চিকিৎসা",
  organic: "জৈব/প্রাকৃতিক উপায়",
  prevention: "প্রতিরোধ",
  expert_threshold: "কখন অফিসারের সাথে কথা বলবেন",
} as const;

export default function ResultCard({ result }: { result: PredictionResponse }) {
  return (
    <section className="rounded-xl border border-leaf-100 bg-white p-4" data-testid="result-card">
      {/* Photo-quality notes (blurry / too dark) — advisory, diagnosis shown below */}
      {result.photo_warnings && result.photo_warnings.length > 0 && (
        <div className="mb-3 space-y-2" data-testid="photo-warnings">
          {result.photo_warnings.map((w, i) => (
            <p key={i} className="rounded-lg bg-warn-100 p-3 text-sm leading-relaxed text-warn-700">
              {w}
            </p>
          ))}
        </div>
      )}

      {/* Cross-crop blight note (model torn between potato/tomato same disease) */}
      {result.family_note && (
        <p className="mb-3 rounded-lg bg-warn-100 p-3 text-xs leading-relaxed text-warn-700" data-testid="family-note">
          {result.family_note}
        </p>
      )}

      {/* Top line: crop + disease + confidence */}
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-bold" data-testid="disease-bn">
            {result.disease_bn}
          </h2>
          <p className="text-sm text-stone-500">
            ফসল: {cropLabel(result.crop)} · {result.disease}
          </p>
        </div>
        <span className="shrink-0 rounded-full bg-leaf-100 px-3 py-1 text-sm font-bold text-leaf-700">
          {pct(result.confidence)}
        </span>
      </div>

      {/* Advice sections, if the guideline exists for this class */}
      {result.advice.source !== "none" && result.advice.symptoms !== null && (
        <div className="space-y-2 text-sm" data-testid="advice-block">
          {(Object.keys(BN_LABELS) as (keyof typeof BN_LABELS)[]).map((key) =>
            result.advice[key] ? (
              <details key={key} className="rounded-lg bg-leaf-50 p-3" open={key === "treatment"}>
                <summary className="cursor-pointer font-bold text-leaf-700">
                  {BN_LABELS[key]}
                </summary>
                <p className="mt-1 leading-relaxed">{result.advice[key]}</p>
              </details>
            ) : null,
          )}
          <p className="text-xs text-stone-400">
            {(result.advice.source === "llm" && "পরামর্শ: স্ট্যাটিক গাইডলাইনভিত্তিক (LLM সহজ ভাষায় লিখেছে)") ||
              "পরামর্শ: কিউরেটেড গাইডলাইন"}
          </p>
        </div>
      )}

      {/* Top-3 alternatives (skip rank 1, already shown above) */}
      <div className="mt-4" data-testid="top3">
        <h3 className="mb-2 text-sm font-bold text-stone-700">শীর্ষ-৩ বিকল্প</h3>
        <ul className="space-y-1">
          {result.top3.slice(1).map((p) => (
            <li key={p.class_name} className="flex items-center justify-between rounded bg-stone-50 px-3 py-1.5 text-sm">
              <span>{p.class_name_bn}</span>
              <span className="text-stone-500">{pct(p.confidence)}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

/** All 11 crops the current model can output (backend/app/data/labels.json). */
const CROP_BN: Record<string, string> = {
  rice: "ধান",
  potato: "আলু",
  tomato: "টম্যাটো",
  chilli: "মরিচ",
  maize: "ভুট্টা",
  grape: "আঙুর",
  cauliflower: "ফুলকপি",
  cucumber: "শসা",
  papaya: "পেঁপে",
  mango: "আম",
  okra: "ঢেঁড়স",
};

function cropLabel(crop: string): string {
  return CROP_BN[crop] ?? crop;
}
