"use client";

/**
 * UncertainNotice — the honest fallback when confidence < threshold.
 *
 * UX principle (from user feedback): refusing with NO guidance makes the app
 * useless ("what's the point of the app then?"). So this screen shows BOTH:
 *   1. a clear "not sure" banner + agriculture-officer advice (safety), and
 *   2. the MOST LIKELY candidate's full treatment advice, prominently labeled
 *      LOW-CONFIDENCE, so the farmer gets actionable guidance while being
 *      told to verify. The guideline text is still only ever the curated
 *      Bangla JSON — never invented.
 */

import type { PredictionResponse } from "@/lib/types";

function pct(c: number): string {
  return `${Math.round(c * 100)}%`;
}

const BN_LABELS = {
  symptoms: "লক্ষণ",
  treatment: "চিকিৎসা",
  organic: "জৈব/প্রাকৃতিক উপায়",
  prevention: "প্রতিরোধ",
} as const;

export default function UncertainNotice({ result }: { result: PredictionResponse }) {
  const best = result.top3[0];
  const advice = result.advice;

  return (
    <div className="space-y-3">
      {/* 1) Honest banner — safety first */}
      <section className="rounded-xl border border-warn-500 bg-warn-100 p-4" data-testid="uncertain-notice">
        <h2 className="text-lg font-bold text-warn-700">⚠️ নিশ্চিত বলা যাচ্ছে না</h2>
        <p className="mt-1 text-sm leading-relaxed">
          ছবিটি থেকে যথেষ্ট আস্থায় রোগ শনাক্ত করা যায়নি। নিচে{" "}
          <strong>সবচেয়ে সম্ভাব্য রোগ</strong>-এর পরামর্শ দেওয়া হলো — অনুগ্রহ করে
          সেটি ভালো করে মিলিয়ে নিন, এবং নিশ্চিত হতে কৃষি অফিসারের সাথেও কথা বলুন।
        </p>
        {result.family_note && (
          <p className="mt-2 rounded-lg bg-white p-3 text-xs leading-relaxed text-warn-700" data-testid="family-note">
            {result.family_note}
          </p>
        )}
      </section>

      {/* When even the CROP is a guess (<25%): coach a retake, demote advice */}
      {result.crop_unclear && (
        <section className="rounded-xl bg-warn-100 border border-warn-500 p-4" data-testid="crop-unclear-guide">
          <p className="text-sm leading-relaxed">
            🔍 <strong>ফসলটিও অস্পষ্ট হয়ে গেছে</strong> (আস্থা খুব কম)। ভালো নির্ণয়
            পেতে: সকাল/বিকেল দিন-আলোয়, একটি <strong>সন্দেহভাজন পাতা কাছ থেকে</strong> তুলুন —
            পাতা যেন ফ্রেমের বেশিরভাগ জুড়ে থাকে এবং রোগের দাগ স্পষ্ট দেখা যায়।
          </p>
        </section>
      )}

      {/* 2) Best guess advice, clearly labeled LOW-CONFIDENCE */}
      {advice.source !== "none" && advice.treatment && (
        <section className="rounded-xl border border-leaf-100 bg-white p-4" data-testid="best-guess-advice">
          <div className="mb-2 flex items-start justify-between gap-3">
            <div>
              <h3 className="font-bold text-leaf-700">
                সম্ভাব্য রোগ: {result.disease_bn}
                {result.crop_unclear && (
                  <span className="ml-2 text-xs font-normal text-warn-700">(ফসল অস্পষ্ট — নিচের নির্দেশনা মানুন)</span>
                )}
              </h3>
              <p className="text-xs text-stone-500">
                মিলের সম্ভাবনা কম ({pct(result.confidence)}) — এই পরামর্শ সাবধানে নিন
              </p>
            </div>
          </div>

          <div className="space-y-2 text-sm">
            {(Object.keys(BN_LABELS) as (keyof typeof BN_LABELS)[]).map((key) =>
              advice[key] ? (
                <details key={key} className="rounded-lg bg-leaf-50 p-3" open={key === "treatment"}>
                  <summary className="cursor-pointer font-bold text-leaf-700">{BN_LABELS[key]}</summary>
                  <p className="mt-1 leading-relaxed">{advice[key]}</p>
                </details>
              ) : null,
            )}
          </div>
        </section>
      )}

      {/* 3) The full candidate list (kept for transparency) */}
      <section className="rounded-xl bg-white p-4 text-sm">
        <h3 className="mb-2 font-bold text-stone-700">সব সম্ভাবনা</h3>
        <ul className="space-y-1" data-testid="uncertain-candidates">
          {result.top3.map((p) => (
            <li key={p.class_name} className="flex items-center justify-between rounded bg-stone-50 px-3 py-1.5">
              <span>{p.class_name_bn}</span>
              <span className="text-stone-500">{pct(p.confidence)}</span>
            </li>
          ))}
        </ul>
        {advice.expert_threshold && (
          <p className="mt-3 rounded-lg bg-warn-100 p-3 text-xs leading-relaxed">
            🧑‍🌾 {advice.expert_threshold}
          </p>
        )}
      </section>
    </div>
  );
}
