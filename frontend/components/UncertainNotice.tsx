"use client";

/**
 * UncertainNotice — the honest fallback demanded by the project spec:
 * when the model's confidence is below the calibrated threshold, the app
 * refuses to guess and tells the farmer to consult the agriculture officer,
 * while still showing the most likely candidates for reference.
 */

import type { PredictionResponse } from "@/lib/types";

export default function UncertainNotice({ result }: { result: PredictionResponse }) {
  return (
    <section className="rounded-xl border border-warn-500 bg-warn-100 p-4" data-testid="uncertain-notice">
      <h2 className="text-lg font-bold text-warn-700">⚠️ নিশ্চিত বলা যাচ্ছে না</h2>
      <p className="mt-1 text-sm leading-relaxed">
        ছবিটি থেকে নিশ্চিতভাবে রোগ শনাক্ত করা যায়নি।
        <strong> তাই আমরা কোনো ওষুধের পরামর্শ দিচ্ছি না।</strong> সঠিক চিকিৎসার
        জন্য স্থানীয় <strong>কৃষি অফিসারের সাথে কথা বলুন</strong> — উপজেলা
        কৃষি সম্প্রসারণ অফিসে যোগাযোগ করুন।
      </p>
      <p className="mt-2 text-xs text-stone-500">
        সবচেয়ে সম্ভাব্য ক্লাসগুলো (শতভাগ আস্থা কম):
      </p>
      <ul className="mt-1 space-y-1 text-sm">
        {result.top3.map((p) => (
          <li key={p.class_name} className="flex items-center justify-between rounded bg-white px-3 py-1.5">
            <span>{p.class_name_bn}</span>
            <span className="text-stone-500">{Math.round(p.confidence * 100)}%</span>
          </li>
        ))}
      </ul>
      {result.advice.expert_threshold && (
        <p className="mt-3 rounded-lg bg-white p-3 text-sm">{result.advice.expert_threshold}</p>
      )}
    </section>
  );
}
