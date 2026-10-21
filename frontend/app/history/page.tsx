"use client";

/**
 * History page — recent scans from the backend, newest first.
 * Client component so it can refetch on demand (small user base, MVP).
 */

import { useCallback, useEffect, useState } from "react";

import { getHistory } from "@/lib/api";
import type { HistoryItem } from "@/lib/types";

function cropIcon(crop: string): string {
  return { rice: "🌾", potato: "🥔", tomato: "🍅" }[crop] ?? "🌿";
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString("bn-BD", {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export default function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [state, setState] = useState<"loading" | "error" | "done" | "empty">("loading");
  const [errorMessage, setErrorMessage] = useState("");

  const load = useCallback(async () => {
    setState("loading");
    try {
      const data = await getHistory(50, 0);
      setItems(data.items);
      setTotal(data.total);
      setState(data.items.length === 0 ? "empty" : "done");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err));
      setState("error");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <section className="rounded-xl border border-leaf-100 bg-white p-4">
      <div className="mb-3 flex items-center justify-between">
        <h1 className="text-lg font-bold">📅 স্ক্যান ইতিহাস</h1>
        <button
          type="button"
          onClick={() => void load()}
          className="rounded-lg border border-stone-300 px-3 py-1 text-sm text-stone-600"
          data-testid="refresh"
        >
          রিফ্রেশ
        </button>
      </div>

      {state === "loading" && <p className="animate-pulse text-stone-500">লোড হচ্ছে…</p>}

      {state === "error" && (
        <p className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{errorMessage}</p>
      )}

      {state === "empty" && (
        <p className="text-stone-500" data-testid="empty">
          এখনো কোনো স্ক্যান নেই। <a href="/" className="text-leaf-700 underline">পাতার ছবি দিয়ে শুরু করুন</a>।
        </p>
      )}

      {state === "done" && (
        <>
          <p className="mb-2 text-sm text-stone-500">মোট {total}টি স্ক্যান</p>
          <ul className="space-y-2" data-testid="history-list">
            {items.map((it) => (
              <li key={it.id} className="flex items-center justify-between rounded-lg bg-stone-50 px-3 py-2">
                <div>
                  <p className="font-semibold">
                    {cropIcon(it.crop)} {cropBN(it.crop)} — {diseaseShort(it.disease)}
                  </p>
                  <p className="text-xs text-stone-500">{formatDate(it.created_at)}</p>
                </div>
                <div className="text-right">
                  <p className="text-sm font-bold text-leaf-700">
                    {Math.round(it.confidence * 100)}%
                  </p>
                  {it.is_uncertain ? (
                    <p className="text-xs text-warn-700">অনিশ্চিত</p>
                  ) : (
                    <p className="text-xs text-stone-400">নিশ্চিত</p>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </>
      )}

      <p className="mt-4 text-sm">
        <a href="/" className="text-leaf-700 underline">← নতুন স্ক্যান করুন</a>
      </p>
    </section>
  );
}

/* Small display helpers (machine name -> Bangla) */
function cropBN(crop: string): string {
  return { rice: "ধান", potato: "আলু", tomato: "টম্যাটো" }[crop] ?? crop;
}
function diseaseShort(disease: string): string {
  // The API sends full machine names ("rice_brown_spot"); match by keyword.
  const keywords: Array<[string, string]> = [
    ["healthy", "সুস্থ পাতা"],
    ["late_blight", "অনাগাম পাতা পোড়া"],
    ["early_blight", "আগাম পাতা পোড়া"],
    ["blast", "ব্লাস্ট"],
    ["brown_spot", "বাদামি দাগ"],
    ["bacterial_spot", "ব্যাকটেরিয়াল স্পট"],
    ["leaf_mold", "লিফ মোল্ড"],
  ];
  for (const [key, bn] of keywords) {
    if (disease.includes(key)) return bn;
  }
  return disease;
}
