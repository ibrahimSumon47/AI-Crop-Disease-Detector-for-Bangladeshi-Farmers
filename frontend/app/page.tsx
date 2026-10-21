"use client";

/**
 * Home: upload -> diagnose -> result.
 * State machine keeps the flow explicit and testable:
 *   idle (upload form) -> loading -> done (result / uncertain) | error
 */

import { useState } from "react";

import ResultCard from "@/components/ResultCard";
import UncertainNotice from "@/components/UncertainNotice";
import UploadForm from "@/components/UploadForm";
import { predictLeaf } from "@/lib/api";
import type { PredictionResponse } from "@/lib/types";

type Phase =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "done"; result: PredictionResponse };

export default function HomePage() {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });

  async function handleSubmit(file: File) {
    setPhase({ kind: "loading" });
    try {
      const result = await predictLeaf(file);
      setPhase({ kind: "done", result });
    } catch (err) {
      setPhase({ kind: "error", message: err instanceof Error ? err.message : String(err) });
    }
  }

  function reset() {
    setPhase({ kind: "idle" });
  }

  return (
    <div className="space-y-4">
      {phase.kind === "idle" && <UploadForm onSubmit={handleSubmit} />}

      {phase.kind === "loading" && (
        <div className="rounded-xl border border-leaf-100 bg-white p-8 text-center" data-testid="loading">
          <p className="animate-pulse text-lg font-semibold text-leaf-700">
            ছবি বিশ্লেষণ হচ্ছে…
          </p>
          <p className="mt-1 text-sm text-stone-500">কয়েক সেকেন্ড লাগতে পারে</p>
        </div>
      )}

      {phase.kind === "error" && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm" data-testid="error">
          <p className="font-bold text-red-700">সমস্যা হয়েছে</p>
          <p className="mt-1">{phase.message}</p>
        </div>
      )}

      {phase.kind === "done" &&
        (phase.result.is_uncertain ? (
          <UncertainNotice result={phase.result} />
        ) : (
          <ResultCard result={phase.result} />
        ))}

      {phase.kind !== "idle" && (
        <button
          type="button"
          onClick={reset}
          className="w-full rounded-lg border border-stone-300 bg-white px-4 py-2 font-semibold text-stone-700"
          data-testid="scan-again"
        >
          🔄 আবার স্ক্যান করুন
        </button>
      )}
    </div>
  );
}
