/**
 * Backend API client.
 *
 * BACKEND_URL comes from NEXT_PUBLIC_BACKEND_URL (set in .env.local, e.g.
 * "http://localhost:8000" in dev or the Render/HF Spaces URL in prod).
 * Defaulting to localhost keeps `npm run dev` working without any config.
 */

import type { HistoryResponse, PredictionResponse } from "./types";

export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

/**
 * POST /predict — upload a leaf photo and get the diagnosis.
 * Throws with a readable message when the backend rejects the image.
 */
export async function predictLeaf(file: File): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file, file.name);

  let response: Response;
  try {
    response = await fetch(`${BACKEND_URL}/predict`, { method: "POST", body: form });
  } catch {
    // Network/CORS/DNS failure — the message must guide the farmer/user.
    throw new Error(
      `সার্ভারে সংযোগ করা যায়নি (${BACKEND_URL})। ইন্টারনেট চালু আছে কিনা দেখে আবার চেষ্টা করুন।`,
    );
  }

  if (!response.ok) {
    // FastAPI validation errors come as {detail: ...}
    let detail = `সার্ভার ত্রুটি (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return response.json();
}

/** GET /history — recent scans (newest first). */
export async function getHistory(
  limit = 20,
  offset = 0,
): Promise<HistoryResponse> {
  const response = await fetch(
    `${BACKEND_URL}/history?limit=${limit}&offset=${offset}`,
  );
  if (!response.ok) {
    throw new Error(`হিস্টরি আনতে সমস্যা হয়েছে (${response.status})।`);
  }
  return response.json();
}
