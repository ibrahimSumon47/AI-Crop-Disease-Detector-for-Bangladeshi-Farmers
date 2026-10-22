/**
 * Types mirroring backend/app/schemas.py — the single source of truth is
 * the FastAPI OpenAPI schema; update both together when the API changes.
 */

export type Confidence = number; // 0..1

/** One ranked prediction (rank 1 = most likely). */
export interface PredictionItem {
  rank: number;
  class_name: string;
  class_name_bn: string;
  // The model now covers 11 crops (see backend/app/data/labels.json);
  // this stays a plain string so new crops do not break the UI.
  crop: string;
  confidence: Confidence;
}

/** Bangla advice block (static JSON, optionally rewritten by the LLM). */
export interface Advice {
  symptoms: string | null;
  treatment: string | null;
  organic: string | null;
  prevention: string | null;
  expert_threshold: string | null; // when to call the agriculture officer
  source: "guideline" | "llm" | "none";
}

/** Full POST /predict response. */
export interface PredictionResponse {
  crop: PredictionItem["crop"];
  disease: string;
  disease_bn: string;
  confidence: Confidence;
  top3: PredictionItem[];
  is_uncertain: boolean;
  advice: Advice;
  /** Present when the model is torn between the same disease on potato vs tomato. */
  family_note: string | null;
  /** Bangla photo-quality notes (blurry / too dark). Advisory only. */
  photo_warnings?: string[];
  /** True when even the top-1 is below 0.25 — the crop itself is a guess. */
  crop_unclear: boolean;
  scan_id: number | null;
  created_at: string;
}

/** One row of GET /history. */
export interface HistoryItem {
  id: number;
  created_at: string;
  filename: string | null;
  crop: string;
  disease: string;
  confidence: Confidence;
  is_uncertain: boolean;
}

export interface HistoryResponse {
  items: HistoryItem[];
  total: number;
}
