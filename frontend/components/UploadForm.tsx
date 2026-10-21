"use client";

/**
 * UploadForm — the photo upload step for farmers.
 *
 * Mobile-first: accept="image/*" + capture="environment" opens the phone
 * camera on Android; on desktop it opens the file picker. Shows a local
 * preview and a big submit button. Syncs state up to the parent page.
 */

import { useRef, useState } from "react";

interface Props {
  onSubmit: (file: File) => void; // parent runs the API call
  busy?: boolean; // parent disables UI during upload
}

export default function UploadForm({ onSubmit, busy = false }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  function pick(f: File | undefined) {
    if (!f) return;
    setFile(f);
    // Revoke the previous object URL so we don't leak memory.
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(f));
  }

  return (
    <section className="rounded-xl border border-leaf-100 bg-white p-4">
      <h2 className="mb-3 text-lg font-bold">১. পাতার ছবি নিন</h2>
      <p className="mb-3 text-sm text-stone-600">
        আক্রান্ত পাতাটি সরাসরি দিন-আলোয়, যেন দাগগুলো স্পষ্ট দেখা যায়।
      </p>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        data-testid="file-input"
        onChange={(e) => pick(e.target.files?.[0])}
      />

      {previewUrl ? (
        <div className="mb-3">
          {/* eslint-disable-next-line @next/next/no-img-element -- object URL, no Next/image processing needed */}
          <img
            src={previewUrl}
            alt="আপলোড করা পাতার প্রিভিউ"
            className="mx-auto max-h-64 rounded-lg border border-stone-200 object-contain"
          />
        </div>
      ) : (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={busy}
          className="flex h-40 w-full flex-col items-center justify-center rounded-lg border-2 border-dashed border-leaf-500 bg-leaf-50 text-leaf-700 hover:bg-leaf-100"
          data-testid="pick-button"
        >
          <span className="text-4xl">📷</span>
          <span className="mt-2 font-semibold">ছবি তুলুন / সিলেক্ট করুন</span>
        </button>
      )}

      <div className="mt-3 flex gap-2">
        {file && (
          <button
            type="button"
            onClick={() => {
              setFile(null);
              if (previewUrl) URL.revokeObjectURL(previewUrl);
              setPreviewUrl(null);
            }}
            disabled={busy}
            className="rounded-lg border border-stone-300 px-4 py-2 text-stone-600"
            data-testid="change-photo"
          >
            অন্য ছবি
          </button>
        )}
        <button
          type="button"
          onClick={() => file && onSubmit(file)}
          disabled={!file || busy}
          className="flex-1 rounded-lg bg-leaf-600 px-4 py-2 font-bold text-white disabled:opacity-40"
          data-testid="diagnose-button"
        >
          {busy ? "পরীক্ষা চলছে…" : "রোগ নির্ণয় করুন"}
        </button>
      </div>
    </section>
  );
}
