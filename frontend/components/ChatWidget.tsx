"use client";

/**
 * ChatWidget — floating Bangla chatbot for farmers.
 *
 * Talks to POST /chat (backend). The backend answers from the curated
 * Bangla guideline JSON (grounded LLM when a key is configured, otherwise
 * the key-free rules-based responder), so a wrong answer cannot be invented.
 * Scope per product decision: disease/treatment questions + app-usage help.
 */

import { useRef, useState } from "react";

type Msg = { role: "user" | "assistant"; content: string };

const GREETING: Msg = {
  role: "assistant",
  content:
    "নমস্কার! আমি কৃষি সহায়ক 🌾 রোগ, চিকিৎসা বা অ্যাপ ব্যবহার নিয়ে প্রশ্ন লিখুন — যেমন: \"ধানের ব্লাস্ট কীভাবে সারাবো?\"",
};

export default function ChatWidget() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Msg[]>([GREETING]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    const next: Msg[] = [...messages, { role: "user", content: text }];
    setMessages(next);
    setInput("");
    setBusy(true);
    try {
      const resp = await fetch(`/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: next.map(({ role, content }) => ({ role, content })) }),
      });
      const body = await resp.json();
      setMessages([...next, { role: "assistant", content: body.reply ?? "উত্তর পাওয়া যায়নি।" }]);
    } catch {
      setMessages([...next, { role: "assistant", content: "সংযোগে সমস্যা হয়েছে — আবার চেষ্টা করুন।" }]);
    } finally {
      setBusy(false);
      requestAnimationFrame(() => listRef.current?.scrollTo({ top: 999999 }));
    }
  }

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label="চ্যাটবট খুলুন"
        data-testid="chat-open"
        className="fixed bottom-5 right-5 z-50 flex h-14 w-14 items-center justify-center rounded-full bg-leaf-600 text-2xl text-white shadow-lg"
      >
        💬
      </button>
    );
  }

  return (
    <div className="fixed bottom-5 right-5 z-50 flex h-[70vh] max-h-[560px] w-[min(92vw,380px)] flex-col overflow-hidden rounded-xl border border-leaf-100 bg-white shadow-2xl">
      <div className="flex items-center justify-between bg-leaf-600 px-4 py-2.5 text-white">
        <span className="font-bold">🌾 কৃষি সহায়ক</span>
        <button type="button" onClick={() => setOpen(false)} aria-label="বন্ধ করুন" className="text-xl leading-none">
          ×
        </button>
      </div>
      <div ref={listRef} className="flex-1 space-y-2 overflow-y-auto p-3" data-testid="chat-list">
        {messages.map((m, i) => (
          <div key={i} className={m.role === "user" ? "flex justify-end" : "flex justify-start"}>
            <p
              className={`max-w-[85%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm leading-relaxed ${
                m.role === "user" ? "bg-leaf-600 text-white" : "bg-stone-100 text-stone-800"
              }`}
            >
              {m.content}
            </p>
          </div>
        ))}
        {busy && <p className="animate-pulse text-center text-xs text-stone-400">লিখছে…</p>}
      </div>
      <div className="flex gap-2 border-t border-stone-200 p-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
          placeholder="প্রশ্ন লিখুন…"
          className="flex-1 rounded-lg border border-stone-300 px-3 py-2 text-sm"
          data-testid="chat-input"
        />
        <button
          type="button"
          onClick={() => void send()}
          disabled={busy}
          className="rounded-lg bg-leaf-600 px-4 py-2 text-sm font-bold text-white disabled:opacity-40"
          data-testid="chat-send"
        >
          পাঠান
        </button>
      </div>
    </div>
  );
}
