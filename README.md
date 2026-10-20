# 🌿 AI Crop Disease Detector for Bangladeshi Farmers

**Mobile-friendly web app that identifies crop diseases from a leaf photo — in Bangla.**

A farmer uploads (or captures) a photo of a plant leaf using a $100 Android phone. The app
detects the crop and the most likely disease, shows a confidence score with top-3
alternatives, and gives **treatment advice written in simple Bangla**. If the model is not
confident, it honestly says *"আমি নিশ্চিত নই — কৃষি কর্মকর্তার সাথে পরামর্শ করুন"* instead of
guessing, because a wrong pesticide recommendation can destroy a farmer's season.

![Status: In Progress](https://img.shields.io/badge/status-in%20progress-yellow)
![Crops](https://img.shields.io/badge/crops-rice%20%7C%20potato%20%7C%20tomato-green)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)

> ## 🇧🇩 কৃষকদের জন্য, বাংলায়
> এই অ্যাপটি পাতার ছবি দেখে ধান, আলু ও টম্যাটোর রোগ শনাক্ত করে এবং সহজ বাংলায় চিকিৎসা
> পরামর্শ দেয়। নিশ্চিত না হলে অ্যাপটি অনুমান করে না — স্পষ্টভাবে বলে যে কৃষি কর্মকর্তার
> পরামর্শ নেওয়া উচিত।

---

## ✨ Planned Features

| Feature | Description |
| --- | --- |
| 📷 **Scan a leaf** | Upload or capture a photo (`capture="environment"` for the back camera), with drag-and-drop on desktop |
| 🧠 **AI diagnosis** | Crop + disease prediction from ~10 disease classes (+ "healthy" per crop) for rice, potato, tomato |
| 📊 **Confidence + top-3** | Confidence bar and three alternatives, so the farmer sees the whole picture |
| 🟢 / 🟡 **Honest uncertainty** | Below the calibrated confidence threshold, the app refuses to guess and advises consulting an agriculture officer |
| 💊 **Bangla treatment advice** | Curated guideline per disease: symptoms, treatment, organic option, prevention, "when to call an expert" |
| 🗂 **Scan history** | Past scans saved in SQLite with results, list + detail + delete |
| 🌐 **Bangla-first UI** | Default Bangla interface with an optional English toggle |

**Supported crops:** rice (ধান), potato (আলু), tomato (টম্যাটো)

---

## 🛠 Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | Next.js (App Router, TypeScript) + Tailwind CSS, Noto Sans Bengali font |
| Backend | FastAPI (Python), SQLite via SQLAlchemy, ONNX Runtime for inference |
| ML | Transfer learning — MobileNetV2 / EfficientNet-B0 (ImageNet pre-trained), TensorFlow/Keras or PyTorch, exported to ONNX |
| Advice pipeline | Curated Bangla guideline JSON (static, expert-reviewed) + optional LLM layer that only rewrites/extends from that JSON — never invents doses or products |
| Hosting (free tier) | Vercel (frontend) + Hugging Face Spaces (Docker) or Render (backend + model) |
| CI | GitHub Actions (backend tests + frontend lint/build) |

---

## 🧭 How it works (request flow)

```
photo upload → FastAPI /predict → validate & preprocess image
→ ONNX model inference (softmax) → confidence threshold check
   ├─ confident → guideline lookup → (optional LLM rewrite) → return diagnosis + advice
   └─ not confident → "পরামর্শ করুন কৃষি কর্মকর্তার সাথে" → return is_uncertain=true
→ save scan metadata to SQLite → JSON response
```

Full details: [docs/architecture.md](docs/architecture.md)

---

## 📁 Project Structure

```
.
├── frontend/            # Next.js app (Bangla-first UI)
├── backend/             # FastAPI app
│   ├── app/
│   │   ├── main.py config.py db.py models.py schemas.py
│   │   ├── routers/     # /predict, /history
│   │   ├── services/    # preprocess, ONNX inference, advice
│   │   └── data/        # model.onnx, labels.json, guidelines_bn.json
│   └── tests/
├── ml/
│   ├── notebooks/       # Kaggle/Colab training notebook (free GPU)
│   ├── src/             # data prep, train, evaluate, calibrate, export
│   └── reports/         # metrics, confusion matrix, threshold, history
├── docs/                # architecture, deployment, model card, screenshots
└── .github/workflows/   # CI
```

---

## 🚧 Project Roadmap

- [x] **Phase 1** — Repository setup
- [ ] **Phase 2** — Data + ML (PlantVillage / PlantDoc / rice leaf dataset, transfer learning, ONNX export)
- [ ] **Phase 3** — Backend (FastAPI, inference service, Bangla advice, history, tests, Docker)
- [ ] **Phase 4** — Frontend (Next.js, Bangla/English, upload, result card, history)
- [ ] **Phase 5** — Integration + QA (docker-compose, field test with real phone photos)
- [ ] **Phase 6** — Deployment (HF Spaces/Render + Vercel)
- [ ] **Phase 7** — Documentation, model card, release v1.0.0

---

## ⚠️ Important Limitations

- **This is a research/educational prototype. Advice text MUST be reviewed by a
  qualified agriculture expert before any real-world use.** The app is not a
  replacement for a plant-pathology expert or an agriculture officer.
- Model metrics and accuracy will be published **only after real training runs** —
  nothing is fabricated ahead of time.
- Performance on field photos (real phones, bad light) will differ from clean
  laboratory datasets; a field-test gap will be measured and reported honestly.

---

## 📜 License

MIT — see [LICENSE](LICENSE).
