# Deployment Guide (free tiers)

Target setup, exactly as planned in the project spec:

| Piece | Host | Source |
| --- | --- | --- |
| Frontend (Next.js) | Vercel (free) | this GitHub repo, `frontend/` dir |
| Backend + model (FastAPI + ONNX) | Render (free) **or** Hugging Face Spaces | this GitHub repo |

Order matters: deploy the **backend first**, then point the frontend at its URL.

---

## 1. Backend — Render (recommended, ~10 minutes)

The repo root already contains `render.yaml` (Render Blueprint), so no manual
service configuration is needed.

1. Sign in at <https://dashboard.render.com> — you only need one account and
   it must belong to the repository owner.
2. **New + → Blueprint** → select "AI-Crop-Disease-Detector-for-Bangladeshi-Farmers"
   → Configure/Apply. Render reads [`render.yaml`](../render.yaml):
   - build: `pip install -r requirements.txt` (rootDir `backend`)
   - start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - health check: `/health`
3. In the service → **Environment**, set (KEEP the secret out of the repo):
   - `ALLOWED_ORIGINS` = your dev origin now
     (`http://localhost:3000`), and later add the Vercel URL, comma-separated.
   - `CONFIDENCE_THRESHOLD` is already `0.97` (calibrated, see
     `ml/reports/threshold.json`).
   - `LLM_API_KEY` — leave empty. The static Bangla guidelines work fine
     without it; this is the only optional secret.
4. Deploy. When it finishes, open `https://<service>.onrender.com/health`
   and confirm `{"status": "ok", "model_loaded": true}`.

**Free-tier notes (be honest about them):** the service sleeps after ~15 min
of inactivity; the first request then takes ~30–60 s to cold-start.
SQLite (`app_data.db`) resets on each redeploy — acceptable for scan history
in this MVP; switch `DATABASE_URL` to a managed Postgres later if needed.

### Alternative — Hugging Face Spaces (Docker)

1. Create a new Space, SDK = **Docker**, owner = your account.
2. Add these Space secrets/README variables:
   - Space port: 7860 (README `app_port: 7860`)
3. In `backend/Dockerfile` set `ENV PORT=7860` (already documented in-file).
4. Set `ALLOWED_ORIGINS` in the Space's environment/secret section like on Render.
5. The Dockerfile copies `backend/app` including `model.onnx`, so no extra steps.

> Whichever backend host you choose, the deployment URLs and CORS rules are
> identical — only the dashboard differs.

---

## 2. Frontend — Vercel (free, ~5 minutes)

1. Sign in at <https://vercel.com> with the repository owner's account.
2. **Add New → Project** → import
   `ibrahimSumon47/AI-Crop-Disease-Detector-for-Bangladeshi-Farmers`.
3. Set **Root Directory = `frontend`** (Vercel detects Next.js via
   `frontend/vercel.json` + package.json scripts).
4. **Environment Variables** (Production and Preview both):
   - `NEXT_PUBLIC_BACKEND_URL` = the Render/HF backend URL from step 1,
     e.g. `https://crop-disease-api.onrender.com`
5. Deploy. Vercel builds with `npm run build` (passes with no warnings).

## 3. Close the CORS loop (REQUIRED)

The browser enforces CORS, so the backend must explicitly allow the final
frontend origin:

1. Copy your Vercel URL (e.g. `https://crop-detector.vercel.app`).
2. Render dashboard → backend service → Environment → `ALLOWED_ORIGINS` =
   comma-separated list:
   `http://localhost:3000,https://crop-detector.vercel.app`
   (add a 3rd entry if you changed the Vercel domain).
3. Wait for the auto-redeploy, then test in the browser:
   upload a leaf photo on the Vercel site → you should get a diagnosis or
   the honest "নিশ্চিত না" screen. The scan appears under `/history`.

## 4. Verify the deployed stack (checklist)

- [ ] `https://<backend>/health` → `model_loaded: true`
- [ ] `https://<frontend>/` loads the Bangla UI
- [ ] Upload from the site returns a result (or the uncertain screen)
- [ ] `/history` shows the scan you just made
- [ ] An invalid file (e.g. `.txt`) shows the 422 error message
- [ ] Take screenshots for `docs/` (assignment evidence)

## 5. Environment variable summary

| Variable | Where | Value |
| --- | --- | --- |
| `ALLOWED_ORIGINS` | Render/HF env | dev + production frontend origins |
| `CONFIDENCE_THRESHOLD` | Render/HF env | `0.97` |
| `DATABASE_URL` | Render/HF env | default SQLite (or Postgres later) |
| `LLM_API_KEY` | Render/HF secret (optional) | empty by default |
| `NEXT_PUBLIC_BACKEND_URL` | Vercel env | backend base URL |
