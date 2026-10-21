"""
Central configuration for the backend, read from environment variables.

Why a single Settings object: every module (CORS, DB, inference, threshold)
imports the same `settings` instance, so deployment behavior is controlled
in ONE place and there are no scattered os.getenv() calls.

All values have safe defaults so `uvicorn app.main:app` works out of the box
in development; production values come from the environment / .env file
(see .env.example in the repo root).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/app -> backend  (the repo's backend dir)
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed application settings, overridable via environment variables."""

    # Load a local .env if present (dev convenience); real env vars win.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- HTTP / CORS -------------------------------------------------------
    # Comma-separated list in the env var; rendered as a list here.
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # --- Database -----------------------------------------------------------
    # SQLite file, created automatically on startup.
    database_url: str = f"sqlite:///{(BASE_DIR / 'app_data.db').as_posix()}"

    # --- Inference ------------------------------------------------------------
    model_path: str = str((BASE_DIR / "app" / "data" / "model.onnx").as_posix())
    labels_path: str = str((BASE_DIR / "app" / "data" / "labels.json").as_posix())

    # Below this softmax confidence the API answers "not sure" instead of
    # guessing. Calibrated on the validation split (ml/reports/threshold.json).
    confidence_threshold: float = 0.97

    # --- Upload limits -------------------------------------------------------
    max_upload_mb: int = 5          # reject images larger than this
    min_image_px: int = 64          # reject tiny/accidental uploads
    input_size: int = 224           # model input (MobileNetV2), H == W

    # --- Optional LLM advice layer ------------------------------------------
    # Empty key => guideline JSON only (static fallback), no external calls.
    llm_api_key: str = ""
    llm_api_url: str = "https://api.openai.com/v1/chat/completions"
    llm_model: str = "gpt-4o-mini"

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse 'a,b,c' -> ['a', 'b', 'c'] for FastAPI's CORSMiddleware."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
