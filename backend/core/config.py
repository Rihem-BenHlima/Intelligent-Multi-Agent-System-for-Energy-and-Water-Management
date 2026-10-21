"""
core/config.py
──────────────
Pydantic Settings — reads from environment / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, model_validator
from typing import List
from pathlib import Path
from urllib.parse import quote


class Settings(BaseSettings):
    # hide_input_in_errors: a failed validation must NOT print the raw environment
    # (which contains ANTHROPIC_API_KEY etc.) into logs / tracebacks.
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore", hide_input_in_errors=True)

    # ── MySQL ─────────────────────────────────────────────────────────────
    MYSQL_HOST:     str = "localhost"
    MYSQL_PORT:     int = 3306
    MYSQL_DB:       str = "energy_mgmt"
    MYSQL_USER:     str = "user"
    # MODIFY: set MYSQL_PASSWORD in the root .env. There is deliberately NO
    # default: the app refuses to start without a real (non-placeholder) value.
    MYSQL_PASSWORD: str

    @property
    def database_url(self) -> str:
        return (
            f"mysql+aiomysql://{quote(self.MYSQL_USER, safe='')}:{quote(self.MYSQL_PASSWORD, safe='')}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}"
        )

    # ── Redis ──────────────────────────────────────────────────────────────────
    # MODIFY: set REDIS_PASSWORD in the root .env (no default, see above).
    REDIS_PASSWORD: str
    
    # docker-compose passes its own REDIS_URL (host "redis") to the container.
    REDIS_URL: str = ""

    @field_validator("MYSQL_PASSWORD", "REDIS_PASSWORD")
    @classmethod
    def _reject_missing_or_placeholder(cls, v: str, info) -> str:
        if not v or v.strip().upper().startswith("CHANGE_ME"):
            raise ValueError(
                f"{info.field_name} is empty or still the CHANGE_ME placeholder — "
                "set a strong value in the root .env (see .env.example)"
            )
        return v

    @model_validator(mode="after")
    def _build_redis_url(self):
        if not self.REDIS_URL:
            self.REDIS_URL = f"redis://:{quote(self.REDIS_PASSWORD, safe='')}@localhost:6379/0"
        return self

    # ── Claude ─────────────────────────────────────────────────────────────────
    # MODIFY: set ANTHROPIC_API_KEY in .env (never hardcode a key here)
    ANTHROPIC_API_KEY: str = ""

    # ── CORS ───────────────────────────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000,http://localhost:5174,http://127.0.0.1:5174"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    ML_MODELS_DIR: str = str(Path(__file__).parent.parent / "ml_models")
    DATA_PATH:     str = str(Path(__file__).parent.parent / "data" / "06_validated.csv")
    SHORT_TERM_DATA_PATH: str = str(Path(__file__).parent.parent / "data" / "02_features_15m.csv")

    # ── Prediction limits ──────────────────────────────────────────────────────
    MAX_HORIZON_DAYS: int = 7


settings = Settings()
