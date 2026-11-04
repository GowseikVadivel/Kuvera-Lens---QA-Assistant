"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    qdrant_url: str | None
    qdrant_api_key: str | None

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            qdrant_url=os.environ.get("QDRANT_URL"),
            qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
        )

    def require_groq_api_key(self) -> str:
        if not self.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is required for answer generation. "
                "Set it in the environment or deployment secrets."
            )
        return self.groq_api_key

    def require_qdrant_cloud_config(self) -> tuple[str, str]:
        url = self.qdrant_url
        api_key = self.qdrant_api_key
        if not url or not api_key:
            missing = [
                name
                for name, value in (("QDRANT_URL", url), ("QDRANT_API_KEY", api_key))
                if not value
            ]
            raise RuntimeError(
                "Qdrant Cloud configuration is incomplete. Set "
                + ", ".join(missing)
                + " in the environment or deployment secrets."
            )
        return url, api_key
