from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    api_key: str


def load_openai_compatible_api(path: str | Path | None = None) -> ApiConfig:
    """Load an OpenAI-compatible embedding endpoint without logging secrets."""
    env_key = os.environ.get("OPENAI_API_KEY", "")
    env_base = os.environ.get("OPENAI_BASE_URL", "")
    if env_key:
        return ApiConfig(
            base_url=(env_base or "https://api.openai.com/v1").rstrip("/"),
            api_key=env_key,
        )

    if not path:
        raise ValueError("OPENAI_API_KEY is not set and no API file was provided")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"API file not found: {path}. Set OPENAI_API_KEY and optional OPENAI_BASE_URL instead."
        )

    raw = path.read_bytes()
    text = None
    for encoding in ("utf-8", "gbk", "gb18030"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="ignore")

    base_match = re.search(r"(https?://[^\s'\"`]+/v1)", text)
    key_match = re.search(r"Authorization:\s*Bearer\s+([A-Za-z0-9_\-]+)", text)
    if not key_match:
        key_match = re.search(r"Bearer\s+([A-Za-z0-9_\-]+)", text)
    if not base_match or not key_match:
        raise ValueError(f"Could not parse base_url/api_key from {path}")
    return ApiConfig(base_url=base_match.group(1).rstrip("/"), api_key=key_match.group(1))
