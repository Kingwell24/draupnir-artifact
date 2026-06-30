from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


MOJIBAKE_REPLACEMENTS = {
    "\ufffd": "",
}


def ascii_clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("->", " -> ")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonical_token(value: object) -> str:
    text = ascii_clean(value).lower()
    text = text.replace(" -> ", "->").replace("=>", "->")
    text = text.replace("::", ".")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip(" _.,;")


def token_to_words(token: str) -> str:
    token = canonical_token(token)
    token = token.replace(":", " ")
    token = token.replace("->", ".")
    token = re.sub(r"[\[\]{}<>\"'`]", " ", token)
    token = re.sub(r"[_/\\|,;]+", " ", token)
    token = re.sub(r"\s+", " ", token)
    return token.strip()


def ordered_unique(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        token = canonical_token(value)
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def compact_sentence(label: str, value: object) -> str:
    cleaned = ascii_clean(value)
    if not cleaned:
        return ""
    return f"{label}: {cleaned}"
