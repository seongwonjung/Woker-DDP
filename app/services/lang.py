"""Language helpers used across the pipeline."""

from __future__ import annotations


def normalize_lang_code(value: str | None) -> str | None:
    """Normalize language inputs to lower-case ISO-639-1 codes.

    Returns ``None`` for unknown values or when auto-detection is requested.
    Accepts common aliases (e.g. ``"kr"`` → ``"ko"``) and language names.
    """

    if not value:
        return None
    v = str(value).strip().lower()
    if not v or v in {"auto", "none", "null"}:
        return None
    v = v.replace("_", "-")
    if "-" in v:
        v = v.split("-", 1)[0]
    alias = {
        "kr": "ko",
        "jp": "ja",
        "cn": "zh",
        "tw": "zh",
        "ua": "uk",
        "he": "he",
    }
    names = {
        "english": "en",
        "korean": "ko",
        "japanese": "ja",
        "chinese": "zh",
        "mandarin": "zh",
        "cantonese": "zh",
        "german": "de",
        "french": "fr",
        "spanish": "es",
        "portuguese": "pt",
        "brazilian": "pt",
        "russian": "ru",
        "arabic": "ar",
        "hindi": "hi",
        "vietnamese": "vi",
        "indonesian": "id",
        "thai": "th",
        "turkish": "tr",
        "italian": "it",
        "polish": "pl",
        "ukrainian": "uk",
        "dutch": "nl",
        "hebrew": "he",
    }
    if v in names:
        return names[v]
    if v in alias:
        return alias[v]
    if len(v) == 2 and v.isalpha():
        return v
    return None


__all__ = ["normalize_lang_code"]
