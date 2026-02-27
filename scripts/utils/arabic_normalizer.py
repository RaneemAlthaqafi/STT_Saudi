"""
Arabic text normalization for ASR evaluation.

Follows the Open Arabic ASR Leaderboard (OALL) methodology:
  1. Unicode NFKC
  2. Whisper BasicTextNormalizer
  3. Remove diacritics (tashkeel)
  4. Normalize alef variants → ا
  5. Normalize alef maqsura ى → ي
  6. Do NOT normalize taa marbouta (OALL standard)
  7. Remove tatweel ـ
  8. Remove punctuation
  9. Normalize Arabic-Indic digits → Western
  10. Lowercase, collapse whitespace

Sources:
  - Open Arabic ASR Leaderboard: https://arxiv.org/html/2412.13788v1
  - SADA ASR Paper: https://arxiv.org/html/2508.12968v1
  - whisper_normalizer: pip install whisper-normalizer
"""

import re
import unicodedata

try:
    from whisper_normalizer.basic import BasicTextNormalizer
    _WHISPER_NORMALIZER = BasicTextNormalizer()
except ImportError:
    _WHISPER_NORMALIZER = None

# Arabic diacritics (tashkeel) — always remove for eval
_DIACRITICS = re.compile(
    r'[\u0617-\u061A\u064B-\u065F\u0670'
    r'\u06D6-\u06DC\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED]'
)

# Punctuation: Arabic + Latin + general Unicode
_PUNCTUATION = re.compile(
    r'[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E'
    r'\u00A0-\u00BF\u060C\u061B\u061F\u0640\u066A-\u066D\u06D4'
    r'\u2000-\u206F\uFD3E-\uFD3F\uFE50-\uFE6F'
    r'\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65]'
)

# Arabic-Indic digits → Western
_INDIC_DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
_EXTENDED_INDIC_DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')

# Zero-width characters
_ZERO_WIDTH = re.compile(r'[\u200b-\u200f\u202a-\u202e\ufeff]')


def normalize_arabic_for_eval(text: str) -> str:
    """
    Normalize Arabic text for ASR evaluation (OALL standard).

    This is the single normalization function used across all evaluation:
    benchmark, per-model eval, and post-training eval.
    """
    if not text:
        return ""

    # 1. Unicode NFKC
    text = unicodedata.normalize("NFKC", text)

    # 2. Whisper BasicTextNormalizer (punctuation, lowercase, whitespace)
    if _WHISPER_NORMALIZER is not None:
        text = _WHISPER_NORMALIZER(text)

    # 3. Remove diacritics
    text = _DIACRITICS.sub('', text)

    # 4. Normalize alef variants → bare alef
    text = re.sub(r'[\u0623\u0625\u0622\u0671]', '\u0627', text)

    # 5. Normalize alef maqsura → yaa
    text = re.sub(r'\u0649', '\u064A', text)

    # 6. taa marbouta: NOT normalized (OALL standard)

    # 7. Remove tatweel (kashida)
    text = re.sub(r'\u0640', '', text)

    # 8. Remove remaining punctuation
    text = _PUNCTUATION.sub(' ', text)

    # 9. Normalize digits
    text = text.translate(_INDIC_DIGITS)
    text = text.translate(_EXTENDED_INDIC_DIGITS)

    # 10. Remove zero-width characters
    text = _ZERO_WIDTH.sub('', text)

    # 11. Lowercase + collapse whitespace
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_arabic_for_training(text: str) -> str:
    """
    Lighter normalization for training data (preserve more info).

    Only does: alef normalization, alef maqsura, tatweel removal,
    zero-width removal, whitespace collapse.
    Does NOT remove diacritics or punctuation.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    # Normalize alef variants
    text = re.sub(r'[\u0623\u0625\u0622\u0671]', '\u0627', text)

    # Normalize alef maqsura
    text = re.sub(r'\u0649', '\u064A', text)

    # Remove tatweel
    text = re.sub(r'\u0640', '', text)

    # Remove zero-width
    text = _ZERO_WIDTH.sub('', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
