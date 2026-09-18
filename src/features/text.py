"""Нормализация и стемминг (Snowball через PyStemmer)."""

from __future__ import annotations

import re
from functools import lru_cache

import Stemmer

_TOKEN_RE = re.compile(r"[a-zа-я0-9]+")
_STEMMER = Stemmer.Stemmer("russian")

# Короткий список служебных слов, найденных среди «слов запроса, которых нет в объявлении».
STOP_WORDS = frozenset(
    [
        "в",
        "во",
        "на",
        "по",
        "для",
        "с",
        "со",
        "из",
        "из-за",
        "к",
        "ко",
        "у",
        "о",
        "об",
        "от",
        "до",
        "за",
        "под",
        "над",
        "при",
        "без",
        "и",
        "или",
        "а",
        "но",
        "не",
        "ни",
        "что",
        "как",
        "это",
        "то",
        "же",
        "ли",
        "бы",
        "мне",
        "меня",
        "нужно",
        "нужен",
        "нужна",
        "надо",
    ]
)


def normalize(text: str | None) -> str:
    """Нижний регистр, ё→е, неразрывные пробелы → обычные."""
    # Пропуски из pandas приходят как NaN (float), а не None — отсекаем любой не-str.
    if not isinstance(text, str):
        return ""
    return text.lower().replace("ё", "е").replace("\xa0", " ")


def tokenize(text: str | None) -> list[str]:
    return _TOKEN_RE.findall(normalize(text))


@lru_cache(maxsize=2_000_000)
def stem(word: str) -> str:
    # Кэш важен: словарь корпуса — сотни тысяч слов при ~100 млн словоупотреблений.
    return _STEMMER.stemWord(word)


def stems(text: str | None, drop_stop_words: bool = False) -> list[str]:
    toks = tokenize(text)
    if drop_stop_words:
        toks = [t for t in toks if t not in STOP_WORDS]
    return [stem(t) for t in toks]


def stem_string(text: str | None, drop_stop_words: bool = False) -> str:
    """Стеммы через пробел — удобный формат для хранения в parquet и подачи в BM25."""
    return " ".join(stems(text, drop_stop_words))
