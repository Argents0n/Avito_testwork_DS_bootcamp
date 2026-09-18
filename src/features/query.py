"""Признаки текста запроса: топонимы, намерения, модификаторы."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from .text import STOP_WORDS, stem, stems, tokenize

# Намерения, которые меняют смысл запроса: «скупка телевизоров» — это не ремонт телевизоров.
INTENT_WORDS: dict[str, list[str]] = {
    "buy": ["скупка", "скупаю", "куплю", "купить", "выкуп", "выкупим", "приму", "прием", "сдать", "продать"],
    "rent": ["аренда", "аренду", "прокат", "напрокат", "снять", "сдам", "посуточно", "почасово"],
    "learn": ["обучение", "курсы", "курс", "урок", "уроки", "научиться", "репетитор", "подготовка"],
    "job": ["работа", "вакансия", "требуется", "подработка"],
}
# Модификаторы не описывают услугу; в тексте объявления их обычно нет, требовать их нельзя.
MODIFIER_WORDS = [
    "недорого",
    "дешево",
    "дешевле",
    "бесплатно",
    "срочно",
    "круглосуточно",
    "выезд",
    "выездом",
    "дому",
    "рядом",
    "частный",
    "частник",
    "профессиональный",
    "качественно",
    "хороший",
    "лучший",
]
_INTENT_STEMS = {intent: {stem(w) for w in words} for intent, words in INTENT_WORDS.items()}
_MODIFIER_STEMS = {stem(w) for w in MODIFIER_WORDS}

# Типовые слова адресов: это не топонимы, а их «обвязка».
_ADDRESS_STOP_WORDS = frozenset(
    [
        "улица",
        "ул",
        "область",
        "обл",
        "район",
        "р",
        "н",
        "проспект",
        "пр",
        "кт",
        "городской",
        "округ",
        "поселок",
        "посёлок",
        "пос",
        "переулок",
        "пер",
        "шоссе",
        "село",
        "деревня",
        "д",
        "край",
        "республика",
        "респ",
        "микрорайон",
        "мкр",
        "бульвар",
        "б",
        "р",
        "проезд",
        "набережная",
        "наб",
        "площадь",
        "пл",
        "территория",
        "тер",
        "садовое",
        "товарищество",
        "снт",
        "сельское",
        "поселение",
        "муниципальный",
        "квартал",
        "корпус",
        "строение",
        "стр",
        "линия",
        "тупик",
        "аллея",
        "дом",
        "город",
        "г",
        "станция",
        "метро",
        "км",
    ]
)


def build_gazetteer(
    addresses: Iterable[str], titles: Iterable[str], min_address_df: int = 20, max_title_ratio: float = 0.12
) -> set[str]:
    """Словарь стеммов-топонимов из адресов объявлений."""
    address_df: Counter[str] = Counter()
    for a in addresses:
        address_df.update({s for s in stems(a) if len(s) >= 3 and not any(ch.isdigit() for ch in s)})
    title_df: Counter[str] = Counter()
    for t in titles:
        title_df.update(set(stems(t)))
    stop = {stem(w) for w in _ADDRESS_STOP_WORDS} | {stem(w) for w in STOP_WORDS}
    return {
        s
        for s, n in address_df.items()
        if n >= min_address_df and title_df.get(s, 0) <= n * max_title_ratio and s not in stop
    }


def query_features(text: str, gazetteer: set[str]) -> dict[str, object]:
    toks = tokenize(text)
    content = [t for t in toks if t not in STOP_WORDS]
    st = [stem(t) for t in content]
    st_set = set(st)
    return {
        "q_norm": " ".join(toks),
        "q_stems": " ".join(st),
        "q_n_words": len(content),
        "q_intent": ",".join(sorted(i for i, words in _INTENT_STEMS.items() if st_set & words)),
        "q_has_modifier": bool(st_set & _MODIFIER_STEMS),
        "q_toponyms": " ".join(s for s in st if s in gazetteer),
        "q_has_latin": bool(re.search(r"[a-z]", " ".join(toks))),
        "q_has_digits": bool(re.search(r"\d", " ".join(toks))),
    }
