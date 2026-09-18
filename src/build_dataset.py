"""Сборка data/processed из сырых parquet.

Статистики по разметке (запрос→подкатегория, регион→города, популярность) сюда не кладём:
их считают внутри фолда по fit-части.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .features.geo import location_centroids, location_level
from .features.params import extract_item_fields, extract_search_filters
from .features.query import build_gazetteer, query_features
from .features.text import normalize, stem_string

RAW = Path("dataset")
OUT = Path("data/processed")

SEARCH_COLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
ITEM_COLS = [
    "item_id",
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
    "item_category_id",
    "item_microcat_id",
    "item_price",
    "item_rating",
    "item_rating_reviews_count",
    "item_location_id",
    "item_latitude",
    "item_longitude",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
]

# Цена ≤ 0 означает «договорная», ≥ 10^8 — мусор вида 999999999999 (EDA, раздел 8).
PRICE_MAX_VALID = 1e8
# Короткие описания («Звоните») совпадают случайно, для поиска дублей их не учитываем.
MIN_DESC_LEN_FOR_DUP = 50
# Поля параметров, которые идут в текстовый индекс; остальное (график, цены, дни) — шум.
PARAMS_TEXT_FIELDS = ["vid", "tip", "tip_auto", "subject", "price_list", "brands"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _parallel_map(func, values: list, chunksize: int = 2000) -> list:
    """Парсинг и стемминг — чистый Python, поэтому распараллеливаем процессами."""
    with ProcessPoolExecutor() as ex:
        return list(ex.map(func, values, chunksize=chunksize))


def _item_text_row(args: tuple[str, str, str]) -> tuple[str, str, str]:
    title, params_text, desc = args
    return stem_string(title), stem_string(params_text), stem_string(desc)


def _desc_hash(desc: str | None) -> str | None:
    d = normalize(desc).strip()
    if len(d) < MIN_DESC_LEN_FOR_DUP:
        return None
    return hashlib.md5(d.encode()).hexdigest()


def build_items(corpus: pd.DataFrame, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Приоритет у версии из корпуса: именно среди неё ищем на бенчмарке.
    corpus = corpus[ITEM_COLS].assign(in_corpus=True)
    train_items = train[ITEM_COLS].drop_duplicates("item_id")
    n_conflicts = train[ITEM_COLS[:4]].drop_duplicates().item_id.duplicated().sum()
    log(f"items: в трейне {len(train_items)} уникальных item_id, с разными версиями текста {n_conflicts}")
    train_only = train_items[~train_items.item_id.isin(corpus.item_id)].assign(in_corpus=False)
    items = pd.concat([corpus, train_only], ignore_index=True)
    log(f"items: всего {len(items)} (корпус {len(corpus)}, только трейн {len(train_only)})")

    log("items: разбор параметров")
    fields = pd.DataFrame(_parallel_map(extract_item_fields, items.item_infm_params_text.tolist()))
    items = pd.concat([items.reset_index(drop=True), fields], axis=1)

    items["lat"] = items.item_latitude.astype(float)
    items["lon"] = items.item_longitude.astype(float)
    price = items.item_price.astype(float)
    items["price_clean"] = price.where((price > 0) & (price < PRICE_MAX_VALID))
    items["log_reviews"] = np.log1p(items.item_rating_reviews_count.fillna(0))
    items["is_services_category"] = items.item_category_id == 114

    log("items: дубли описаний")
    items["desc_hash"] = [_desc_hash(d) for d in items.item_description_raw]
    dup_counts = items.loc[items.in_corpus & items.desc_hash.notna()].groupby("desc_hash").size()
    # Размер группы одинаковых описаний в корпусе: шаблонные объявления сети в разных городах.
    items["desc_dup_count"] = items.desc_hash.map(dup_counts).fillna(1).astype(int)

    log("items: стемминг текстов (самый долгий шаг)")
    params_text = items[PARAMS_TEXT_FIELDS].agg(" | ".join, axis=1)
    rows = _parallel_map(
        _item_text_row, list(zip(items.item_title_raw, params_text, items.item_description_raw)), chunksize=500
    )
    items_text = pd.DataFrame(rows, columns=["title_stems", "params_stems", "desc_stems"])
    items_text.insert(0, "item_id", items.item_id.values)

    items = items.drop(columns=["item_description_raw", "item_latitude", "item_longitude", "item_price", "desc_hash"])
    return items, items_text


def _session_id(row: tuple) -> str:
    """Стабильный id сессии трейна: 16 hex-символов, как формат id в данных."""
    return hashlib.md5("\x1f".join(map(str, row)).encode()).hexdigest()[:16]


def build_queries(
    train: pd.DataFrame, bench: pd.DataFrame, items: pd.DataFrame, gazetteer: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Сессия трейна = одинаковые признаки запроса. Так трейн устроен так же, как бенчмарк:
    # один запрос → несколько выбранных объявлений (EDA, раздел 7).
    train = train.copy()
    train["query_id"] = [_session_id(r) for r in train[SEARCH_COLS].itertuples(index=False)]
    pairs = train.groupby(["query_id", "item_id"]).size().rename("n_choices").reset_index()

    tq = train.drop_duplicates("query_id")[["query_id", *SEARCH_COLS]].assign(source="train")
    bq = bench[["query_id", *SEARCH_COLS]].assign(source="bench")
    queries = pd.concat([tq, bq], ignore_index=True)
    stats = pairs.groupby("query_id").agg(n_relevant=("item_id", "size"), n_choices=("n_choices", "sum"))
    queries = queries.merge(stats, on="query_id", how="left")

    log(f"queries: {len(tq)} сессий трейна, {len(bq)} запросов бенчмарка; разбор фильтров")
    filters = pd.DataFrame(_parallel_map(extract_search_filters, queries.search_infm_params_text.tolist()))
    qf = pd.DataFrame([query_features(q, gazetteer) for q in queries.search_query])
    queries = pd.concat([queries.reset_index(drop=True), filters, qf], axis=1)

    # Уровень локации и центр города: без разметки, только по координатам объявлений.
    item_locs = set(items.item_location_id.unique())
    queries["loc_level"] = location_level(queries.search_location_id, item_locs)
    cent = location_centroids(items)
    queries = queries.merge(cent, left_on="search_location_id", right_on="location_id", how="left").drop(
        columns="location_id"
    )
    return queries, pairs


def report(items: pd.DataFrame, queries: pd.DataFrame, pairs: pd.DataFrame) -> None:
    """Санити-чеки: заполненность полей и совпадение фильтров должны сойтись с EDA."""
    corpus = items[items.in_corpus]
    fill_cols = ["vid", "tip", "tip_auto", "subject", "price_list", "brands", "address"]
    log("заполненность полей в корпусе:\n" + (corpus[fill_cols] != "").mean().round(3).to_string())
    bool_cols = [
        c
        for c in items.columns
        if c.startswith(("work_", "travel_"))
        or c in ("online_booking", "has_contract", "has_guarantee", "works_with_legal")
    ]
    log("флаги в корпусе:\n" + corpus[bool_cols].mean().round(3).to_string())
    log(
        f"опыт указан: {corpus.experience_years.notna().mean():.3f}, "
        f"цена валидна: {corpus.price_clean.notna().mean():.3f}, "
        f"в группах дублей: {(corpus.desc_dup_count > 1).mean():.3f}"
    )

    j = pairs.merge(queries[queries.source == "train"], on="query_id").merge(items, on="item_id")
    checks = {
        "Вид услуги (EDA 98.4 %)": ("f_vid", "vid"),
        "Тип услуги (EDA 95.4 %)": ("f_tip", "tip"),
        "Предмет (EDA 97.9 %)": ("f_subject", "subject"),
    }
    for name, (f, i) in checks.items():
        m = j[f] != ""
        ok = [fv in iv for fv, iv in zip(j.loc[m, f], j.loc[m, i])]
        log(f"совпадение фильтра {name}: {np.mean(ok):.3f} на {m.sum()} парах")
    m = j.f_tip_auto != ""
    ok = [fv in p for fv, p in zip(j.loc[m, "f_tip_auto"], j.loc[m, "item_infm_params_text"])]
    log(f"совпадение фильтра Тип автосервиса (EDA 95.6 %, подстрока): {np.mean(ok):.3f}")
    m = j.f_online_booking
    log(f"онлайн-запись у выбранных при фильтре: {j.loc[m, 'online_booking'].mean():.3f} (EDA 0.58)")
    city = j.loc_level == "city"
    log(
        f"тот же город для городских запросов: "
        f"{(j.loc[city, 'search_location_id'] == j.loc[city, 'item_location_id']).mean():.3f} (EDA 0.932)"
    )

    for src in ("train", "bench"):
        q = queries[queries.source == src]
        log(
            f"{src}: регион {(q.loc_level == 'region').mean():.3f}, есть Вид {(q.f_vid != '').mean():.3f}, "
            f"топоним {(q.q_toponyms != '').mean():.3f}, намерение {(q.q_intent != '').mean():.3f}, "
            f"модификатор {q.q_has_modifier.mean():.3f}, латиница {q.q_has_latin.mean():.3f}"
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log("чтение сырых данных")
    train = pd.read_parquet(RAW / "train.parquet")
    corpus = pd.read_parquet(RAW / "benchmark_items.parquet")
    bench = pd.read_parquet(RAW / "benchmark_queries.parquet")

    items, items_text = build_items(corpus, train)

    log("газеттир топонимов")
    gazetteer = build_gazetteer(items.address, items.item_title_raw)
    (OUT / "gazetteer.txt").write_text("\n".join(sorted(gazetteer)), encoding="utf-8")
    log(f"газеттир: {len(gazetteer)} стеммов")

    queries, pairs = build_queries(train, bench, items, gazetteer)

    log("запись parquet")
    items.to_parquet(OUT / "items.parquet", index=False)
    items_text.to_parquet(OUT / "items_text.parquet", index=False)
    queries.to_parquet(OUT / "queries.parquet", index=False)
    pairs.to_parquet(OUT / "train_pairs.parquet", index=False)

    report(items, queries, pairs)
    log("готово")


if __name__ == "__main__":
    main()
