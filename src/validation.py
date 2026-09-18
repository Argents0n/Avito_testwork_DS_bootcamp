"""Локальная валидация: режем трейн по тексту запроса и подгоняем страты под бенчмарк.

Корпус для val = корпус бенчмарка + правильные объявления val-запросов.
Статистики по разметке считаем только по fit-части, иначе метрика врёт.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
VALIDATION = Path("data/validation")

FREQ_BINS = [-1, 0, 1, 3, 10, 100, np.inf]
FREQ_LABELS = ["0", "1", "2-3", "4-10", "11-100", ">100"]
# Размер городского пула — число объявлений корпуса бенчмарка в городе запроса. Без этой оси val-запросы
# оказывались в городах поменьше (медиана пула 1 686 против 2 406 в бенчмарке, p10 115 против 396),
# то есть с меньшим числом конкурентов и завышенным recall.
POOL_BINS = [-1, 300, 1000, 3000, 10000, np.inf]
POOL_LABELS = ["<=300", "301-1k", "1k-3k", "3k-10k", ">10k"]
STRATA = ["freq_bucket", "loc_level", "has_vid", "pool_bucket"]


def _freq_bucket(rows: pd.Series) -> pd.Series:
    return pd.cut(rows, FREQ_BINS, labels=FREQ_LABELS).astype(str)


def build_split(n_val: int, seed: int, name: str, exclude_split: str | None = None, native_only: bool = False) -> Path:
    """exclude_split — имя уже построенного сплита, чьи val-запросы нельзя трогать."""
    rng = np.random.default_rng(seed)
    queries = pd.read_parquet(PROCESSED / "queries.parquet")
    pairs = pd.read_parquet(PROCESSED / "train_pairs.parquet")
    items = pd.read_parquet(PROCESSED / "items.parquet", columns=["item_id", "in_corpus", "item_location_id"])

    train = queries[queries.source == "train"].copy()
    bench = queries[queries.source == "bench"].copy()
    if exclude_split:
        excluded = set()
        for ex in exclude_split.split(","):
            excluded |= set(pd.read_parquet(VALIDATION / ex / "val_queries.parquet").query_id)
        train = train[~train.query_id.isin(excluded)]
        pairs = pairs[~pairs.query_id.isin(excluded)]
    corpus_item_ids = set(items.loc[items.in_corpus, "item_id"])
    native_sessions = set(
        pairs.assign(native=pairs.item_id.isin(corpus_item_ids))
        .groupby("query_id")
        .native.all()
        .pipe(lambda x: x[x].index)
    )
    corpus_loc_counts = items[items.in_corpus].groupby("item_location_id").size()
    for df in (train, bench):
        df["has_vid"] = df.f_vid != ""
        pool = df.search_location_id.map(corpus_loc_counts).fillna(0)
        df["pool_bucket"] = np.where(
            df.loc_level == "region", "region", pd.cut(pool, POOL_BINS, labels=POOL_LABELS).astype(str)
        )

    # Частота текста = число строк трейна (как в EDA: «медианная частота запроса бенчмарка в трейне»).
    rows_by_text = train.groupby("search_query").n_choices.sum()
    sessions_by_text = train.groupby("search_query").size()

    # Целевое распределение страт — по бенчмарку против полного трейна.
    bench["freq_bucket"] = _freq_bucket(bench.search_query.map(rows_by_text).fillna(0))
    target = bench.groupby(STRATA).size() / len(bench)

    # Кандидаты в val и их частота текста в fit после выемки этой сессии.
    n_sessions = train.search_query.map(sessions_by_text)
    fit_rows = train.search_query.map(rows_by_text) - train.n_choices
    train["freq_bucket"] = np.where(n_sessions == 1, "0", _freq_bucket(fit_rows.where(n_sessions > 1, 0)))
    # Сессия с единственным текстом, но частотой > 0 невозможна; сессия текста с ≥2 сессиями не может дать «0».
    train = train[(n_sessions == 1) | (train.freq_bucket != "0")]
    if native_only:
        train = train[train.query_id.isin(native_sessions)]

    shuffled = train.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    used_texts: set[str] = set()
    chosen = []
    shortfall = {}
    for key, share in target.sort_values().items():  # редкие страты первыми, чтобы им хватило текстов
        want = int(round(share * n_val))
        cell = shuffled[(shuffled[STRATA] == pd.Series(key, index=STRATA)).all(axis=1)]
        cell = cell[~cell.search_query.isin(used_texts)].drop_duplicates("search_query").head(want)
        if len(cell) < want:
            shortfall[str(key)] = (want, len(cell))
        used_texts.update(cell.search_query)
        chosen.append(cell)
    val = pd.concat(chosen)

    val_pairs = pairs[pairs.query_id.isin(val.query_id)]
    corpus_ids = set(items.loc[items.in_corpus, "item_id"])
    val_item_ids = set(val_pairs.item_id)
    split_corpus = pd.DataFrame({"item_id": sorted(corpus_ids | val_item_ids)})

    fit_pairs = pairs[~pairs.query_id.isin(val.query_id)]
    meta = {
        "name": name,
        "seed": seed,
        "n_val_requested": n_val,
        "n_val": int(len(val)),
        "n_val_relevant_pairs": int(len(val_pairs)),
        "n_corpus": int(len(split_corpus)),
        "strata_shortfall": shortfall,
        "strata": pd.DataFrame(
            {
                "bench": target,
                "val": val.groupby(STRATA).size() / len(val),
            }
        )
        .fillna(0)
        .round(4)
        .reset_index()
        .astype(str)
        .to_dict("records"),
        # Диагностика рисков расхождения с бенчмарком:
        "val_positive_share_in_bench_corpus": round(len(val_item_ids & corpus_ids) / len(val_item_ids), 4),
        "val_positive_share_chosen_in_fit": round(len(val_item_ids & set(fit_pairs.item_id)) / len(val_item_ids), 4),
        "val_mean_n_relevant": round(float(val.n_relevant.mean()), 3),
        "val_mean_inverse_n_relevant": round(float((1 / val.n_relevant).mean()), 3),
        "bench_category0_share": round(float((bench.search_category == 0).mean()), 4),
        "val_category0_share": round(float((val.search_category == 0).mean()), 4),
        "val_toponym_share": round(float((val.q_toponyms != "").mean()), 4),
        "bench_toponym_share": round(float((bench.q_toponyms != "").mean()), 4),
    }
    # Плотность конкурентов: сколько объявлений корпуса в городе запроса (только городской уровень).
    loc_counts = items[items.item_id.isin(split_corpus.item_id)].groupby("item_location_id").size()
    for label, df in (("val", val), ("bench", bench)):
        city = df[df.loc_level == "city"]
        meta[f"{label}_city_pool_quantiles"] = (
            city.search_location_id.map(loc_counts).fillna(0).quantile([0.1, 0.5, 0.9]).round(0).tolist()
        )

    out = VALIDATION / name
    out.mkdir(parents=True, exist_ok=True)
    val[["query_id", *STRATA]].to_parquet(out / "val_queries.parquet", index=False)
    val_pairs.to_parquet(out / "val_pairs.parquet", index=False)
    split_corpus.to_parquet(out / "corpus_ids.parquet", index=False)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


@dataclass
class Split:
    """Всё, что нужно гипотезе: на чём учиться, на чём мерить, где искать."""

    name: str
    queries: pd.DataFrame  # все запросы (трейн + бенчмарк) с признаками
    fit_queries: pd.DataFrame  # сессии трейна без val — только на них считать статистики по разметке
    fit_pairs: pd.DataFrame
    val_queries: pd.DataFrame  # val-запросы с признаками и стратами
    truth: dict[str, set[str]]  # query_id → множество правильных item_id
    corpus_ids: pd.Index  # где искать


def load_split(name: str) -> Split:
    base = VALIDATION / name
    queries = pd.read_parquet(PROCESSED / "queries.parquet")
    pairs = pd.read_parquet(PROCESSED / "train_pairs.parquet")
    val_q = pd.read_parquet(base / "val_queries.parquet")
    val_pairs = pd.read_parquet(base / "val_pairs.parquet")
    val_ids = set(val_q.query_id)
    fit_queries = queries[(queries.source == "train") & ~queries.query_id.isin(val_ids)]
    truth = val_pairs.groupby("query_id").item_id.agg(set).to_dict()
    return Split(
        name=name,
        queries=queries,
        fit_queries=fit_queries,
        fit_pairs=pairs[pairs.query_id.isin(fit_queries.query_id)],
        # loc_level уже есть в queries: берём из файла страт только недостающие колонки, иначе будут _x/_y.
        val_queries=queries.merge(val_q[["query_id", *val_q.columns.difference(queries.columns)]], on="query_id"),
        truth=truth,
        corpus_ids=pd.Index(pd.read_parquet(base / "corpus_ids.parquet").item_id),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default=None)
    parser.add_argument("--exclude-split", default=None, help="имена сплитов через запятую")
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    name = args.name or f"val_n{args.n}_s{args.seed}"
    out = build_split(args.n, args.seed, name, args.exclude_split, args.native_only)
    (out / "exclude_split.txt").write_text(args.exclude_split or "", encoding="utf-8")
    print((out / "meta.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
