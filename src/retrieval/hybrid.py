"""Гибрид BM25 + энкодер внутри гео-пулов (базовое решение до переранжирования)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..metrics import K
from .bm25 import BM25
from .pools import WeightedGeo, top_from_tiers

N_BM25 = 40


@dataclass
class HybridIndex:
    items: pd.DataFrame  # корпус в фиксированном порядке (item_id, item_location_id, vid, log_reviews)
    pools: WeightedGeo
    bm25: BM25
    doc_emb: np.ndarray  # эмбеддинги документов в порядке items, L2-нормированы
    tie: np.ndarray  # малая добавка популярности для разведения равных скоров


def build_index(
    items: pd.DataFrame,
    items_all: pd.DataFrame,
    texts: pd.DataFrame,
    doc_emb: np.ndarray,
    region_share: dict[int, pd.Series],
) -> HybridIndex:
    """items и texts должны быть в одном порядке (по item_id)."""
    bm25 = BM25().fit((texts.title_stems + " " + texts.params_stems + " " + texts.desc_stems).tolist())
    popularity = items.log_reviews.to_numpy(dtype=np.float32)
    tie = 1e-3 * popularity / max(float(popularity.max()), 1.0)
    return HybridIndex(items, WeightedGeo(items, items_all, region_share), bm25, doc_emb.astype(np.float32), tie)


def retrieve(
    index: HybridIndex, queries: pd.DataFrame, query_emb: np.ndarray, n_bm25: int = N_BM25, k: int = K
) -> dict[str, list[str]]:
    """queries — строки queries.parquet (нужны query_id, search_location_id, loc_level, f_vid, q_stems)."""
    item_ids = index.items.item_id.to_numpy()
    all_idx = index.pools.all_idx
    out = {}
    for i, q in enumerate(queries.itertuples(index=False)):
        geo, mult = index.pools.pool(q.search_location_id, q.loc_level)
        geo_vid = index.pools.with_vid(geo, q.f_vid)

        s = index.bm25.scores(q.q_stems)
        bm25_tiers = [geo_vid[s[geo_vid] > 0], geo[s[geo] > 0], np.flatnonzero(s > 0), geo_vid, all_idx]
        first = top_from_tiers(s * mult + index.tie, bm25_tiers, k=n_bm25)

        dense_key = (index.doc_emb @ query_emb[i]) * mult + index.tie
        # Уже взятые BM25 исключаются явно, чтобы энкодер заполнил ровно k − N свободных мест.
        rest = top_from_tiers(dense_key, [geo_vid, geo, all_idx], k=k - len(first), exclude=first)
        out[q.query_id] = item_ids[np.concatenate([first, rest])[:k]].tolist()
    return out
