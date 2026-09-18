"""Кандидаты и признаки для переранжирования.

Кандидаты — объединение списков BM25, e5 и USER-base внутри гео-пулов.
Все статистики по разметке приходят снаружи (StatsContext) и считаются без размечаемых запросов.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.microcat import MicrocatPredictor
from ..metrics import K
from ..retrieval.bm25 import BM25
from ..retrieval.hybrid import N_BM25
from ..retrieval.pools import WeightedGeo, region_share_from_fit, top_from_tiers
from .neighbors import QueryNeighbors


def _km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Гаверсинус в км; аргументы в градусах, поддерживает broadcasting."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


DEPTH = 500
# Версия признаков — часть имени кэша, чтобы не смешивать с прошлыми прогонами.
FEATURE_VERSION = "v5"
# Глубина списка «похожие запросы трейна»: он дополняет BM25 и e5, а не заменяет их.
KNN_DEPTH = 300
# Ядро региона — города с первыми 90 % выборов (как REGION_CORE_SHARE в H2).
REGION_CORE_SHARE = 0.9

# fmt: off
FEATURE_GROUPS = {
    "text": ["bm25_all", "bm25_title", "bm25_params", "bm25_desc", "bm25_rel_max", "cos", "cos_gap_max",
             "rank_bm25", "rank_dense", "in_base50",
             "cov_title", "cov_params", "cov_desc", "cov_all", "all_in_title", "bigram_title"],
    "geo": ["geo_mult", "same_city", "region_city_share", "dist_km",
            "city_center_dist_km", "dist_core_km", "region_city_rank"],
    "cat": ["vid_match", "tip_match", "microcat_prob", "microcat_rank", "tip_auto_match", "subject_match"],
    "item": ["log_reviews", "rating", "phone_hidden", "msg_forbidden", "log_desc_dup", "work_remote",
             "travel_city", "online_booking", "price_list_size", "experience_years", "is_services"],
    "query": ["q_n_words", "is_region", "has_vid", "has_tip", "log_pool", "log_text_rows", "n_cand"],
    "semantic": ["cos_user", "cos_user_gap_max", "rank_user", "rank_knn", "knn_cos_centroid", "knn_cos_max",
                 "knn_top_sim", "knn_microcat_share"],
    "hist": ["log_item_choices", "same_text_choices", "hist_text_max_sim", "hist_text_mean_sim", "hist_n_texts"],
}
# fmt: on
# Признаки объявления, распределённые по-разному у объявлений корпуса и трейна: модель могла бы выучить
# «происхождение», которого на бенчмарке нет (H6).
ORIGIN_LEAKY_FEATURES = ["log_desc_dup", "price_list_size", "experience_years", "is_services"]
# Запросозависимая история: похож ли запрос на тексты, по которым объявление уже выбирали.
V5_NEW_FEATURES = ["hist_text_max_sim", "hist_text_mean_sim", "hist_n_texts"]
# Признаки, добавленные во второй версии, — для замера их вклада отдельно.
V2_NEW_FEATURES = {
    "cov_title",
    "cov_params",
    "cov_desc",
    "cov_all",
    "all_in_title",
    "bigram_title",
    "city_center_dist_km",
    "dist_core_km",
    "region_city_rank",
    "tip_auto_match",
    "subject_match",
}


@dataclass
class StatsContext:
    """Статистики по разметке для конкретного набора размечаемых запросов (без них самих)."""

    region_share: dict[int, pd.Series]
    microcat: MicrocatPredictor
    text_rows: dict[str, int]
    item_choices: dict[str, int]
    text_item_choices: dict[tuple[str, str], int]
    neighbors: QueryNeighbors | None = None

    @classmethod
    def from_fit(
        cls,
        fit_pairs: pd.DataFrame,
        fit_queries: pd.DataFrame,
        items_all: pd.DataFrame,
        with_neighbors: bool = False,
        neighbor_model: str = "e5-small",
    ) -> StatsContext:
        joined = fit_pairs.merge(fit_queries[["query_id", "search_query"]], on="query_id")
        return cls(
            neighbors=(
                QueryNeighbors(fit_pairs, fit_queries, items_all, item_model=neighbor_model, query_model=neighbor_model)
                if with_neighbors
                else None
            ),
            region_share=region_share_from_fit(fit_pairs, fit_queries, items_all),
            microcat=MicrocatPredictor(fit_pairs, fit_queries, items_all),
            text_rows=joined.groupby("search_query").n_choices.sum().to_dict(),
            item_choices=fit_pairs.groupby("item_id").n_choices.sum().to_dict(),
            text_item_choices=joined.groupby(["search_query", "item_id"]).n_choices.sum().to_dict(),
        )


class CandidateBuilder:
    """Индексы корпуса, общие для всех запросов: BM25 (все поля и по отдельности), эмбеддинги, гео."""

    def __init__(
        self,
        items_all: pd.DataFrame,
        texts: pd.DataFrame,
        corpus_ids: pd.Index,
        doc_emb: np.ndarray,
        doc_emb_user: np.ndarray | None = None,
        doc_emb_neighbors: np.ndarray | None = None,
    ):
        items = items_all.set_index("item_id").loc[corpus_ids].reset_index()
        texts = texts.set_index("item_id").loc[corpus_ids]
        self.items = items
        self.items_all = items_all
        self.item_ids = items.item_id.to_numpy()
        self.bm25_all = BM25().fit((texts.title_stems + " " + texts.params_stems + " " + texts.desc_stems).tolist())
        self.bm25_title = BM25().fit(texts.title_stems.tolist())
        self.bm25_params = BM25().fit(texts.params_stems.tolist())
        self.bm25_desc = BM25().fit(texts.desc_stems.tolist())
        self.doc_emb = doc_emb.astype(np.float32)
        self.doc_emb_user = doc_emb_user.astype(np.float32) if doc_emb_user is not None else None
        # Эмбеддинги корпуса в пространстве «похожих запросов» (e5-small); по умолчанию — основной энкодер.
        self.doc_emb_neighbors = doc_emb_neighbors.astype(np.float32) if doc_emb_neighbors is not None else self.doc_emb
        popularity = items.log_reviews.to_numpy(dtype=np.float32)
        self.tie = 1e-3 * popularity / max(float(popularity.max()), 1.0)
        self.loc = items.item_location_id.to_numpy()
        self.lat = items.lat.to_numpy(dtype=np.float64)
        self.lon = items.lon.to_numpy(dtype=np.float64)
        self.vid = items.vid.to_numpy()
        self.tip = items.tip.to_numpy()
        self.microcat = items.item_microcat_id.to_numpy()
        self.tip_auto = items.tip_auto.to_numpy()
        self.subject = items.subject.to_numpy()
        # Пробелы по краям, чтобы проверка «пара слов подряд» не цепляла части других стеммов.
        self.title_padded = (" " + texts.title_stems + " ").to_numpy()
        # Центры городов по координатам всех объявлений (без разметки): расстояния «город — город».
        cent = items_all.dropna(subset=["lat", "lon"]).groupby("item_location_id")[["lat", "lon"]].median()
        self.city_lat, self.city_lon = cent.lat, cent.lon
        self.item_city_lat = pd.Series(self.loc).map(cent.lat).to_numpy(dtype=np.float64)
        self.item_city_lon = pd.Series(self.loc).map(cent.lon).to_numpy(dtype=np.float64)
        # Статические признаки объявления — один раз на корпус.
        self.item_static = pd.DataFrame(
            {
                "log_reviews": items.log_reviews.astype(np.float32),
                "rating": items.item_rating.astype(np.float32),
                "phone_hidden": items.item_is_phone_hidden.astype(np.float32),
                "msg_forbidden": items.item_is_message_forbidden.astype(np.float32),
                "log_desc_dup": np.log1p(items.desc_dup_count).astype(np.float32),
                "work_remote": items.work_remote.astype(np.float32),
                "travel_city": items.travel_city.astype(np.float32),
                "online_booking": items.online_booking.astype(np.float32),
                "price_list_size": items.price_list_size.astype(np.float32),
                "experience_years": items.experience_years.astype(np.float32),
                "is_services": items.is_services_category.astype(np.float32),
            }
        ).to_numpy()

    def build(
        self,
        queries: pd.DataFrame,
        query_emb: np.ndarray,
        ctx: StatsContext,
        truth: dict[str, set[str]] | None = None,
        keep_neg: int | None = None,
        seed: int = 0,
        depth: int = DEPTH,
        query_emb_user: np.ndarray | None = None,
        query_emb_neighbors: np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """Возвращает (строки кандидатов с признаками и меткой, базовый топ-50 рабочего гибрида)."""
        rng = np.random.default_rng(seed)
        geo = WeightedGeo(self.items, self.items_all, ctx.region_share)
        all_idx = geo.all_idx
        chunks, base = [], {}
        for i, q in enumerate(queries.itertuples(index=False)):
            pool, mult = geo.pool(q.search_location_id, q.loc_level)
            pool_vid = geo.with_vid(pool, q.f_vid)

            s = self.bm25_all.scores(q.q_stems)
            d = self.doc_emb @ query_emb[i]
            bm25_tiers = [pool_vid[s[pool_vid] > 0], pool[s[pool] > 0], np.flatnonzero(s > 0), pool_vid, all_idx]
            list_b = top_from_tiers(s * mult + self.tie, bm25_tiers, k=depth)
            list_d = top_from_tiers(d * mult + self.tie, [pool_vid, pool, all_idx], k=depth)

            # Базовый ответ = ровно рабочий гибрид (H1 + H2): первые 40 BM25 + добивка энкодером.
            first = list_b[:N_BM25]
            rest = top_from_tiers(d * mult + self.tie, [pool_vid, pool, all_idx], k=K - len(first), exclude=first)
            base50 = np.concatenate([first, rest])
            base[q.query_id] = self.item_ids[base50].tolist()

            lists = [list_b, list_d, base50]
            use_user = self.doc_emb_user is not None and query_emb_user is not None
            if use_user:
                d_user = self.doc_emb_user @ query_emb_user[i]
                list_u = top_from_tiers(d_user * mult + self.tie, [pool_vid, pool, all_idx], k=depth)
                lists.append(list_u)
            nb_emb = query_emb_neighbors if query_emb_neighbors is not None else query_emb
            hits = ctx.neighbors.query(nb_emb[i]) if ctx.neighbors is not None else None
            if hits is not None:
                d_knn = self.doc_emb_neighbors @ hits.centroid
                list_k = top_from_tiers(d_knn * mult + self.tie, [pool_vid, pool, all_idx], k=KNN_DEPTH)
                lists.append(list_k)
            cand = np.unique(np.concatenate(lists))
            rank_b = np.full(len(self.item_ids), depth, dtype=np.float32)
            rank_b[list_b] = np.arange(len(list_b))
            rank_d = np.full(len(self.item_ids), depth, dtype=np.float32)
            rank_d[list_d] = np.arange(len(list_d))
            nan = np.full(len(cand), np.nan, dtype=np.float32)
            if use_user:
                rank_u = np.full(len(self.item_ids), depth, dtype=np.float32)
                rank_u[list_u] = np.arange(len(list_u))
                cu = d_user[cand]
                sem_user = {"cos_user": cu, "cos_user_gap_max": cu - cu.max(), "rank_user": rank_u[cand]}
            else:
                sem_user = {"cos_user": nan, "cos_user_gap_max": nan, "rank_user": nan}
            if hits is not None:
                rank_k = np.full(len(self.item_ids), KNN_DEPTH, dtype=np.float32)
                rank_k[list_k] = np.arange(len(list_k))
                sem_knn = {
                    "rank_knn": rank_k[cand],
                    "knn_cos_centroid": d_knn[cand],
                    "knn_cos_max": (self.doc_emb_neighbors[cand] @ hits.centroids.T).max(axis=1),
                    "knn_top_sim": np.full(len(cand), hits.sims[0]),
                    "knn_microcat_share": pd.Series(self.microcat[cand]).map(hits.microcat).fillna(0).to_numpy(),
                }
            else:
                sem_knn = {
                    c: nan for c in ["rank_knn", "knn_cos_centroid", "knn_cos_max", "knn_top_sim", "knn_microcat_share"]
                }

            sc = s[cand]
            dc = d[cand]
            probs = ctx.microcat.predict(q.q_stems)
            mc = pd.Series(self.microcat[cand])
            share = ctx.region_share.get(q.search_location_id) if q.loc_level == "region" else None
            if q.loc_level == "city" and not np.isnan(q.loc_lat):
                la1, lo1 = np.radians(q.loc_lat), np.radians(q.loc_lon)
                la2, lo2 = np.radians(self.lat[cand]), np.radians(self.lon[cand])
                h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
                dist = 2 * 6371 * np.arcsin(np.sqrt(h))
            else:
                dist = np.full(len(cand), np.nan)
            text_rows = ctx.text_rows.get(q.search_query, 0)
            cand_ids = self.item_ids[cand]
            if hits is not None:
                h_max, h_mean, h_cnt = ctx.neighbors.item_text_sims(nb_emb[i], cand_ids)
            else:
                h_max = h_mean = np.full(len(cand), np.nan, np.float32)
                h_cnt = np.zeros(len(cand), np.float32)

            # --- v2: покрытие слов запроса по полям ---
            q_terms = list(dict.fromkeys(q.q_stems.split()))
            n_terms = max(len(q_terms), 1)
            cov_title = self.bm25_title.term_hits(q.q_stems, cand) / n_terms
            bigrams = [f" {a} {b} " for a, b in zip(q_terms, q_terms[1:])]
            bigram_title = (
                np.array([any(bg in t for bg in bigrams) for t in self.title_padded[cand]])
                if bigrams
                else np.full(len(cand), -1.0)
            )

            # --- v2: гео для обоих уровней локации ---
            if q.loc_level == "city" and not np.isnan(q.loc_lat):
                city_center_dist = _km(q.loc_lat, q.loc_lon, self.item_city_lat[cand], self.item_city_lon[cand])
                dist_core = dist
                region_rank = np.full(len(cand), np.nan)
            elif share is not None:
                core = share.index[share.cumsum().shift(fill_value=0) < REGION_CORE_SHARE]
                core = [c for c in core if c in self.city_lat.index]
                clat, clon = self.city_lat.loc[core].to_numpy(), self.city_lon.loc[core].to_numpy()
                dist_core = _km(self.lat[cand][:, None], self.lon[cand][:, None], clat[None, :], clon[None, :]).min(
                    axis=1
                )
                city_center_dist = _km(
                    self.item_city_lat[cand][:, None], self.item_city_lon[cand][:, None], clat[None, :], clon[None, :]
                ).min(axis=1)
                region_rank = (
                    pd.Series(self.loc[cand]).map(pd.Series(np.arange(len(share)), index=share.index)).to_numpy()
                )
            else:
                city_center_dist = dist_core = region_rank = np.full(len(cand), np.nan)

            f = {
                "bm25_all": sc,
                "bm25_title": self.bm25_title.scores(q.q_stems)[cand],
                "bm25_params": self.bm25_params.scores(q.q_stems)[cand],
                "bm25_desc": self.bm25_desc.scores(q.q_stems)[cand],
                "bm25_rel_max": sc / max(float(sc.max()), 1e-6),
                "cos": dc,
                "cos_gap_max": dc - dc.max(),
                "rank_bm25": rank_b[cand],
                "rank_dense": rank_d[cand],
                "in_base50": np.isin(cand, base50),
                "cov_title": cov_title,
                "cov_params": self.bm25_params.term_hits(q.q_stems, cand) / n_terms,
                "cov_desc": self.bm25_desc.term_hits(q.q_stems, cand) / n_terms,
                "cov_all": self.bm25_all.term_hits(q.q_stems, cand) / n_terms,
                "all_in_title": cov_title >= 1.0,
                "bigram_title": bigram_title,
                "geo_mult": mult[cand],
                "same_city": self.loc[cand] == q.search_location_id,
                "region_city_share": (
                    pd.Series(self.loc[cand]).map(share).fillna(0).to_numpy()
                    if share is not None
                    else np.zeros(len(cand))
                ),
                "dist_km": dist,
                "city_center_dist_km": city_center_dist,
                "dist_core_km": dist_core,
                "region_city_rank": region_rank,
                "vid_match": (self.vid[cand] == q.f_vid).astype(np.float32) if q.f_vid else np.full(len(cand), -1.0),
                "tip_match": (self.tip[cand] == q.f_tip).astype(np.float32) if q.f_tip else np.full(len(cand), -1.0),
                "microcat_prob": mc.map(probs).fillna(0).to_numpy() if not probs.empty else np.zeros(len(cand)),
                "microcat_rank": (
                    mc.map(pd.Series(np.arange(len(probs)), index=probs.index)).fillna(len(probs) + 1).to_numpy()
                    if not probs.empty
                    else np.full(len(cand), 999.0)
                ),
                "tip_auto_match": (
                    np.array([q.f_tip_auto in x for x in self.tip_auto[cand]], dtype=np.float32)
                    if q.f_tip_auto
                    else np.full(len(cand), -1.0)
                ),
                "subject_match": (
                    np.array([q.f_subject in x for x in self.subject[cand]], dtype=np.float32)
                    if q.f_subject
                    else np.full(len(cand), -1.0)
                ),
                "q_n_words": np.full(len(cand), q.q_n_words),
                "is_region": np.full(len(cand), q.loc_level == "region"),
                "has_vid": np.full(len(cand), q.f_vid != ""),
                "has_tip": np.full(len(cand), q.f_tip != ""),
                "log_pool": np.full(len(cand), np.log1p(len(pool))),
                "log_text_rows": np.full(len(cand), np.log1p(text_rows)),
                "n_cand": np.full(len(cand), len(cand)),
                **sem_user,
                **sem_knn,
                "log_item_choices": np.log1p([ctx.item_choices.get(x, 0) for x in cand_ids]),
                "same_text_choices": np.array([ctx.text_item_choices.get((q.search_query, x), 0) for x in cand_ids]),
                "hist_text_max_sim": h_max,
                "hist_text_mean_sim": h_mean,
                "hist_n_texts": h_cnt,
            }
            df = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
            static = pd.DataFrame(self.item_static[cand], columns=FEATURE_GROUPS["item"])
            df = pd.concat([df, static], axis=1)
            df.insert(0, "item_id", cand_ids)
            df.insert(0, "query_id", q.query_id)
            if truth is not None:
                df["label"] = np.isin(cand_ids, list(truth.get(q.query_id, ()))).astype(np.int8)
                if keep_neg is not None:
                    neg = np.flatnonzero(df.label.to_numpy() == 0)
                    keep = np.concatenate(
                        [
                            np.flatnonzero(df.label.to_numpy() == 1),
                            rng.choice(neg, size=min(keep_neg, len(neg)), replace=False),
                        ]
                    )
                    df = df.iloc[np.sort(keep)].reset_index(drop=True)
            chunks.append(df)
            if i % 2000 == 0:
                print(f"    кандидаты: {i}/{len(queries)}", flush=True)
        return pd.concat(chunks, ignore_index=True), base


def feature_columns(groups: Iterable[str]) -> list[str]:
    return [c for g in groups for c in FEATURE_GROUPS[g]]
