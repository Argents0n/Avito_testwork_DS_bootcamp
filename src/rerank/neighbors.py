"""Похожие запросы трейна: центры выборов и подкатегории соседей.

Для новых текстов это единственный коллаборативный сигнал — своей истории у них нет.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..embed_corpus import ensure_embeddings, ensure_query_embeddings

# Поиск ближайших текстов — в пространстве исходного e5-small (эмбеддинги всех текстов трейна уже посчитаны).
MODEL = "e5-small"
K_NEIGHBORS = 20


@dataclass
class NeighborHits:
    sims: np.ndarray  # косинусы K ближайших текстов, по убыванию
    centroids: np.ndarray  # (K, dim) центры выборов этих текстов
    centroid: np.ndarray  # общий центр: взвешенное по сходству среднее, L2-нормирован
    microcat: pd.Series  # взвешенная доля подкатегорий среди выборов соседей


class QueryNeighbors:
    def __init__(
        self,
        fit_pairs: pd.DataFrame,
        fit_queries: pd.DataFrame,
        items_all: pd.DataFrame,
        item_model: str = MODEL,
        query_model: str = MODEL,
    ):
        """item_model — энкодер «центров выборов». Кандидаты сравниваются с центром в этом же пространстве."""
        joined = fit_pairs.merge(fit_queries[["query_id", "search_query"]], on="query_id").merge(
            items_all[["item_id", "item_microcat_id"]], on="item_id"
        )
        by_text_item = joined.groupby(["search_query", "item_id"]).n_choices.sum().reset_index()
        texts = pd.Index(by_text_item.search_query.unique())
        self.texts = texts

        # Центры выборов: сумма эмбеддингов выбранных объявлений с весом числа выборов, затем нормировка.
        item_ids = pd.Index(by_text_item.item_id.unique())
        item_emb = ensure_embeddings(item_model, item_ids).astype(np.float32)
        rows = texts.get_indexer(by_text_item.search_query)
        cols = item_ids.get_indexer(by_text_item.item_id)
        weights = by_text_item.n_choices.to_numpy(dtype=np.float32)
        centroids = np.zeros((len(texts), item_emb.shape[1]), dtype=np.float32)
        np.add.at(centroids, rows, item_emb[cols] * weights[:, None])
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-6)
        self.centroids = centroids

        mc = joined.groupby(["search_query", "item_microcat_id"]).n_choices.sum()
        mc = mc / mc.groupby(level=0).transform("sum")
        self.microcat = {t: g.droplevel(0) for t, g in mc.groupby(level=0)}

        self.text_emb = ensure_query_embeddings(query_model, texts.tolist()).astype(np.float32)

        # v5: для каждого объявления — тексты fit-части, по которым его выбирали (индексы в self.texts).
        # Нужен запросозависимый признак истории: «похож ли текущий запрос на то, за чем это объявление выбирали».
        # groupby.agg не может вернуть массив на группу — берём позиции строк каждой группы через indices.
        positions = by_text_item.groupby("item_id").indices
        self.item_texts: dict[str, np.ndarray] = {item: rows[pos] for item, pos in positions.items()}

    def item_text_sims(self, query_emb: np.ndarray, item_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(max, mean) косинус запроса к текстам, по которым выбирали каждое объявление, и число таких текстов."""
        n = len(item_ids)
        mx, mean, cnt = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32), np.zeros(n, np.float32)
        lists = [self.item_texts.get(x) for x in item_ids]
        have = np.flatnonzero([x is not None for x in lists])
        if len(have) == 0:
            return mx, mean, cnt
        idx = np.concatenate([lists[j] for j in have])
        lengths = np.array([len(lists[j]) for j in have])
        sims = self.text_emb[idx] @ query_emb
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        mx[have] = np.maximum.reduceat(sims, starts)
        mean[have] = np.add.reduceat(sims, starts) / lengths
        cnt[have] = lengths
        return mx, mean, cnt

    def query(self, query_emb: np.ndarray, k: int = K_NEIGHBORS) -> NeighborHits:
        sims = self.text_emb @ query_emb
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        s = sims[top]
        w = np.clip(s, 0, None)
        centroid = (self.centroids[top] * w[:, None]).sum(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-6)
        mc = pd.concat([self.microcat[self.texts[j]] * wj for j, wj in zip(top, w)])
        mc = mc.groupby(level=0).sum()
        mc = mc / max(float(mc.sum()), 1e-6)
        return NeighborHits(sims=s, centroids=self.centroids[top], centroid=centroid, microcat=mc)
