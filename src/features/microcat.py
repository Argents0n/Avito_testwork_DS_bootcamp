"""Предсказание подкатегории по стеммам запроса (статистика fit-части).

Вес стемма = 1 / log(1 + частота): редкие слова однозначнее частых.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MicrocatPredictor:
    def __init__(self, fit_pairs: pd.DataFrame, fit_queries: pd.DataFrame, items_all: pd.DataFrame):
        f = fit_pairs.merge(fit_queries[["query_id", "q_stems"]], on="query_id").merge(
            items_all[["item_id", "item_microcat_id"]], on="item_id"
        )
        rows = f.assign(stem=f.q_stems.str.split()).explode("stem").dropna(subset=["stem"])
        counts = rows.groupby(["stem", "item_microcat_id"]).n_choices.sum()
        probs = counts / counts.groupby(level=0).transform("sum")
        self.by_stem: dict[str, tuple[np.ndarray, np.ndarray]] = {
            stem: (g.index.get_level_values(1).to_numpy(), g.to_numpy(dtype=np.float64))
            for stem, g in probs.groupby(level=0)
        }
        freq = rows.groupby("stem").size()
        self.weight = (1.0 / np.log1p(freq)).to_dict()

    def predict(self, q_stems: str) -> pd.Series:
        """Распределение подкатегорий по убыванию вероятности; пустое, если ни одного знакомого стемма."""
        acc: dict[int, float] = {}
        # сортировка не для красоты: без неё порядок обхода set зависит от хеша строк,
        # сумма float складывается иначе и ответ гуляет между запусками
        for stem in sorted(set(q_stems.split())):
            if stem not in self.by_stem:
                continue
            w = self.weight[stem]
            for m, p in zip(*self.by_stem[stem]):
                acc[m] = acc.get(m, 0.0) + w * p
        if not acc:
            return pd.Series(dtype=float)
        s = pd.Series(acc)
        return (s / s.sum()).sort_values(ascending=False)

    def likely_set(self, q_stems: str, coverage: float) -> set[int] | None:
        """Минимальный набор подкатегорий с суммарной вероятностью ≥ coverage; None — предсказания нет."""
        p = self.predict(q_stems)
        if p.empty:
            return None
        keep = p.cumsum().shift(fill_value=0) < coverage
        return set(p.index[keep])
