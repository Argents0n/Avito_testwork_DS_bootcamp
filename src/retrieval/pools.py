"""Гео-пулы, веса городов и отбор топ-K ярусами."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import K


def top_from_tiers(
    key: np.ndarray, tiers: list[np.ndarray], k: int = K, exclude: np.ndarray | None = None
) -> np.ndarray:
    """Топ-k индексов: ярусы по порядку, внутри яруса — по убыванию key, без повторов."""
    picked: list[np.ndarray] = []
    taken = np.zeros(len(key), dtype=bool)
    if exclude is not None:
        taken[exclude] = True
    need = k
    for tier in tiers:
        if need == 0:
            break
        cand = tier[~taken[tier]]
        if len(cand) == 0:
            continue
        if len(cand) > need:
            cand = cand[np.argpartition(-key[cand], need - 1)[:need]]
        cand = cand[np.argsort(-key[cand], kind="stable")]
        picked.append(cand)
        taken[cand] = True
        need -= len(cand)
    return np.concatenate(picked) if picked else np.array([], dtype=int)


class GeoPools:
    """Индексы объявлений корпуса по локациям и фильтру «Вид услуги»."""

    def __init__(self, items: pd.DataFrame, region_cities: dict[int, set[int]]):
        self.loc_index = {loc: np.asarray(idx) for loc, idx in items.groupby("item_location_id").indices.items()}
        self.vid = items.vid.to_numpy()
        self.region_cities = region_cities
        self.all_idx = np.arange(len(items))

    def geo(self, location_id: int, loc_level: str) -> np.ndarray:
        if loc_level == "city":
            return self.loc_index.get(location_id, np.array([], dtype=int))
        cities = self.region_cities.get(location_id)
        if not cities:
            return self.all_idx
        return np.concatenate([self.loc_index[c] for c in cities if c in self.loc_index])

    def with_vid(self, pool: np.ndarray, f_vid: str) -> np.ndarray:
        return pool[self.vid[pool] == f_vid] if f_vid else pool


def region_cities_from_fit(
    fit_pairs: pd.DataFrame, fit_queries: pd.DataFrame, items_all: pd.DataFrame
) -> dict[int, set[int]]:
    fit = fit_pairs.merge(fit_queries[["query_id", "search_location_id", "loc_level"]], on="query_id")
    fit = fit.merge(items_all[["item_id", "item_location_id"]], on="item_id")
    return fit[fit.loc_level == "region"].groupby("search_location_id").item_location_id.agg(set).to_dict()


# --- H2: иерархия гео (docs/hypotheses.md) ---
NEIGHBOR_RADIUS_KM = 100
NEIGHBOR_WEIGHT = 0.6
REGION_CORE_SHARE = 0.9
REGION_TAIL_WEIGHT = 0.8


def region_share_from_fit(
    fit_pairs: pd.DataFrame, fit_queries: pd.DataFrame, items_all: pd.DataFrame
) -> dict[int, pd.Series]:
    """Регион → доли выборов по городам (по убыванию). Считается ТОЛЬКО по fit-части разметки."""
    fit = fit_pairs.merge(fit_queries[["query_id", "search_location_id", "loc_level"]], on="query_id")
    fit = fit.merge(items_all[["item_id", "item_location_id"]], on="item_id")
    counts = fit[fit.loc_level == "region"].groupby(["search_location_id", "item_location_id"]).n_choices.sum()
    out = {}
    for region, s in counts.groupby(level=0):
        s = s.droplevel(0)
        out[region] = (s / s.sum()).sort_values(ascending=False)
    return out


def city_neighbors(items_all: pd.DataFrame, radius_km: float) -> dict[int, np.ndarray]:
    """Соседние города в радиусе radius_km по центрам локаций (медиана координат объявлений, без разметки)."""
    cent = items_all.dropna(subset=["lat", "lon"]).groupby("item_location_id")[["lat", "lon"]].median()
    lat, lon = np.radians(cent.lat.to_numpy()), np.radians(cent.lon.to_numpy())
    ids = cent.index.to_numpy()
    out = {}
    for i, loc in enumerate(ids):
        h = np.sin((lat - lat[i]) / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin((lon - lon[i]) / 2) ** 2
        dist = 2 * 6371 * np.arcsin(np.sqrt(h))
        out[loc] = ids[(dist <= radius_km) & (ids != loc)]
    return out


class WeightedGeo:
    """Гео-пул с весами вместо плоского множества (H2)."""

    def __init__(self, items: pd.DataFrame, items_all: pd.DataFrame, region_share: dict[int, pd.Series]):
        self.loc_index = {loc: np.asarray(idx) for loc, idx in items.groupby("item_location_id").indices.items()}
        self.vid = items.vid.to_numpy()
        self.n = len(items)
        self.all_idx = np.arange(self.n)
        self.region_share = region_share
        self.neighbors = city_neighbors(items_all, NEIGHBOR_RADIUS_KM)

    def _idx(self, locations) -> np.ndarray:
        parts = [self.loc_index[c] for c in locations if c in self.loc_index]
        return np.concatenate(parts) if parts else np.array([], dtype=int)

    def pool(self, location_id: int, loc_level: str) -> tuple[np.ndarray, np.ndarray]:
        """(индексы пула, множители для всех объявлений корпуса; вне пула множитель 1 — для глобальных ярусов)."""
        mult = np.ones(self.n, dtype=np.float32)
        if loc_level == "city":
            core = self.loc_index.get(location_id, np.array([], dtype=int))
            near = self._idx(self.neighbors.get(location_id, []))
            mult[near] = NEIGHBOR_WEIGHT
            return np.concatenate([core, near]), mult
        share = self.region_share.get(location_id)
        if share is None:
            return self.all_idx, mult
        tail = share.index[share.cumsum().shift(fill_value=0) >= REGION_CORE_SHARE]
        mult[self._idx(tail)] = REGION_TAIL_WEIGHT
        return self._idx(share.index), mult

    def with_vid(self, pool: np.ndarray, f_vid: str) -> np.ndarray:
        return pool[self.vid[pool] == f_vid] if f_vid else pool
