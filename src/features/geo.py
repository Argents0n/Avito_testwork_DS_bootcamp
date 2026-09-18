"""Гео: уровень локации (город/регион), центры городов, расстояния."""

from __future__ import annotations

import numpy as np
import pandas as pd

LOC_LEVEL_CITY = "city"
LOC_LEVEL_REGION = "region"

EARTH_RADIUS_KM = 6371.0


def location_centroids(items: pd.DataFrame) -> pd.DataFrame:
    """Центр каждой локации объявлений = медиана координат (устойчива к объявлениям с чужими координатами)."""
    valid = items.dropna(subset=["lat", "lon"])
    return (
        valid.groupby("item_location_id")
        .agg(loc_lat=("lat", "median"), loc_lon=("lon", "median"), loc_n_items=("lat", "size"))
        .reset_index()
        .rename(columns={"item_location_id": "location_id"})
    )


def location_level(search_location_ids: pd.Series, item_location_ids: set[int]) -> pd.Series:
    """city — если id встречается у объявлений, иначе region."""
    return np.where(search_location_ids.isin(item_location_ids), LOC_LEVEL_CITY, LOC_LEVEL_REGION)


def haversine_km(lat1, lon1, lat2, lon2):
    """Расстояние по большому кругу; работает поэлементно с numpy-массивами."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
