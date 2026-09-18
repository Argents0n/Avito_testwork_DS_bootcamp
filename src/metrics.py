"""Recall@50 и разбивка по сегментам."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

K = 50


def recall_per_query(predictions: Mapping[str, Iterable[str]], truth: Mapping[str, set[str]], k: int = K) -> pd.Series:
    """Recall@k для каждого запроса из truth."""
    out = {}
    for qid, rel in truth.items():
        pred = list(dict.fromkeys(predictions.get(qid, [])))[:k]
        out[qid] = len(rel.intersection(pred)) / len(rel) if rel else np.nan
    return pd.Series(out, name=f"recall@{k}")


def recall_at_k(predictions: Mapping[str, Iterable[str]], truth: Mapping[str, set[str]], k: int = K) -> float:
    return float(recall_per_query(predictions, truth, k).mean())


def recall_report(per_query: pd.Series, segments: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Средний recall и число запросов по каждому значению сегментных колонок."""
    df = segments[columns].join(per_query.rename("recall"), how="inner")
    parts = [pd.DataFrame({"segment": "ALL", "value": "", "n": len(df), "recall": df.recall.mean()}, index=[0])]
    for col in columns:
        g = df.groupby(col, observed=True).recall.agg(["size", "mean"]).reset_index()
        parts.append(pd.DataFrame({"segment": col, "value": g[col].astype(str), "n": g["size"], "recall": g["mean"]}))
    return pd.concat(parts, ignore_index=True)
