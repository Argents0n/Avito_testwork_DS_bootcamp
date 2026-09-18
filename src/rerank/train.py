"""Обучение переранжирования и сравнение вариантов на валидации.

HistGradientBoosting, а не LightGBM: у lightgbm на macOS колесо требует системную libomp.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ..embed_corpus import ensure_embeddings, ensure_query_embeddings
from ..metrics import K, recall_per_query, recall_report
from ..validation import load_split
from .extra_features import add_extra_features, extra_columns, extra_item_table
from .features import FEATURE_GROUPS, FEATURE_VERSION, V5_NEW_FEATURES, CandidateBuilder, StatsContext, feature_columns

PROCESSED = "data/processed"
OUT = Path("data/rerank")
# Основной энкодер — дообученный e5 (H8); в нём же ищем похожие запросы.
MODEL = "e5-small-ft"
# v5: «похожие запросы» и запросозависимая история — в пространстве дообученного энкодера.
NEIGHBOR_MODEL = "e5-small-ft"
# Второй энкодер (v3): глубже 300 находит правильные объявления, которых нет у BM25 и e5-small.
USER_MODEL = "user-base"
NEG_PER_QUERY = 100
SEED = 13
# v5: ансамбль из моделей с разными random_state — разброс одной модели ±0.0008 (H12), усреднение его гасит.
ENSEMBLE_SEEDS = (13, 1, 2, 3, 4)
HGB_PARAMS = dict(
    max_iter=600,
    learning_rate=0.05,
    max_leaf_nodes=63,
    min_samples_leaf=100,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=30,
)

ITEM_COLS = [
    "item_id",
    "item_location_id",
    "vid",
    "tip",
    "item_microcat_id",
    "log_reviews",
    "lat",
    "lon",
    "item_rating",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
    "desc_dup_count",
    "work_remote",
    "travel_city",
    "online_booking",
    "price_list_size",
    "experience_years",
    "is_services_category",
    "tip_auto",
    "subject",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def top_k_by_score(df: pd.DataFrame, score: np.ndarray, k: int = K) -> dict[str, list[str]]:
    ranked = df[["query_id", "item_id"]].assign(score=score).sort_values(["query_id", "score"], ascending=[True, False])
    return ranked.groupby("query_id", sort=False).head(k).groupby("query_id").item_id.agg(list).to_dict()


def sample_training_rows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Все позитивы + до NEG_PER_QUERY случайных негативов на запрос; запросы без позитива в кандидатах."""
    has_pos = df.groupby("query_id").label.transform("max") == 1
    df = df[has_pos]
    neg = df[df.label == 0]
    neg = (
        neg.assign(_r=rng.random(len(neg))).sort_values("_r").groupby("query_id").head(NEG_PER_QUERY).drop(columns="_r")
    )
    return pd.concat([df[df.label == 1], neg], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", default="val_n5000_s42")
    parser.add_argument("--train", default="rr_train_n40000_s7")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    s_val = load_split(args.val)
    s_tr = load_split(args.train)
    val_ids = set(s_val.val_queries.query_id)

    train_path = OUT / f"feat_{args.train}_{FEATURE_VERSION}.parquet"
    val_path = OUT / f"feat_{args.val}_{FEATURE_VERSION}.parquet"
    base_path = OUT / f"base50_{args.val}_{FEATURE_VERSION}.parquet"
    if not (train_path.exists() and val_path.exists() and base_path.exists()):
        items_all = pd.read_parquet(f"{PROCESSED}/items.parquet", columns=ITEM_COLS)
        corpus = s_val.corpus_ids.union(s_tr.corpus_ids)
        log(f"общий корпус {len(corpus)}; эмбеддинги")
        doc_emb = ensure_embeddings(MODEL, corpus)
        texts = pd.read_parquet(f"{PROCESSED}/items_text.parquet")
        texts = texts[texts.item_id.isin(corpus)]
        log("индексы BM25 (все поля + заголовок / параметры / описание)")
        builder = CandidateBuilder(
            items_all,
            texts,
            corpus,
            doc_emb,
            ensure_embeddings(USER_MODEL, corpus),
            ensure_embeddings(NEIGHBOR_MODEL, corpus),
        )
        del texts

        # Статистики для обучающих запросов: без них самих и без валидации.
        tr_fit_q = s_tr.fit_queries[~s_tr.fit_queries.query_id.isin(val_ids)]
        ctx_tr = StatsContext.from_fit(
            s_tr.fit_pairs[s_tr.fit_pairs.query_id.isin(tr_fit_q.query_id)],
            tr_fit_q,
            items_all,
            with_neighbors=True,
            neighbor_model=NEIGHBOR_MODEL,
        )
        ctx_val = StatsContext.from_fit(
            s_val.fit_pairs, s_val.fit_queries, items_all, with_neighbors=True, neighbor_model=NEIGHBOR_MODEL
        )

        tq = s_tr.val_queries.reset_index(drop=True)
        vq = s_val.val_queries.reset_index(drop=True)
        log(f"кандидаты и признаки: обучение {len(tq)} запросов")
        tq_texts, vq_texts = tq.search_query.tolist(), vq.search_query.tolist()
        feat_tr, _ = builder.build(
            tq,
            ensure_query_embeddings(MODEL, tq_texts),
            ctx_tr,
            s_tr.truth,
            keep_neg=NEG_PER_QUERY,
            seed=SEED,
            query_emb_user=ensure_query_embeddings(USER_MODEL, tq_texts),
            query_emb_neighbors=ensure_query_embeddings(NEIGHBOR_MODEL, tq_texts),
        )
        feat_tr.to_parquet(train_path, index=False)
        log(f"кандидаты и признаки: валидация {len(vq)} запросов")
        feat_val, base = builder.build(
            vq,
            ensure_query_embeddings(MODEL, vq_texts),
            ctx_val,
            s_val.truth,
            query_emb_user=ensure_query_embeddings(USER_MODEL, vq_texts),
            query_emb_neighbors=ensure_query_embeddings(NEIGHBOR_MODEL, vq_texts),
        )
        feat_val.to_parquet(val_path, index=False)
        pd.DataFrame([(q, i) for q, items in base.items() for i in items], columns=["query_id", "item_id"]).to_parquet(
            base_path, index=False
        )
    feat_tr = pd.read_parquet(train_path)
    feat_val = pd.read_parquet(val_path)
    base = pd.read_parquet(base_path).groupby("query_id").item_id.agg(list).to_dict()
    log(
        f"строк: обучение {len(feat_tr)}, валидация {len(feat_val)}; "
        f"позитив в кандидатах: обучение {feat_tr.groupby('query_id').label.max().mean():.3f}, "
        f"валидация {feat_val.groupby('query_id').label.max().mean():.3f}"
    )

    train_rows = sample_training_rows(feat_tr, rng)
    del feat_tr
    log(f"обучающая выборка после сэмплирования негативов: {len(train_rows)} строк, позитивов {train_rows.label.sum()}")
    log("дешёвые признаки (H12)")
    extra_items = extra_item_table()
    all_queries = pd.read_parquet(f"{PROCESSED}/queries.parquet")
    train_rows = add_extra_features(train_rows, extra_items, all_queries)
    feat_val = add_extra_features(feat_val, extra_items, all_queries)
    del extra_items

    segments = s_val.val_queries.set_index("query_id")
    seg_cols = ["loc_level", "has_vid", "freq_bucket"]
    results = {"base_hybrid": recall_per_query(base, s_val.truth)}
    ceiling = feat_val.groupby("query_id").apply(lambda g: g.label.sum(), include_groups=False)
    results["ceiling_candidates"] = (
        (ceiling / pd.Series({q: len(r) for q, r in s_val.truth.items()})).reindex(list(s_val.truth)).fillna(0)
    )

    all_cols = feature_columns(FEATURE_GROUPS) + extra_columns()
    variants = {
        "rr_all": all_cols,
        # Вклад новых частей v5 по отдельности: запросозависимая история и дешёвые признаки.
        "rr_all_no_histq": [c for c in all_cols if c not in V5_NEW_FEATURES],
        "rr_all_no_extra": [c for c in all_cols if c not in extra_columns()],
    }
    ens_models, ens_score = [], None
    for name, cols in variants.items():
        log(f"{name}: обучение на {len(cols)} признаках")
        model = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED).fit(train_rows[cols], train_rows.label)
        log(f"{name}: итераций {model.n_iter_}")
        score = model.predict_proba(feat_val[cols])[:, 1]
        results[name] = recall_per_query(top_k_by_score(feat_val, score), s_val.truth)
        joblib.dump({"model": model, "columns": cols}, OUT / f"model_{name}_{FEATURE_VERSION}.joblib")
        if name == "rr_all":
            ens_models, ens_score = [model], score

    # Ансамбль rr_all по сидам: среднее вероятностей (шкала у моделей одна — тот же алгоритм и признаки).
    for seed in ENSEMBLE_SEEDS[1:]:
        log(f"rr_all: сид {seed}")
        model = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=seed).fit(
            train_rows[all_cols], train_rows.label
        )
        ens_models.append(model)
        ens_score = ens_score + model.predict_proba(feat_val[all_cols])[:, 1]
    results["rr_all_ens5"] = recall_per_query(top_k_by_score(feat_val, ens_score / len(ens_models)), s_val.truth)
    joblib.dump({"models": ens_models, "columns": all_cols}, OUT / f"model_rr_all_ens5_{FEATURE_VERSION}.joblib")

    table = {}
    for name, pq in results.items():
        rep = recall_report(pq, segments, seg_cols).set_index(["segment", "value"])
        table[name] = rep.recall
        counts = rep.n
    out = pd.DataFrame(table)
    out.insert(0, "n", counts)
    pd.set_option("display.width", 250)
    print(out.T.round(4).to_string())
    out.T.to_csv(f"logs/h4_rerank_{FEATURE_VERSION}_{args.train}.csv")


if __name__ == "__main__":
    main()
