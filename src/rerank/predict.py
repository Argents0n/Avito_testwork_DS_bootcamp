"""Ответ переранжированием: кандидаты → признаки → модель → топ-50."""

from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pandas as pd

from ..embed_corpus import ensure_embeddings, ensure_query_embeddings
from ..metrics import recall_at_k
from ..submission import RAW, validate_answer, write_answer
from ..validation import load_split
from .extra_features import add_extra_features, extra_item_table
from .features import FEATURE_VERSION, CandidateBuilder, StatsContext
from .train import ITEM_COLS, MODEL, NEIGHBOR_MODEL, USER_MODEL, top_k_by_score

PROCESSED = "data/processed"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(
    model_names: list[str],
    corpus_ids: pd.Index,
    queries: pd.DataFrame,
    fit_queries: pd.DataFrame,
    fit_pairs: pd.DataFrame,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, list[str]]]:
    """Возвращает ({модель: ответ переранжирования}, ответ базового гибрида)."""
    items_all = pd.read_parquet(f"{PROCESSED}/items.parquet", columns=ITEM_COLS)
    texts = pd.read_parquet(f"{PROCESSED}/items_text.parquet")
    texts = texts[texts.item_id.isin(corpus_ids)]
    log(f"индексы: корпус {len(corpus_ids)}")
    builder = CandidateBuilder(
        items_all,
        texts,
        corpus_ids,
        ensure_embeddings(MODEL, corpus_ids),
        ensure_embeddings(USER_MODEL, corpus_ids),
        ensure_embeddings(NEIGHBOR_MODEL, corpus_ids),
    )
    del texts
    ctx = StatsContext.from_fit(fit_pairs, fit_queries, items_all, with_neighbors=True, neighbor_model=NEIGHBOR_MODEL)
    queries = queries.reset_index(drop=True)
    texts_q = queries.search_query.tolist()
    log(f"кандидаты и признаки: {len(queries)} запросов")
    feats, base = builder.build(
        queries,
        ensure_query_embeddings(MODEL, texts_q),
        ctx,
        query_emb_user=ensure_query_embeddings(USER_MODEL, texts_q),
        query_emb_neighbors=ensure_query_embeddings(NEIGHBOR_MODEL, texts_q),
    )
    # Дешёвые признаки (H12) присоединяются так же, как в обучении.
    feats = add_extra_features(feats, extra_item_table(), queries)
    results = {}
    for name in model_names:
        bundle = joblib.load(f"data/rerank/model_{name}_{FEATURE_VERSION}.joblib")
        # Ансамбль по сидам хранится списком моделей — усредняем вероятности.
        models = bundle["models"] if "models" in bundle else [bundle["model"]]
        score = np.mean([m.predict_proba(feats[bundle["columns"]])[:, 1] for m in models], axis=0)
        reranked = top_k_by_score(feats, score)
        # Страховка: кандидатов ≥ 500 на запрос, но если бы их оказалось меньше 50 — добиваем базовым гибридом.
        for qid, items in reranked.items():
            if len(items) < 50:
                reranked[qid] = list(dict.fromkeys(items + base[qid]))[:50]
        results[name] = reranked
    return results, base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        nargs="+",
        default=["rr_no_hist"],
        help="в режиме --split можно несколько; для ответа на бенчмарк берётся первая",
    )
    parser.add_argument("--split")
    parser.add_argument("--out", default="submissions/03_h4_rerank.csv")
    args = parser.parse_args()

    if args.split:
        split = load_split(args.split)
        results, base = run(args.model, split.corpus_ids, split.val_queries, split.fit_queries, split.fit_pairs)
        log(f"{args.split}: base {recall_at_k(base, split.truth):.4f}")
        for name, reranked in results.items():
            recall = recall_at_k(reranked, split.truth)
            # Калибровка по 6 отправкам (docs/hypotheses.md, H6): платформа ≈ 0.913 × валидация + 0.040.
            log(f"{args.split}: {name} {recall:.4f} → прогноз платформы {0.913 * recall + 0.0397:.4f}")
        return

    queries = pd.read_parquet(f"{PROCESSED}/queries.parquet")
    train_q, bench_q = queries[queries.source == "train"], queries[queries.source == "bench"]
    pairs = pd.read_parquet(f"{PROCESSED}/train_pairs.parquet")
    corpus_ids = pd.Index(pd.read_parquet(RAW / "benchmark_items.parquet", columns=["item_id"]).item_id)
    results, _ = run(args.model[:1], corpus_ids, bench_q, train_q, pairs)
    reranked = results[args.model[0]]
    lengths = np.array([len(v) for v in reranked.values()])
    log(f"ответов {len(reranked)}, длина min {lengths.min()}, max {lengths.max()}")
    write_answer(reranked, args.out)
    errors = validate_answer(args.out, set(bench_q.query_id), set(corpus_ids))
    if errors:
        log(f"ОШИБКИ ФОРМАТА ({len(errors)}):\n" + "\n".join(errors[:20]))
        raise SystemExit(1)
    log(f"{args.out} записан и прошёл проверку формата")


if __name__ == "__main__":
    main()
