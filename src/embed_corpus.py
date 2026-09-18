"""Эмбеддинги объявлений и текстов запросов с кэшем на диске."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .features.text import normalize
from .retrieval.encoder import ENCODERS, Encoder

EMB_DIR = Path("data/embeddings")
PROCESSED = Path("data/processed")
RAW = Path("dataset")

# Длина описания в символах: весь текст документа всё равно обрезается до max_length токенов,
# а заголовок и классификация идут первыми, чтобы гарантированно попасть в окно.
DESC_CHARS = 400
PRICE_LIST_CHARS = 200
CHUNK = 20_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def doc_texts(item_ids: pd.Index) -> list[str]:
    """Текст документа: заголовок. Вид / Тип / предмет. Прайс-лист. Начало описания."""
    items = pd.read_parquet(
        PROCESSED / "items.parquet", columns=["item_id", "item_title_raw", "vid", "tip", "subject", "price_list"]
    )
    items = items.set_index("item_id").loc[item_ids]
    desc = (
        pd.concat(
            [
                pd.read_parquet(RAW / "benchmark_items.parquet", columns=["item_id", "item_description_raw"]),
                pd.read_parquet(RAW / "train.parquet", columns=["item_id", "item_description_raw"]),
            ]
        )
        .drop_duplicates("item_id")
        .set_index("item_id")
        .item_description_raw.reindex(item_ids)
    )

    texts = []
    for (title, vid, tip, subject, price_list), d in zip(
        items[["item_title_raw", "vid", "tip", "subject", "price_list"]].itertuples(index=False), desc
    ):
        meta = " / ".join(x for x in (vid, tip, subject) if x)
        parts = [title, meta, price_list[:PRICE_LIST_CHARS], normalize(d)[:DESC_CHARS] if isinstance(d, str) else ""]
        texts.append(". ".join(p for p in parts if p))
    return texts


def ensure_embeddings(model_key: str, item_ids: pd.Index, max_length: int = 128) -> np.ndarray:
    """Эмбеддинги в порядке item_ids; недостающие считаются и дописываются в кэш."""
    out_dir = EMB_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_path, emb_path = out_dir / "ids.parquet", out_dir / "emb.npy"
    if ids_path.exists():
        cached_ids = pd.read_parquet(ids_path).item_id
        cached = np.load(emb_path)
    else:
        cached_ids, cached = pd.Series([], dtype=str), None

    missing = pd.Index(item_ids).difference(pd.Index(cached_ids))
    if len(missing):
        log(f"{model_key}: кодирую {len(missing)} объявлений порциями по {CHUNK}")
        texts = doc_texts(missing)
        encoder = Encoder(ENCODERS[model_key], max_length=max_length)
        # Порции с сохранением после каждой: на сотнях тысяч объявлений процесс может прерваться
        # (сон ноутбука, конец сессии), а без этого теряется вся работа.
        for start in range(0, len(missing), CHUNK):
            t0 = time.time()
            emb = encoder.encode_docs(texts[start : start + CHUNK]).astype(np.float16)
            cached_ids = pd.concat(
                [pd.Series(cached_ids), pd.Series(missing[start : start + CHUNK])], ignore_index=True
            )
            cached = emb if cached is None else np.vstack([cached, emb])
            pd.DataFrame({"item_id": cached_ids}).to_parquet(ids_path, index=False)
            np.save(emb_path, cached)
            log(
                f"{model_key}: {min(start + CHUNK, len(missing))}/{len(missing)}, "
                f"{len(emb) / (time.time() - t0):.0f} объявлений/с"
            )

    pos = pd.Index(cached_ids).get_indexer(item_ids)
    return cached[pos]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(ENCODERS))
    parser.add_argument("--split", default="val_n5000_s42")
    args = parser.parse_args()
    ids = pd.Index(pd.read_parquet(f"data/validation/{args.split}/corpus_ids.parquet").item_id)
    emb = ensure_embeddings(args.model, ids)
    log(f"{args.model}: готово, {emb.shape}")


if __name__ == "__main__":
    main()


def ensure_query_embeddings(model_key: str, texts: list[str]) -> np.ndarray:
    """Эмбеддинги текстов запросов (с префиксом «query: ») с кэшем по тексту."""
    out_dir = EMB_DIR / f"queries_{model_key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_path, emb_path = out_dir / "texts.parquet", out_dir / "emb.npy"
    if ids_path.exists():
        cached_texts = pd.read_parquet(ids_path).text
        cached = np.load(emb_path)
    else:
        cached_texts, cached = pd.Series([], dtype=str), None
    missing = pd.Index(pd.unique(pd.Series(texts))).difference(pd.Index(cached_texts))
    if len(missing):
        log(f"{model_key}: кодирую {len(missing)} текстов запросов")
        emb = Encoder(ENCODERS[model_key]).encode_queries(missing.tolist()).astype(np.float16)
        cached_texts = pd.concat([pd.Series(cached_texts), pd.Series(missing)], ignore_index=True)
        cached = emb if cached is None else np.vstack([cached, emb])
        pd.DataFrame({"text": cached_texts}).to_parquet(ids_path, index=False)
        np.save(emb_path, cached)
    return cached[pd.Index(cached_texts).get_indexer(texts)]
