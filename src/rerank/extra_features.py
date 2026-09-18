"""Дешёвые признаки из полей данных, присоединяются к кандидатам по item_id / query_id."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.text import stems

PROCESSED = "data/processed"
RAW = "dataset"


# Слова в объявлении, подтверждающие намерение запроса (стеммы сравниваются со стеммами заголовка и прайс-листа).
INTENT_ITEM_WORDS = {
    "buy": ["скупка", "выкуп", "куплю", "покупка", "прием", "дорого", "оценка"],
    "rent": ["аренда", "прокат", "посуточно", "сдаю", "сдается", "почасово"],
    "learn": ["обучение", "курсы", "урок", "занятия", "репетитор", "подготовка", "школа"],
    "job": ["работа", "вакансия", "требуется", "подработка"],
}
AT_HOME_WORDS = {"дому", "выезд", "выездом", "выездной", "приезд"}
REMOTE_WORDS = {"онлайн", "online", "дистанционно", "удаленно", "удаленный", "zoom", "скайп", "skype"}

EXTRA_GROUPS = {
    "price": ["price_rel_city_mc", "price_missing"],
    "intent": ["intent_match", "intent_mismatch"],
    "modifier": ["mod_home_match", "mod_remote_match"],
    "toponym": ["toponym_in_address"],
    "online": ["online_booking_match"],
    "flags": [
        "work_online",
        "work_at_client",
        "work_at_self",
        "travel_region",
        "travel_zones",
        "travel_none",
        "has_contract",
        "has_guarantee",
        "works_with_legal",
    ],
    "twins": ["twins_city", "twin_rank_reviews"],
    "text_len": ["title_words", "desc_chars"],
    "cat0": ["query_cat0"],
}


def extra_item_table() -> pd.DataFrame:
    """Признаки объявлений, не зависящие от запроса (по всем объявлениям: корпус + трейн)."""
    items = pd.read_parquet(
        f"{PROCESSED}/items.parquet",
        columns=[
            "item_id",
            "item_title_raw",
            "item_microcat_id",
            "item_location_id",
            "price_clean",
            "log_reviews",
            "address",
            "price_list",
            *EXTRA_GROUPS["flags"],
        ],
    )
    texts = pd.read_parquet(f"{PROCESSED}/items_text.parquet", columns=["item_id", "title_stems"])
    items = items.merge(texts, on="item_id")
    desc = pd.concat(
        [
            pd.read_parquet(f"{RAW}/benchmark_items.parquet", columns=["item_id", "item_description_raw"]),
            pd.read_parquet(f"{RAW}/train.parquet", columns=["item_id", "item_description_raw"]),
        ]
    )
    desc = desc.drop_duplicates("item_id").set_index("item_id").item_description_raw.str.len()
    items["desc_chars"] = items.item_id.map(desc).fillna(0).astype(np.float32)
    items["title_words"] = items.title_stems.str.split().str.len().astype(np.float32)

    med = items.groupby(["item_location_id", "item_microcat_id"]).price_clean.transform("median")
    items["price_rel_city_mc"] = (items.price_clean / med).astype(np.float32)
    items["price_missing"] = items.price_clean.isna().astype(np.float32)

    key = items.item_title_raw.str.lower().str.strip()
    grp = items.groupby([items.item_location_id, key])
    items["twins_city"] = grp.item_id.transform("size").astype(np.float32)
    items["twin_rank_reviews"] = grp.log_reviews.rank(ascending=False, method="average").astype(np.float32)

    # Стеммы заголовка + прайс-листа для намерения и модификаторов; стеммы адреса для топонимов.
    items["text_stem_set"] = [set(t.split()) | set(stems(p)) for t, p in zip(items.title_stems, items.price_list)]
    items["addr_stem_set"] = [set(stems(a)) for a in items.address]
    for c in EXTRA_GROUPS["flags"]:
        items[c] = items[c].astype(np.float32)
    return items


def add_extra_features(feat: pd.DataFrame, items: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "item_id",
        "price_rel_city_mc",
        "price_missing",
        "twins_city",
        "twin_rank_reviews",
        "title_words",
        "desc_chars",
        "text_stem_set",
        "addr_stem_set",
        *EXTRA_GROUPS["flags"],
    ]
    q = queries.set_index("query_id")[["q_intent", "q_stems", "q_toponyms", "f_online_booking", "search_category"]]
    df = feat.merge(items[cols], on="item_id", how="left").join(q, on="query_id")

    intent_stems = {k: {stems(w)[0] for w in v} for k, v in INTENT_ITEM_WORDS.items()}
    home = {stems(w)[0] for w in AT_HOME_WORDS}
    remote = {stems(w)[0] for w in REMOTE_WORDS}
    intent_match, intent_mismatch, mod_home, mod_remote, topo = [], [], [], [], []
    # travel_city и work_remote уже есть в признаках v4 (группа item), остальные флаги присоединены выше.
    for intents, qs, tops, text_set, addr_set, at_client, travel_city, w_remote, w_online in zip(
        df.q_intent,
        df.q_stems,
        df.q_toponyms,
        df.text_stem_set,
        df.addr_stem_set,
        df.work_at_client,
        df.travel_city,
        df.work_remote,
        df.work_online,
    ):
        its = [i for i in intents.split(",") if i]
        if its:
            hit = any(text_set & intent_stems[i] for i in its)
            intent_match.append(float(hit))
            intent_mismatch.append(float(not hit))
        else:
            intent_match.append(-1.0)
            intent_mismatch.append(-1.0)
        q_set = set(qs.split())
        mod_home.append(float(at_client > 0 or travel_city > 0) if q_set & home else -1.0)
        mod_remote.append(float(w_remote > 0 or w_online > 0) if q_set & remote else -1.0)
        topo.append(float(bool(set(tops.split()) & addr_set)) if tops else -1.0)
    df["intent_match"] = np.asarray(intent_match, np.float32)
    df["intent_mismatch"] = np.asarray(intent_mismatch, np.float32)
    df["mod_home_match"] = np.asarray(mod_home, np.float32)
    df["mod_remote_match"] = np.asarray(mod_remote, np.float32)
    df["toponym_in_address"] = np.asarray(topo, np.float32)
    df["online_booking_match"] = np.where(df.f_online_booking, df.online_booking, -1.0).astype(np.float32)
    df["query_cat0"] = (df.search_category == 0).astype(np.float32)
    return df.drop(
        columns=[
            "text_stem_set",
            "addr_stem_set",
            "q_intent",
            "q_stems",
            "q_toponyms",
            "f_online_booking",
            "search_category",
        ]
    )


def extra_columns() -> list[str]:
    return [c for group in EXTRA_GROUPS.values() for c in group]
