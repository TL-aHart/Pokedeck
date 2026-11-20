"""
File Name: clean_data.py
Course: CMPSC 445 Fall 2026
Description:
clean_cards: missing values, reprints collapsed to one row per unique playable card
clean_decks: join to cards, keep legal 20 card decks, merge rare archetypes into other
clean_matchups: win rates 0 to 1, mirror rows, known archetypes only
All functions take and return DataFrames (no file io)
"""

import numpy as np
import pandas as pd

from core import config
from core.utils import normalize_name, slugify

VARIANT = ["name_key", "type", "element", "stage", "health", "retreatCost", "weakness", "evolvesFrom"]


def _stage(s):
    return "none" if pd.isna(s) else str(int(s)) if isinstance(s, (int, float)) else str(s).lower()


def clean_cards(dataset):
    missing_values = dataset.isna().sum()
    print("clean_cards: missing values before cleaning")
    print(missing_values[missing_values > 0].to_string())

    df = dataset.copy()
    df["name_key"] = df["name"].apply(normalize_name)
    df["type"] = df["type"].str.lower()
    df["element"] = df["element"].fillna("none").str.lower() # trainers have no element
    df["stage"] = df["stage"].apply(_stage)
    df["health"] = pd.to_numeric(df["health"], errors="coerce").fillna(0).astype(int)
    df["retreatCost"] = pd.to_numeric(df["retreatCost"], errors="coerce").fillna(0).astype(int)
    df["weakness"] = df["weakness"].fillna("none").str.lower() # dragon, colorless have none
    df["evolvesFrom"] = df["evolvesFrom"].fillna("").apply(lambda s: normalize_name(s) if s else "")
    df["is_ex"] = (df["type"] == "pokemon") & df["name"].str.lower().str.endswith(" ex")
    df = df.sort_values(["release_date", "set", "number"]).drop_duplicates(["set", "number"])

    # reprints (other set, rarity, art) collapse to a unique card
    cards = (df.groupby(VARIANT, sort=False, dropna=False)
             .agg(name=("name", "first"), first_set=("set", "first"), first_number=("number", "first"),
                  release_date=("release_date", "first"), n_prints=("set", "size"), is_ex=("is_ex", "first"))
             .reset_index().sort_values(["release_date", "first_set", "first_number"]).reset_index(drop=True))
    # same name, different stats gets _v2, _v3
    cards["variant_rank"] = cards.groupby("name_key").cumcount() + 1
    cards["card_id"] = [slugify(n) + ("" if r == 1 else f"_v{r}") for n, r in zip(cards["name"], cards["variant_rank"])]
    cards["is_primary"] = cards["variant_rank"] == 1 # variant used when a deck lists a bare name
    assert cards["card_id"].is_unique
    print(f"clean_cards: {len(cards)} unique playable cards")
    return cards.drop(columns=["variant_rank"])


# scraped decks show up here, a row per card (deck_id, archetype, card_name, count)
def clean_decks(raw_decks, cards):
    df = raw_decks.copy()
    df["deck_id"] = df["deck_id"].astype(str).str.strip()
    df["archetype_raw"] = df["archetype"].astype(str).str.strip()
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    lookup = cards[cards["is_primary"]].set_index("name_key")["card_id"]
    df["card_id"] = df["card_name"].apply(normalize_name).map(lookup)

    unmatched = df[df["card_id"].isna()]
    if len(unmatched):
        print(f"clean_decks: unmatched card names {sorted(unmatched['card_name'].unique())[:10]}")
    deck_cards = (df.dropna(subset=["card_id"]).groupby(["deck_id", "card_id"], as_index=False)
                  .agg(count=("count", "sum"), archetype_raw=("archetype_raw", "first")))

    stats = deck_cards.groupby("deck_id")["count"].agg(["sum", "max"])
    valid = (stats["sum"] == config.DECK_SIZE) & (stats["max"] <= config.MAX_COPIES)
    valid &= ~stats.index.isin(unmatched["deck_id"])
    print(f"clean_decks: {len(stats)} decks, {int(valid.sum())} legal")
    deck_cards = deck_cards[deck_cards["deck_id"].isin(stats.index[valid])]

    signature = deck_cards.sort_values("card_id").groupby("deck_id").apply(
        lambda g: "|".join(f"{c}:{n}" for c, n in zip(g["card_id"], g["count"])), include_groups=False)
    deck_cards = deck_cards[~deck_cards["deck_id"].isin(signature[signature.duplicated()].index)]   # exact duplicates

    decks = deck_cards.groupby("deck_id", as_index=False).agg(archetype_raw=("archetype_raw", "first"))
    sizes = decks["archetype_raw"].value_counts()
    rare = sizes[sizes < config.MIN_DECKS_PER_ARCHETYPE].index
    decks["archetype"] = np.where(decks["archetype_raw"].isin(rare), config.OTHER_LABEL, decks["archetype_raw"])
    deck_cards = deck_cards.merge(decks[["deck_id", "archetype"]], on="deck_id")
    print(f"clean_decks: {len(decks)} decks, {decks['archetype'].nunique()} archetypes, merged to other: {list(rare)}")
    return decks, deck_cards


def clean_matchups(raw_matchups, decks):
    df = raw_matchups.copy()
    df["win_rate_a"] = pd.to_numeric(df["win_rate_a"], errors="coerce")
    df.loc[df["win_rate_a"] > 1, "win_rate_a"] /= 100 # percent to fraction
    df = df.dropna(subset=["win_rate_a"])
    known = set(decks["archetype"]) - {config.OTHER_LABEL}
    df = df[df["archetype_a"].isin(known) & df["archetype_b"].isin(known)]
    mirror = df.rename(columns={"archetype_a": "archetype_b", "archetype_b": "archetype_a"}).assign(
        win_rate_a=lambda d: 1 - d["win_rate_a"])
    return pd.concat([df, mirror]).drop_duplicates(["archetype_a", "archetype_b"]).reset_index(drop=True)
