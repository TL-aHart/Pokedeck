"""
File Name: load_data.py
Course: CMPSC 445 Fall 2026
Description:
load_cards: github card database, cached in data/raw
load_decks, load_matchups: optional scraped csv files, None if absent
"""

import json

import pandas as pd
import requests

from core import config


def download_json(file_name, force=False):
    path = config.RAW_DIR / file_name
    if force or not path.exists():
        resp = requests.get(config.GITHUB_URL + file_name, timeout=60)
        resp.raise_for_status()
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(resp.text, encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def load_cards(force=False):
    #one row per printed card, has set release date
    dataset = pd.DataFrame(download_json("cards.extra.json", force))
    dataset["packs"] = dataset["packs"].apply(lambda p: "|".join(p) if isinstance(p, list) else "")
    sets = download_json("sets.json", force)
    set_dates = pd.DataFrame([{"set": s["code"], "release_date": s["releaseDate"]}
                              for group in sets.values() for s in group])
    dataset = dataset.merge(set_dates, on="set", how="left")
    print(f"load_cards: {len(dataset)} printed cards")
    return dataset


# new card source, return the same columns as load_cards
def load_decks():
    if not config.DECKS_CSV.exists():
        print("load_decks: no decks.csv in data/raw, deck stages skipped")
        return None
    return pd.read_csv(config.DECKS_CSV)


def load_matchups():
    return pd.read_csv(config.MATCHUPS_CSV) if config.MATCHUPS_CSV.exists() else None
