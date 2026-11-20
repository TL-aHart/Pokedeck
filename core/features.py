"""
File Name: features.py
Course: CMPSC 445 Fall 2026
Description:
card_features: one feature vector per unique card
pokemon_dataset: Pokemon cards, X = stats and weakness, label = CARD_LABEL (element by default)
deck_features: one feature vector per 20 card deck, label = archetype
deck_matrix: deck x card copy counts
Datasets use columns id, features, label (label last). Features are not scaled here
"""

import pandas as pd

from core import config

ELEMENTS = config.ENERGY_TYPES + ["none"]


def card_features(cards):
    features = cards[["card_id"]].copy()
    for t in config.CARD_TYPES:
        features[f"is_{t}"] = (cards["type"] == t).astype(int)
    for e in ELEMENTS:
        features[f"element_{e}"] = (cards["element"] == e).astype(int)
    for s in config.STAGES:
        features[f"stage_{s}"] = (cards["stage"] == s).astype(int)
    for w in ELEMENTS:
        features[f"weak_{w}"] = (cards["weakness"] == w).astype(int)
    features["health"] = cards["health"]
    features["retreat_cost"] = cards["retreatCost"]
    features["is_ex"] = cards["is_ex"].astype(int)
    features["evolves_from_other"] = (cards["evolvesFrom"] != "").astype(int)
    return features


def pokemon_dataset(cards, features, label=config.CARD_LABEL):
    #label columns are dropped from X so the label never leaks into the features
    pokemon = cards[(cards["type"] == "pokemon") & (cards["element"] != "none")]
    dataset = features.set_index("card_id").loc[pokemon["card_id"]]
    dataset = dataset.drop(columns=[c for c in dataset.columns if c.startswith(f"{label}_")])
    dataset = dataset.loc[:, dataset.nunique() > 1]                # drop constant columns (is_pokemon, ...)
    dataset["label"] = pokemon.set_index("card_id")[label]
    return dataset.rename_axis("id").reset_index()


def main_attackers(deck_cards, cards):
    #highest HP ex Pokemon in the deck, else highest HP Pokemon
    dc = deck_cards.merge(cards[["card_id", "name", "type", "element", "health", "is_ex"]], on="card_id")
    pokemon = dc[dc["type"] == "pokemon"].sort_values(["is_ex", "health", "count"], ascending=False)
    return (pokemon.groupby("deck_id").first()[["name", "element"]]
            .rename(columns={"name": "main_attacker", "element": "main_element"}).reset_index())


def deck_features(deck_cards, decks, cards):
    dc = deck_cards.merge(cards.drop(columns=["name"]), on="card_id")
    pokemon = dc[dc["type"] == "pokemon"]
    n_pokemon = pokemon.groupby("deck_id")["count"].sum()
    X = pd.DataFrame(index=sorted(dc["deck_id"].unique()))
    for t in config.CARD_TYPES:
        X[f"n_{t}"] = dc[dc["type"] == t].groupby("deck_id")["count"].sum()
    for s in config.STAGES:
        X[f"n_stage_{s}"] = pokemon[pokemon["stage"] == s].groupby("deck_id")["count"].sum()
    X["n_ex"] = pokemon[pokemon["is_ex"]].groupby("deck_id")["count"].sum()
    X["n_evolving"] = pokemon[pokemon["evolvesFrom"] != ""].groupby("deck_id")["count"].sum()
    X["n_unique_cards"] = dc.groupby("deck_id")["card_id"].nunique()
    weighted = pokemon.assign(hp=pokemon["health"] * pokemon["count"], rc=pokemon["retreatCost"] * pokemon["count"])
    X["mean_hp"] = weighted.groupby("deck_id")["hp"].sum() / n_pokemon
    X["max_hp"] = pokemon.groupby("deck_id")["health"].max()
    X["mean_retreat"] = weighted.groupby("deck_id")["rc"].sum() / n_pokemon
    for e in config.ENERGY_TYPES:                                  # type mix of the deck
        X[f"frac_{e}"] = pokemon[pokemon["element"] == e].groupby("deck_id")["count"].sum() / n_pokemon
    for e in config.ENERGY_TYPES:                                  # exposure to each attacking type
        X[f"weak_to_{e}"] = pokemon[pokemon["weakness"] == e].groupby("deck_id")["count"].sum() / n_pokemon
    main = main_attackers(deck_cards, cards).set_index("deck_id")
    for e in config.ENERGY_TYPES:
        X[f"main_{e}"] = (main["main_element"] == e).astype(int)
    X = X.fillna(0)
    X["label"] = decks.set_index("deck_id")["archetype"]
    return X.rename_axis("id").reset_index()


def deck_matrix(deck_cards, decks):
    X = deck_cards.pivot_table(index="deck_id", columns="card_id", values="count", aggfunc="sum", fill_value=0)
    X["label"] = decks.set_index("deck_id")["archetype"]
    return X.rename_axis("id").reset_index()
