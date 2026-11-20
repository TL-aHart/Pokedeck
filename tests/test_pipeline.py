"""
File Name: test_pipeline.py
Course: CMPSC 445 Fall 2026
Description:
1. data checks: card tables load, no missing values in features
2. split checks: no overlap, 60/20/20, same Pokemon name never in two sets, train only scaling
3. system check: a decision tree trained on X_train beats the majority class baseline on X_val
4. plug in check: deck functions run on a small in memory deck table built from real card names
   (format check only, never written to data/)
Run: pytest tests
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

import main
from core import config, features, split_data
from core.clean_data import clean_decks
from core.utils import load_cards, load_split, normalize_name, scale_features

NAME = "pokemon_cards"


@pytest.fixture(scope="session", autouse=True)
def prepared_data():
    if not (config.SPLIT_DIR / f"{NAME}_train.csv").exists():
        main.run()


def test_cards_and_features():
    cards = load_cards()
    card_features = pd.read_csv(config.CARD_FEATURES_CSV)
    assert cards["card_id"].is_unique
    assert (cards.groupby("name_key")["is_primary"].sum() == 1).all()
    assert list(card_features["card_id"]) == list(cards["card_id"])
    assert card_features.isna().sum().sum() == 0


def test_type_chart():
    chart = pd.read_csv(config.TYPE_CHART_CSV, index_col="attacker")
    assert list(chart.index) == config.ENERGY_TYPES and list(chart.columns) == config.ENERGY_TYPES
    assert chart.values.min() >= 0 and chart.values.max() <= 1


def test_split_sizes_and_no_overlap():
    ids = {w: set(pd.read_csv(config.SPLIT_DIR / f"{NAME}_{w}.csv")["id"]) for w in ["train", "val", "test"]}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    total = sum(len(v) for v in ids.values())
    assert abs(len(ids["train"]) / total - 0.6) < 0.03
    assert abs(len(ids["val"]) / total - 0.2) < 0.03 and abs(len(ids["test"]) / total - 0.2) < 0.03


def test_same_pokemon_name_not_in_two_sets():
    names = load_cards().set_index("card_id")["name_key"]
    seen = {}
    for w in ["train", "val", "test"]:
        for card_id in pd.read_csv(config.SPLIT_DIR / f"{NAME}_{w}.csv")["id"]:
            assert seen.setdefault(names[card_id], w) == w


def test_label_not_in_features():
    X, y = load_split(NAME, "train")
    assert not any(c.startswith(f"{config.CARD_LABEL}_") for c in X.columns)
    assert X.isna().sum().sum() == 0 and y.notna().all()


def test_scaler_fits_train_only():
    X_train, _ = load_split(NAME, "train")
    X_val, _ = load_split(NAME, "val")
    _, train, val = scale_features(X_train, X_val)
    assert train.mean().abs().max() < 1e-6
    assert val.mean().abs().max() > 1e-6


def test_model_beats_majority_baseline():
    X_train, y_train = load_split(NAME, "train")
    X_val, y_val = load_split(NAME, "val")
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train).score(X_val, y_val)
    tree = DecisionTreeClassifier(max_depth=6, random_state=config.RANDOM_STATE).fit(X_train, y_train)
    accuracy = tree.score(X_val, y_val)
    print(f"majority baseline {baseline:.3f}, decision tree {accuracy:.3f}")
    assert accuracy > baseline


def test_deck_plug_in_with_in_memory_decks():
    cards = load_cards()
    primary = cards[cards["is_primary"]]
    pool = list(primary[primary["type"] == "pokemon"]["name"].sample(40, random_state=1)) + \
        list(primary[primary["type"] != "pokemon"]["name"].sample(10, random_state=1))
    rng = np.random.default_rng(config.RANDOM_STATE)
    rows = []
    for a in range(3):
        for d in range(config.MIN_DECKS_PER_ARCHETYPE + 2):
            for name in rng.choice(pool, 10, replace=False):
                rows.append({"deck_id": f"t{a}_{d}", "archetype": f"A{a}", "card_name": name, "count": 2})
    decks, deck_cards = clean_decks(pd.DataFrame(rows), cards)
    assert (deck_cards.groupby("deck_id")["count"].sum() == config.DECK_SIZE).all()
    decks = decks.merge(features.main_attackers(deck_cards, cards), on="deck_id")
    dataset = features.deck_features(deck_cards, decks, cards)
    matrix = features.deck_matrix(deck_cards, decks)
    assert dataset.isna().sum().sum() == 0 and len(dataset) == len(matrix) == len(decks)
    assignment = split_data.assign_split(dataset)
    assert set(assignment) == {"train", "val", "test"}


def test_normalize_name():
    assert normalize_name("Poké Ball") == "poke ball"
    assert normalize_name("Pikachu-ex") == "pikachu ex"
    assert normalize_name("Pikachu ex (A1 096)") == "pikachu ex"
