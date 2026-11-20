"""
File Name: utils.py
Course: CMPSC 445 Fall 2026
Description:
normalize_name: one comparable key for card names from any source
load_split: X, y for a train, val, or test set
scale_features: StandardScaler fit on the training set only
"""

import re
import unicodedata

import pandas as pd
from sklearn.preprocessing import StandardScaler

from core import config

# trailing set tags, (A1 123) or A1-123
SET_TAG = re.compile(r"\s*[\(\[]?\b[A-Z]{1,5}\d?[a-z]?[\s-]\d{1,3}[\)\]]?$")
NON_FEATURES = ["id", "fold", "label"]


def normalize_name(name):
    s = SET_TAG.sub("", str(name).strip())
    s = s.replace("♀", " f").replace("♂", " m") # nidoran
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = s.replace("’", "'").replace("`", "'").lower()
    s = re.sub(r"[\s-]ex$", " ex", s)
    return re.sub(r"\s+", " ", s).strip()


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", normalize_name(name)).strip("_")


def load_cards():
    cards = pd.read_csv(config.CARDS_CSV)
    cards["evolvesFrom"] = cards["evolvesFrom"].fillna("") # empty string, not NaN
    return cards


def load_split(name, which):
    #name: pokemon_cards, decks, deck_matrix. which: train, val, test. Returns X, y
    dataset = pd.read_csv(config.SPLIT_DIR / f"{name}_{which}.csv")
    X = dataset.drop(columns=[c for c in NON_FEATURES if c in dataset.columns])
    y = dataset["label"]
    return X, y


def scale_features(X_train, *others):
    #fit on X_train only. returns scaler, X_train scaled, others scaled
    scaler = StandardScaler()
    train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    rest = [pd.DataFrame(scaler.transform(o), columns=o.columns, index=o.index) for o in others]
    return (scaler, train, *rest)
