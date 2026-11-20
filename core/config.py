"""
File Name: config.py
Course: CMPSC 445 Fall 2026
Description: paths, constants, seed shared by core modules
"""

from pathlib import Path

# folders
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLIT_DIR = PROCESSED_DIR / "splits"
FIGURES_DIR = ROOT / "out" / "figures"
SUMMARY_TXT = ROOT / "out" / "summary.txt"

# github source (flibustier pokemon-tcg-pocket-database, MIT)
GITHUB_URL = "https://raw.githubusercontent.com/flibustier/pokemon-tcg-pocket-database/main/dist/"
GITHUB_FILES = ["cards.extra.json", "sets.json"]

# PLUG IN: scraped decks, csv columns: deck_id, archetype, card_name, count
DECKS_CSV = RAW_DIR / "decks.csv"
# PLUG IN: scraped matchups, csv columns: archetype_a, archetype_b, win_rate_a
MATCHUPS_CSV = RAW_DIR / "matchups.csv"

# processed files
CARDS_CSV = PROCESSED_DIR / "cards.csv"
CARD_FEATURES_CSV = PROCESSED_DIR / "card_features.csv"
TYPE_CHART_CSV = PROCESSED_DIR / "type_chart.csv"
DECK_CARDS_CSV = PROCESSED_DIR / "deck_cards.csv"
DECKS_CSV_CLEAN = PROCESSED_DIR / "decks.csv"
DECK_FEATURES_CSV = PROCESSED_DIR / "deck_features.csv"
DECK_MATRIX_CSV = PROCESSED_DIR / "deck_matrix.csv"
ARCHETYPES_CSV = PROCESSED_DIR / "archetypes.csv"
MATCHUPS_CLEAN_CSV = PROCESSED_DIR / "matchups.csv"

# game constants
DECK_SIZE = 20
MAX_COPIES = 2
ENERGY_TYPES = ["grass", "fire", "water", "lightning", "psychic",
                "fighting", "darkness", "metal", "dragon", "colorless"]
CARD_TYPES = ["pokemon", "supporter", "item", "tool", "stadium", "fossil"]
STAGES = ["basic", "1", "2"]

# modelling constants
RANDOM_STATE = 42
N_FOLDS = 5 # 60/20/20 split = 3 train folds, 1 val fold, 1 test fold
CARD_LABEL = "element" # PLUG IN: any card column to predict
MIN_DECKS_PER_ARCHETYPE = 8 # rarer archetypes merge into other
OTHER_LABEL = "other"