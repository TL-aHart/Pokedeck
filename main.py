"""
File Name: main.py
Course: CMPSC 445 Fall 2026
Project: PokeDeck, Pokemon TCG Pocket counter deck recommender
Description:
Runs the data pipeline: load, clean, features, split, explore
Cards (github) always run. Deck stages run when data/raw/decks.csv exists
Outputs: data/processed, out/figures, out/summary.txt
"""

from core import config, explore_data, features, load_data, split_data, type_chart
from core.clean_data import clean_cards, clean_decks, clean_matchups


def run():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # cards
    dataset = load_data.load_cards()
    cards = clean_cards(dataset)
    card_features = features.card_features(cards)
    chart = type_chart.build_type_chart(cards)
    cards.to_csv(config.CARDS_CSV, index=False)
    card_features.to_csv(config.CARD_FEATURES_CSV, index=False)
    chart.to_csv(config.TYPE_CHART_CSV)

    pokemon = features.pokemon_dataset(cards, card_features)
    names = cards.set_index("card_id")["name_key"]
    split_data.split_dataset(pokemon, "pokemon_cards", groups=pokemon["id"].map(names))
    explore_data.explore_cards(dataset, cards, card_features, chart)

    # PLUG IN FOR REILLY: ml cluster, classify, recommendion process here, read data with core.utils.load_split

    # decks
    raw_decks = load_data.load_decks()
    if raw_decks is not None:
        decks, deck_cards = clean_decks(raw_decks, cards)
        main = features.main_attackers(deck_cards, cards)
        decks = decks.merge(main, on="deck_id")
        deck_dataset = features.deck_features(deck_cards, decks, cards)
        matrix = features.deck_matrix(deck_cards, decks)
        decks.to_csv(config.DECKS_CSV_CLEAN, index=False)
        deck_cards.to_csv(config.DECK_CARDS_CSV, index=False)
        deck_dataset.to_csv(config.DECK_FEATURES_CSV, index=False)
        matrix.to_csv(config.DECK_MATRIX_CSV, index=False)
        (decks.groupby("archetype").agg(n_decks=("deck_id", "size"), main_attacker=("main_attacker", lambda s: s.mode()[0]),
                                        main_element=("main_element", lambda s: s.mode()[0]))
         .reset_index().to_csv(config.ARCHETYPES_CSV, index=False))
        assignment = split_data.split_dataset(deck_dataset, "decks")
        split_data.split_dataset(matrix, "deck_matrix", assignment=assignment)
        explore_data.explore_decks(deck_dataset)
        raw_matchups = load_data.load_matchups()
        if raw_matchups is not None:
            clean_matchups(raw_matchups, decks).to_csv(config.MATCHUPS_CLEAN_CSV, index=False)
    print("main: done")


if __name__ == "__main__":
    run()
