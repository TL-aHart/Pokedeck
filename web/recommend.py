"""
File Name: recommend.py
Course: CMPSC 445 Fall 2026
Project: PokeDeck, Pokemon TCG Pocket counter deck recommender
Description:
    ***** Mock recommender. Replace `recommend()` with a call to the ML model! *****
        Whatever is returned must keep the same JSON shape
"""
from core import config, features
from core.utils import normalize_name, load_cards
from core.type_chart import type_advantage
import pandas as pd

CARDS = load_cards()
PRIMARY = CARDS[CARDS["is_primary"]]

TYPES = [t.capitalize() for t in config.ENERGY_TYPES]   # UI labels
def to_elements(ui_type): return ui_type.lower()        # UI to data
def to_ui(element): return element.capitalize()         # data to UI

ICON_PATH = {"Fire": "a/ad", "Water": "1/11", "Grass": "2/2e", "Lightning": "0/04", "Psychic": "e/ef",
             "Fighting": "4/48", "Darkness": "a/ab", "Metal": "6/64", "Dragon": "8/8a", "Colorless": "1/1d"}

CHART = pd.read_csv(config.TYPE_CHART_CSV, index_col="attacker")
def best_counter(defender):
    col = CHART[defender]
    return col.idxmax() if col.max() > 0 else "metal"   # nothing is super effective against dragon, so... metal?

DECKS = {
    "Fire": [(2, "Charmander"), (2, "Charmeleon"), (2, "Charizard ex"), (2, "Moltres ex"), (2, "Ninetales"), (1, "Vulpix"), (1, "Ponyta")],
    "Water": [(2, "Froakie"), (2, "Frogadier"), (2, "Greninja"), (2, "Articuno ex"), (2, "Squirtle"), (2, "Staryu")],
    "Lightning": [(2, "Pikachu ex"), (2, "Zapdos ex"), (2, "Voltorb"), (2, "Electrode"), (2, "Blitzle"), (2, "Zebstrika")],
    "Fighting": [(2, "Machop"), (2, "Machoke"), (2, "Machamp ex"), (2, "Marshadow"), (2, "Hitmonlee"), (2, "Primeape")],
    "Darkness": [(2, "Koffing"), (2, "Weezing"), (2, "Darkrai ex"), (2, "Zorua"), (2, "Zoroark"), (2, "Arbok")],
    "Psychic": [(2, "Ralts"), (2, "Kirlia"), (2, "Gardevoir"), (2, "Mewtwo ex"), (2, "Mew ex"), (2, "Jynx")],
    "Grass": [(2, "Celebi ex"), (2, "Exeggcute"), (2, "Exeggutor ex"), (2, "Serperior"), (2, "Snivy"), (2, "Servine")],
    "Metal": [(2, "Skarmory"), (2, "Bronzor"), (2, "Bronzong"), (2, "Dialga ex"), (2, "Melmetal"), (2, "Meltan")],
}

DECK_TRAINERS = [(2, "Professor's Research"), (2, "Poké Ball"), (2, "Sabrina"), (1, "Giovanni"), (1, "X Speed")]
DECK_SIZE = config.DECK_SIZE

LOOKUP = PRIMARY.set_index("name_key")["card_id"]

def to_card_ids(names):
    return [LOOKUP.get(normalize_name(n)) for n in names]

def type_icon(t):
    return f"https://archives.bulbagarden.net/media/upload/{ICON_PATH[t]}/{t}-attack.png"

def recommend(enemy_type, known_count, basis):
    element = to_elements(enemy_type)
    counter_el  = best_counter(element)
    counter = to_ui(counter_el)
    share = type_advantage(CHART, counter_el, element)
    pokemon = list(DECKS[counter])
    trainers = list(DECK_TRAINERS)
    fill = DECK_SIZE - sum(n for n, _ in pokemon) - sum(n for n, _ in trainers)
    if fill > 0:
        trainers.append((fill, "Potion"))

    win = min(0.84, 0.58 + known_count * 0.035 + (len(enemy_type) % 3) * 0.01)
    reason = (f"Nothing hits {enemy_type} for weakness damage, so high-HP {counter} attackers are the steadiest pick."
     if share == 0 else
     f"{share:.0%} of {enemy_type} Pokémon are weak to {counter}, so these attackers deal bonus damage.")

    to_cards = lambda lst: [{"n": n, "name": name} for n, name in lst]
    return {
        "title": f"{counter} deck",
        "icon": type_icon(counter),
        "reason": reason,
        "basis": basis,
        "winPct": round(win * 100),
        "confidence": "High" if known_count >= 6 else "Medium" if known_count >= 2 else "Low",
        "groups": [
            {"name": "Pokémon", "cards": to_cards(pokemon)},
            {"name": "Trainers", "cards": to_cards(trainers)},
        ],
    }

# Helper function
def deck_from_upload(df):
    """returns (enemy_element, deck_feature_row, n_cards, unknown_names) or raises ValueError"""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()
    if "card_name" not in df.columns:
        raise ValueError("File needs a card_name column.")
    if "count" not in df.columns:
        df["count"] = 1
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(1).astype(int)
    df["card_id"] = to_card_ids(df["card_name"])

    unknown = df.loc[df["card_id"].isna(), "card_name"].astype(str).tolist()
    deck_cards = (df.dropna(subset=["card_id"]).groupby("card_id", as_index=False)["count"].sum()
                  .assign(deck_id="upload"))
    pokemon = deck_cards.merge(CARDS[["card_id", "type", "element"]], on="card_id")
    pokemon = pokemon[pokemon["type"] == "pokemon"]
    if pokemon.empty:
        raise ValueError("No Pokémon in the file matched the card list.")
    enemy_el = pokemon.groupby("element")["count"].sum().idxmax()
    decks = pd.DataFrame({"deck_id": ["upload"], "archetype": ["unknown"]})
    feats = features.deck_features(deck_cards, decks, CARDS).drop(columns=["id", "label"])
    return enemy_el, feats, int(deck_cards["count"].sum()), unknown