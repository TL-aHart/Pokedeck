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

# PLACEHOLDER UNTIL THE MODEL IS UP AND RUNNING, GET RID OF TRY CLAUSE LATER
try:
    from core.recommender import recommend_counter
    MODEL_READY = True
except ModuleNotFoundError as e:
    if e.name != "core.recommender":   # Reilly's file exists but THAT FILE is missing a package
        raise
    recommend_counter, MODEL_READY = None, False

CARDS = load_cards()
PRIMARY = CARDS[CARDS["is_primary"]]
NAMES = CARDS.set_index("card_id")["name"]
KINDS = CARDS.set_index("card_id")["type"]
TYPES = [t.capitalize() for t in config.ENERGY_TYPES]   # UI labels
ICON_PATH = {"Fire": "a/ad", "Water": "1/11", "Grass": "2/2e", "Lightning": "0/04", "Psychic": "e/ef",
             "Fighting": "4/48", "Darkness": "a/ab", "Metal": "6/64", "Dragon": "8/8a", "Colorless": "1/1d"}
CHART = pd.read_csv(config.TYPE_CHART_CSV, index_col="attacker")
DECK_TRAINERS = [(2, "Professor's Research"), (2, "Poké Ball"), (2, "Sabrina"), (1, "Giovanni"), (1, "X Speed")]
DECK_SIZE = config.DECK_SIZE
LOOKUP = PRIMARY.set_index("name_key")["card_id"]

# THIS IS FAKE DATA, NEED TO IMPLEMENT DECK SCRAPING
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


def to_elements(ui_type):
    return ui_type.lower()        # UI to data

def to_ui(element):
    return element.capitalize()         # data to UI

def best_counter(defender):
    col = CHART[defender]
    return col.idxmax() if col.max() > 0 else "metal"   # nothing is super effective against dragon, so... metal?

def to_card_ids(names):
    return [LOOKUP.get(normalize_name(n)) for n in names]

def type_icon(t):
    return f"https://archives.bulbagarden.net/media/upload/{ICON_PATH[t]}/{t}-attack.png"

def recommend(enemy_type, card_ids, basis, features=None):
    element = to_elements(enemy_type)
    pick = recommend_counter if MODEL_READY else mock_counter
    return build_response(pick(element, card_ids, features), element, basis)

def mock_counter(enemy_element, enemy_card_ids, enemy_features=None):
    """Stand-in for Reilly's recommend_counter. Same inputs, same output."""
    counter_el = best_counter(enemy_element)
    cards = DECKS[to_ui(counter_el)] + DECK_TRAINERS          # [(count, name), ...]
    ids = to_card_ids([name for _, name in cards])
    return {
        "counter_element": counter_el,
        "archetype": None,
        "deck": [(cid, n) for (n, _), cid in zip(cards, ids)],  # [(card_id, count), ...]
        "win_prob": None,      # STILL TO BE IMPLEMENTED
        "confidence": None,
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
    return enemy_el, feats, int(deck_cards["count"].sum()), unknown, deck_cards["card_id"].tolist()

def check_deck(deck):
    """Trust but verify, DON'T send the browser a deck that breaks the game rules"""
    unknown = [cid for cid, _ in deck if cid not in NAMES.index]
    total = sum(n for _, n in deck)
    if unknown or total != config.DECK_SIZE or any(n > config.MAX_COPIES for _, n in deck):
        raise ValueError(f"Recommender returned an invalid deck: total={total}, unknown={unknown[:3]}")

def build_response(result, enemy_element, basis):
    counter_el = result["counter_element"]
    counter = to_ui(counter_el)
    check_deck(result["deck"])
    to_cards = lambda keep: [{"n": n, "name": NAMES[cid]} for cid, n in result["deck"] if keep(KINDS[cid])]
    share = type_advantage(CHART, counter_el, enemy_element)
    enemy = to_ui(enemy_element)
    reason = result.get("reason") or (
        f"Nothing hits {enemy} for weakness damage, so high-HP {counter} attackers are the steadiest pick."
        if share == 0 else
        f"{share:.0%} of {enemy} Pokémon are weak to {counter}, so these attackers deal bonus damage.")
    win, conf = result.get("win_prob"), result.get("confidence")
    return {
        "title": result.get("archetype") or f"{counter} deck",
        "icon": type_icon(counter),
        "reason": reason,
        "basis": basis,
        "winPct": None if win is None else round(win * 100),
        "confidence": "—" if conf is None else "High" if conf >= 0.75 else "Medium" if conf >= 0.5 else "Low",
        "groups": [{"name": "Pokémon", "cards": to_cards(lambda t: t == "pokemon")},
                   {"name": "Trainers", "cards": to_cards(lambda t: t != "pokemon")}],
    }