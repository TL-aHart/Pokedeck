"""
File Name: app.py
Course: CMPSC 445 Fall 2026
Project: PokeDeck, Pokemon TCG Pocket counter deck recommender
Description:
    Flask routes for the PokeDeck counter deck recommender
"""

from flask import Flask, jsonify, render_template, request
import pandas as pd
from web.recommend import TYPES, recommend, type_icon, PRIMARY, CARDS, deck_from_upload, to_ui, to_card_ids

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload cap

CARD_DB = {
    "pokemon": [[r.name, r.element.capitalize()] for r in PRIMARY[PRIMARY.type == "pokemon"].itertuples()],
    "trainer": [[n, "Supporter"] for n in PRIMARY[PRIMARY.type == "supporter"].name],
    "item": [[n, t.capitalize()] for n, t in PRIMARY[PRIMARY.type.isin(["item", "tool"])][["name", "type"]].values],
}

@app.get("/")
def index():
    types = [{"name": t, "icon": type_icon(t)} for t in TYPES]
    return render_template("index.html", types=types, card_db=CARD_DB)


@app.get("/api/cards")
def get_cards():
    return jsonify(CARD_DB)


@app.post("/api/recommend/scout")
def recommend_scout():
    data = request.get_json(silent=True) or {}
    enemy = data.get("type")
    if enemy not in TYPES:
        return jsonify(error="Pick the enemy type to continue."), 400
    known = [c.strip() for c in data.get("pokemon", []) if c and c.strip()][:4]
    known += [c.strip() for c in (data.get("trainer"), data.get("item")) if c and c.strip()]
    ids = to_card_ids(known)
    basis = [f"Enemy: {enemy}"] + [name if cid else f"Unrecognized: {name}"
                                   for name, cid in zip(known, ids)]
    card_ids = [cid for cid in ids if cid]
    return jsonify(recommend(enemy, len(card_ids), basis))


@app.post("/api/recommend/upload")
def recommend_upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Upload a file to continue."), 400
    ext = f.filename.rsplit(".", 1)[-1].lower()
    try:
        if ext == "csv":
            df = pd.read_csv(f)
        elif ext == "xlsx":
            df = pd.read_excel(f)
        else:
            return jsonify(error=f'"{f.filename}" isn\'t a CSV or XLSX file.'), 400
    except Exception:
        return jsonify(error=f'Couldn\'t read "{f.filename}". Check that it has a header row and at least one card.'), 400

    df = df.dropna(how="all")
    if df.empty:
        return jsonify(error=f'"{f.filename}" has no card rows.'), 400

    try:
        enemy_el, feats, n_cards, unknown = deck_from_upload(df)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    enemy = to_ui(enemy_el)
    basis = [f"{n_cards} cards from {f.filename}", f"Detected type: {enemy}"]
    basis += [f"Unrecognized: {n}" for n in unknown[:3]]
    return jsonify(recommend(enemy, n_cards, basis))


if __name__ == "__main__":
    app.run(debug=True)
