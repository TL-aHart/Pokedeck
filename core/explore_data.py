"""
File Name: explore_data.py
Course: CMPSC 445 Fall 2026
Description:
explore_cards: out/summary.txt (missing values, statistics, outliers) and card figures
explore_decks: deck figures, only when real decks exist
Figures save to out/figures
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from core import config

SURFACE, INK, MUTED, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#2a78d6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=150)
    plt.close(fig)
    print(f"explore: saved out/figures/{name}")


def _barh(labels, values, title, xlabel, name):
    fig, ax = plt.subplots(figsize=(8, max(3, 0.38 * len(labels) + 1.2)))
    ax.barh(labels, values, color=BLUE, height=0.65)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:,.0f}", va="center", fontsize=10)
    ax.set(title=title, xlabel=xlabel, xlim=(0, max(values) * 1.12))
    _save(fig, name)


def explore_cards(dataset, cards, features, chart):
    #dataset: raw printed cards, cards: cleaned. features: card feature vectors
    pokemon = cards[cards["type"] == "pokemon"]
    mean, std = pokemon["health"].mean(), pokemon["health"].std()
    outliers = pokemon[((pokemon["health"] - mean) / std).abs() > 3]
    missing_values = dataset.isna().sum()
    lines = [f"printed cards: {len(dataset)}", f"unique playable cards: {len(cards)}", "",
             "missing values in raw dataset:", missing_values[missing_values > 0].to_string(), "",
             "unique cards by type:", cards["type"].value_counts().to_string(), "",
             "pokemon by element (label classes):", pokemon["element"].value_counts().to_string(), "",
             f"pokemon HP: mean {mean:.1f}, std {std:.1f}, outliers (|z| > 3): {len(outliers)}",
             outliers[["name", "health"]].drop_duplicates().to_string(index=False), "",
             "pokemon HP by stage:", pokemon.groupby("stage")["health"].describe().round(1).to_string()]
    config.SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)
    config.SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("explore: saved out/summary.txt")

    c = cards["type"].value_counts()
    _barh(c.index.str.capitalize(), c.values, "Unique playable cards by type", "cards", "fig01_card_types.png")
    e = pokemon["element"].value_counts()
    _barh(e.index.str.capitalize(), e.values, "Unique Pokemon cards by energy type", "cards", "fig02_pokemon_by_element.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    stages = ["basic", "1", "2"]
    bp = ax.boxplot([pokemon.loc[pokemon["stage"] == s, "health"] for s in stages],
                    tick_labels=["Basic", "Stage 1", "Stage 2"], patch_artist=True, widths=0.5,
                    medianprops={"color": INK, "linewidth": 1.5})
    for b in bp["boxes"]:
        b.set(facecolor="#cde2fb", edgecolor=BLUE)
    ax.set(title="Pokemon HP by evolution stage", ylabel="HP")
    ax.grid(axis="x", visible=False)
    _save(fig, "fig03_hp_by_stage.png")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(chart.values, cmap=SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(10), [t.capitalize() for t in chart.columns], rotation=45, ha="right")
    ax.set_yticks(range(10), [t.capitalize() for t in chart.index])
    ax.grid(False)
    for i in range(10):
        for j in range(10):
            if chart.values[i, j] > 0:
                ax.text(j, i, f"{chart.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if chart.values[i, j] > 0.5 else INK)
    ax.set(title="Type chart from card weaknesses", xlabel="defending type", ylabel="attacking type")
    fig.colorbar(im, ax=ax, shrink=0.8, label="share of defenders weak to attacker")
    _save(fig, "fig04_type_chart.png")

    X = features.set_index("card_id")
    X = X[X["is_pokemon"] == 1][["health", "retreat_cost", "is_ex", "evolves_from_other", "stage_basic", "stage_1", "stage_2"]]
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    ax.set_title("Pokemon feature correlation")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    _save(fig, "fig05_feature_correlation.png")


# runs only when there is a deck, add deck figures here
def explore_decks(deck_dataset):
    X = deck_dataset.drop(columns=["id", "label"])
    X = X.loc[:, X.std() > 0]
    counts = deck_dataset["label"].value_counts()
    _barh(counts.index, counts.values, "Decks per archetype", "decks", "fig06_decks_per_archetype.png")

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(9, 7))
    top = list(counts.index[:8])
    rest = ~deck_dataset["label"].isin(top).values
    ax.scatter(components[rest, 0], components[rest, 1], s=28, c="#b9b8b1", edgecolors=SURFACE, label="Other archetypes")
    for archetype, color in zip(top, SERIES):
        m = (deck_dataset["label"] == archetype).values
        ax.scatter(components[m, 0], components[m, 1], s=36, c=color, edgecolors=SURFACE, label=archetype)
    ax.set(title="2 component PCA of deck feature vectors",
           xlabel=f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.0%})",
           ylabel=f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _save(fig, "fig07_pca_decks.png")

    ks = range(1, min(16, len(deck_dataset)))
    inertia = [KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(X_scaled).inertia_ for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(list(ks), inertia, color=BLUE, linewidth=2, marker="o")
    ax.set(title="K-means elbow preview", xlabel="k", ylabel="inertia")
    _save(fig, "fig08_kmeans_elbow.png")
