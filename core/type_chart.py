"""
File Name: type_chart.py
Course: CMPSC 445 Fall 2026
Description:
type_chart: share of defending type Pokemon weak to each attacking type, from card weaknesses
type_advantage: lookup for the type only baseline (0 to 1)
"""

import pandas as pd

from core import config


def build_type_chart(cards):
    pokemon = cards[(cards["type"] == "pokemon") & cards["is_primary"] & (cards["element"] != "none")]
    chart = pd.DataFrame(0.0, index=config.ENERGY_TYPES, columns=config.ENERGY_TYPES)
    chart.index.name = "attacker"
    for defender, group in pokemon.groupby("element"):
        for attacker, share in group["weakness"].value_counts(normalize=True).items():
            if attacker in chart.index:
                chart.loc[attacker, defender] = share
    return chart


def type_advantage(chart, attacker, defender):
    return float(chart.loc[attacker, defender]) if attacker in chart.index and defender in chart.columns else 0.0
