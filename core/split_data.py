"""
File Name: split_data.py
Course: CMPSC 445 Fall 2026
Description:
60/20/20 train, val, test split, stratified by label (StratifiedGroupKFold, 5 folds:
test = fold 0, val = fold 1, train = folds 2 to 4)
groups keep related instances together (same Pokemon name never in two sets)
train file gets a fold column for K-fold cross-validation
Files: data/processed/splits/{name}_{train,val,test}.csv
"""

import warnings

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from core import config


def assign_split(dataset, groups=None):
    #returns Series id -> train, val, test
    groups = dataset["id"] if groups is None else groups
    sgkf = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    which = pd.Series("train", index=dataset["id"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                            # tiny classes have fewer members than folds
        for k, (_, idx) in enumerate(sgkf.split(dataset, dataset["label"], groups)):
            if k == 0:
                which.iloc[idx] = "test"
            elif k == 1:
                which.iloc[idx] = "val"
    return which


def split_dataset(dataset, name, groups=None, assignment=None):
    #assignment: reuse a split (deck_matrix reuses the deck_features split)
    if assignment is None:
        assignment = assign_split(dataset, groups)
    config.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for which in ["train", "val", "test"]:
        part = dataset[dataset["id"].map(assignment) == which].reset_index(drop=True)
        if which == "train":
            part.insert(1, "fold", -1)
            sgkf = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g = part["id"] if groups is None else part["id"].map(dict(zip(dataset["id"], groups)))
                for k, (_, idx) in enumerate(sgkf.split(part, part["label"], g)):
                    part.loc[idx, "fold"] = k
        part.to_csv(config.SPLIT_DIR / f"{name}_{which}.csv", index=False)
        print(f"split_dataset: {name}_{which} {len(part)} instances")
    return assignment
