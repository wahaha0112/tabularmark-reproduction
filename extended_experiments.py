"""Extended TabularMark experiments omitted from the upstream runnable artifact.

The paper reports these experiments but does not publish executable notebooks for
most of them.  This module reconstructs the stated protocols from the paper.  Two
protocol substitutions are deliberately labelled in the output: a deterministic
constraint-repair proxy for HoloClean (Table 12), and a class-conditional Gaussian
generator for the unavailable language-model generator (Table 15).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

import reproduce as core


ZOO_COLUMNS = [
    "animal_name",
    "hair",
    "feathers",
    "eggs",
    "milk",
    "airborne",
    "aquatic",
    "predator",
    "toothed",
    "backbone",
    "breathes",
    "venomous",
    "fins",
    "legs",
    "tail",
    "domestic",
    "catsize",
    "type",
]


def load_zoo() -> pd.DataFrame:
    return pd.read_csv(core.DATA / "zoo" / "zoo.data", header=None, names=ZOO_COLUMNS)


def load_iris_frame() -> pd.DataFrame:
    bundle = load_iris(as_frame=True)
    frame = bundle.frame.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
            "target": "class",
        }
    )
    frame["class"] = frame["class"].astype(int) + 1
    return frame


def classifier_accuracy(
    original: pd.DataFrame,
    training_data: pd.DataFrame,
    target: str,
    drop: tuple[str, ...] = (),
) -> float:
    features = [column for column in original.columns if column != target and column not in drop]
    indices = np.arange(len(original))
    train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, n_jobs=16, random_state=42)
    model.fit(training_data.loc[train_indices, features], training_data.loc[train_indices, target])
    prediction = model.predict(original.loc[test_indices, features])
    return accuracy_score(original.loc[test_indices, target], prediction)


def housing_mse(original: pd.DataFrame, training_data: pd.DataFrame) -> float:
    train_indices, test_indices = train_test_split(
        np.arange(len(original)), test_size=0.3, random_state=42
    )
    feature_columns = [column for column in original.columns if column != "MEDV"]
    train_frame = training_data.loc[train_indices].dropna(subset=feature_columns)
    test_frame = original.loc[test_indices].dropna(subset=feature_columns)
    model = LinearRegression()
    model.fit(train_frame[feature_columns], train_frame["MEDV"])
    prediction = model.predict(test_frame[feature_columns])
    return mean_squared_error(test_frame["MEDV"], prediction)


def table5_zoo_non_intrusiveness() -> tuple[pd.DataFrame, dict[str, object]]:
    zoo = load_zoo()
    categories = np.sort(zoo["type"].unique())
    marked, keys = core.embed_categorical(zoo, "type", 15, core.SEED)
    table = pd.DataFrame(
        [
            {
                "Dataset": "Do",
                "Z-score": core.detect_categorical(zoo, "type", keys, categories),
                "Accuracy": classifier_accuracy(zoo, zoo, "type", ("animal_name",)),
            },
            {
                "Dataset": "Dw",
                "Z-score": core.detect_categorical(marked, "type", keys, categories),
                "Accuracy": classifier_accuracy(zoo, marked, "type", ("animal_name",)),
            },
        ]
    )
    return table, {"original": zoo, "marked": marked, "keys": keys, "categories": categories}


def constraint_repair_proxy(
    original: pd.DataFrame, marked: pd.DataFrame, target: str, determinant: str
) -> pd.DataFrame:
    """Repair target values that violate a deterministic modal dependency.

    This is intentionally a transparent, deterministic stand-in for the old
    PostgreSQL/TensorFlow HoloClean stack.  The output protocol field prevents it
    from being mistaken for an exact HoloClean rerun.
    """

    repaired = marked.copy()
    mapping = (
        original.groupby(determinant, dropna=False)[target]
        .agg(lambda values: Counter(values).most_common(1)[0][0])
        .to_dict()
    )
    expected = original[determinant].map(mapping)
    mask = expected.notna() & (repaired[target] != expected)
    repaired.loc[mask, target] = expected.loc[mask]
    return repaired


def table12_data_cleaning() -> pd.DataFrame:
    specs = [
        ("Adult", core.DATA / "holoclean" / "Adult1100.csv", "Income", "Occupation"),
        ("Hospital", core.DATA / "holoclean" / "hospital.csv", "Condition", "MeasureCode"),
    ]
    rows = []
    for position, (name, path, target, determinant) in enumerate(specs):
        original = pd.read_csv(path)
        categories = np.sort(original[target].dropna().unique())
        marked, keys = core.embed_categorical(
            original, target, min(300, len(original)), core.SEED + position
        )
        repaired = constraint_repair_proxy(original, marked, target, determinant)
        rows.append(
            {
                "Dataset": name,
                "Dw": core.detect_categorical(marked, target, keys, categories),
                "Da": core.detect_categorical(repaired, target, keys, categories),
                "Protocol": "constraint-repair proxy (not HoloClean inference)",
            }
        )
    return pd.DataFrame(rows)


def anonymize_in_blocks(frame: pd.DataFrame, columns: list[str], k: int = 3) -> pd.DataFrame:
    anonymized = frame.copy()
    order = anonymized.sort_values(columns, kind="mergesort").index.to_numpy()
    for start in range(0, len(order), k):
        group = order[start : start + k]
        if len(group) < k:
            group = order[max(0, len(order) - k) :]
        anonymized.loc[group, columns] = anonymized.loc[group, columns].mean(axis=0).to_numpy()
    return anonymized


def table13_14_iris() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    iris = load_iris_frame()
    categories = np.sort(iris["class"].unique())
    marked, keys = core.embed_categorical(iris, "class", 20, core.SEED)
    table13 = pd.DataFrame(
        [
            {"Dataset": "Do", "Accuracy": classifier_accuracy(iris, iris, "class")},
            {"Dataset": "Dw", "Accuracy": classifier_accuracy(iris, marked, "class")},
        ]
    )

    selected_sets = [
        ["petal_width"],
        ["petal_length", "petal_width"],
        ["sepal_width", "petal_length", "petal_width"],
    ]
    all_features = [column for column in iris.columns if column != "class"]
    table14_rows = []
    for selected in selected_sets:
        anonymized = anonymize_in_blocks(marked, selected, k=3)
        matching_columns = [column for column in all_features if column not in selected]
        detected, matched_count = core.matched_categorical_z(
            iris, anonymized, "class", keys, matching_columns, categories
        )
        table14_rows.append(
            {
                "#columns": len(selected),
                "Z-score": detected,
                "Accuracy": classifier_accuracy(iris, anonymized, "class"),
                "Matched key cells": matched_count,
            }
        )
    return table13, pd.DataFrame(table14_rows), {
        "original": iris,
        "marked": marked,
        "keys": keys,
        "categories": categories,
    }


def gaussian_class_generator(
    training_data: pd.DataFrame, target: str, feature_columns: list[str], seed: int
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    generated_parts = []
    for label, group in training_data.groupby(target):
        matrix = group[feature_columns].to_numpy(float)
        covariance = np.cov(matrix, rowvar=False) + np.eye(len(feature_columns)) * 1e-5
        generated = rng.multivariate_normal(matrix.mean(axis=0), covariance, size=len(group))
        part = pd.DataFrame(generated, columns=feature_columns)
        part[target] = label
        generated_parts.append(part)
    return pd.concat(generated_parts, ignore_index=True).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)


def table15_iris_generation(iris_artifacts: dict[str, object]) -> pd.DataFrame:
    original = iris_artifacts["original"]
    marked = iris_artifacts["marked"]
    features = [column for column in original.columns if column != "class"]
    generated_original = gaussian_class_generator(original, "class", features, core.SEED)
    generated_marked = gaussian_class_generator(marked, "class", features, core.SEED)
    return pd.DataFrame(
        [
            {
                "Protocol": "class-conditional Gaussian proxy (paper LLM generator unavailable)",
                "Do": classifier_accuracy(original, original, "class"),
                "Do_prime": classifier_accuracy(original, generated_original, "class"),
                "Dw": classifier_accuracy(original, marked, "class"),
                "Dw_prime": classifier_accuracy(original, generated_marked, "class"),
            }
        ]
    )


def table16_hog(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    hog = data["hog"]
    categories = np.sort(hog["target"].unique())
    marked, keys = core.embed_categorical(hog, "target", 150, core.SEED)
    rows = []
    for index, proportion in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
        attacked = core.randomize_categorical(
            marked, "target", categories, proportion, core.SEED + 1600 + index
        )
        rows.append(
            {
                "Alteration (%)": int(proportion * 100),
                "Z-score": core.detect_categorical(attacked, "target", keys, categories),
                "Accuracy": classifier_accuracy(hog, attacked, "target"),
            }
        )
    return pd.DataFrame(rows)


def table17_housing(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    housing = data["housing"]
    marked, keys = core.embed_continuous(housing, "MEDV", 50, 25.0, 500, core.SEED)
    detect = core.continuous_detector(housing, "MEDV", keys, 25.0, 500)
    rows = []
    for index, proportion in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
        attacked = core.randomize_continuous(
            marked, "MEDV", 25.0, proportion, core.SEED + 1700 + index
        )
        rows.append(
            {
                "Alteration (%)": int(proportion * 100),
                "Z-score": detect(attacked),
                "MSE": housing_mse(housing, attacked),
            }
        )
    return pd.DataFrame(rows)


def table18_zoo(zoo_artifacts: dict[str, object]) -> pd.DataFrame:
    original = zoo_artifacts["original"]
    marked = zoo_artifacts["marked"]
    keys = zoo_artifacts["keys"]
    categories = zoo_artifacts["categories"]
    rows = []
    for index, proportion in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
        attacked = core.randomize_categorical(
            marked, "type", categories, proportion, core.SEED + 1800 + index
        )
        rows.append(
            {
                "Alteration (%)": int(proportion * 100),
                "Z-score": core.detect_categorical(attacked, "type", keys, categories),
                "Accuracy": classifier_accuracy(
                    original, attacked, "type", ("animal_name",)
                ),
            }
        )
    return pd.DataFrame(rows)


def run_extended(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    table5, zoo_artifacts = table5_zoo_non_intrusiveness()
    table13, table14, iris_artifacts = table13_14_iris()
    return {
        "table5_zoo": table5,
        "table12_data_cleaning": table12_data_cleaning(),
        "table13_iris": table13,
        "table14_k_anonymization": table14,
        "table15_generation_proxy": table15_iris_generation(iris_artifacts),
        "table16_hog_alteration": table16_hog(data),
        "table17_housing_alteration": table17_housing(data),
        "table18_zoo_alteration": table18_zoo(zoo_artifacts),
    }
