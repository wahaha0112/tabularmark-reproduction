#!/usr/bin/env python3
"""Reproduce the main quantitative TabularMark experiments.

The upstream artifact is a collection of scripts and notebooks with inconsistent
paths and a few state-management bugs. This runner preserves the published
algorithm and experimental parameters while making every stage deterministic.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, auc, f1_score, mean_squared_error, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets"
OUTPUT = ROOT / "outputs"
SEED = 10000


@dataclass(frozen=True)
class WatermarkKey:
    row: int
    seed: int


def forest_columns() -> list[str]:
    columns = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
    columns.extend(f"Wilderness_Area_{i}" for i in range(1, 5))
    columns.extend(f"Soil_Type_{i}" for i in range(1, 41))
    columns.append("Cover_Type")
    return columns


def load_datasets() -> dict[str, pd.DataFrame]:
    forest_path = DATA / "covertype" / "cover_type_with_columns.csv"
    if forest_path.exists():
        forest = pd.read_csv(forest_path)
    else:
        forest = pd.read_csv(
            DATA / "covertype" / "covtype.data",
            header=None,
            names=forest_columns(),
        )
        forest.to_csv(forest_path, index=False)

    return {
        "synthetic": pd.read_csv(DATA / "synthetic_dataset" / "synthetic_data.csv"),
        "forest": forest,
        "hog": pd.read_csv(DATA / "HOG" / "digits_HOG.csv"),
        "housing": pd.read_csv(DATA / "boston_housing_prices" / "HousingData.csv"),
    }


def make_keys(length: int, count: int, seed: int) -> list[WatermarkKey]:
    rng = np.random.RandomState(seed)
    divide_seeds = rng.randint(0, 2**32 - 1, size=count, dtype=np.uint32)
    rows = rng.choice(length, size=count, replace=False)
    return [WatermarkKey(int(row), int(divide_seed)) for row, divide_seed in zip(rows, divide_seeds)]


def green_category_set(categories: np.ndarray, seed: int, gamma: float = 0.5) -> np.ndarray:
    shuffled = np.array(categories, copy=True)
    rng = np.random.RandomState(seed)
    rng.shuffle(shuffled)
    green_count = max(1, min(len(shuffled) - 1, int(len(shuffled) * gamma)))
    return shuffled[:green_count]


def embed_categorical(
    original: pd.DataFrame,
    column: str,
    count: int,
    seed: int,
    gamma: float = 0.5,
) -> tuple[pd.DataFrame, list[WatermarkKey]]:
    marked = original.copy()
    categories = np.sort(original[column].unique())
    keys = make_keys(len(original), count, seed)
    for key in keys:
        rng = np.random.RandomState(key.seed)
        shuffled = np.array(categories, copy=True)
        rng.shuffle(shuffled)
        green_count = max(1, min(len(shuffled) - 1, int(len(shuffled) * gamma)))
        green = shuffled[:green_count]
        marked.loc[key.row, column] = rng.choice(green)
    return marked, keys


def green_bin_mask(seed: int, bins: int, gamma: float) -> np.ndarray:
    rng = np.random.RandomState(seed)
    order = np.arange(bins)
    rng.shuffle(order)
    green_count = max(1, min(bins - 1, int(bins * gamma)))
    mask = np.zeros(bins, dtype=bool)
    mask[order[:green_count]] = True
    return mask


def continuous_perturbation(seed: int, bound: float, bins: int, gamma: float) -> float:
    rng = np.random.RandomState(seed)
    edges = np.linspace(-bound, bound, bins + 1)
    order = np.arange(bins)
    rng.shuffle(order)
    green_count = max(1, min(bins - 1, int(bins * gamma)))
    green_bins = order[:green_count]
    values = [rng.uniform(edges[i], np.nextafter(edges[i + 1], edges[i])) for i in green_bins]
    return float(rng.choice(values))


def embed_continuous(
    original: pd.DataFrame,
    column: str,
    count: int,
    bound: float,
    bins: int,
    seed: int,
    gamma: float = 0.5,
) -> tuple[pd.DataFrame, list[WatermarkKey]]:
    marked = original.copy()
    keys = make_keys(len(original), count, seed)
    for key in keys:
        marked.loc[key.row, column] += continuous_perturbation(key.seed, bound, bins, gamma)
    return marked, keys


def z_score(green_count: int, total: int, expected_green: float = 0.5) -> float:
    if total == 0:
        return float("nan")
    denominator = math.sqrt(total * expected_green * (1.0 - expected_green))
    return (green_count - expected_green * total) / denominator


def detect_categorical(
    suspicious: pd.DataFrame,
    column: str,
    keys: list[WatermarkKey],
    categories: np.ndarray,
    gamma: float = 0.5,
) -> float:
    green_count = 0
    for key in keys:
        green = green_category_set(categories, key.seed, gamma)
        green_count += bool(suspicious.loc[key.row, column] in green)
    return z_score(green_count, len(keys), 0.5)


def continuous_detector(
    original: pd.DataFrame,
    column: str,
    keys: list[WatermarkKey],
    bound: float,
    bins: int,
    gamma: float = 0.5,
):
    masks = [green_bin_mask(key.seed, bins, gamma) for key in keys]
    original_values = original.loc[[key.row for key in keys], column].to_numpy(float)
    scale = bins / (2.0 * bound)

    def detect(suspicious: pd.DataFrame) -> float:
        values = suspicious.loc[[key.row for key in keys], column].to_numpy(float)
        differences = values - original_values
        positions = np.floor((differences + bound) * scale).astype(int)
        green_count = 0
        for position, mask in zip(positions, masks):
            if 0 <= position < bins and mask[position]:
                green_count += 1
        return z_score(green_count, len(keys), gamma)

    return detect


def randomize_categorical(
    data: pd.DataFrame, column: str, categories: np.ndarray, proportion: float, seed: int
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    attacked = data.copy()
    count = int(round(proportion * len(attacked)))
    rows = rng.choice(len(attacked), size=count, replace=False)
    attacked.loc[rows, column] = rng.choice(categories, size=count)
    return attacked


def randomize_continuous(
    data: pd.DataFrame, column: str, bound: float, proportion: float, seed: int
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    attacked = data.copy()
    count = int(round(proportion * len(attacked)))
    rows = rng.choice(len(attacked), size=count, replace=False)
    attacked.loc[rows, column] += rng.uniform(-bound, bound, size=count)
    return attacked


def table2_detectability(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, float | str]] = []
    artifacts: dict[str, object] = {}

    continuous_specs = [
        ("Synthetic", data["synthetic"], "dimension_0", 300, 40.0, 500),
        ("Boston Housing", data["housing"], "MEDV", 50, 25.0, 500),
    ]
    for name, original, column, count, bound, bins in continuous_specs:
        marked, keys = embed_continuous(original, column, count, bound, bins, SEED)
        detect = continuous_detector(original, column, keys, bound, bins)
        perturbed = randomize_continuous(original, column, bound, 1.0, SEED + 913)
        rows.append(
            {
                "Dataset": name,
                "Do": detect(original),
                "Dw": detect(marked),
                "Dp": detect(perturbed),
            }
        )
        artifacts[name] = {"marked": marked, "keys": keys, "detector": detect}

    categorical_specs = [
        ("Forest", data["forest"], "Cover_Type", 400),
        ("HOG", data["hog"], "target", 150),
    ]
    for name, original, column, count in categorical_specs:
        categories = np.sort(original[column].unique())
        marked, keys = embed_categorical(original, column, count, SEED)
        perturbed = randomize_categorical(original, column, categories, 1.0, SEED + 913)
        rows.append(
            {
                "Dataset": name,
                "Do": detect_categorical(original, column, keys, categories),
                "Dw": detect_categorical(marked, column, keys, categories),
                "Dp": detect_categorical(perturbed, column, keys, categories),
            }
        )
        artifacts[name] = {"marked": marked, "keys": keys, "categories": categories}

    return pd.DataFrame(rows), artifacts


def forest_f1(
    original: pd.DataFrame,
    training_data: pd.DataFrame,
    xgb_device: str,
    gpu_id: int = 0,
) -> np.ndarray:
    labels = LabelEncoder().fit(original["Cover_Type"])
    indices = np.arange(len(original))
    train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=42)
    model_parameters: dict[str, object] = {
        "n_estimators": 30,
        "max_depth": 10,
        "n_jobs": 8,
        "random_state": 42,
        "verbosity": 0,
    }
    if xgb_device == "gpu":
        model_parameters.update(
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            gpu_id=gpu_id,
        )
    else:
        model_parameters["tree_method"] = "hist"
    model = XGBClassifier(
        **model_parameters,
    )
    model.fit(
        training_data.drop(columns=["Cover_Type"]).iloc[train_indices],
        labels.transform(training_data["Cover_Type"].iloc[train_indices]),
    )
    predictions = model.predict(original.drop(columns=["Cover_Type"]).iloc[test_indices])
    truth = labels.transform(original["Cover_Type"].iloc[test_indices])
    return f1_score(truth, predictions, average=None)


def non_intrusiveness(
    data: dict[str, pd.DataFrame],
    detectability_artifacts: dict[str, object],
    xgb_device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forest = data["forest"]
    forest_marked = detectability_artifacts["Forest"]["marked"]
    if xgb_device == "gpu":
        with ThreadPoolExecutor(max_workers=2) as executor:
            original_future = executor.submit(forest_f1, forest, forest, xgb_device, 0)
            marked_future = executor.submit(forest_f1, forest, forest_marked, xgb_device, 1)
            f1_original = original_future.result()
            f1_marked = marked_future.result()
    else:
        f1_original = forest_f1(forest, forest, xgb_device)
        f1_marked = forest_f1(forest, forest_marked, xgb_device)
    table3 = pd.DataFrame(
        [
            {"Dataset": "Do", "Category 2": f1_original[1], "Category 4": f1_original[3], "Category 6": f1_original[5]},
            {"Dataset": "Dw", "Category 2": f1_marked[1], "Category 4": f1_marked[3], "Category 6": f1_marked[5]},
        ]
    )

    hog = data["hog"]
    hog_marked = detectability_artifacts["HOG"]["marked"]
    hog_train, hog_test = train_test_split(np.arange(len(hog)), test_size=0.3, random_state=42)
    hog_results = []
    for label, training_data in [("Do", hog), ("Dw", hog_marked)]:
        model = RandomForestClassifier(n_estimators=100, n_jobs=16, random_state=42)
        model.fit(training_data.drop(columns=["target"]).iloc[hog_train], training_data["target"].iloc[hog_train])
        prediction = model.predict(hog.drop(columns=["target"]).iloc[hog_test])
        hog_results.append((label, accuracy_score(hog["target"].iloc[hog_test], prediction)))

    housing = data["housing"]
    housing_marked = detectability_artifacts["Boston Housing"]["marked"]
    housing_train, housing_test = train_test_split(np.arange(len(housing)), test_size=0.3, random_state=42)
    housing_results = []
    test_mask = ~housing.drop(columns=["MEDV"]).iloc[housing_test].isna().any(axis=1)
    clean_test = housing.iloc[housing_test].loc[test_mask]
    for label, training_data in [("Do", housing), ("Dw", housing_marked)]:
        train_frame = training_data.iloc[housing_train]
        train_frame = train_frame.loc[~train_frame.drop(columns=["MEDV"]).isna().any(axis=1)]
        model = LinearRegression()
        model.fit(train_frame.drop(columns=["MEDV"]), train_frame["MEDV"])
        prediction = model.predict(clean_test.drop(columns=["MEDV"]))
        housing_results.append((label, mean_squared_error(clean_test["MEDV"], prediction)))

    table4 = pd.DataFrame(
        [
            {
                "Dataset": label,
                "HOG Accuracy": dict(hog_results)[label],
                "Boston Housing MSE": dict(housing_results)[label],
            }
            for label in ("Do", "Dw")
        ]
    )
    return table3, table4


def matched_categorical_z(
    original: pd.DataFrame,
    suspicious: pd.DataFrame,
    column: str,
    keys: list[WatermarkKey],
    key_columns: list[str],
    categories: np.ndarray,
) -> tuple[float, int]:
    lookup: dict[tuple[object, ...], object] = {}
    for values in suspicious[key_columns + [column]].itertuples(index=False, name=None):
        lookup.setdefault(tuple(values[:-1]), values[-1])

    green_count = 0
    matched = 0
    for key in keys:
        signature = tuple(original.loc[key.row, key_columns])
        if signature not in lookup:
            continue
        matched += 1
        green = green_category_set(categories, key.seed)
        green_count += bool(lookup[signature] in green)
    return z_score(green_count, matched, 0.5), matched


def robustness(data: dict[str, pd.DataFrame], xgb_device: str) -> dict[str, pd.DataFrame]:
    forest = data["forest"]
    categories = np.sort(forest["Cover_Type"].unique())
    marked, keys = embed_categorical(forest, "Cover_Type", 300, SEED)
    proportions = [0.2, 0.4, 0.6, 0.8, 1.0]

    alteration_z = []
    attacked_datasets = []
    for position, proportion in enumerate(proportions):
        attacked = randomize_categorical(
            marked, "Cover_Type", categories, proportion, seed=12138 + position
        )
        alteration_z.append(
            {
                "Alteration (%)": int(proportion * 100),
                "Z-score": detect_categorical(attacked, "Cover_Type", keys, categories),
            }
        )
        attacked_datasets.append(attacked)

    if xgb_device == "gpu":
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(forest_f1, forest, attacked, xgb_device, index % 2)
                for index, attacked in enumerate(attacked_datasets)
            ]
            f1_results = [future.result() for future in futures]
    else:
        f1_results = [forest_f1(forest, attacked, xgb_device) for attacked in attacked_datasets]

    alteration_f1 = []
    for proportion, f1 in zip(proportions, f1_results):
        alteration_f1.append(
            {
                "Alteration (%)": int(proportion * 100),
                "Category 2": f1[1],
                "Category 4": f1[3],
                "Category 6": f1[5],
            }
        )

    insertion_rows = []
    deletion_rows = []
    rng = np.random.RandomState(SEED)
    for proportion in proportions:
        insert_count = int(proportion * len(marked))
        inserted = pd.concat(
            [marked, marked.iloc[rng.choice(len(marked), size=insert_count, replace=True)]],
            ignore_index=True,
        ).sample(frac=1.0, random_state=SEED + int(proportion * 100)).reset_index(drop=True)

        delete_count = int(proportion * len(marked))
        keep = np.ones(len(marked), dtype=bool)
        keep[rng.choice(len(marked), size=delete_count, replace=False)] = False
        deleted = marked.loc[keep].sample(frac=1.0, random_state=SEED + int(proportion * 100)).reset_index(drop=True)

        insertion_record: dict[str, float | int] = {"Insertion (%)": int(proportion * 100)}
        deletion_record: dict[str, float | int] = {"Deletion (%)": int(proportion * 100)}
        for key_count, key_columns in [
            (2, ["Elevation", "Aspect"]),
            (3, ["Elevation", "Aspect", "Slope"]),
        ]:
            insertion_z, insertion_matches = matched_categorical_z(
                forest, inserted, "Cover_Type", keys, key_columns, categories
            )
            deletion_z, deletion_matches = matched_categorical_z(
                forest, deleted, "Cover_Type", keys, key_columns, categories
            )
            insertion_record[f"Z-score ({key_count} attrs)"] = insertion_z
            insertion_record[f"Matched ({key_count} attrs)"] = insertion_matches
            deletion_record[f"Z-score ({key_count} attrs)"] = deletion_z
            deletion_record[f"Matched ({key_count} attrs)"] = deletion_matches
        insertion_rows.append(insertion_record)
        deletion_rows.append(deletion_record)

    return {
        "table6_alteration_z": pd.DataFrame(alteration_z),
        "table7_alteration_f1": pd.DataFrame(alteration_f1),
        "tables8_9_insertion": pd.DataFrame(insertion_rows),
        "tables10_11_deletion": pd.DataFrame(deletion_rows),
    }


def roc_experiment(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    specifications = [
        ("Synthetic", data["synthetic"], "dimension_0", 300, 40.0, 500),
        ("Boston Housing", data["housing"], "MEDV", 50, 25.0, 500),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    result_rows = []
    proportions = [0.2, 0.4, 0.6, 0.8, 1.0]
    for axis, (name, original, column, count, bound, bins) in zip(axes, specifications):
        marked, keys = embed_continuous(original, column, count, bound, bins, SEED)
        detect = continuous_detector(original, column, keys, bound, bins)
        scores = []
        labels = []
        for repetition in range(100):
            proportion = proportions[repetition % len(proportions)]
            scores.append(detect(randomize_continuous(marked, column, bound, proportion, SEED + repetition)))
            labels.append(1)
            scores.append(detect(randomize_continuous(original, column, bound, proportion, SEED + 1000 + repetition)))
            labels.append(0)
        fpr, tpr, _ = roc_curve(labels, scores)
        area = auc(fpr, tpr)
        result_rows.append({"Dataset": name, "AUC": area})
        axis.plot(fpr, tpr, linewidth=2.2, label=f"AUC = {area:.3f}")
        axis.plot([0, 1], [0, 1], linestyle="--", color="0.5")
        axis.set_title(name)
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.legend(loc="lower right")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT / "figure6_roc.png", dpi=220)
    plt.close(figure)
    return pd.DataFrame(result_rows)


def logistic_accuracy(original: pd.DataFrame, training_data: pd.DataFrame) -> float:
    indices = np.arange(len(original))
    train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(training_data[["dimension_0", "dimension_1"]].iloc[train_indices], training_data["target"].iloc[train_indices])
    predictions = model.predict(original[["dimension_0", "dimension_1"]].iloc[test_indices])
    return accuracy_score(original["target"].iloc[test_indices], predictions)


def tradeoff_grid(
    original: pd.DataFrame,
    values: list[float | int],
    mode: str,
) -> pd.DataFrame:
    proportions = [0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for value_index, value in enumerate(values):
        count = int(value) if mode == "n" else 300
        bound = float(value) if mode == "p" else 40.0
        gamma = float(value) if mode == "gamma" else 0.5
        if mode == "gamma":
            bound = 30.0
        marked, keys = embed_continuous(
            original, "dimension_0", count, bound, 500, SEED + value_index, gamma
        )
        detect = continuous_detector(original, "dimension_0", keys, bound, 500, gamma)
        attack_bound = bound if mode == "gamma" else 40.0
        for proportion_index, proportion in enumerate(proportions):
            attacked = randomize_continuous(
                marked,
                "dimension_0",
                attack_bound,
                proportion,
                SEED + 100 * value_index + proportion_index,
            )
            rows.append(
                {
                    "mode": mode,
                    "value": value,
                    "attack_percent": int(proportion * 100),
                    "z_score": detect(attacked),
                    "accuracy": logistic_accuracy(original, attacked),
                }
            )
    return pd.DataFrame(rows)


def plot_tradeoff(results: pd.DataFrame, filename: str, label: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for value, group in results.groupby("value", sort=False):
        axes[0].plot(group["attack_percent"], group["accuracy"], marker="o", label=f"{label}={value:g}")
        axes[1].plot(group["attack_percent"], group["z_score"], marker="o", label=f"{label}={value:g}")
    axes[0].set_ylabel("Classification Accuracy")
    axes[1].set_ylabel("Z-score")
    for axis in axes:
        axis.set_xlabel("Alteration Proportion (%)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT / filename, dpi=220)
    plt.close(figure)


def tradeoffs(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    synthetic = data["synthetic"].copy()
    if "target" not in synthetic:
        rng = np.random.RandomState(42)
        weights = rng.uniform(-1, 1, 2)
        probabilities = expit(synthetic[["dimension_0", "dimension_1"]].to_numpy() @ weights)
        synthetic["target"] = np.random.RandomState(42).binomial(1, probabilities)

    p_results = tradeoff_grid(synthetic, [10.0, 20.0, 30.0, 40.0, 50.0], "p")
    n_results = tradeoff_grid(synthetic, [100, 200, 300, 400, 500], "n")
    gamma_results = tradeoff_grid(synthetic, [0.25, 0.33, 0.5, 0.67, 0.75], "gamma")
    plot_tradeoff(p_results, "figure8_p_tradeoff.png", "p")
    plot_tradeoff(n_results, "figure9_n_tradeoff.png", "n")
    plot_tradeoff(gamma_results, "figure10_gamma_tradeoff.png", "gamma")
    return {
        "table20_p_tradeoff": p_results,
        "table22_n_tradeoff": n_results,
        "figure10_gamma_data": gamma_results,
    }


def save_table(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(OUTPUT / f"{name}.csv", index=False, float_format="%.6f")
    print(f"\n{name}\n{frame.to_string(index=False)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["core", "robustness", "tradeoffs", "all"],
        default="all",
    )
    parser.add_argument(
        "--xgb-device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="XGBoost execution device; auto uses CUDA when /dev/nvidia0 is present.",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    data = load_datasets()

    xgb_device = args.xgb_device
    if xgb_device == "auto":
        xgb_device = "gpu" if Path("/dev/nvidia0").exists() else "cpu"

    metadata = {
        "stage": args.stage,
        "seed": SEED,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "xgb_device": xgb_device,
    }

    if args.stage in {"core", "all"}:
        table2, artifacts = table2_detectability(data)
        save_table("table2_detectability", table2)
        table3, table4 = non_intrusiveness(data, artifacts, xgb_device)
        save_table("table3_forest_f1", table3)
        save_table("table4_more_datasets", table4)
        save_table("figure6_roc_data", roc_experiment(data))

    if args.stage in {"robustness", "all"}:
        for name, frame in robustness(data, xgb_device).items():
            save_table(name, frame)

    if args.stage in {"tradeoffs", "all"}:
        for name, frame in tradeoffs(data).items():
            save_table(name, frame)

    metadata["elapsed_seconds"] = time.time() - started
    metadata_path = OUTPUT / f"run_metadata_{args.stage}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nCompleted in {metadata['elapsed_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
