"""Deterministic reconstruction of the Figure 7 baseline comparison."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import reproduce as core


ARTIFACT = core.ROOT / "experiments" / "compare" / "classfication_task" / "cover_type"


def artifact_text(name: str) -> str:
    return (ARTIFACT / name).read_text(encoding="utf-8").strip()


def python_green_domain(categories: list[int], seed: int) -> list[int]:
    shuffled = list(categories)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled[: len(shuffled) // 2]


def tabularmark_baseline(original: pd.DataFrame) -> tuple[pd.DataFrame, list[int], list[int]]:
    indices = [int(value) for value in artifact_text("tabularmark_index.txt").splitlines()]
    seeds = [int(value) for value in artifact_text("tabularmark_seed.txt").splitlines()]
    categories = sorted(int(value) for value in original["Cover_Type"].unique())
    marked = original.copy()
    for row, seed in zip(indices, seeds):
        rng = random.Random(seed)
        green = python_green_domain(categories, seed)
        marked.loc[row, "Cover_Type"] = rng.choice(green)
    return marked, indices, seeds


def tabularmark_mismatch(
    attacked: pd.DataFrame, indices: list[int], seeds: list[int]
) -> float:
    categories = sorted(int(value) for value in attacked["Cover_Type"].unique())
    green = sum(
        int(attacked.loc[row, "Cover_Type"]) in python_green_domain(categories, seed)
        for row, seed in zip(indices, seeds)
    )
    return (len(indices) - green) / len(indices)


def histogram_group(key: int, private_key: str, groups: int) -> int:
    digest = hashlib.sha256()
    digest.update(f"{private_key}{key}".encode())
    inner = digest.hexdigest()
    digest.update(f"{private_key}{inner}".encode())
    return int(digest.hexdigest(), 16) % groups


def histogram_baseline(
    original: pd.DataFrame,
) -> tuple[pd.DataFrame, str, dict[int, float], set[int]]:
    private_key = "b80d"
    watermark = artifact_text("histogram_mark.txt")
    marked = original.copy()
    marked["group_number"] = marked["primary_key"].map(
        lambda value: histogram_group(int(value), private_key, len(watermark))
    )
    minimum = marked["Cover_Type"].min()
    maximum = marked["Cover_Type"].max()
    midpoint = (minimum + maximum) / 2
    peak: dict[int, float] = {}
    excluded: set[int] = set()
    for group_number, bit in enumerate(watermark):
        group_mask = marked["group_number"] == group_number
        group = marked.loc[group_mask]
        usable = group.loc[
            (group["Cover_Type"] != minimum) & (group["Cover_Type"] != maximum)
        ]
        counts = Counter(np.abs(usable["Cover_Type"] - midpoint))
        p = float(counts.most_common(1)[0][0]) if counts else 1.0
        peak[group_number] = p
        excluded.update(
            int(value)
            for value in group.loc[
                (group["Cover_Type"] == minimum) | (group["Cover_Type"] == maximum),
                "primary_key",
            ]
        )
        values = marked.loc[group_mask, "Cover_Type"] - midpoint
        changed = np.where(
            (values == p) & (bit == "0"),
            values,
            np.where(
                (values == p) & (bit == "1"),
                values + 1,
                np.where(
                    (values == -p) & (bit == "0"),
                    values,
                    np.where(
                        (values == -p) & (bit == "1"),
                        values - 1,
                        np.where(values >= p + 1, values + 1, np.where(values <= -(p + 1), values - 1, values)),
                    ),
                ),
            ),
        )
        boundary = (marked.loc[group_mask, "Cover_Type"] == minimum) | (
            marked.loc[group_mask, "Cover_Type"] == maximum
        )
        current = marked.loc[group_mask, "Cover_Type"].to_numpy(copy=True)
        current[~boundary.to_numpy()] = changed[~boundary.to_numpy()] + midpoint
        marked.loc[group_mask, "Cover_Type"] = current
    return marked, watermark, peak, excluded


def histogram_mismatch(
    attacked: pd.DataFrame, watermark: str, peak: dict[int, float], excluded: set[int]
) -> float:
    minimum = attacked["Cover_Type"].min()
    maximum = attacked["Cover_Type"].max()
    midpoint = (minimum + maximum) / 2
    detected = []
    for group_number in range(len(watermark)):
        group = attacked.loc[attacked["group_number"] == group_number]
        error = group["Cover_Type"] - midpoint
        usable = ~group["primary_key"].isin(excluded)
        zeros = (usable & (error == peak[group_number])).sum()
        ones = (usable & ((error == peak[group_number] + 1) | (error == peak[group_number] - 1))).sum()
        detected.append("0" if zeros > ones else "1")
    return sum(a != b for a, b in zip(watermark, detected)) / len(watermark)


def semantic_hash(value: str) -> str:
    return hashlib.blake2b(value.encode()).hexdigest()


def semantic_baseline(
    original: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str, str, dict[str, str]]:
    owner_key = artifact_text("semantic_Ks.txt")
    attribute_key = artifact_text("semantic_k.txt")
    watermark = artifact_text("semantic_mark.txt")
    marked = original.copy()
    records: dict[str, str] = {}
    for row_index, row in marked.iterrows():
        key_hash = semantic_hash(f"{row['primary_key']}{attribute_key}")
        owner_hash = semantic_hash(f"{owner_key}{key_hash}")
        if int(owner_hash, 16) % 35 != 0:
            continue
        bit_index = int(semantic_hash(f"{owner_key}{key_hash}1"), 16) % len(watermark)
        old_value = int(row["Cover_Type"])
        if watermark[bit_index] == "1":
            new_value = math.floor((old_value + 5) / 2)
            marked.at[row_index, "Cover_Type"] = new_value
            records[key_hash] = semantic_hash(
                f"{new_value}{attribute_key}{abs(old_value - new_value)}"
            )
        else:
            records[key_hash] = semantic_hash(f"{old_value}{attribute_key}")
    return marked, watermark, owner_key, attribute_key, records


def semantic_mismatch(
    attacked: pd.DataFrame,
    watermark: str,
    owner_key: str,
    attribute_key: str,
    records: dict[str, str],
) -> float:
    counts = np.zeros((len(watermark), 2), dtype=int)
    for _, row in attacked.iterrows():
        key_hash = semantic_hash(f"{row['primary_key']}{attribute_key}")
        if key_hash not in records:
            continue
        owner_hash = semantic_hash(f"{owner_key}{key_hash}")
        if int(owner_hash, 16) % 35 != 0:
            continue
        bit_index = int(semantic_hash(f"{owner_key}{key_hash}1"), 16) % len(watermark)
        value = int(row["Cover_Type"])
        if records[key_hash] == semantic_hash(f"{value}{attribute_key}"):
            counts[bit_index, 0] += 1
        elif any(
            records[key_hash] == semantic_hash(f"{value}{attribute_key}{difference}")
            for difference in (0, 1, 2)
        ):
            counts[bit_index, 1] += 1
    detected = ["0" if zero > one else "1" for zero, one in counts]
    return sum(a != b for a, b in zip(watermark, detected)) / len(watermark)


def attack(frame: pd.DataFrame, proportion: float, seed: int) -> pd.DataFrame:
    attacked = frame.copy()
    rng = np.random.RandomState(seed)
    count = int(len(attacked) * proportion)
    rows = rng.choice(len(attacked), size=count, replace=False)
    attacked.loc[rows, "Cover_Type"] = rng.randint(1, 8, size=count)
    return attacked


def model_f1(original: pd.DataFrame, training: pd.DataFrame, device: str) -> np.ndarray:
    feature_columns = [
        column for column in original.columns if column not in {"primary_key", "Cover_Type"}
    ]
    train_rows, test_rows = train_test_split(
        np.arange(len(original)), test_size=0.3, random_state=42
    )
    parameters: dict[str, object] = {
        "n_estimators": 10,
        "random_state": 42,
        "n_jobs": 8,
        "verbosity": 0,
    }
    if device == "gpu":
        parameters.update(tree_method="gpu_hist", predictor="gpu_predictor", gpu_id=0)
    else:
        parameters["tree_method"] = "hist"
    model = XGBClassifier(**parameters)
    model.fit(
        training.loc[train_rows, feature_columns],
        training.loc[train_rows, "Cover_Type"].astype(int) - 1,
    )
    prediction = model.predict(original.loc[test_rows, feature_columns])
    truth = original.loc[test_rows, "Cover_Type"].astype(int) - 1
    return f1_score(truth, prediction, average=None, labels=np.arange(7), zero_division=0)


def run_comparison(device: str) -> dict[str, pd.DataFrame]:
    original = pd.read_csv(core.DATA / "compare" / "covtype_with_key.subset.data")
    tabular, tabular_indices, tabular_seeds = tabularmark_baseline(original)
    histogram, histogram_watermark, histogram_peak, histogram_excluded = histogram_baseline(original)
    semantic, semantic_watermark, owner_key, attribute_key, semantic_records = semantic_baseline(original)

    f1_rows = []
    for name, frame in [
        ("Original", original),
        ("TabularMark", tabular),
        ("HistMark", histogram),
        ("SemMark", semantic),
    ]:
        scores = model_f1(original, frame, device)
        for category, score in enumerate(scores, start=1):
            f1_rows.append({"Scheme": name, "Category": category, "F1-score": score})
    f1_frame = pd.DataFrame(f1_rows)

    mismatch_rows = []
    for index, proportion in enumerate([0.0, 0.2, 0.4, 0.6]):
        tabular_attacked = attack(tabular, proportion, core.SEED + 7100 + index)
        histogram_attacked = attack(histogram, proportion, core.SEED + 7200 + index)
        semantic_attacked = attack(semantic, proportion, core.SEED + 7300 + index)
        mismatch_rows.extend(
            [
                {
                    "Scheme": "TabularMark",
                    "Alteration (%)": int(proportion * 100),
                    "Mismatch Percentage": tabularmark_mismatch(
                        tabular_attacked, tabular_indices, tabular_seeds
                    ),
                },
                {
                    "Scheme": "HistMark",
                    "Alteration (%)": int(proportion * 100),
                    "Mismatch Percentage": histogram_mismatch(
                        histogram_attacked,
                        histogram_watermark,
                        histogram_peak,
                        histogram_excluded,
                    ),
                },
                {
                    "Scheme": "SemMark",
                    "Alteration (%)": int(proportion * 100),
                    "Mismatch Percentage": semantic_mismatch(
                        semantic_attacked,
                        semantic_watermark,
                        owner_key,
                        attribute_key,
                        semantic_records,
                    ),
                },
            ]
        )
    mismatch_frame = pd.DataFrame(mismatch_rows)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for scheme, group in f1_frame.groupby("Scheme", sort=False):
        axes[0].plot(group["Category"], group["F1-score"], marker="o", label=scheme)
    for scheme, group in mismatch_frame.groupby("Scheme", sort=False):
        axes[1].plot(
            group["Alteration (%)"],
            group["Mismatch Percentage"] * 100,
            marker="o",
            label=scheme,
        )
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("F1-score")
    axes[1].set_xlabel("Alteration Proportion (%)")
    axes[1].set_ylabel("Mismatch Percentage (%)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(core.OUTPUT / "figure7_comparison.png", dpi=220)
    plt.close(figure)
    return {
        "figure7_f1": f1_frame,
        "figure7_mismatch": mismatch_frame,
    }
