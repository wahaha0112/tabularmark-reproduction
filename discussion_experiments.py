"""Reconstructed discussion experiments for Tables 25-27 and Figure 11."""

from __future__ import annotations

import bisect
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import reproduce as core
from extended_experiments import housing_mse


def normal_green_perturbation(
    seed: int, bound: float, bins: int, sigma: float = 1.0
) -> float:
    rng = np.random.RandomState(seed)
    mask = core.green_bin_mask(seed, bins, 0.5)
    scale = bins / (2.0 * bound)
    for _ in range(100000):
        value = float(rng.normal(0.0, sigma))
        if not -bound <= value < bound:
            continue
        position = int(math.floor((value + bound) * scale))
        if 0 <= position < bins and mask[position]:
            return value
    raise RuntimeError("Failed to sample normal perturbation in a green domain")


def embed_continuous_normal(
    original: pd.DataFrame,
    column: str,
    count: int,
    bound: float,
    bins: int,
    seed: int,
) -> tuple[pd.DataFrame, list[core.WatermarkKey]]:
    marked = original.copy()
    keys = core.make_keys(len(original), count, seed)
    for key in keys:
        marked.loc[key.row, column] += normal_green_perturbation(key.seed, bound, bins)
    return marked, keys


def tables25_26(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    housing = data["housing"]
    uniform, _ = core.embed_continuous(housing, "MEDV", 50, 25.0, 500, core.SEED)
    normal, normal_keys = embed_continuous_normal(
        housing, "MEDV", 50, 25.0, 500, core.SEED
    )
    table25 = pd.DataFrame(
        [
            {
                "Do": housing_mse(housing, housing),
                "Completely Random": housing_mse(housing, uniform),
                "Normal Distribution": housing_mse(housing, normal),
            }
        ]
    )

    detect = core.continuous_detector(housing, "MEDV", normal_keys, 25.0, 500)
    rows = []
    for index, proportion in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
        attacked = core.randomize_continuous(
            normal, "MEDV", 25.0, proportion, core.SEED + 2600 + index
        )
        rows.append(
            {
                "Alteration (%)": int(proportion * 100),
                "Z-score": detect(attacked),
                "MSE": housing_mse(housing, attacked),
            }
        )
    return table25, pd.DataFrame(rows)


def tmc_shapley_values(
    features: np.ndarray,
    target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    permutations: int = 16,
) -> np.ndarray:
    """Small deterministic TMC-Shapley estimate for data-point contributions."""

    rng = np.random.RandomState(core.SEED + 2700)
    values = np.zeros(len(features), dtype=float)
    baseline_prediction = np.full(len(validation_target), target.mean())
    baseline_utility = -mean_squared_error(validation_target, baseline_prediction)
    for _ in range(permutations):
        order = rng.permutation(len(features))
        previous = baseline_utility
        selected: list[int] = []
        for row in order:
            selected.append(int(row))
            model = LinearRegression()
            model.fit(features[selected], target[selected])
            utility = -mean_squared_error(
                validation_target, model.predict(validation_features)
            )
            values[row] += utility - previous
            previous = utility
    return values / permutations


def embed_selected_rows(
    original: pd.DataFrame,
    rows: np.ndarray,
    column: str,
    bound: float,
    bins: int,
) -> pd.DataFrame:
    marked = original.copy()
    rng = np.random.RandomState(core.SEED)
    seeds = rng.randint(0, 2**32 - 1, size=len(rows), dtype=np.uint32)
    for row, seed in zip(rows, seeds):
        marked.loc[int(row), column] += core.continuous_perturbation(
            int(seed), bound, bins, 0.5
        )
    return marked


def table27_shapley(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    housing = data["housing"].dropna().reset_index(drop=True)
    feature_columns = [column for column in housing.columns if column != "MEDV"]
    train_rows, validation_rows = train_test_split(
        np.arange(len(housing)), test_size=0.3, random_state=42
    )
    values = tmc_shapley_values(
        housing.loc[train_rows, feature_columns].to_numpy(float),
        housing.loc[train_rows, "MEDV"].to_numpy(float),
        housing.loc[validation_rows, feature_columns].to_numpy(float),
        housing.loc[validation_rows, "MEDV"].to_numpy(float),
    )
    # The paper calls these the "smallest" values.  We use the smallest
    # magnitudes: points whose estimated marginal contribution is closest to
    # zero are the least consequential to perturb.  This also avoids treating a
    # noisy negative estimate as a highly disposable sample.
    ranked_training_positions = np.argsort(np.abs(values))
    selected_rows = train_rows[ranked_training_positions[:50]]
    random_rows = np.random.RandomState(core.SEED + 2701).choice(
        train_rows, size=50, replace=False
    )
    random_marked = embed_selected_rows(
        housing, random_rows, "MEDV", 25.0, 500
    )
    shapley_marked = embed_selected_rows(
        housing, selected_rows, "MEDV", 25.0, 500
    )
    return pd.DataFrame(
        [
            {
                "Completely Random": housing_mse(housing, random_marked),
                "Shapley Strategy": housing_mse(housing, shapley_marked),
                "Estimator": "16-permutation TMC-Shapley",
            }
        ]
    )


def signature(row: pd.Series, columns: list[str]) -> tuple[float, ...]:
    return tuple(float(row[column]) for column in columns)


def naive_match_all(
    original: pd.DataFrame,
    suspicious: pd.DataFrame,
    key_columns: list[str],
) -> dict[int, float]:
    """Paper Algorithm 3: match every suspicious tuple row by row."""

    original_signatures = [
        signature(row, key_columns) for _, row in original.iterrows()
    ]
    found: dict[int, float] = {}
    for _, row in suspicious.iterrows():
        candidate = signature(row, key_columns)
        for original_index, expected in enumerate(original_signatures):
            if candidate == expected:
                found.setdefault(original_index, float(row["MEDV"]))
                break
    return found


def binary_match_all(
    original: pd.DataFrame,
    suspicious: pd.DataFrame,
    key_columns: list[str],
) -> dict[int, float]:
    pairs = sorted(
        (signature(row, key_columns), int(index))
        for index, row in original.iterrows()
    )
    signatures = [pair[0] for pair in pairs]
    found: dict[int, float] = {}
    for _, row in suspicious.iterrows():
        candidate = signature(row, key_columns)
        position = bisect.bisect_left(signatures, candidate)
        if position < len(pairs) and signatures[position] == candidate:
            found.setdefault(pairs[position][1], float(row["MEDV"]))
    return found


def matched_continuous_z(
    original: pd.DataFrame,
    keys: list[core.WatermarkKey],
    values: list[float | None],
    bound: float,
    bins: int,
) -> float:
    green = 0
    matched = 0
    scale = bins / (2.0 * bound)
    for key, value in zip(keys, values):
        if value is None:
            continue
        difference = value - float(original.loc[key.row, "MEDV"])
        position = int(math.floor((difference + bound) * scale))
        if 0 <= position < bins:
            matched += 1
            green += bool(core.green_bin_mask(key.seed, bins, 0.5)[position])
    return core.z_score(green, matched, 0.5)


def figure11_matching(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    housing = data["housing"].dropna().reset_index(drop=True)
    marked, keys = core.embed_continuous(
        housing, "MEDV", 50, 25.0, 500, core.SEED
    )
    key_columns = [column for column in housing.columns if column != "MEDV"][:3]
    rng = np.random.RandomState(core.SEED + 1100)
    rows = []
    for proportion in [0.2, 0.4, 0.6, 0.8, 1.0]:
        insert_count = int(round(len(marked) * proportion))
        inserted = marked.iloc[rng.choice(len(marked), size=insert_count, replace=True)].copy()
        for column in key_columns:
            inserted[column] += rng.normal(0, 1e-4, size=insert_count)
        suspicious = pd.concat([marked, inserted], ignore_index=True).sample(
            frac=1.0, random_state=core.SEED + int(proportion * 100)
        ).reset_index(drop=True)

        started = time.perf_counter()
        naive_matches = naive_match_all(housing, suspicious, key_columns)
        naive_time = time.perf_counter() - started
        started = time.perf_counter()
        binary_matches = binary_match_all(housing, suspicious, key_columns)
        binary_time = time.perf_counter() - started
        naive_values = [naive_matches.get(key.row) for key in keys]
        binary_values = [binary_matches.get(key.row) for key in keys]
        rows.append(
            {
                "Insertion (%)": int(proportion * 100),
                "Tuple-by-tuple Z-score": matched_continuous_z(
                    housing, keys, naive_values, 25.0, 500
                ),
                "Binary-search Z-score": matched_continuous_z(
                    housing, keys, binary_values, 25.0, 500
                ),
                "Tuple-by-tuple time (s)": naive_time,
                "Binary-search time (s)": binary_time,
            }
        )
    frame = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(frame["Insertion (%)"], frame["Tuple-by-tuple Z-score"], marker="o", label="tuple-by-tuple")
    axes[0].plot(frame["Insertion (%)"], frame["Binary-search Z-score"], marker="s", label="binary search")
    axes[0].set_ylabel("Z-score")
    axes[1].plot(frame["Insertion (%)"], frame["Tuple-by-tuple time (s)"], marker="o", label="tuple-by-tuple")
    axes[1].plot(frame["Insertion (%)"], frame["Binary-search time (s)"], marker="s", label="binary search")
    axes[1].set_ylabel("Time cost (s)")
    for axis in axes:
        axis.set_xlabel("Insertion Proportion (%)")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(core.OUTPUT / "figure11_matching.png", dpi=220)
    plt.close(figure)
    return frame


def run_discussion(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    table25, table26 = tables25_26(data)
    return {
        "table25_noise_selection": table25,
        "table26_normal_alteration": table26,
        "table27_shapley": table27_shapley(data),
        "figure11_matching_data": figure11_matching(data),
    }
