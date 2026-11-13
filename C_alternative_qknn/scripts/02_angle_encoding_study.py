
"""
02_angle_encoding_study.py
==========================

Project C — QKNN representation study
Study 02: Angle encoding, feature relevance, and scaling sensitivity.

PURPOSE
-------
This script consolidates the angle-encoding experiments into one reproducible
study. It compares:

    1. Translation baseline:
       K(x, y) = mean_i cos^2(pi * (x_i - y_i) / 2)

    2. Angle-mean encoding:
       K(x, y) = mean_i cos^2(alpha * (x_i - y_i) / 2)

    3. Angle-product encoding:
       K(x, y) = product_i cos^2(alpha * (x_i - y_i) / 2)

The angle values are:
    pi/2, 3pi/4, pi, 3pi/2, 2pi

The angle-mean encoding at alpha=pi is mathematically identical to the
translation baseline, so it is not evaluated twice.

EXPERIMENTS
-----------
A. Angle parameter comparison:
   Compare the different alpha values.

B. Feature relevance:
   Use 2, 4, or 8 informative features and add nested sets of independent
   noise features: 0, 2, 6, 14, 30, or 62.

C. Scaling sensitivity:
   Compare:
       - MinMax [0, 1]
       - MinMax [-1, 1]
       - StandardScaler, clipped to [-3, 3], then mapped to [0, 1]

All scalers are fitted on the training split only.

IMPORTANT INTERPRETATION
-----------------------
This script computes exact similarity matrices classically using NumPy.
It does NOT simulate finite-shot measurement, execute Qiskit circuits, or
measure quantum hardware runtime. It therefore studies the mathematical
similarity functions and their induced neighbour rankings, not quantum
speedup or circuit-level cost.

OUTPUTS
-------
angle_encoding_results.csv
angle_encoding_paired_differences.csv
angle_encoding_summary.csv
angle_encoding_paired_summary.csv
angle_encoding_kernel_diagnostics.csv
angle_encoding_kernel_summary.csv

EXAMPLE
-------
python scripts/02_angle_encoding_study.py \
    --seeds 10 \
    --n-samples 1200 \
    --outdir results/angle_encoding_study

A smaller test run:
python scripts/02_angle_encoding_study.py \
    --seeds 2 \
    --n-samples 500 \
    --informative-features 2 4 \
    --noise-features 0 6 14 \
    --outdir results/angle_encoding_test
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

ANGLE_VALUES = {
    "pi_half": np.pi / 2,
    "three_pi_fourths": 3 * np.pi / 4,
    "three_pi_half": 3 * np.pi / 2,
    "two_pi": 2 * np.pi,
}

SCALING_METHODS = [
    "minmax_01",
    "minmax_m11",
    "standard_clip3_to_01",
]

DEFAULT_INFORMATIVE_FEATURES = [2, 4, 8]
DEFAULT_NOISE_FEATURES = [0, 2, 6, 14, 30, 62]
DEFAULT_K_VALUES = [1, 3, 5, 9]


# ---------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------

def generate_dataset(
    n_samples: int,
    n_informative: int,
    max_noise_features: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a binary classification dataset.

    The informative core is generated once. Noise features are generated
    independently and appended in a fixed order. Consequently, the feature
    sets are nested:

        core
        core + first 2 noise features
        core + first 6 noise features
        ...

    This makes comparisons across noise levels paired and controlled.
    """

    X_core, y = make_classification(
        n_samples=n_samples,
        n_features=n_informative,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=1.0,
        flip_y=0.01,
        random_state=seed,
    )

    rng = np.random.default_rng(seed + 100_000)

    if max_noise_features > 0:
        X_noise = rng.uniform(
            low=0.0,
            high=1.0,
            size=(n_samples, max_noise_features),
        )
        X = np.column_stack((X_core, X_noise))
    else:
        X = X_core.copy()

    return X.astype(float), y.astype(int)


# ---------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------

def scale_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit the scaler on training data only, then transform train and test.

    MinMax transformations are not clipped. A test value outside the
    training range may therefore map slightly outside the nominal interval.

    For standard_clip3_to_01:
        1. Standardise using training mean and standard deviation.
        2. Clip both train and test values to [-3, 3].
        3. Map [-3, 3] linearly to [0, 1].
    """

    if method == "minmax_01":
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(X_train), scaler.transform(X_test)

    if method == "minmax_m11":
        scaler = MinMaxScaler(feature_range=(-1, 1))
        return scaler.fit_transform(X_train), scaler.transform(X_test)

    if method == "standard_clip3_to_01":
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(X_train)
        test_scaled = scaler.transform(X_test)

        train_scaled = np.clip(train_scaled, -3.0, 3.0)
        test_scaled = np.clip(test_scaled, -3.0, 3.0)

        train_scaled = (train_scaled + 3.0) / 6.0
        test_scaled = (test_scaled + 3.0) / 6.0

        return train_scaled, test_scaled

    raise ValueError(f"Unknown scaling method: {method}")


# ---------------------------------------------------------------------
# Similarity matrices
# ---------------------------------------------------------------------

def compute_similarity(
    X_query: np.ndarray,
    X_train: np.ndarray,
    encoding: str,
    alpha: float | None = None,
) -> np.ndarray:
    """
    Compute an exact similarity matrix.

    Rows correspond to query samples; columns correspond to training
    samples.

    The implementation loops over features to avoid constructing a large
    (n_query, n_train, n_features) temporary array.

    encoding:
        translation
        angle_mean
        angle_product
    """

    n_query, n_features = X_query.shape
    n_train = X_train.shape[0]

    if n_features == 0:
        raise ValueError("The dataset must contain at least one feature.")

    if encoding == "translation":
        alpha = np.pi
        mode = "mean"
    elif encoding == "angle_mean":
        if alpha is None:
            raise ValueError("alpha is required for angle_mean.")
        mode = "mean"
    elif encoding == "angle_product":
        if alpha is None:
            raise ValueError("alpha is required for angle_product.")
        mode = "product"
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    if mode == "mean":
        similarity = np.zeros((n_query, n_train), dtype=float)

        for feature in range(n_features):
            delta = (
                X_query[:, feature, None]
                - X_train[None, :, feature]
            )
            similarity += np.cos(alpha * delta / 2.0) ** 2

        similarity /= n_features

    else:
        similarity = np.ones((n_query, n_train), dtype=float)

        for feature in range(n_features):
            delta = (
                X_query[:, feature, None]
                - X_train[None, :, feature]
            )
            component = np.cos(alpha * delta / 2.0) ** 2
            similarity *= component

    # Protect against tiny floating-point excursions.
    return np.clip(similarity, 0.0, 1.0)


# ---------------------------------------------------------------------
# k-nearest-neighbour prediction
# ---------------------------------------------------------------------

def get_neighbours(
    similarity: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return indices of the k highest-similarity training samples."""

    if k < 1 or k > similarity.shape[1]:
        raise ValueError(
            f"k={k} is invalid for {similarity.shape[1]} training samples."
        )

    # argpartition avoids sorting every training sample.
    indices = np.argpartition(
        -similarity,
        kth=k - 1,
        axis=1,
    )[:, :k]

    return indices


def predict_from_neighbours(
    neighbour_indices: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    """Majority-vote prediction with deterministic tie handling."""

    predictions = np.empty(neighbour_indices.shape[0], dtype=int)

    for row in range(neighbour_indices.shape[0]):
        labels = y_train[neighbour_indices[row]]
        values, counts = np.unique(labels, return_counts=True)

        # np.unique sorts labels, so ties resolve to the smaller label.
        predictions[row] = values[np.argmax(counts)]

    return predictions


def accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(np.mean(y_true == y_pred))


def neighbour_jaccard(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    """
    Mean per-query Jaccard similarity between two neighbour sets.
    Neighbour order is intentionally ignored.
    """

    scores = []

    for ref_row, cand_row in zip(reference, candidate):
        ref_set = set(ref_row.tolist())
        cand_set = set(cand_row.tolist())

        union = ref_set | cand_set
        intersection = ref_set & cand_set

        scores.append(
            len(intersection) / len(union) if union else 1.0
        )

    return float(np.mean(scores))


# ---------------------------------------------------------------------
# Dataset descriptors for C3 analysis
# ---------------------------------------------------------------------

def dataset_descriptors(
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate basic dataset characteristics.

    These descriptors can later be related to which encoding performs
    best. They are descriptive statistics, not causal explanations.
    """

    n_samples, n_features = X.shape

    if n_features > 1:
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        off_diagonal = corr[~np.eye(n_features, dtype=bool)]
        mean_abs_correlation = (
            float(np.mean(np.abs(off_diagonal)))
            if off_diagonal.size
            else 0.0
        )
    else:
        mean_abs_correlation = 0.0

    class_means = []
    class_stds = []

    for label in np.unique(y):
        class_data = X[y == label]
        class_means.append(np.mean(class_data, axis=0))
        class_stds.append(np.std(class_data, axis=0))

    if len(class_means) >= 2:
        centroid_distance = float(
            np.linalg.norm(class_means[0] - class_means[1])
        )
        pooled_std = float(
            np.sqrt(np.mean(np.square(class_stds)))
        )
        centroid_separation = centroid_distance / (pooled_std + 1e-12)
    else:
        centroid_separation = np.nan

    return {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "mean_abs_feature_correlation": mean_abs_correlation,
        "class_centroid_separation": centroid_separation,
    }


# ---------------------------------------------------------------------
# Kernel diagnostics
# ---------------------------------------------------------------------

def kernel_diagnostics(
    similarity: np.ndarray,
) -> Dict[str, float]:
    """Summarise the distribution of similarity values."""

    values = similarity.ravel()

    return {
        "kernel_mean": float(np.mean(values)),
        "kernel_std": float(np.std(values)),
        "kernel_median": float(np.median(values)),
        "fraction_below_1e-12": float(np.mean(values < 1e-12)),
        "fraction_exact_zero": float(np.mean(values == 0.0)),
        "fraction_above_099": float(np.mean(values > 0.99)),
        "fraction_above_09999": float(np.mean(values > 0.9999)),
    }


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> None:

    os.makedirs(args.outdir, exist_ok=True)

    result_rows: List[dict] = []
    diagnostic_rows: List[dict] = []

    max_noise = max(args.noise_features)

    for seed in range(args.seeds):
        print(f"\nSeed {seed + 1}/{args.seeds}")

        for n_informative in args.informative_features:

            print(f"  Informative features: {n_informative}")

            X_full, y = generate_dataset(
                n_samples=args.n_samples,
                n_informative=n_informative,
                max_noise_features=max_noise,
                seed=seed,
            )

            # One fixed split is reused for every noise level and scaling.
            all_indices = np.arange(len(y))

            train_indices, test_indices = train_test_split(
                all_indices,
                test_size=args.test_size,
                random_state=seed + 50_000,
                stratify=y,
            )

            y_train = y[train_indices]
            y_test = y[test_indices]

            for n_noise in args.noise_features:

                n_features = n_informative + n_noise
                X = X_full[:, :n_features]

                descriptors = dataset_descriptors(X, y)

                X_train_raw = X[train_indices]
                X_test_raw = X[test_indices]

                for scaling in SCALING_METHODS:

                    X_train, X_test = scale_data(
                        X_train_raw,
                        X_test_raw,
                        scaling,
                    )

                    # -------------------------------------------------
                    # Compute all similarity matrices
                    # -------------------------------------------------

                    similarities: Dict[str, np.ndarray] = {}

                    similarities["translation"] = compute_similarity(
                        X_test,
                        X_train,
                        encoding="translation",
                    )

                    for angle_name, alpha in ANGLE_VALUES.items():
                        similarities[f"angle_mean_{angle_name}"] = (
                            compute_similarity(
                                X_test,
                                X_train,
                                encoding="angle_mean",
                                alpha=alpha,
                            )
                        )

                        similarities[f"angle_product_{angle_name}"] = (
                            compute_similarity(
                                X_test,
                                X_train,
                                encoding="angle_product",
                                alpha=alpha,
                            )
                        )

                    # -------------------------------------------------
                    # Kernel diagnostics
                    # -------------------------------------------------

                    for model_name, sim in similarities.items():
                        diag = kernel_diagnostics(sim)

                        diagnostic_rows.append({
                            "seed": seed,
                            "informative_features": n_informative,
                            "noise_features": n_noise,
                            "total_features": n_features,
                            "scaling": scaling,
                            "model": model_name,
                            **descriptors,
                            **diag,
                        })

                    # -------------------------------------------------
                    # Neighbour selection and classification
                    # -------------------------------------------------

                    for k in args.k_values:

                        if k > len(train_indices):
                            continue

                        baseline_neighbours = get_neighbours(
                            similarities["translation"],
                            k,
                        )

                        for model_name, sim in similarities.items():

                            neighbours = (
                                baseline_neighbours
                                if model_name == "translation"
                                else get_neighbours(sim, k)
                            )

                            predictions = predict_from_neighbours(
                                neighbours,
                                y_train,
                            )

                            result_rows.append({
                                "seed": seed,
                                "informative_features": n_informative,
                                "noise_features": n_noise,
                                "total_features": n_features,
                                "informative_fraction": (
                                    n_informative / n_features
                                ),
                                "scaling": scaling,
                                "model": model_name,
                                "k": k,
                                "accuracy": accuracy_score(
                                    y_test,
                                    predictions,
                                ),
                                "neighbour_jaccard_vs_translation": (
                                    1.0
                                    if model_name == "translation"
                                    else neighbour_jaccard(
                                        baseline_neighbours,
                                        neighbours,
                                    )
                                ),
                                **descriptors,
                            })

            print("    Completed all noise levels and scaling methods.")

    # -----------------------------------------------------------------
    # Convert to DataFrames
    # -----------------------------------------------------------------

    results = pd.DataFrame(result_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)

    if results.empty:
        raise RuntimeError("No experiment results were generated.")

    # -----------------------------------------------------------------
    # Paired differences against translation
    # -----------------------------------------------------------------

    baseline = results[
        results["model"] == "translation"
    ][[
        "seed",
        "informative_features",
        "noise_features",
        "scaling",
        "k",
        "accuracy",
    ]].rename(columns={
        "accuracy": "translation_accuracy"
    })

    alternatives = results[
        results["model"] != "translation"
    ].copy()

    paired = alternatives.merge(
        baseline,
        on=[
            "seed",
            "informative_features",
            "noise_features",
            "scaling",
            "k",
        ],
        how="inner",
        validate="many_to_one",
    )

    paired["accuracy_difference"] = (
        paired["accuracy"] - paired["translation_accuracy"]
    )

    paired["accuracy_difference_pp"] = (
        100.0 * paired["accuracy_difference"]
    )

    # -----------------------------------------------------------------
    # Aggregate summaries
    # -----------------------------------------------------------------

    summary_columns = [
        "accuracy",
        "neighbour_jaccard_vs_translation",
    ]

    summary = (
        results
        .groupby([
            "informative_features",
            "noise_features",
            "total_features",
            "scaling",
            "model",
            "k",
        ], as_index=False)[summary_columns]
        .agg(["mean", "std"])
    )

    # Flatten the multi-index columns generated by aggregation.
    summary.columns = [
        "_".join(
            str(part) for part in column if str(part) != ""
        ).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]

    paired_summary = (
        paired
        .groupby([
            "informative_features",
            "noise_features",
            "scaling",
            "model",
            "k",
        ], as_index=False)
        .agg(
            mean_accuracy_difference_pp=(
                "accuracy_difference_pp", "mean"
            ),
            std_accuracy_difference_pp=(
                "accuracy_difference_pp", "std"
            ),
            wins=(
                "accuracy_difference_pp",
                lambda x: int(np.sum(x > 1e-12)),
            ),
            ties=(
                "accuracy_difference_pp",
                lambda x: int(np.sum(np.abs(x) <= 1e-12)),
            ),
            losses=(
                "accuracy_difference_pp",
                lambda x: int(np.sum(x < -1e-12)),
            ),
            paired_runs=("accuracy_difference_pp", "count"),
        )
    )

    kernel_summary = (
        diagnostics
        .groupby([
            "informative_features",
            "noise_features",
            "scaling",
            "model",
        ], as_index=False)
        .agg({
            "kernel_mean": ["mean", "std"],
            "kernel_std": ["mean", "std"],
            "kernel_median": ["mean", "std"],
            "fraction_below_1e-12": ["mean", "std"],
            "fraction_exact_zero": ["mean", "std"],
            "fraction_above_099": ["mean", "std"],
            "fraction_above_09999": ["mean", "std"],
        })
    )

    kernel_summary.columns = [
        "_".join(
            str(part) for part in column if str(part) != ""
        ).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in kernel_summary.columns
    ]

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    output_files = {
        "angle_encoding_results.csv": results,
        "angle_encoding_paired_differences.csv": paired,
        "angle_encoding_summary.csv": summary,
        "angle_encoding_paired_summary.csv": paired_summary,
        "angle_encoding_kernel_diagnostics.csv": diagnostics,
        "angle_encoding_kernel_summary.csv": kernel_summary,
    }

    for filename, dataframe in output_files.items():
        path = os.path.join(args.outdir, filename)
        dataframe.to_csv(path, index=False)
        print(f"Saved: {path}")

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------

    print("\n" + "=" * 78)
    print("ANGLE ENCODING STUDY — COMPLETED")
    print("=" * 78)

    print(f"Seeds:                 {args.seeds}")
    print(f"Samples per dataset:   {args.n_samples}")
    print(f"Informative features:  {args.informative_features}")
    print(f"Noise features:        {args.noise_features}")
    print(f"k values:              {args.k_values}")
    print(f"Scaling methods:       {SCALING_METHODS}")
    print(f"Result rows:           {len(results)}")
    print(f"Paired comparison rows:{len(paired)}")
    print(f"Output directory:      {args.outdir}")

    print("\nModels evaluated:")
    for model in sorted(results["model"].unique()):
        print(f"  - {model}")

    print("\nMean accuracy by model:")
    console_summary = (
        results
        .groupby("model")["accuracy"]
        .mean()
        .sort_values(ascending=False)
    )

    print(console_summary.to_string())

    print("\nMean accuracy difference vs translation (percentage points):")
    if not paired.empty:
        print(
            paired
            .groupby("model")["accuracy_difference_pp"]
            .mean()
            .sort_values(ascending=False)
            .round(3)
            .to_string()
        )

    print("\nNOTE:")
    print(
        "These are exact classical evaluations of the specified "
        "similarity formulas."
    )
    print(
        "They are not finite-shot QKNN circuit simulations or "
        "quantum-runtime measurements."
    )


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Study angle encoding, feature relevance, and scaling "
            "sensitivity for QKNN similarity functions."
        )
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of random seeds.",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=1200,
        help="Number of samples generated per seed/informative setting.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Fraction of samples reserved for testing.",
    )

    parser.add_argument(
        "--informative-features",
        type=int,
        nargs="+",
        default=DEFAULT_INFORMATIVE_FEATURES,
        help="Numbers of informative features to test.",
    )

    parser.add_argument(
        "--noise-features",
        type=int,
        nargs="+",
        default=DEFAULT_NOISE_FEATURES,
        help="Numbers of appended noise features to test.",
    )

    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=DEFAULT_K_VALUES,
        help="Neighbour counts for kNN.",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="results/angle_encoding_study",
        help="Directory for CSV outputs.",
    )

    args = parser.parse_args()

    if args.seeds < 1:
        parser.error("--seeds must be at least 1.")

    if args.n_samples < 20:
        parser.error("--n-samples must be at least 20.")

    if not 0.0 < args.test_size < 1.0:
        parser.error("--test-size must be between 0 and 1.")

    if any(value < 1 for value in args.informative_features):
        parser.error("All informative feature counts must be >= 1.")

    if any(value < 0 for value in args.noise_features):
        parser.error("Noise feature counts cannot be negative.")

    if any(value < 1 for value in args.k_values):
        parser.error("All k values must be >= 1.")

    if len(set(args.noise_features)) != len(args.noise_features):
        parser.error("Noise feature counts must be unique.")

    args.informative_features = sorted(set(args.informative_features))
    args.noise_features = sorted(set(args.noise_features))
    args.k_values = sorted(set(args.k_values))

    return args


if __name__ == "__main__":
    arguments = parse_args()
    run_experiment(arguments)


