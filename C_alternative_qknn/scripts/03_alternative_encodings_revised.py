#!/usr/bin/env python3
"""
03_alternative_encodings.py
===========================

Project C — QKNN representation study
Study 03: controlled comparison of alternative data encodings.

RESEARCH QUESTIONS
------------------
1. Under a common state-fidelity definition, how do different encodings
   induce different nearest-neighbour geometries?
2. How do encoding-specific similarity constructions compare?
3. Which dataset properties (continuous/discrete, sparsity, correlation,
   dimensionality, direction-vs-magnitude structure, periodicity, etc.)
   are associated with useful encodings?
4. What are the first-order qubit/resource implications?

COMPARISON TRACKS
-----------------
A. COMMON FIDELITY
   All methods in this track return a fidelity-like quantity in [0,1]:
     * amplitude_fidelity
     * phase_product_fidelity_pi
     * angle_product_fidelity_{alpha}
     * basis_exact_fidelity_{bits}bit

   This is the fairest available comparison of state representations.
   It does NOT make their state-preparation costs equal.

B. ENCODING-SPECIFIC / CLASSICAL SIMILARITIES
     * translation_mean
     * angle_mean_{alpha}
     * basis_hamming_{bits}bit

   These are useful alternatives, but they are not all quantum state
   fidelities. Do not mix this track with the common-fidelity track when
   making claims about which state encoding is best.

IMPORTANT MATHEMATICAL NOTES
----------------------------
* Amplitude fidelity uses nonnegative, L2-normalised feature vectors.
  It is squared cosine similarity and is invariant to positive rescaling
  of an individual sample. This can help direction-based problems and
  hurt magnitude-based problems.
* Phase product fidelity at alpha=pi is exactly the same kernel as
  angle-product fidelity at alpha=pi. Both are retained as an equivalence
  check, not as independent methods.
* Angle-mean and translation-mean similarities are averages of
  per-feature fidelities. They are not the fidelity of a product state.
* Basis exact fidelity is 1 only when every quantised bit matches.
  Hamming similarity is a separate, classical distance-derived measure.
* All similarities are evaluated classically and exactly. This script
  does not execute quantum circuits or finite-shot measurements.
* Qubit counts are first-order encoding estimates. Circuit depth, gate
  count after transpilation, state-preparation cost, runtime and shots
  are NOT measured here.

DATASETS
--------
Real:
  iris, wine, breast_cancer, digits

Controlled synthetic:
  directional_clusters       amplitude's direction invariance may help
  radial_magnitude            amplitude's norm invariance should hurt
  binary_hamming              naturally discrete, local bit corruption
  sparse_signal               sparse informative coordinates + noise
  periodic_features           periodic feature geometry
  correlated_redundant        redundant correlated dimensions
  anisotropic_blobs            rotated/anisotropic continuous geometry
  moons, circles               nonlinear low-dimensional manifolds
  high_dim_weak_signal         many noise dimensions, weak signal
  heavy_tailed                 outlier-prone continuous data
  ordinal_discrete             ordered discrete levels
  imbalanced                   minority-class stress test

The "theoretical expectation" fields are hypotheses, not guaranteed
outcomes. A mismatch between hypothesis and empirical results is useful.

PREPROCESSING / FAIRNESS
-----------------------
* Same train/test split for all methods within a dataset/seed.
* Stratified split.
* Constant features removed using TRAINING DATA ONLY.
* Scaling fitted on training data only.
* Two scaling conditions are evaluated.
* Same k values and train/test samples across methods.
* Reports accuracy, balanced accuracy, macro-F1 and neighbour Jaccard.
* Dataset descriptors are computed from the training split after
  constant-feature removal; original and retained dimensions are recorded.

OUTPUTS
-------
encoding_results.csv
encoding_paired_differences.csv
encoding_summary.csv
encoding_paired_summary.csv
encoding_kernel_diagnostics.csv
encoding_kernel_summary.csv
encoding_dataset_characteristics.csv
encoding_equivalence_checks.csv
encoding_dataset_ranking.csv
encoding_resource_estimates.csv
encoding_hypothesis_map.csv

EXAMPLE
-------
python scripts/03_alternative_encodings.py --seeds 5 --outdir results/alternative_encodings

Fuller run:
python scripts/03_alternative_encodings.py --seeds 10 --outdir results/alternative_encodings
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_blobs,
    make_circles,
    make_classification,
    make_moons,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATASET_INFO = {
    "iris": ("real", "Mixed continuous measurements; small, low-dimensional benchmark."),
    "wine": ("real", "Continuous chemical measurements with different feature scales."),
    "breast_cancer": ("real", "Continuous, correlated biomedical measurements."),
    "digits": ("real", "Integer-valued pixel intensities; high-dimensional with constant pixels."),
    "directional_clusters": ("directional", "Class depends primarily on vector direction; amplitude's norm invariance may help."),
    "radial_magnitude": ("magnitude", "Class depends on radius along similar directions; amplitude's norm invariance should hurt."),
    "binary_hamming": ("binary", "Binary prototype classes with independent bit flips; Hamming geometry is a plausible fit."),
    "sparse_signal": ("sparse", "Sparse informative coordinates embedded in many zero/noise coordinates."),
    "periodic_features": ("periodic", "Features live on a periodic domain; phase periodicity may help for suitable alpha."),
    "correlated_redundant": ("correlated", "Strongly correlated and duplicated/redundant dimensions."),
    "anisotropic_blobs": ("continuous", "Continuous clusters with rotated anisotropic covariance."),
    "moons": ("manifold", "Nonlinear two-dimensional manifold."),
    "circles": ("manifold", "Concentric circular structure."),
    "high_dim_weak_signal": ("high_dimensional", "Many dimensions, few informative features and weak class separation."),
    "heavy_tailed": ("heavy_tailed", "Continuous data with heavy-tailed feature distributions and outliers."),
    "ordinal_discrete": ("ordinal", "Ordered discrete feature levels; quantisation resolution matters."),
    "imbalanced": ("imbalanced", "Severe class imbalance; accuracy alone is insufficient."),
}

DEFAULT_DATASETS = list(DATASET_INFO)
SCALING_METHODS = ["minmax_01", "standard_clip3_to_01"]
DEFAULT_K_VALUES = [1, 3, 5, 9]
ANGLE_ALPHAS = {
    "pi_half": np.pi / 2,
    "pi": np.pi,
    "three_pi_half": 3 * np.pi / 2,
    "two_pi": 2 * np.pi,
}


# ---------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------

def _normalise_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def make_dataset(name: str, seed: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create a real or controlled synthetic classification dataset."""
    rng = np.random.default_rng(seed)

    if name == "iris":
        d = load_iris()
        return d.data.astype(float), d.target.astype(int)
    if name == "wine":
        d = load_wine()
        return d.data.astype(float), d.target.astype(int)
    if name == "breast_cancer":
        d = load_breast_cancer()
        return d.data.astype(float), d.target.astype(int)
    if name == "digits":
        d = load_digits()
        return d.data.astype(float), d.target.astype(int)

    if name == "directional_clusters":
        # Three separated directions, with independent positive radial
        # variation. Class identity is primarily angular, not magnitude.
        n_classes, d = 3, 12
        prototypes = _normalise_rows(rng.uniform(0.05, 1.0, (n_classes, d)))
        labels = rng.integers(0, n_classes, n_samples)
        radial = rng.uniform(0.35, 2.5, n_samples)
        X = prototypes[labels] * radial[:, None]
        X += rng.normal(0, 0.035, X.shape)
        X = np.maximum(X, 0.0)
        return X, labels

    if name == "radial_magnitude":
        # All classes share approximately the same direction. Label is
        # determined by radius. Amplitude encoding removes this information.
        d = 12
        direction = np.linspace(0.5, 1.5, d)
        direction = direction / np.linalg.norm(direction)
        labels = rng.integers(0, 3, n_samples)
        radii = np.choose(labels, [0.55, 1.35, 2.3])
        X = radii[:, None] * direction[None, :]
        X += rng.normal(0, 0.035, (n_samples, d))
        X = np.maximum(X, 0.0)
        return X, labels

    if name == "binary_hamming":
        # Four binary class prototypes, corrupted by independent bit flips.
        d, n_classes = 16, 4
        prototypes = np.array([
            [0, 0, 0, 0] * 4,
            [0, 0, 1, 1] * 4,
            [1, 1, 0, 0] * 4,
            [1, 1, 1, 1] * 4,
        ], dtype=float)
        labels = rng.integers(0, n_classes, n_samples)
        X = prototypes[labels].copy()
        flips = rng.random((n_samples, d)) < 0.10
        X = np.logical_xor(X.astype(bool), flips).astype(float)
        return X, labels

    if name == "sparse_signal":
        n_features, n_informative = 64, 6
        labels = rng.integers(0, 2, n_samples)
        # Six informative coordinates are active for every sample; the
        # remaining coordinates are exactly zero for most samples.
        X = np.zeros((n_samples, n_features), dtype=float)
        signs = np.where(labels == 0, -1.0, 1.0)
        X[:, :n_informative] = (
            signs[:, None] * 0.9 + rng.normal(0, 0.16, (n_samples, n_informative))
        )
        noise_mask = rng.random((n_samples, n_features - n_informative)) < 0.08
        noise_values = rng.normal(0, 0.7, noise_mask.shape)
        X[:, n_informative:] = noise_mask * noise_values
        # This produces actual row-wise sparsity rather than merely
        # low-magnitude dense Gaussian noise.
        return X, labels

    if name == "periodic_features":
        # Two periodic coordinates on [0,1). Class regions are separated
        # on a torus; values near 0 and 1 are neighbours in the true geometry.
        labels = rng.integers(0, 2, n_samples)
        centres = np.array([[0.02, 0.5], [0.52, 0.5]])
        X = (centres[labels] + rng.normal(0, 0.10, (n_samples, 2))) % 1.0
        extra = rng.uniform(0, 1, (n_samples, 4))
        return np.column_stack([X, extra]), labels

    if name == "correlated_redundant":
        labels = rng.integers(0, 2, n_samples)
        latent = rng.normal(0, 1, (n_samples, 4))
        latent[:, 0] += np.where(labels == 0, -0.8, 0.8)
        redundant = np.column_stack([
            latent,
            latent[:, 0] + rng.normal(0, 0.025, n_samples),
            latent[:, 1] * 1.5 + rng.normal(0, 0.025, n_samples),
            latent[:, 0],  # exact duplicate
            rng.normal(0, 1, (n_samples, 8)),
        ])
        return redundant, labels

    if name == "anisotropic_blobs":
        X, y = make_blobs(
            n_samples=n_samples, centers=3, n_features=4,
            cluster_std=2.0, random_state=seed,
        )
        transform = np.array([
            [0.6, -0.6, 0.2, 0.0],
            [0.4, 0.8, 0.0, 0.2],
            [0.0, 0.3, 1.2, 0.4],
            [0.2, 0.0, 0.3, 0.9],
        ])
        return X @ transform, y

    if name == "moons":
        return make_moons(n_samples=n_samples, noise=0.22, random_state=seed)
    if name == "circles":
        return make_circles(n_samples=n_samples, noise=0.12, factor=0.35, random_state=seed)

    if name == "high_dim_weak_signal":
        X, y = make_classification(
            n_samples=n_samples, n_features=80, n_informative=5,
            n_redundant=5, n_repeated=0, n_classes=2,
            n_clusters_per_class=2, class_sep=0.55, flip_y=0.04,
            random_state=seed,
        )
        return X, y

    if name == "heavy_tailed":
        labels = rng.integers(0, 2, n_samples)
        base = rng.standard_t(df=2.5, size=(n_samples, 10))
        base[:, :3] += np.where(labels[:, None] == 0, -0.55, 0.55)
        return base, labels

    if name == "ordinal_discrete":
        labels = rng.integers(0, 3, n_samples)
        latent = rng.normal(0, 1, (n_samples, 8))
        latent[:, :3] += np.choose(labels, [-0.9, 0.0, 0.9])[:, None]
        # Quantise to five ordered levels.
        X = np.digitize(latent, bins=[-1.0, -0.35, 0.35, 1.0]).astype(float)
        return X, labels

    if name == "imbalanced":
        X, y = make_classification(
            n_samples=n_samples, n_features=12, n_informative=4,
            n_redundant=3, n_classes=2, weights=[0.90, 0.10],
            class_sep=0.8, flip_y=0.03, random_state=seed,
        )
        return X, y

    raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------
# Train-only feature filtering and scaling
# ---------------------------------------------------------------------

def remove_constant_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit zero-variance mask on training data only; apply it to test data."""
    variances = np.var(X_train, axis=0)
    mask = variances > 0.0
    if not np.any(mask):
        # Defensive fallback: retain the highest-variance feature.
        mask[int(np.argmax(variances))] = True
    return X_train[:, mask], X_test[:, mask], mask


def scale_data(
    X_train: np.ndarray, X_test: np.ndarray, method: str
) -> Tuple[np.ndarray, np.ndarray]:
    if method == "minmax_01":
        scaler = MinMaxScaler(feature_range=(0, 1))
        A = scaler.fit_transform(X_train)
        B = scaler.transform(X_test)
        return np.clip(A, 0.0, 1.0), np.clip(B, 0.0, 1.0)

    if method == "standard_clip3_to_01":
        scaler = StandardScaler()
        A = np.clip(scaler.fit_transform(X_train), -3.0, 3.0)
        B = np.clip(scaler.transform(X_test), -3.0, 3.0)
        return (A + 3.0) / 6.0, (B + 3.0) / 6.0

    raise ValueError(f"Unknown scaling method: {method}")


# ---------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------

def translation_mean_similarity(Xq: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    out = np.zeros((len(Xq), len(Xt)), dtype=float)
    for j in range(Xq.shape[1]):
        delta = Xq[:, j, None] - Xt[None, :, j]
        out += np.cos(np.pi * delta / 2.0) ** 2
    return np.clip(out / Xq.shape[1], 0.0, 1.0)


def angle_mean_similarity(Xq: np.ndarray, Xt: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros((len(Xq), len(Xt)), dtype=float)
    for j in range(Xq.shape[1]):
        delta = Xq[:, j, None] - Xt[None, :, j]
        out += np.cos(alpha * delta / 2.0) ** 2
    return np.clip(out / Xq.shape[1], 0.0, 1.0)


def angle_product_fidelity(Xq: np.ndarray, Xt: np.ndarray, alpha: float) -> np.ndarray:
    out = np.ones((len(Xq), len(Xt)), dtype=float)
    for j in range(Xq.shape[1]):
        delta = Xq[:, j, None] - Xt[None, :, j]
        out *= np.cos(alpha * delta / 2.0) ** 2
    return np.clip(out, 0.0, 1.0)


def amplitude_fidelity(Xq: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    """Squared overlap of real, nonnegative, L2-normalised amplitude states."""
    qn = np.linalg.norm(Xq, axis=1, keepdims=True)
    tn = np.linalg.norm(Xt, axis=1, keepdims=True)
    q = Xq / np.maximum(qn, 1e-12)
    t = Xt / np.maximum(tn, 1e-12)
    return np.clip((q @ t.T) ** 2, 0.0, 1.0)


def quantise_to_bits(X: np.ndarray, bits: int) -> np.ndarray:
    levels = 2 ** bits
    q = np.rint(np.clip(X, 0.0, 1.0) * (levels - 1)).astype(np.int64)
    shifts = np.arange(bits - 1, -1, -1)
    binary = ((q[..., None] >> shifts) & 1).astype(np.uint8)
    return binary.reshape(len(X), -1)


def basis_similarities(
    Xq: np.ndarray, Xt: np.ndarray, bits: int
) -> Tuple[np.ndarray, np.ndarray]:
    qb, tb = quantise_to_bits(Xq, bits), quantise_to_bits(Xt, bits)
    matching = np.sum(qb[:, None, :] == tb[None, :, :], axis=2)
    total = qb.shape[1]
    exact = (matching == total).astype(float)
    hamming_sim = matching / total
    return exact, hamming_sim


def compute_similarities(
    Xq: np.ndarray, Xt: np.ndarray, basis_bits: List[int]
) -> Dict[str, Tuple[np.ndarray, str, str]]:
    """
    Return model -> (similarity matrix, comparison_track, representation).
    The track labels prevent common-fidelity and alternative-score claims
    from being conflated.
    """
    models: Dict[str, Tuple[np.ndarray, str, str]] = {}
    models["amplitude_fidelity"] = (
        amplitude_fidelity(Xq, Xt), "common_fidelity", "amplitude"
    )
    models["phase_product_fidelity_pi"] = (
        angle_product_fidelity(Xq, Xt, np.pi), "common_fidelity", "phase"
    )

    for label, alpha in ANGLE_ALPHAS.items():
        models[f"angle_product_fidelity_{label}"] = (
            angle_product_fidelity(Xq, Xt, alpha),
            "common_fidelity", "angle_product",
        )
        models[f"angle_mean_{label}"] = (
            angle_mean_similarity(Xq, Xt, alpha),
            "encoding_specific", "angle_mean",
        )

    models["translation_mean"] = (
        translation_mean_similarity(Xq, Xt),
        "encoding_specific", "translation_baseline",
    )

    for bits in basis_bits:
        exact, hamming = basis_similarities(Xq, Xt, bits)
        models[f"basis_exact_fidelity_{bits}bit"] = (
            exact, "common_fidelity", "basis",
        )
        models[f"basis_hamming_{bits}bit"] = (
            hamming, "encoding_specific", "basis_hamming",
        )
    return models


# ---------------------------------------------------------------------
# Dataset descriptors
# ---------------------------------------------------------------------

def safe_mean_abs_correlation(X: np.ndarray) -> float:
    if X.shape[1] < 2:
        return 0.0
    # Remove constant columns defensively before correlation calculation.
    keep = np.var(X, axis=0) > 0.0
    Xv = X[:, keep]
    if Xv.shape[1] < 2:
        return 0.0
    corr = np.corrcoef(Xv, rowvar=False)
    upper = corr[np.triu_indices_from(corr, k=1)]
    upper = upper[np.isfinite(upper)]
    return float(np.mean(np.abs(upper))) if upper.size else 0.0


def calculate_dataset_characteristics(
    X_train: np.ndarray, y_train: np.ndarray,
    original_features: int, retained_features: int,
    dataset_name: str,
) -> Dict[str, float]:
    labels, counts = np.unique(y_train, return_counts=True)
    proportions = counts / counts.sum()
    entropy = float(-np.sum(proportions * np.log2(proportions + 1e-15)))
    imbalance = float(counts.max() / counts.min())
    class_centres, within_vars = [], []

    for label in labels:
        values = X_train[y_train == label]
        centre = values.mean(axis=0)
        class_centres.append(centre)
        within_vars.append(float(np.mean(np.sum((values - centre) ** 2, axis=1))))

    centre_distances = [
        np.linalg.norm(class_centres[i] - class_centres[j])
        for i in range(len(class_centres))
        for j in range(i + 1, len(class_centres))
    ]
    between = float(np.mean(centre_distances)) if centre_distances else np.nan
    within = float(np.sqrt(np.mean(within_vars))) if within_vars else np.nan

    X_nonzero = np.mean(np.abs(X_train) > 1e-12)
    sparsity = 1.0 - float(X_nonzero)
    return {
        "dataset_family": DATASET_INFO[dataset_name][0],
        "dataset_hypothesis": DATASET_INFO[dataset_name][1],
        "n_train": int(len(y_train)),
        "original_n_features": int(original_features),
        "retained_n_features": int(retained_features),
        "removed_constant_features": int(original_features - retained_features),
        "n_classes": int(len(labels)),
        "imbalance_ratio": imbalance,
        "class_entropy_bits": entropy,
        "mean_abs_feature_correlation": safe_mean_abs_correlation(X_train),
        "training_zero_fraction": sparsity,
        "between_centroid_distance": between,
        "within_class_spread": within,
        "centroid_separation_ratio": float(between / (within + 1e-12)),
    }


# ---------------------------------------------------------------------
# kNN and diagnostics
# ---------------------------------------------------------------------

def get_neighbours(similarity: np.ndarray, k: int) -> np.ndarray:
    k = min(k, similarity.shape[1])
    # Stable full sort makes tie handling deterministic across methods.
    return np.argsort(-similarity, axis=1, kind="mergesort")[:, :k]


def predict_from_neighbours(indices: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    pred = np.empty(indices.shape[0], dtype=int)
    for i, row in enumerate(indices):
        labels, counts = np.unique(y_train[row], return_counts=True)
        pred[i] = labels[np.argmax(counts)]
    return pred


def mean_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    scores = []
    for x, y in zip(a, b):
        sx, sy = set(x.tolist()), set(y.tolist())
        scores.append(len(sx & sy) / len(sx | sy) if sx | sy else 1.0)
    return float(np.mean(scores))


def similarity_diagnostics(matrix: np.ndarray) -> Dict[str, float]:
    v = matrix.ravel()
    return {
        "similarity_mean": float(np.mean(v)),
        "similarity_std": float(np.std(v)),
        "similarity_median": float(np.median(v)),
        "fraction_below_1e_12": float(np.mean(v < 1e-12)),
        "fraction_exact_zero": float(np.mean(v == 0.0)),
        "fraction_above_099": float(np.mean(v > 0.99)),
        "fraction_above_09999": float(np.mean(v > 0.9999)),
        "unique_similarity_fraction": float(len(np.unique(v)) / max(len(v), 1)),
    }


# ---------------------------------------------------------------------
# Resource estimates (not circuit measurements)
# ---------------------------------------------------------------------

def resource_estimates(n_features: int, basis_bits: List[int]) -> List[Dict[str, object]]:
    """First-order logical encoding size only; excludes circuit synthesis."""
    amp_qubits = int(np.ceil(np.log2(max(2, n_features))))
    rows = [
        {
            "model_family": "amplitude",
            "retained_features": n_features,
            "basis_bits_per_feature": np.nan,
            "estimated_data_qubits": amp_qubits,
            "encoding_operations_lower_bound": np.nan,
            "resource_note": "log2(d) qubits after padding; arbitrary state preparation may be O(d) or more",
        },
        {
            "model_family": "phase_product",
            "retained_features": n_features,
            "basis_bits_per_feature": np.nan,
            "estimated_data_qubits": n_features,
            "encoding_operations_lower_bound": n_features,
            "resource_note": "one phase-encoded qubit per feature; operation count is a simple feature-wise estimate",
        },
        {
            "model_family": "angle_product",
            "retained_features": n_features,
            "basis_bits_per_feature": np.nan,
            "estimated_data_qubits": n_features,
            "encoding_operations_lower_bound": n_features,
            "resource_note": "one rotation-encoded qubit per feature; excludes state preparation and transpilation",
        },
    ]
    for bits in basis_bits:
        rows.append({
            "model_family": f"basis_{bits}bit",
            "retained_features": n_features,
            "basis_bits_per_feature": bits,
            "estimated_data_qubits": n_features * bits,
            "encoding_operations_lower_bound": n_features * bits,
            "resource_note": "one computational-basis qubit per quantisation bit",
        })
    return rows


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> None:
    os.makedirs(args.outdir, exist_ok=True)
    result_rows, diagnostic_rows, dataset_rows = [], [], []
    equivalence_rows, resource_rows, hypothesis_rows = [], [], []

    for dataset_name in args.datasets:
        print("\\n" + "=" * 78)
        print(f"DATASET: {dataset_name} — {DATASET_INFO[dataset_name][0]}")
        print("=" * 78)

        for seed in range(args.seeds):
            X, y = make_dataset(dataset_name, seed, args.synthetic_samples)
            if len(y) > args.max_samples:
                keep, _ = train_test_split(
                    np.arange(len(y)), train_size=args.max_samples,
                    random_state=seed + 1000, stratify=y,
                )
                X, y = X[keep], y[keep]

            train_idx, test_idx = train_test_split(
                np.arange(len(y)), test_size=args.test_size,
                random_state=seed + 2000, stratify=y,
            )
            Xtr0, Xte0 = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            Xtr, Xte, feature_mask = remove_constant_features(Xtr0, Xte0)
            n_original, n_retained = X.shape[1], Xtr.shape[1]

            descriptors = calculate_dataset_characteristics(
                Xtr, ytr, n_original, n_retained, dataset_name
            )
            dataset_rows.append({
                "dataset": dataset_name, "seed": seed, **descriptors
            })
            hypothesis_rows.append({
                "dataset": dataset_name,
                "dataset_family": DATASET_INFO[dataset_name][0],
                "theoretical_hypothesis": DATASET_INFO[dataset_name][1],
                "original_n_features": n_original,
                "retained_n_features": n_retained,
            })

            for scaling in SCALING_METHODS:
                A, B = scale_data(Xtr, Xte, scaling)
                models = compute_similarities(B, A, args.basis_bits)

                # Exact mathematical equivalence: phase(pi) == angle-product(pi).
                phase = models["phase_product_fidelity_pi"][0]
                angle_pi = models["angle_product_fidelity_pi"][0]
                equivalence_rows.append({
                    "dataset": dataset_name, "seed": seed,
                    "scaling": scaling,
                    "comparison": "phase_product_pi_vs_angle_product_pi",
                    "max_absolute_difference": float(np.max(np.abs(phase - angle_pi))),
                    "mathematically_expected_equal": True,
                })

                for model, (matrix, track, representation) in models.items():
                    diagnostic_rows.append({
                        "dataset": dataset_name, "seed": seed,
                        "scaling": scaling, "model": model,
                        "comparison_track": track,
                        "representation": representation,
                        **descriptors,
                        **similarity_diagnostics(matrix),
                    })

                baseline_neighbours_by_k = {
                    k: get_neighbours(models["translation_mean"][0], k)
                    for k in args.k_values if k <= len(ytr)
                }

                for k, baseline_neighbours in baseline_neighbours_by_k.items():
                    for model, (matrix, track, representation) in models.items():
                        neighbours = (
                            baseline_neighbours if model == "translation_mean"
                            else get_neighbours(matrix, k)
                        )
                        pred = predict_from_neighbours(neighbours, ytr)
                        result_rows.append({
                            "dataset": dataset_name, "seed": seed,
                            "scaling": scaling, "model": model,
                            "comparison_track": track,
                            "representation": representation, "k": k,
                            "accuracy": accuracy_score(yte, pred),
                            "balanced_accuracy": balanced_accuracy_score(yte, pred),
                            "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
                            "neighbour_jaccard_vs_translation": (
                                1.0 if model == "translation_mean"
                                else mean_jaccard(baseline_neighbours, neighbours)
                            ),
                            **descriptors,
                        })

            for row in resource_estimates(n_retained, args.basis_bits):
                resource_rows.append({
                    "dataset": dataset_name, "seed": seed, **row
                })

            print(
                f"  Seed {seed + 1}/{args.seeds}: {len(y)} samples, "
                f"{n_original}->{n_retained} features after train-only variance filter"
            )

    results = pd.DataFrame(result_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    dataset_characteristics = pd.DataFrame(dataset_rows)
    equivalence = pd.DataFrame(equivalence_rows)
    resources = pd.DataFrame(resource_rows)
    hypotheses = pd.DataFrame(hypothesis_rows).drop_duplicates(
        subset=["dataset", "dataset_family", "theoretical_hypothesis"]
    )

    if results.empty:
        raise RuntimeError("No results were generated.")

    # Paired comparison against translation baseline (within same dataset/seed/
    # scaling/k). The comparison_track column is retained for filtering.
    baseline = results[results.model == "translation_mean"][[
        "dataset", "seed", "scaling", "k", "accuracy",
        "balanced_accuracy", "macro_f1",
    ]].rename(columns={
        "accuracy": "translation_accuracy",
        "balanced_accuracy": "translation_balanced_accuracy",
        "macro_f1": "translation_macro_f1",
    })
    alternatives = results[results.model != "translation_mean"].copy()
    paired = alternatives.merge(
        baseline, on=["dataset", "seed", "scaling", "k"],
        how="inner", validate="many_to_one",
    )
    paired["accuracy_difference_pp"] = 100 * (
        paired.accuracy - paired.translation_accuracy
    )
    paired["balanced_accuracy_difference_pp"] = 100 * (
        paired.balanced_accuracy - paired.translation_balanced_accuracy
    )
    paired["macro_f1_difference_pp"] = 100 * (
        paired.macro_f1 - paired.translation_macro_f1
    )

    summary = results.groupby(
        ["dataset", "comparison_track", "scaling", "model", "k"],
        as_index=False,
    ).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        neighbour_jaccard_mean=("neighbour_jaccard_vs_translation", "mean"),
        neighbour_jaccard_std=("neighbour_jaccard_vs_translation", "std"),
    )

    paired_summary = paired.groupby(
        ["dataset", "comparison_track", "scaling", "model", "k"],
        as_index=False,
    ).agg(
        mean_accuracy_difference_pp=("accuracy_difference_pp", "mean"),
        std_accuracy_difference_pp=("accuracy_difference_pp", "std"),
        mean_balanced_accuracy_difference_pp=("balanced_accuracy_difference_pp", "mean"),
        mean_macro_f1_difference_pp=("macro_f1_difference_pp", "mean"),
        wins=("accuracy_difference_pp", lambda x: int(np.sum(x > 1e-12))),
        ties=("accuracy_difference_pp", lambda x: int(np.sum(np.abs(x) <= 1e-12))),
        losses=("accuracy_difference_pp", lambda x: int(np.sum(x < -1e-12))),
        paired_runs=("accuracy_difference_pp", "count"),
    )

    kernel_summary = diagnostics.groupby(
        ["dataset", "comparison_track", "scaling", "model"],
        as_index=False,
    ).agg(
        similarity_mean=("similarity_mean", "mean"),
        similarity_std_mean=("similarity_std", "mean"),
        similarity_median=("similarity_median", "mean"),
        fraction_below_1e_12=("fraction_below_1e_12", "mean"),
        fraction_exact_zero=("fraction_exact_zero", "mean"),
        fraction_above_099=("fraction_above_099", "mean"),
        unique_similarity_fraction=("unique_similarity_fraction", "mean"),
    )

    dataset_ranking = results.groupby(
        ["dataset", "comparison_track", "model"], as_index=False
    ).agg(
        mean_accuracy=("accuracy", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_macro_f1=("macro_f1", "mean"),
    )
    dataset_ranking["accuracy_rank_within_track"] = dataset_ranking.groupby(
        ["dataset", "comparison_track"]
    )["mean_accuracy"].rank(method="min", ascending=False)

    outputs = {
        "encoding_results.csv": results,
        "encoding_paired_differences.csv": paired,
        "encoding_summary.csv": summary,
        "encoding_paired_summary.csv": paired_summary,
        "encoding_kernel_diagnostics.csv": diagnostics,
        "encoding_kernel_summary.csv": kernel_summary,
        "encoding_dataset_characteristics.csv": dataset_characteristics,
        "encoding_equivalence_checks.csv": equivalence,
        "encoding_dataset_ranking.csv": dataset_ranking,
        "encoding_resource_estimates.csv": resources,
        "encoding_hypothesis_map.csv": hypotheses,
    }
    for filename, frame in outputs.items():
        path = os.path.join(args.outdir, filename)
        frame.to_csv(path, index=False)
        print(f"Saved: {path}")

    print("\\n" + "=" * 78)
    print("REVISED ALTERNATIVE ENCODING STUDY — COMPLETED")
    print("=" * 78)
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}; k: {args.k_values}; scaling: {SCALING_METHODS}")
    print(f"Result rows: {len(results):,}; paired rows: {len(paired):,}")

    for track in ["common_fidelity", "encoding_specific"]:
        subset = results[results.comparison_track == track]
        if subset.empty:
            continue
        print(f"\\nMean accuracy by model — {track}:")
        print(subset.groupby("model").accuracy.mean().sort_values(ascending=False).round(4).to_string())

    print("\\nPhase/angle-product(pi) maximum absolute difference:",
          equivalence.max_absolute_difference.max())
    print("\\nInterpretation guardrails:")
    print("* Compare common_fidelity models with one another for representation comparisons.")
    print("* Treat encoding_specific models as a separate track.")
    print("* Theoretical dataset labels are hypotheses, not guaranteed winners.")
    print("* Resource CSV contains logical estimates, not measured circuit depth/runtime.")
    print("* No quantum circuits or finite-shot measurements were run.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled comparison of QKNN data encodings across diverse datasets."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        choices=DEFAULT_DATASETS)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--synthetic-samples", type=int, default=1200)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--basis-bits", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--outdir", type=str, default="results/alternative_encodings")
    args = parser.parse_args()

    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    if args.max_samples < 20 or args.synthetic_samples < 20:
        parser.error("--max-samples and --synthetic-samples must be >= 20")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    if any(k < 1 for k in args.k_values):
        parser.error("All k values must be >= 1")
    if any(b < 1 or b > 8 for b in args.basis_bits):
        parser.error("Basis bit depths must be between 1 and 8")

    args.datasets = list(dict.fromkeys(args.datasets))
    args.k_values = sorted(set(args.k_values))
    args.basis_bits = sorted(set(args.basis_bits))
    return args


if __name__ == "__main__":
    run_experiment(parse_args())
