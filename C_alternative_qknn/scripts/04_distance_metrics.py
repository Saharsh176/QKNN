#!/usr/bin/env python3
"""
04_distance_metrics.py
======================

Multi-seed validation of distance metrics for k-NN.

Continuous track (identical train-fitted MinMax [0,1] inputs):
    - Euclidean
    - Manhattan
    - Cosine

Quantised/discrete track:
    - Hamming similarity at 2, 4 and 8 uniform quantisation levels.
      Quantisation boundaries are fitted on training data only.

Main research question:
    Can measurable dataset characteristics help predict which metric
    performs best?

Design:
    - Same stratified train/test split for every metric within each seed.
    - All preprocessing is fitted on training data only.
    - Default seeds are 0,...,19 (20 independent repetitions).
    - Per-dataset paired comparisons against Euclidean.
    - Bootstrap confidence intervals over paired seed-level differences.
    - Wilcoxon signed-rank tests across seeds and Holm correction within
      each dataset/k family.
    - Dataset descriptors computed on ORIGINAL, unscaled training features.
    - Hamming is reported as a separate representation track, not as a
      directly equivalent continuous-metric comparison.
    - One Excel workbook, separate worksheet per result table.

No quantum circuits, finite shots, QRAM or quantum speedup are evaluated.

Example:
    python 04_distance_metrics.py --seeds 0 1 2 ... 19

Quick smoke test:
    python 04_distance_metrics.py --seeds 0 --k-values 1 3 \
        --outdir results/distance_metrics_smoke

Dependencies:
    numpy pandas scipy scikit-learn openpyxl
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon

from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    make_classification, make_moons, make_circles
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------
# 1. DATASETS
# ---------------------------------------------------------------------

def make_synthetic_datasets(seed, n=900):
    rng = np.random.default_rng(seed)
    data = {}

    # Directional clusters: direction is informative, radius is nuisance.
    y = rng.integers(0, 3, n)
    angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
    theta = angles[y] + rng.normal(0, 0.24, n)
    radius = 1.0 + rng.normal(0, 0.08, n)
    X = np.column_stack([
        radius*np.cos(theta), radius*np.sin(theta),
        rng.normal(0, 0.04, (n, 4))
    ])
    data["directional_clusters"] = (X, y)

    # Radial magnitude: radius carries class information.
    y = rng.integers(0, 3, n)
    theta = rng.uniform(0, 2*np.pi, n)
    radius = np.array([0.45, 1.0, 1.7])[y] + rng.normal(0, 0.08, n)
    X = np.column_stack([
        radius*np.cos(theta), radius*np.sin(theta),
        rng.normal(0, 0.025, (n, 3))
    ])
    data["radial_magnitude"] = (X, y)

    # Binary prototypes with class-specific bits and bit-flip noise.
    y = rng.integers(0, 3, n)
    prototypes = rng.integers(0, 2, (3, 24))
    prototypes[:, :3] = 0
    prototypes[np.arange(3), np.arange(3)] = 1
    X = prototypes[y].copy()
    X = np.logical_xor(X, rng.random(X.shape) < 0.10).astype(float)
    data["binary_hamming"] = (X, y)

    # Sparse block signal.
    y = rng.integers(0, 3, n)
    X = np.zeros((n, 40))
    blocks = [np.arange(0, 5), np.arange(10, 15), np.arange(20, 25)]
    for i, c in enumerate(y):
        X[i, blocks[c]] = rng.normal(1.0, 0.25, 5)
        noise_idx = rng.choice(40, 2, replace=False)
        X[i, noise_idx] += rng.normal(0, 0.15, 2)
    data["sparse_signal"] = (X, y)

    # Periodic features.
    y = rng.integers(0, 3, n)
    t = rng.uniform(0, 2*np.pi, n)
    phase = np.array([0, 2*np.pi/3, 4*np.pi/3])[y]
    X = np.column_stack([
        np.sin(t + phase), np.cos(t + phase),
        np.sin(2*t + phase), np.cos(2*t + phase),
        rng.normal(0, 0.08, n)
    ])
    data["periodic_features"] = (X, y)

    data["correlated_redundant"] = make_classification(
        n_samples=n, n_features=20, n_informative=5, n_redundant=10,
        n_classes=3, n_clusters_per_class=1, flip_y=0.04,
        random_state=seed
    )

    X, y = make_classification(
        n_samples=n, n_features=8, n_informative=5, n_redundant=1,
        n_classes=3, n_clusters_per_class=1, flip_y=0.04,
        random_state=seed
    )
    transform = np.diag([3.0, 0.45, 1.8, 0.7, 1.0, 1.0, 1.0, 1.0])
    data["anisotropic_blobs"] = (X @ transform, y)

    data["moons"] = make_moons(n_samples=n, noise=0.18, random_state=seed)
    data["circles"] = make_circles(
        n_samples=n, noise=0.10, factor=0.45, random_state=seed
    )

    X, y = make_classification(
        n_samples=n, n_features=80, n_informative=5, n_redundant=5,
        n_classes=3, n_clusters_per_class=1, class_sep=0.65,
        flip_y=0.08, random_state=seed
    )
    X += rng.normal(0, 1.5, X.shape)
    data["high_dim_weak_signal"] = (X, y)

    y = rng.integers(0, 3, n)
    centers = np.array([[-1.5, 0], [1.5, 0], [0, 1.5]])
    X = centers[y] + rng.standard_t(2.5, size=(n, 2))*0.35
    X = np.column_stack([X, rng.standard_t(2.5, size=(n, 5))*0.25])
    data["heavy_tailed"] = (X, y)

    y = rng.integers(0, 3, n)
    centers = np.array([[1, 1, 2, 1], [3, 2, 1, 3], [2, 4, 3, 2]])
    X = centers[y] + rng.choice(
        [-1, 0, 1], size=(n, 4), p=[0.12, 0.76, 0.12]
    )
    data["ordinal_discrete"] = (np.clip(X, 0, 5).astype(float), y)

    data["imbalanced"] = make_classification(
        n_samples=1000, n_features=16, n_informative=6, n_redundant=3,
        n_classes=3, n_clusters_per_class=1, weights=[0.78, 0.17, 0.05],
        class_sep=0.9, flip_y=0.03, random_state=seed
    )
    return data


def load_all_datasets(seed):
    result = {}
    for name, loader in [
        ("iris", load_iris), ("wine", load_wine),
        ("breast_cancer", load_breast_cancer), ("digits", load_digits)
    ]:
        ds = loader()
        result[name] = (ds.data.astype(float), ds.target.astype(int))
    result.update(make_synthetic_datasets(seed))
    return result


# ---------------------------------------------------------------------
# 2. PREPROCESSING AND DATASET DESCRIPTORS
# ---------------------------------------------------------------------

def remove_constant_features(X_train, X_test):
    keep = np.nanstd(X_train, axis=0) > 1e-12
    if not keep.any():
        keep[:] = True
    return X_train[:, keep], X_test[:, keep], keep


def effective_rank(X):
    """Participation-ratio effective rank of the centered feature matrix."""
    Xc = X - np.mean(X, axis=0, keepdims=True)
    singular = np.linalg.svd(Xc, compute_uv=False)
    eig = singular**2
    total = eig.sum()
    if total <= 1e-15:
        return 0.0
    return float(total**2 / np.sum(eig**2))


def dataset_descriptors(X_raw, y, name, seed):
    """Descriptors are computed before scaling, on training data only."""
    X = np.asarray(X_raw, dtype=float)
    variances = np.var(X, axis=0)
    nonconstant = variances > 1e-12
    Xv = X[:, nonconstant]
    variances = variances[nonconstant]

    if Xv.shape[1] > 1:
        corr = np.corrcoef(Xv, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        abs_corr = np.abs(corr[np.triu_indices_from(corr, k=1)])
        mean_abs_corr = float(abs_corr.mean()) if abs_corr.size else 0.0
    else:
        mean_abs_corr = 0.0

    _, class_counts = np.unique(y, return_counts=True)
    row_norms = np.linalg.norm(X, axis=1)
    feature_mean_abs = np.mean(np.abs(X), axis=0)
    feature_std = np.std(X, axis=0)
    cv = feature_std / np.maximum(feature_mean_abs, 1e-12)

    # A centroid-separation proxy: between-class centroid dispersion
    # divided by pooled within-class RMS spread.
    overall = X.mean(axis=0)
    centroids = []
    within_sse = 0.0
    within_n = 0
    for label in np.unique(y):
        Xi = X[y == label]
        cent = Xi.mean(axis=0)
        centroids.append(cent)
        within_sse += float(np.sum((Xi - cent)**2))
        within_n += Xi.shape[0]
    between = float(np.mean([
        np.sum((c - overall)**2) for c in centroids
    ]))
    within_rms = np.sqrt(within_sse / max(within_n * X.shape[1], 1))
    centroid_separation = np.sqrt(between) / max(within_rms, 1e-12)

    return {
        "dataset": name,
        "seed": seed,
        "n_train": len(y),
        "n_features_original": X.shape[1],
        "n_features_nonconstant": int(nonconstant.sum()),
        "n_classes": len(class_counts),
        "class_imbalance_ratio": float(class_counts.max()/class_counts.min()),
        "raw_zero_fraction": float(np.mean(np.isclose(X, 0.0, atol=1e-12))),
        "raw_sparsity_fraction": float(np.mean(np.isclose(X, 0.0, atol=1e-12))),
        "mean_abs_feature_value": float(np.mean(np.abs(X))),
        "mean_row_norm": float(row_norms.mean()),
        "std_row_norm": float(row_norms.std()),
        "row_norm_cv": float(row_norms.std()/max(row_norms.mean(), 1e-12)),
        "feature_variance_mean": float(np.mean(variances)) if len(variances) else 0.0,
        "feature_variance_cv": float(np.std(variances)/max(np.mean(variances), 1e-12)) if len(variances) else 0.0,
        "feature_cv_mean": float(np.mean(cv)),
        "mean_abs_feature_correlation": mean_abs_corr,
        "effective_rank": effective_rank(Xv) if Xv.size else 0.0,
        "effective_rank_fraction": (
            effective_rank(Xv)/Xv.shape[1] if Xv.shape[1] else 0.0
        ),
        "centroid_separation_proxy": float(centroid_separation)
    }


# ---------------------------------------------------------------------
# 3. PAIRWISE METRICS (CHUNKED TO LIMIT MEMORY)
# ---------------------------------------------------------------------

def pairwise_metric(A, B, metric, chunk_size=96):
    out = np.empty((len(A), len(B)), dtype=float)

    if metric == "euclidean":
        bnorm = np.sum(B*B, axis=1)[None, :]
        for start in range(0, len(A), chunk_size):
            a = A[start:start+chunk_size]
            d2 = np.sum(a*a, axis=1)[:, None] + bnorm - 2*a@B.T
            out[start:start+len(a)] = np.sqrt(np.maximum(d2, 0.0))
        return out

    if metric == "manhattan":
        for start in range(0, len(A), chunk_size):
            a = A[start:start+chunk_size]
            out[start:start+len(a)] = np.abs(
                a[:, None, :] - B[None, :, :]
            ).sum(axis=2)
        return out

    if metric == "cosine":
        bnorm = np.linalg.norm(B, axis=1)
        for start in range(0, len(A), chunk_size):
            a = A[start:start+chunk_size]
            anorm = np.linalg.norm(a, axis=1)
            den = anorm[:, None]*bnorm[None, :]
            sim = np.divide(
                a@B.T, den, out=np.zeros((len(a), len(B))),
                where=den > 1e-12
            )
            both_zero = (anorm[:, None] <= 1e-12) & (bnorm[None, :] <= 1e-12)
            sim[both_zero] = 1.0
            out[start:start+len(a)] = np.clip(sim, -1.0, 1.0)
        return out

    raise ValueError(f"Unknown metric: {metric}")


def median_train_distance(X_train, metric):
    # Subsample at most 500 rows for a stable, bounded-cost scale estimate.
    if len(X_train) > 500:
        idx = np.linspace(0, len(X_train)-1, 500).astype(int)
        sample = X_train[idx]
    else:
        sample = X_train
    D = pairwise_metric(sample, sample, metric)
    vals = D[np.triu_indices(len(sample), k=1)]
    vals = vals[np.isfinite(vals) & (vals > 1e-12)]
    return float(np.median(vals)) if vals.size else 1.0


def quantise_train_test(X_train, X_test, levels):
    lo = X_train.min(axis=0)
    hi = X_train.max(axis=0)
    span = hi-lo
    span[span < 1e-12] = 1.0

    def apply(X):
        z = np.clip((X-lo)/span, 0.0, 1.0)
        q = np.floor(z*levels).astype(np.int16)
        return np.minimum(q, levels-1)

    return apply(X_train), apply(X_test)


def hamming_similarity(A, B, chunk_size=96):
    out = np.empty((len(A), len(B)), dtype=float)
    for start in range(0, len(A), chunk_size):
        a = A[start:start+chunk_size]
        out[start:start+len(a)] = 1.0 - np.mean(
            a[:, None, :] != B[None, :, :], axis=2
        )
    return out


# ---------------------------------------------------------------------
# 4. NEIGHBOURS, VOTING, DIAGNOSTICS
# ---------------------------------------------------------------------

def top_k(S, k):
    # Stable deterministic tie handling. Tie frequency is reported separately.
    return np.argsort(-S, axis=1, kind="stable")[:, :k]


def predict(y_train, neighbours):
    pred = np.empty(len(neighbours), dtype=int)
    for i, ids in enumerate(neighbours):
        labels, counts = np.unique(y_train[ids], return_counts=True)
        pred[i] = labels[counts == counts.max()].min()
    return pred


def jaccard_rows(A, B):
    vals = []
    for a, b in zip(A, B):
        sa, sb = set(a.tolist()), set(b.tolist())
        vals.append(len(sa & sb)/len(sa | sb) if sa | sb else 1.0)
    return float(np.mean(vals))


def diagnostics(S):
    x = S.ravel()
    return {
        "similarity_min": float(x.min()),
        "similarity_mean": float(x.mean()),
        "similarity_std": float(x.std()),
        "similarity_max": float(x.max()),
        "fraction_near_zero": float(np.mean(x <= 1e-12)),
        "fraction_near_one": float(np.mean(x >= 1-1e-12)),
        "unique_similarity_fraction": float(
            len(np.unique(np.round(x, 6)))/max(len(x), 1)
        )
    }


# ---------------------------------------------------------------------
# 5. STATISTICS
# ---------------------------------------------------------------------

def bootstrap_ci(values, n_boot=4000, seed=2026):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).astype(float))


def safe_wilcoxon(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or np.allclose(x, 0):
        return np.nan
    try:
        return float(wilcoxon(x, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def holm_adjust(pvalues):
    """Holm family-wise correction; NaNs remain NaN."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.where(np.isfinite(p))[0]
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m-rank)*p[idx])
        running = max(running, adjusted)
        out[idx] = running
    return out


# ---------------------------------------------------------------------
# 6. EXPERIMENT
# ---------------------------------------------------------------------

def run_experiment(args):
    result_rows, descriptor_rows, diagnostic_rows = [], [], []
    paired_rows, equivalence_rows = [], []

    for seed in args.seeds:
        print(f"\nSeed {seed}")
        for dataset_name, (X, y) in load_all_datasets(seed).items():
            Xtr_raw, Xte_raw, ytr, yte = train_test_split(
                X, y, test_size=args.test_size, random_state=seed,
                stratify=y
            )

            # Describe ORIGINAL features before scaling or quantisation.
            descriptor_rows.append(
                dataset_descriptors(Xtr_raw, ytr, dataset_name, seed)
            )

            Xtr_raw, Xte_raw, keep = remove_constant_features(
                Xtr_raw, Xte_raw
            )

            scaler = MinMaxScaler()
            Xtr = np.clip(scaler.fit_transform(Xtr_raw), 0.0, 1.0)
            Xte = np.clip(scaler.transform(Xte_raw), 0.0, 1.0)

            method_similarities = {}

            # Continuous track: same Xtr/Xte for all three metrics.
            for metric in ("euclidean", "manhattan", "cosine"):
                raw = pairwise_metric(Xte, Xtr, metric)
                if metric == "cosine":
                    S = np.clip((raw+1.0)/2.0, 0.0, 1.0)
                    scale = 1.0
                else:
                    scale = median_train_distance(Xtr, metric)
                    S = np.exp(-raw/max(scale, 1e-12))
                method_similarities[metric] = (
                    S, "continuous", 0, scale
                )

            # Discrete track: explicit train-fitted quantisation.
            for levels in args.hamming_levels:
                Qtr, Qte = quantise_train_test(Xtr, Xte, levels)
                S = hamming_similarity(Qte, Qtr)
                method_similarities[f"hamming_{levels}levels"] = (
                    S, "quantised_discrete", levels, 1.0
                )
                equivalence_rows.append({
                    "dataset": dataset_name, "seed": seed,
                    "method": f"hamming_{levels}levels",
                    "range_check_passed": bool(
                        S.min() >= -1e-12 and S.max() <= 1+1e-12
                    ),
                    "quantisation_levels": levels
                })

            neighbour_cache = {}
            for method, (S, track, levels, scale) in method_similarities.items():
                diag = diagnostics(S)
                diagnostic_rows.append({
                    "dataset": dataset_name, "seed": seed,
                    "method": method, "track": track,
                    "quantisation_levels": levels, **diag
                })

                for k in args.k_values:
                    if k > len(ytr):
                        continue
                    ids = top_k(S, k)
                    neighbour_cache[(method, k)] = ids
                    pred = predict(ytr, ids)

                    result_rows.append({
                        "dataset": dataset_name, "seed": seed,
                        "method": method, "track": track,
                        "quantisation_levels": levels, "k": k,
                        "accuracy": accuracy_score(yte, pred),
                        "balanced_accuracy": balanced_accuracy_score(yte, pred),
                        "macro_f1": f1_score(
                            yte, pred, average="macro", zero_division=0
                        ),
                        "distance_scale": scale,
                        "n_train": len(ytr), "n_test": len(yte),
                        "n_features": Xtr.shape[1],
                        **diag
                    })

            # Paired neighbour-set comparisons to Euclidean.
            for k in args.k_values:
                ref = neighbour_cache.get(("euclidean", k))
                if ref is None:
                    continue
                for method in method_similarities:
                    if method == "euclidean":
                        continue
                    paired_rows.append({
                        "dataset": dataset_name, "seed": seed, "k": k,
                        "reference_method": "euclidean",
                        "comparison_method": method,
                        "comparison_track": method_similarities[method][1],
                        "neighbour_jaccard": jaccard_rows(
                            ref, neighbour_cache[(method, k)]
                        )
                    })

    results = pd.DataFrame(result_rows)
    descriptors = pd.DataFrame(descriptor_rows)
    diagnostics_df = pd.DataFrame(diagnostic_rows)
    paired = pd.DataFrame(paired_rows)
    equivalence = pd.DataFrame(equivalence_rows)

    # Overall summary by track/method/k.
    metric_summary = (
        results.groupby(["track", "method", "k"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_near_zero=("fraction_near_zero", "mean"),
            mean_near_one=("fraction_near_one", "mean"),
            observations=("accuracy", "count")
        )
    )

    # Dataset-level mean and 95% CI over independent seeds.
    dataset_summary_rows = []
    for (ds, track, method, k), group in results.groupby(
        ["dataset", "track", "method", "k"]
    ):
        lo, hi = bootstrap_ci(group["accuracy"].to_numpy(),
                              n_boot=args.bootstrap_samples,
                              seed=args.bootstrap_seed)
        dataset_summary_rows.append({
            "dataset": ds, "track": track, "method": method, "k": k,
            "mean_accuracy": group["accuracy"].mean(),
            "std_accuracy": group["accuracy"].std(),
            "accuracy_ci95_low": lo,
            "accuracy_ci95_high": hi,
            "mean_balanced_accuracy": group["balanced_accuracy"].mean(),
            "mean_macro_f1": group["macro_f1"].mean(),
            "n_seeds": group["seed"].nunique()
        })
    dataset_summary = pd.DataFrame(dataset_summary_rows)

    # Paired seed-level accuracy differences vs Euclidean, separately
    # for each dataset and k. Positive means comparison metric is better.
    eu = results[results.method == "euclidean"][
        ["dataset", "seed", "k", "accuracy"]
    ].rename(columns={"accuracy": "euclidean_accuracy"})
    comp = results[results.method != "euclidean"].merge(
        eu, on=["dataset", "seed", "k"], how="inner"
    )
    comp["accuracy_difference"] = (
        comp["accuracy"] - comp["euclidean_accuracy"]
    )

    paired_stats_rows = []
    for (ds, track, method, k), group in comp.groupby(
        ["dataset", "track", "method", "k"]
    ):
        vals = group["accuracy_difference"].to_numpy()
        lo, hi = bootstrap_ci(
            vals, n_boot=args.bootstrap_samples,
            seed=args.bootstrap_seed + int(k)
        )
        paired_stats_rows.append({
            "dataset": ds, "track": track, "method": method, "k": k,
            "mean_accuracy_difference": float(np.mean(vals)),
            "difference_ci95_low": lo,
            "difference_ci95_high": hi,
            "std_paired_difference": float(np.std(vals, ddof=1)) if len(vals)>1 else np.nan,
            "p_value_wilcoxon": safe_wilcoxon(vals),
            "n_seeds": len(vals)
        })
    paired_differences = pd.DataFrame(paired_stats_rows)

    if not paired_differences.empty:
        paired_differences["p_value_holm"] = np.nan
        for _, idx in paired_differences.groupby(
            ["dataset", "k"]
        ).groups.items():
            locs = list(idx)
            paired_differences.loc[locs, "p_value_holm"] = holm_adjust(
                paired_differences.loc[locs, "p_value_wilcoxon"].to_numpy()
            )
        paired_differences["significant_holm_05"] = (
            paired_differences["p_value_holm"] < 0.05
        )

    paired_summary = (
        paired.groupby(
            ["comparison_track", "comparison_method", "k"],
            as_index=False
        )
        .agg(
            mean_neighbour_jaccard=("neighbour_jaccard", "mean"),
            std_neighbour_jaccard=("neighbour_jaccard", "std"),
            observations=("neighbour_jaccard", "count")
        )
    )

    # Winners are selected within dataset/track/k by mean accuracy.
    rankings = (
        dataset_summary.sort_values(
            ["dataset", "track", "k", "mean_accuracy"],
            ascending=[True, True, True, False],
            kind="stable"
        )
        .groupby(["dataset", "track", "k"], as_index=False)
        .first()
        .rename(columns={"method": "top_method"})
    )

    win_counts = (
        rankings.groupby(["track", "top_method"], as_index=False)
        .size().rename(columns={"size": "dataset_k_wins"})
    )

    # Average descriptors per dataset, then associate with metric winners.
    descriptor_means = (
        descriptors.groupby("dataset", as_index=False)
        .mean(numeric_only=True)
    )
    winner_descriptors = rankings.merge(
        descriptor_means, on="dataset", how="left",
        suffixes=("", "_descriptor")
    )

    methodology = pd.DataFrame([
        ["Seeds", f"{len(args.seeds)} seeds: {args.seeds}"],
        ["Split", "Stratified train/test split; identical within each dataset/seed."],
        ["Scaling", "MinMax [0,1], fitted on training data only."],
        ["Continuous fairness", "Euclidean, Manhattan and cosine receive identical scaled features."],
        ["Euclidean/Manhattan", "Exact distances; exp(-d/median training distance) is strictly monotonic."],
        ["Cosine", "Cosine similarity mapped to [0,1] via (cos+1)/2."],
        ["Hamming", "Fraction of matching positions after train-fitted uniform quantisation."],
        ["Hamming interpretation", "Separate discrete representation track; not a pure metric-only comparison."],
        ["Descriptors", "Computed on original, unscaled training features before MinMax transformation."],
        ["Sparsity", "Raw zero fraction measured before scaling; zero-valued observations are not confused with scaled minima."],
        ["Effective rank", "Participation ratio of singular-value spectrum of centered training matrix."],
        ["Class separation", "Centroid-separation proxy; descriptive, not a formal Bayes-overlap estimate."],
        ["Inference", "Exact classical top-k; deterministic stable tie ordering; majority vote."],
        ["Statistics", "Paired seed-level differences, percentile bootstrap CI, Wilcoxon signed-rank and Holm adjustment."],
        ["Multiplicity", "Holm correction is applied within each dataset/k family across comparison methods."],
        ["Quantum scope", "No quantum circuits, state preparation, shots, QRAM or quantum speedup measured."]
    ], columns=["item", "description"])

    return {
        "results": results,
        "metric_summary": metric_summary,
        "dataset_summary": dataset_summary,
        "paired_differences": paired_differences,
        "paired_summary": paired_summary,
        "dataset_characteristics": descriptors,
        "dataset_rankings": rankings,
        "winner_counts": win_counts,
        "winner_descriptors": winner_descriptors,
        "kernel_diagnostics": diagnostics_df,
        "equivalence_checks": equivalence,
        "methodology": methodology
    }


# ---------------------------------------------------------------------
# 7. EXCEL EXPORT
# ---------------------------------------------------------------------

def save_workbook(tables, outdir, export_csv):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "04_distance_metrics_results.xlsx"

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            sheet = name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            ws.freeze_panes = "A2"
            if ws.max_row > 1:
                ws.auto_filter.ref = ws.dimensions

            for col_cells in ws.columns:
                letter = col_cells[0].column_letter
                sample = col_cells[:min(ws.max_row, 500)]
                width = max(
                    (len(str(c.value)) if c.value is not None else 0)
                    for c in sample
                )
                ws.column_dimensions[letter].width = min(max(width+2, 12), 48)

            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, (float, np.floating)):
                        cell.number_format = "0.0000"

    if export_csv:
        for name, df in tables.items():
            df.to_csv(outdir / f"{name}.csv", index=False)

    return path


# ---------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Study 04: multi-seed distance metric validation."
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 9])
    p.add_argument("--hamming-levels", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--bootstrap-samples", type=int, default=4000)
    p.add_argument("--bootstrap-seed", type=int, default=2026)
    p.add_argument(
        "--outdir", type=Path,
        default=Path("results/distance_metrics_v2")
    )
    p.add_argument("--csv", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.seeds:
        raise ValueError("Provide at least one seed.")
    if any(k < 1 for k in args.k_values):
        raise ValueError("All k values must be positive.")
    if any(q < 2 for q in args.hamming_levels):
        raise ValueError("Hamming quantisation levels must be >= 2.")
    if not 0.05 <= args.test_size <= 0.5:
        raise ValueError("--test-size must be between 0.05 and 0.5.")
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples should be at least 100.")

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("="*78)
    print("QKNN STUDY 04 — MULTI-SEED DISTANCE METRIC VALIDATION")
    print("="*78)
    print(f"Seeds: {args.seeds}")
    print(f"k values: {args.k_values}")
    print(f"Hamming levels: {args.hamming_levels}")
    print(f"Bootstrap samples: {args.bootstrap_samples}")
    print("Dataset descriptors: original, unscaled training features")

    tables = run_experiment(args)
    workbook = save_workbook(tables, args.outdir, args.csv)

    print("\nOVERALL MEAN ACCURACY")
    print("-"*78)
    print(tables["metric_summary"].to_string(index=False))

    print("\nDATASET/K WINNER COUNTS")
    print("-"*78)
    print(tables["winner_counts"].to_string(index=False))

    print("\nOUTPUT")
    print(workbook.resolve())
    if args.csv:
        print(f"CSV exports: {args.outdir.resolve()}")
    print("\nCompleted. This is a classical metric study, not a quantum speedup test.")


if __name__ == "__main__":
    main()
