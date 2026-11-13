#!/usr/bin/env python3
"""
04_distance_metrics_v3.py
=========================

Study 04 v3: do dataset characteristics predict which distance metric works best?

CONTINUOUS TRACK
    Euclidean, Manhattan, cosine distance/similarity.
    All receive exactly the same train-fitted MinMax [0, 1] representation.

DISCRETE TRACK
    Hamming similarity after train-fitted uniform quantisation at 2/4/8 levels.
    This is a quantisation + Hamming pipeline, not a pure metric-only comparison.

VALIDATION
    Repeated stratified K-fold CV (default 5 folds x 3 repeats).
    Every method uses identical folds within a dataset.
    Corrected resampled t-tests account approximately for overlap between
    repeated-CV training sets (Nadeau-Bengio correction).
    Holm correction is applied within each dataset/k comparison family.

DATASET-TO-METRIC EXPERIMENT
    Dataset descriptors are computed on each training fold only.
    Descriptor values and metric outcomes are averaged per dataset.
    A low-complexity Ridge model predicts each continuous metric's accuracy
    advantage over Euclidean.
    Leave-one-dataset-out validation evaluates selection on an unseen dataset.
    With only 17 datasets, this is exploratory evidence, not a deployable
    meta-learning system.

OUTPUT
    One Excel workbook with separate worksheets for every result table.
    Optional CSV export with --csv.

SCOPE
    Classical exact k-NN only. No Qiskit circuits, finite shots, QRAM,
    state-preparation cost, gate counts, or quantum speedup are measured.

Dependencies:
    numpy pandas scipy scikit-learn openpyxl

Example:
    python 04_distance_metrics_v3.py

Quick smoke test:
    python 04_distance_metrics_v3.py --folds 3 --repeats 1 \
        --k-values 1 3 --datasets iris moons --outdir results/04_v3_smoke
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import t as student_t, spearmanr

from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    make_classification, make_moons, make_circles, make_blobs
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------------
# 1. DATASETS: 4 REAL + 13 SYNTHETIC
# ---------------------------------------------------------------------

REAL_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}

SYNTHETIC_NAMES = [
    "directional_clusters", "radial_magnitude", "binary_hamming",
    "sparse_signal", "periodic_features", "correlated_redundant",
    "anisotropic_blobs", "moons", "circles", "high_dim_weak_signal",
    "heavy_tailed", "ordinal_discrete", "imbalanced",
]


def make_synthetic_datasets(seed: int, n: int = 900):
    rng = np.random.default_rng(seed)
    data = {}

    # Direction is informative; radius is nuisance variation.
    y = rng.integers(0, 3, n)
    angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
    theta = angles[y] + rng.normal(0, 0.24, n)
    radius = 1.0 + rng.normal(0, 0.08, n)
    X = np.column_stack([
        radius*np.cos(theta), radius*np.sin(theta),
        rng.normal(0, 0.04, (n, 4))
    ])
    data["directional_clusters"] = (X, y)

    # Magnitude/radius carries the class signal.
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
    for i, label in enumerate(y):
        X[i, blocks[label]] = rng.normal(1.0, 0.25, 5)
        noise_idx = rng.choice(40, 2, replace=False)
        X[i, noise_idx] += rng.normal(0, 0.15, 2)
    data["sparse_signal"] = (X, y)

    # Periodic features.
    y = rng.integers(0, 3, n)
    phase_input = rng.uniform(0, 2*np.pi, n)
    phase = np.array([0, 2*np.pi/3, 4*np.pi/3])[y]
    X = np.column_stack([
        np.sin(phase_input + phase), np.cos(phase_input + phase),
        np.sin(2*phase_input + phase), np.cos(2*phase_input + phase),
        rng.normal(0, 0.08, n)
    ])
    data["periodic_features"] = (X, y)

    data["correlated_redundant"] = make_classification(
        n_samples=n, n_features=20, n_informative=5, n_redundant=10,
        n_classes=3, n_clusters_per_class=1, flip_y=0.04,
        random_state=seed
    )

    # Genuine anisotropic Gaussian blobs (not a transformed classification set).
    X, y = make_blobs(
        n_samples=n, centers=3, n_features=8, cluster_std=1.0,
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


def load_all_datasets(seed: int, n_synthetic: int):
    result = {}
    for name, loader in REAL_LOADERS.items():
        ds = loader()
        result[name] = (ds.data.astype(float), ds.target.astype(int))
    result.update(make_synthetic_datasets(seed, n=n_synthetic))
    return result


# ---------------------------------------------------------------------
# 2. DESCRIPTORS (TRAIN FOLD ONLY, ORIGINAL FEATURE SCALE)
# ---------------------------------------------------------------------

def effective_rank(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(Xc, compute_uv=False)
    eig = singular**2
    total = eig.sum()
    return float(total**2 / np.sum(eig**2)) if total > 1e-15 else 0.0


def dataset_descriptors(X, y, dataset):
    X = np.asarray(X, dtype=float)
    variances = np.var(X, axis=0)
    keep = variances > 1e-12
    Xv = X[:, keep]
    vv = variances[keep]

    if Xv.shape[1] > 1:
        corr = np.corrcoef(Xv, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        tri = np.abs(corr[np.triu_indices_from(corr, k=1)])
        mean_abs_corr = float(tri.mean()) if tri.size else 0.0
    else:
        mean_abs_corr = 0.0

    _, counts = np.unique(y, return_counts=True)
    row_norms = np.linalg.norm(X, axis=1)
    abs_mean = np.mean(np.abs(X), axis=0)
    feature_cv = np.std(X, axis=0) / np.maximum(abs_mean, 1e-12)

    overall = X.mean(axis=0)
    centroids = []
    within_sse = 0.0
    for label in np.unique(y):
        Xi = X[y == label]
        centroid = Xi.mean(axis=0)
        centroids.append(centroid)
        within_sse += float(np.sum((Xi-centroid)**2))
    between = float(np.mean([np.sum((c-overall)**2) for c in centroids]))
    within_rms = np.sqrt(within_sse / max(X.shape[0]*X.shape[1], 1))
    separation = np.sqrt(between) / max(within_rms, 1e-12)

    return {
        "dataset": dataset,
        "n_train": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_features_nonconstant": int(keep.sum()),
        "n_classes": int(len(counts)),
        "class_imbalance_ratio": float(counts.max()/counts.min()),
        "raw_sparsity_fraction": float(np.mean(np.isclose(X, 0.0, atol=1e-12))),
        "row_norm_cv": float(row_norms.std()/max(row_norms.mean(), 1e-12)),
        "feature_variance_cv": float(np.std(vv)/max(np.mean(vv), 1e-12)) if len(vv) else 0.0,
        "feature_cv_mean": float(np.mean(feature_cv)),
        "mean_abs_feature_correlation": mean_abs_corr,
        "effective_rank": effective_rank(Xv) if Xv.size else 0.0,
        "effective_rank_fraction": effective_rank(Xv)/Xv.shape[1] if Xv.shape[1] else 0.0,
        "centroid_separation_proxy": float(separation),
    }


# ---------------------------------------------------------------------
# 3. DISTANCE / SIMILARITY MATRICES
# ---------------------------------------------------------------------

def remove_constant_features(X_train, X_test):
    keep = np.nanstd(X_train, axis=0) > 1e-12
    if not keep.any():
        keep[:] = True
    return X_train[:, keep], X_test[:, keep]


def pairwise_metric(A, B, metric, chunk_size=64):
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

    raise ValueError(metric)


def median_train_distance(X, metric):
    # Deterministic subsample bounds the cost of scale estimation.
    if len(X) > 400:
        idx = np.linspace(0, len(X)-1, 400).astype(int)
        X = X[idx]
    D = pairwise_metric(X, X, metric)
    vals = D[np.triu_indices(len(X), k=1)]
    vals = vals[np.isfinite(vals) & (vals > 1e-12)]
    return float(np.median(vals)) if vals.size else 1.0


def quantise_train_test(X_train, X_test, levels):
    lo, hi = X_train.min(axis=0), X_train.max(axis=0)
    span = hi-lo
    span[span < 1e-12] = 1.0

    def apply(X):
        z = np.clip((X-lo)/span, 0.0, 1.0)
        q = np.floor(z*levels).astype(np.int16)
        return np.minimum(q, levels-1)
    return apply(X_train), apply(X_test)


def hamming_similarity(A, B, chunk_size=64):
    out = np.empty((len(A), len(B)), dtype=float)
    for start in range(0, len(A), chunk_size):
        a = A[start:start+chunk_size]
        out[start:start+len(a)] = 1.0 - np.mean(
            a[:, None, :] != B[None, :, :], axis=2
        )
    return out


def top_k(similarity, k):
    return np.argsort(-similarity, axis=1, kind="stable")[:, :k]


def predict_labels(y_train, neighbour_ids):
    pred = np.empty(len(neighbour_ids), dtype=int)
    for i, ids in enumerate(neighbour_ids):
        labels, counts = np.unique(y_train[ids], return_counts=True)
        pred[i] = labels[counts == counts.max()].min()
    return pred


def neighbour_jaccard(A, B):
    scores = []
    for a, b in zip(A, B):
        sa, sb = set(a.tolist()), set(b.tolist())
        scores.append(len(sa & sb)/len(sa | sb) if sa | sb else 1.0)
    return float(np.mean(scores))


def similarity_diagnostics(S):
    flat = S.ravel()
    return {
        "similarity_min": float(flat.min()),
        "similarity_mean": float(flat.mean()),
        "similarity_std": float(flat.std()),
        "similarity_max": float(flat.max()),
        "fraction_near_zero": float(np.mean(flat <= 1e-12)),
        "fraction_near_one": float(np.mean(flat >= 1-1e-12)),
        "unique_similarity_fraction": float(
            len(np.unique(np.round(flat, 6)))/max(len(flat), 1)
        ),
    }


# ---------------------------------------------------------------------
# 4. CORRECTED RESAMPLED TEST + MULTIPLE-TEST CORRECTION
# ---------------------------------------------------------------------

def corrected_resampled_t(values, test_train_ratio):
    """
    Nadeau-Bengio corrected resampled t-test.

    For n paired fold-level differences d_i:
      SE = sqrt((1/n + n_test/n_train) * sample_variance(d))

    The correction inflates the naive standard error to reflect dependence
    induced by overlapping training sets. It is an approximate procedure;
    p-values should still be treated cautiously, especially with few datasets.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan

    mean = float(np.mean(x))
    variance = float(np.var(x, ddof=1))
    correction = (1.0/n) + float(test_train_ratio)
    se = np.sqrt(max(correction*variance, 0.0))
    if se <= 1e-15:
        return mean, se, np.nan, 1.0 if abs(mean) <= 1e-15 else 0.0

    statistic = mean/se
    pvalue = float(2*student_t.sf(abs(statistic), df=n-1))
    return mean, float(se), float(statistic), pvalue


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.where(np.isfinite(p))[0]
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m-rank)*p[idx]))
        out[idx] = running
    return out


def bootstrap_ci(values, n_boot, seed):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).astype(float))


# ---------------------------------------------------------------------
# 5. EXPERIMENT
# ---------------------------------------------------------------------

def run_experiment(args):
    result_rows, descriptor_rows, diagnostic_rows = [], [], []
    split_rows = []

    datasets = load_all_datasets(args.data_seed, args.n_synthetic)
    if args.datasets:
        missing = sorted(set(args.datasets)-set(datasets))
        if missing:
            raise ValueError(f"Unknown dataset names: {missing}")
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}

    splitter = RepeatedStratifiedKFold(
        n_splits=args.folds, n_repeats=args.repeats,
        random_state=args.split_seed
    )

    for dataset_name, (X, y) in datasets.items():
        print(f"\nDataset: {dataset_name} ({len(y)} rows, {X.shape[1]} features)")
        split_iter = list(splitter.split(X, y))

        for fold_id, (train_idx, test_idx) in enumerate(split_iter):
            repeat_id = fold_id // args.folds
            fold_in_repeat = fold_id % args.folds
            Xtr_raw, Xte_raw = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            descriptor_rows.append({
                "dataset": dataset_name,
                "fold_id": fold_id,
                "repeat": repeat_id,
                "fold": fold_in_repeat,
                **dataset_descriptors(Xtr_raw, ytr, dataset_name)
            })

            Xtr_raw, Xte_raw = remove_constant_features(Xtr_raw, Xte_raw)
            scaler = MinMaxScaler()
            Xtr = np.clip(scaler.fit_transform(Xtr_raw), 0.0, 1.0)
            Xte = np.clip(scaler.transform(Xte_raw), 0.0, 1.0)

            similarities = {}
            for metric in ("euclidean", "manhattan", "cosine"):
                raw = pairwise_metric(Xte, Xtr, metric)
                if metric == "cosine":
                    S = np.clip((raw+1.0)/2.0, 0.0, 1.0)
                    scale = 1.0
                else:
                    scale = median_train_distance(Xtr, metric)
                    S = np.exp(-raw/max(scale, 1e-12))
                similarities[metric] = (S, "continuous", 0, scale)

            for levels in args.hamming_levels:
                Qtr, Qte = quantise_train_test(Xtr, Xte, levels)
                S = hamming_similarity(Qte, Qtr)
                method = f"hamming_{levels}levels"
                similarities[method] = (S, "quantised_discrete", levels, 1.0)

            neighbour_cache = {}
            for method, (S, track, levels, scale) in similarities.items():
                diagnostic_rows.append({
                    "dataset": dataset_name, "fold_id": fold_id,
                    "repeat": repeat_id, "fold": fold_in_repeat,
                    "method": method, "track": track,
                    "quantisation_levels": levels,
                    **similarity_diagnostics(S)
                })

                for k in args.k_values:
                    if k > len(ytr):
                        continue
                    ids = top_k(S, k)
                    neighbour_cache[(method, k)] = ids
                    pred = predict_labels(ytr, ids)
                    result_rows.append({
                        "dataset": dataset_name, "fold_id": fold_id,
                        "repeat": repeat_id, "fold": fold_in_repeat,
                        "n_train": len(train_idx), "n_test": len(test_idx),
                        "method": method, "track": track,
                        "quantisation_levels": levels, "k": k,
                        "accuracy": accuracy_score(yte, pred),
                        "balanced_accuracy": balanced_accuracy_score(yte, pred),
                        "macro_f1": f1_score(
                            yte, pred, average="macro", zero_division=0
                        ),
                    })

            for k in args.k_values:
                base = neighbour_cache.get(("euclidean", k))
                if base is None:
                    continue
                for method, (S, track, levels, scale) in similarities.items():
                    if method == "euclidean":
                        continue
                    ids = neighbour_cache[(method, k)]
                    split_rows.append({
                        "dataset": dataset_name, "fold_id": fold_id,
                        "repeat": repeat_id, "fold": fold_in_repeat,
                        "method": method, "track": track,
                        "quantisation_levels": levels, "k": k,
                        "neighbour_jaccard_vs_euclidean": neighbour_jaccard(ids, base)
                    })

    return (
        pd.DataFrame(result_rows),
        pd.DataFrame(descriptor_rows),
        pd.DataFrame(diagnostic_rows),
        pd.DataFrame(split_rows),
    )


# ---------------------------------------------------------------------
# 6. SUMMARIES, INFERENCE, AND DESCRIPTOR-TO-METRIC MODEL
# ---------------------------------------------------------------------

def build_tables(results, descriptors, diagnostics, neighbour_rows, args):
    metric_summary = (
        results.groupby(["track", "method", "k"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            sd_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            n_dataset_folds=("accuracy", "size"),
        )
    )

    fold_summary = (
        results.groupby(["dataset", "track", "method", "k"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            sd_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            n_folds=("accuracy", "size"),
        )
    )

    # Paired fold-level differences; all methods share the same folds.
    wide = results.pivot_table(
        index=["dataset", "fold_id", "repeat", "fold", "k"],
        columns="method", values="accuracy"
    ).reset_index()
    paired_rows = []
    for _, row in wide.iterrows():
        for method in ("manhattan", "cosine"):
            if method in wide.columns and pd.notna(row.get(method)) and pd.notna(row.get("euclidean")):
                paired_rows.append({
                    "dataset": row["dataset"], "fold_id": row["fold_id"],
                    "repeat": row["repeat"], "fold": row["fold"],
                    "k": row["k"], "method": method,
                    "comparison": f"{method} - euclidean",
                    "accuracy_difference": row[method]-row["euclidean"]
                })
        for method in [c for c in wide.columns if str(c).startswith("hamming_")]:
            if pd.notna(row.get(method)) and pd.notna(row.get("euclidean")):
                paired_rows.append({
                    "dataset": row["dataset"], "fold_id": row["fold_id"],
                    "repeat": row["repeat"], "fold": row["fold"],
                    "k": row["k"], "method": method,
                    "comparison": f"{method} pipeline - euclidean",
                    "accuracy_difference": row[method]-row["euclidean"]
                })
    paired = pd.DataFrame(paired_rows)

    test_train_ratio = 1.0/(args.folds-1)
    test_rows = []
    for (dataset, k, method), group in paired.groupby(["dataset", "k", "method"]):
        mean, se, stat, p = corrected_resampled_t(
            group["accuracy_difference"], test_train_ratio
        )
        test_rows.append({
            "dataset": dataset, "k": k, "method": method,
            "comparison": group["comparison"].iloc[0],
            "mean_paired_difference": mean,
            "corrected_standard_error": se,
            "corrected_t_statistic": stat,
            "p_value_raw": p,
            "n_fold_comparisons": len(group),
            "test_train_ratio_used": test_train_ratio,
            "test_type": "Nadeau-Bengio corrected resampled t-test",
        })
    statistical_tests = pd.DataFrame(test_rows)
    if len(statistical_tests):
        statistical_tests["p_value_holm"] = np.nan
        # Correct across all non-Euclidean comparisons within each dataset/k family.
        for (_, _), idx in statistical_tests.groupby(["dataset", "k"]).groups.items():
            statistical_tests.loc[idx, "p_value_holm"] = holm_adjust(
                statistical_tests.loc[idx, "p_value_raw"].to_numpy()
            )

    paired_summary_rows = []
    for (dataset, k, method), group in paired.groupby(["dataset", "k", "method"]):
        ci_low, ci_high = bootstrap_ci(
            group["accuracy_difference"], args.bootstrap_samples,
            args.bootstrap_seed + abs(hash((dataset, k, method))) % 100000
        )
        paired_summary_rows.append({
            "dataset": dataset, "k": k, "method": method,
            "comparison": group["comparison"].iloc[0],
            "mean_difference": group["accuracy_difference"].mean(),
            "median_difference": group["accuracy_difference"].median(),
            "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high,
            "n_fold_comparisons": len(group),
            "interpretation": (
                "Quantisation+Hamming pipeline comparison"
                if str(method).startswith("hamming_")
                else "Continuous metric comparison on identical features"
            )
        })
    paired_summary = pd.DataFrame(paired_summary_rows)

    # Dataset-level descriptors: average across training folds.
    descriptor_cols = [
        c for c in descriptors.columns
        if c not in {"dataset", "fold_id", "repeat", "fold", "n_train"}
    ]
    descriptor_means = descriptors.groupby("dataset", as_index=False)[descriptor_cols].mean()

    # Dataset-level mean accuracy for continuous methods only.
    continuous = (
        results[results["track"] == "continuous"]
        .groupby(["dataset", "method", "k"], as_index=False)
        .agg(mean_accuracy=("accuracy", "mean"))
    )
    continuous_wide = continuous.pivot_table(
        index=["dataset", "k"], columns="method", values="mean_accuracy"
    ).reset_index()

    association_rows = []
    feature_names = [
        "n_features", "n_features_nonconstant", "class_imbalance_ratio",
        "raw_sparsity_fraction", "row_norm_cv", "feature_variance_cv",
        "feature_cv_mean", "mean_abs_feature_correlation",
        "effective_rank_fraction", "centroid_separation_proxy"
    ]
    for k in args.k_values:
        wk = continuous_wide[continuous_wide["k"] == k].copy()
        merged = wk.merge(descriptor_means, on="dataset", how="inner")
        for method in ("manhattan", "cosine"):
            if method not in merged.columns or "euclidean" not in merged.columns:
                continue
            delta = merged[method]-merged["euclidean"]
            for feature in feature_names:
                if feature not in merged.columns:
                    continue
                x = merged[feature].to_numpy(dtype=float)
                y = delta.to_numpy(dtype=float)
                if len(x) >= 4 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
                    rho, p = spearmanr(x, y)
                else:
                    rho, p = np.nan, np.nan
                association_rows.append({
                    "k": k, "metric_advantage": method,
                    "descriptor": feature,
                    "spearman_rho": rho, "p_value_unadjusted": p,
                    "n_datasets": len(merged),
                    "warning": "Exploratory; only 17 datasets and synthetic/real mix"
                })
    descriptor_metric_associations = pd.DataFrame(association_rows)

    # Leave-one-dataset-out metric selection. Targets are deltas vs Euclidean.
    # Each held-out dataset is excluded from model fitting and feature scaling.
    prediction_rows = []
    if len(descriptor_means) >= 5:
        for k in args.k_values:
            wk = continuous_wide[continuous_wide["k"] == k].copy()
            merged = wk.merge(descriptor_means, on="dataset", how="inner")
            features = [f for f in feature_names if f in merged.columns]
            if len(merged) < 5 or not features:
                continue
            Xmeta = merged[features].replace([np.inf, -np.inf], np.nan)
            Xmeta = Xmeta.fillna(Xmeta.median()).to_numpy(dtype=float)
            y_targets = np.column_stack([
                merged["manhattan"].to_numpy()-merged["euclidean"].to_numpy(),
                merged["cosine"].to_numpy()-merged["euclidean"].to_numpy()
            ])
            names = merged["dataset"].tolist()

            for held in range(len(names)):
                train_mask = np.arange(len(names)) != held
                model = make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=args.ridge_alpha)
                )
                model.fit(Xmeta[train_mask], y_targets[train_mask])
                predicted = model.predict(Xmeta[[held]])[0]
                predicted_deltas = {
                    "euclidean": 0.0,
                    "manhattan": float(predicted[0]),
                    "cosine": float(predicted[1])
                }
                selected = max(predicted_deltas, key=predicted_deltas.get)
                actual = {
                    "euclidean": float(merged.iloc[held]["euclidean"]),
                    "manhattan": float(merged.iloc[held]["manhattan"]),
                    "cosine": float(merged.iloc[held]["cosine"])
                }
                best_actual = max(actual, key=actual.get)
                prediction_rows.append({
                    "held_out_dataset": names[held], "k": k,
                    "predicted_manhattan_delta": predicted_deltas["manhattan"],
                    "predicted_cosine_delta": predicted_deltas["cosine"],
                    "selected_metric": selected,
                    "selected_metric_actual_accuracy": actual[selected],
                    "euclidean_actual_accuracy": actual["euclidean"],
                    "oracle_best_metric": best_actual,
                    "oracle_best_accuracy": actual[best_actual],
                    "selection_gain_vs_euclidean": actual[selected]-actual["euclidean"],
                    "regret_vs_oracle": actual[best_actual]-actual[selected],
                    "n_training_datasets": int(train_mask.sum()),
                    "model": f"StandardScaler + Ridge(alpha={args.ridge_alpha})",
                    "warning": "Exploratory LODO result; dataset sample is small"
                })
    heldout_predictions = pd.DataFrame(prediction_rows)

    if len(heldout_predictions):
        meta_summary = (
            heldout_predictions.groupby("k", as_index=False)
            .agg(
                mean_selection_gain=("selection_gain_vs_euclidean", "mean"),
                median_selection_gain=("selection_gain_vs_euclidean", "median"),
                mean_regret_vs_oracle=("regret_vs_oracle", "mean"),
                selected_manhattan_fraction=("selected_metric", lambda x: np.mean(x == "manhattan")),
                selected_cosine_fraction=("selected_metric", lambda x: np.mean(x == "cosine")),
                selected_euclidean_fraction=("selected_metric", lambda x: np.mean(x == "euclidean")),
                n_held_out_datasets=("held_out_dataset", "nunique")
            )
        )
    else:
        meta_summary = pd.DataFrame()

    # Hamming resolution summary: a separate quantised representation track.
    hamming_resolution = (
        results[results["track"] == "quantised_discrete"]
        .groupby(["method", "quantisation_levels", "k"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            n_dataset_folds=("accuracy", "size")
        )
    )

    # Dataset/k winner tables: observational summaries, not held-out claims.
    rankings = []
    for (dataset, track, k), group in results.groupby(["dataset", "track", "k"]):
        means = group.groupby("method")["accuracy"].mean().sort_values(ascending=False)
        rankings.append({
            "dataset": dataset, "track": track, "k": k,
            "observed_winner": means.index[0],
            "winner_accuracy": float(means.iloc[0]),
            "runner_up": means.index[1] if len(means) > 1 else None,
            "runner_up_accuracy": float(means.iloc[1]) if len(means) > 1 else np.nan,
            "winner_margin": float(means.iloc[0]-means.iloc[1]) if len(means) > 1 else np.nan,
            "selection_note": "In-sample dataset summary; not a held-out prediction"
        })
    dataset_rankings = pd.DataFrame(rankings)

    winner_counts = (
        dataset_rankings.groupby(["track", "k", "observed_winner"], as_index=False)
        .size().rename(columns={"size": "dataset_wins"})
    )

    # Descriptor table removes fold-level identifiers from the averaged output.
    dataset_characteristics = descriptor_means.copy()

    # Aggregate neighbour similarity against Euclidean.
    neighbour_summary = (
        neighbour_rows.groupby(["track", "method", "k"], as_index=False)
        .agg(
            mean_neighbour_jaccard=("neighbour_jaccard_vs_euclidean", "mean"),
            sd_neighbour_jaccard=("neighbour_jaccard_vs_euclidean", "std")
        )
    )

    # Workbook methodology/configuration.
    methodology = pd.DataFrame([
        ["Purpose", "Test whether training-fold dataset descriptors predict the best continuous k-NN metric."],
        ["Datasets", "4 real benchmark datasets + 13 synthetic datasets."],
        ["Anisotropic blobs", "Generated with make_blobs and a diagonal anisotropic transformation."],
        ["Validation", f"Repeated stratified {args.folds}-fold CV x {args.repeats} repeats."],
        ["Paired design", "All metrics share the same train/test folds within each dataset."],
        ["Continuous fairness", "Euclidean, Manhattan and cosine use identical train-fitted MinMax [0,1] features."],
        ["Euclidean/Manhattan", "Exact distances; exponential similarity mapping uses a train-only median-distance scale."],
        ["Cosine", "Cosine similarity mapped to [0,1] as (cosine+1)/2."],
        ["Hamming", "Separate train-fitted uniform quantisation + Hamming similarity track."],
        ["Descriptors", "Computed on original, unscaled training-fold features only."],
        ["Descriptor association", "Spearman correlation of descriptor vs metric accuracy advantage over Euclidean."],
        ["Metric selector", "StandardScaler + Ridge; predicts Manhattan/Cosine accuracy deltas vs Euclidean."],
        ["Held-out validation", "Leave-one-dataset-out; all fitting and scaling exclude the held-out dataset."],
        ["Small-sample caveat", "Only 17 datasets; LODO/meta-learning and descriptor associations are exploratory."],
        ["Statistical test", "Nadeau-Bengio corrected resampled t-test on paired fold differences."],
        ["Multiplicity", "Holm correction within each dataset/k family across non-Euclidean comparisons."],
        ["Confidence intervals", "Fold bootstrap intervals are descriptive; folds are dependent and intervals are not independent-sample inference."],
        ["Scoring", "Accuracy, balanced accuracy, macro-F1, plus neighbour Jaccard vs Euclidean."],
        ["Tie handling", "Stable ordering; majority-vote ties resolved by smallest encoded class label."],
        ["Quantum scope", "No quantum circuits, finite shots, QRAM, gate/resource cost, or quantum speedup measured."],
        ["Data seed", args.data_seed],
        ["Split seed", args.split_seed],
        ["Folds x repeats", f"{args.folds} x {args.repeats}"],
        ["k values", str(args.k_values)],
        ["Hamming levels", str(args.hamming_levels)],
        ["Ridge alpha", args.ridge_alpha],
    ], columns=["item", "description"])

    config = pd.DataFrame([
        ["data_seed", args.data_seed],
        ["split_seed", args.split_seed],
        ["folds", args.folds],
        ["repeats", args.repeats],
        ["k_values", str(args.k_values)],
        ["hamming_levels", str(args.hamming_levels)],
        ["n_synthetic", args.n_synthetic],
        ["bootstrap_samples", args.bootstrap_samples],
        ["ridge_alpha", args.ridge_alpha],
        ["datasets_in_run", ", ".join(sorted(results["dataset"].unique()))],
    ], columns=["parameter", "value"])

    return {
        "run_config": config,
        "methodology": methodology,
        "raw_results": results,
        "fold_summary": fold_summary,
        "metric_summary": metric_summary,
        "paired_differences": paired,
        "paired_summary": paired_summary,
        "statistical_tests": statistical_tests,
        "dataset_characteristics": dataset_characteristics,
        "descriptor_metric_associations": descriptor_metric_associations,
        "heldout_dataset_predictions": heldout_predictions,
        "meta_selection_summary": meta_summary,
        "hamming_resolution": hamming_resolution,
        "neighbour_jaccard": neighbour_summary,
        "dataset_rankings": dataset_rankings,
        "winner_counts": winner_counts,
        "kernel_diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------
# 7. EXCEL EXPORT
# ---------------------------------------------------------------------

def save_workbook(tables, outdir, export_csv):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "04_distance_metrics_v3_results.xlsx"

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
        description="Study 04 v3: repeated-CV metric comparison and dataset-to-metric prediction."
    )
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Optional subset of dataset names.")
    p.add_argument("--data-seed", type=int, default=2026)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--n-synthetic", type=int, default=900)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 9])
    p.add_argument("--hamming-levels", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=2026)
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--outdir", type=Path, default=Path("results/distance_metrics_v3"))
    p.add_argument("--csv", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.folds < 2 or args.repeats < 1:
        raise ValueError("--folds must be >=2 and --repeats >=1.")
    if args.n_synthetic < 150:
        raise ValueError("--n-synthetic should be >=150 for stratified 5-fold CV.")
    if any(k < 1 for k in args.k_values):
        raise ValueError("All k values must be positive.")
    if any(q < 2 for q in args.hamming_levels):
        raise ValueError("Hamming levels must be >=2.")
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples should be >=100.")
    if args.ridge_alpha <= 0:
        raise ValueError("--ridge-alpha must be positive.")

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("="*78)
    print("QKNN STUDY 04 v3 — REPEATED-CV METRICS + DATASET-TO-METRIC PREDICTION")
    print("="*78)
    print(f"Folds x repeats: {args.folds} x {args.repeats}")
    print(f"k values: {args.k_values}")
    print(f"Hamming levels: {args.hamming_levels}")
    print("Descriptors: original, unscaled training-fold features")
    print("Inference: Nadeau-Bengio corrected resampled t-test")
    print("Meta-selection: leave-one-dataset-out Ridge")

    results, descriptors, diagnostics, neighbour_rows = run_experiment(args)
    tables = build_tables(results, descriptors, diagnostics, neighbour_rows, args)
    workbook = save_workbook(tables, args.outdir, args.csv)

    print("\nOVERALL MEAN ACCURACY")
    print("-"*78)
    print(tables["metric_summary"].to_string(index=False))

    print("\nHELD-OUT DATASET METRIC-SELECTION SUMMARY")
    print("-"*78)
    if len(tables["meta_selection_summary"]):
        print(tables["meta_selection_summary"].to_string(index=False))
    else:
        print("Not enough datasets for leave-one-dataset-out prediction.")

    print("\nOUTPUT")
    print(workbook.resolve())
    if args.csv:
        print(f"CSV exports: {args.outdir.resolve()}")
    print("\nCompleted. This is a classical metric-selection study, not a quantum speedup test.")


if __name__ == "__main__":
    main()
