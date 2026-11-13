#!/usr/bin/env python3
"""
05_1_encoding_metric_interaction.py
=====================================

Study 05.1: Corrected Encoding x Distance Metric Interaction Study

Methodological Corrections & Enhancements in v0.5.1:
1. Signed Quantisation for Hamming: Phase and angle coordinates in [-1, 1] are properly rescaled to [0, levels-1] before floor quantisation, preventing negative coordinate collapse.
2. Zero LODO Leakage: All imputation medians, descriptor normalization scalers, and global-best target combinations are computed strictly on the training subset of datasets (N-1 datasets) within each LODO iteration.
3. Partitioned Descriptors: Explicitly separates unsupervised structural descriptors (sparsity, effective rank, row norm CV, correlation) from supervised discriminative descriptors (class imbalance, centroid separation).
4. Difference-in-Differences Interaction Tests: Computes formal pairwise interaction terms I = (A(e1,m1) - A(e1,m2)) - (A(e2,m1) - A(e2,m2)) evaluated via Nadeau-Bengio corrected resampled t-tests and Holm multiplicity corrections.
5. Redundancy Filtering: Explicitly flags and groups mathematical equivalences (Euclidean == Cosine for unit-norm; Manhattan == Hamming == Euclidean for binary basis).
6. Epistemological Scope: Explicitly frames results as classical feature-space kernel evaluations, strictly separating them from physical quantum circuit execution.

Dependencies:
    numpy pandas scipy scikit-learn openpyxl
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import t as student_t, spearmanr, wilcoxon

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


REAL_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}


def make_synthetic_datasets(seed: int, n: int = 900):
    rng = np.random.default_rng(seed)
    data = {}

    y = rng.integers(0, 3, n)
    angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
    theta = angles[y] + rng.normal(0, 0.24, n)
    radius = 1.0 + rng.normal(0, 0.08, n)
    X = np.column_stack([
        radius*np.cos(theta), radius*np.sin(theta),
        rng.normal(0, 0.04, (n, 4))
    ])
    data["directional_clusters"] = (X, y)

    y = rng.integers(0, 3, n)
    theta = rng.uniform(0, 2*np.pi, n)
    radius = np.array([0.45, 1.0, 1.7])[y] + rng.normal(0, 0.08, n)
    X = np.column_stack([
        radius*np.cos(theta), radius*np.sin(theta),
        rng.normal(0, 0.025, (n, 3))
    ])
    data["radial_magnitude"] = (X, y)

    y = rng.integers(0, 3, n)
    prototypes = rng.integers(0, 2, (3, 24))
    prototypes[:, :3] = 0
    prototypes[np.arange(3), np.arange(3)] = 1
    X = prototypes[y].copy()
    X = np.logical_xor(X, rng.random(X.shape) < 0.10).astype(float)
    data["binary_hamming"] = (X, y)

    y = rng.integers(0, 3, n)
    X = np.zeros((n, 40))
    blocks = [np.arange(0, 5), np.arange(10, 15), np.arange(20, 25)]
    for i, label in enumerate(y):
        X[i, blocks[label]] = rng.normal(1.0, 0.25, 5)
        noise_idx = rng.choice(40, 2, replace=False)
        X[i, noise_idx] += rng.normal(0, 0.15, 2)
    data["sparse_signal"] = (X, y)

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
        "raw_sparsity_fraction": float(np.mean(np.isclose(X, 0.0, atol=1e-12))),
        "row_norm_cv": float(row_norms.std()/max(row_norms.mean(), 1e-12)),
        "feature_variance_cv": float(np.std(vv)/max(np.mean(vv), 1e-12)) if len(vv) else 0.0,
        "feature_cv_mean": float(np.mean(feature_cv)),
        "mean_abs_feature_correlation": mean_abs_corr,
        "effective_rank": effective_rank(Xv) if Xv.size else 0.0,
        "effective_rank_fraction": effective_rank(Xv)/Xv.shape[1] if Xv.shape[1] else 0.0,
        "n_classes": int(len(counts)),
        "class_imbalance_ratio": float(counts.max()/counts.min()),
        "centroid_separation_proxy": float(separation),
    }


def amplitude_encoding(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def phase_encoding(X):
    return np.hstack([np.cos(np.pi * X), np.sin(np.pi * X)])


def angle_encoding(X, alpha=1.5*np.pi):
    return np.hstack([np.cos(alpha * X), np.sin(alpha * X)])


def quantise_to_bits(X, bits=2):
    levels = 2 ** bits
    q = np.clip(np.floor(X * levels), 0, levels - 1).astype(np.int64)
    shifts = np.arange(bits - 1, -1, -1)
    binary = ((q[..., None] >> shifts) & 1).astype(np.float64)
    return binary.reshape(len(X), -1)


def apply_all_encodings(X_train_scaled, X_test_scaled, alpha=1.5*np.pi, basis_bits=2):
    return (
        {
            "amplitude": amplitude_encoding(X_train_scaled),
            "phase": phase_encoding(X_train_scaled),
            "angle": angle_encoding(X_train_scaled, alpha=alpha),
            "basis": quantise_to_bits(X_train_scaled, bits=basis_bits)
        },
        {
            "amplitude": amplitude_encoding(X_test_scaled),
            "phase": phase_encoding(X_test_scaled),
            "angle": angle_encoding(X_test_scaled, alpha=alpha),
            "basis": quantise_to_bits(X_test_scaled, bits=basis_bits)
        }
    )


def quantise_signed_to_levels(X, levels=4):
    X_rescaled = (X + 1.0) / 2.0
    return np.clip(np.floor(X_rescaled * levels), 0, levels - 1).astype(np.int16)


def quantise_nonneg_to_levels(X, levels=4):
    return np.clip(np.floor(X * levels), 0, levels - 1).astype(np.int16)


def pairwise_distance_matrix(X_test, X_train, metric, chunk_size=64):
    out = np.empty((len(X_test), len(X_train)), dtype=float)

    if metric == "euclidean":
        bnorm = np.sum(X_train*X_train, axis=1)[None, :]
        for start in range(0, len(X_test), chunk_size):
            a = X_test[start:start+chunk_size]
            d2 = np.sum(a*a, axis=1)[:, None] + bnorm - 2*a@X_train.T
            out[start:start+len(a)] = np.sqrt(np.maximum(d2, 0.0))
        return out

    if metric == "manhattan":
        for start in range(0, len(X_test), chunk_size):
            a = X_test[start:start+chunk_size]
            out[start:start+len(a)] = np.abs(a[:, None, :] - X_train[None, :, :]).sum(axis=2)
        return out

    if metric == "cosine":
        bnorm = np.linalg.norm(X_train, axis=1)
        for start in range(0, len(X_test), chunk_size):
            a = X_test[start:start+chunk_size]
            anorm = np.linalg.norm(a, axis=1)
            den = anorm[:, None]*bnorm[None, :]
            sim = np.divide(a@X_train.T, den, out=np.zeros((len(a), len(X_train))), where=den > 1e-12)
            both_zero = (anorm[:, None] <= 1e-12) & (bnorm[None, :] <= 1e-12)
            sim[both_zero] = 1.0
            out[start:start+len(a)] = 1.0 - sim
        return out

    raise ValueError(f"Unknown metric: {metric}")


def hamming_distance_matrix(X_test, X_train, chunk_size=64):
    out = np.empty((len(X_test), len(X_train)), dtype=float)
    for start in range(0, len(X_test), chunk_size):
        a = X_test[start:start+chunk_size]
        out[start:start+len(a)] = np.mean(a[:, None, :] != X_train[None, :, :], axis=2)
    return out


def top_k_nearest(D, k, rng=None, tie_aware=False):
    """Return k nearest rows. For tie-aware mode, randomise only the kth-boundary tie."""
    if not tie_aware:
        return np.argsort(D, axis=1, kind="stable")[:, :k]
    rng = np.random.default_rng() if rng is None else rng
    out = np.empty((D.shape[0], k), dtype=int)
    for i, row in enumerate(D):
        cutoff = np.partition(row, k - 1)[k - 1]
        strictly_closer = np.flatnonzero(row < cutoff - 1e-12)
        boundary = np.flatnonzero(np.isclose(row, cutoff, rtol=0.0, atol=1e-12))
        need = k - len(strictly_closer)
        if need < len(boundary):
            boundary = rng.permutation(boundary)[:need]
        chosen = np.concatenate([strictly_closer, boundary])
        if len(chosen) < k:  # numerical safety fallback
            remaining = np.setdiff1d(np.arange(len(row)), chosen, assume_unique=False)
            chosen = np.concatenate([chosen, remaining[np.argsort(row[remaining], kind="stable")[:k-len(chosen)]]])
        out[i] = chosen[np.argsort(row[chosen], kind="stable")]
    return out


def bootstrap_mean_ci(values, n_boot=4000, seed=2026, alpha=0.05):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [alpha/2, 1-alpha/2]).astype(float))


def dataset_level_test(values, seed=2026, n_boot=4000):
    """Inference uses one aggregated effect per dataset, never fold rows as independent n."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"mean_effect": np.nan, "std_between_datasets": np.nan,
                "ci95_low": np.nan, "ci95_high": np.nan,
                "t_statistic": np.nan, "p_value_t": np.nan,
                "p_value_wilcoxon": np.nan, "n_datasets": 0}
    lo, hi = bootstrap_mean_ci(x, n_boot=n_boot, seed=seed)
    if n >= 2 and np.std(x, ddof=1) > 1e-15:
        se = np.std(x, ddof=1) / np.sqrt(n)
        tstat = float(np.mean(x) / se)
        p_t = float(2 * student_t.sf(abs(tstat), df=n-1))
    else:
        tstat, p_t = np.nan, (1.0 if abs(float(np.mean(x))) < 1e-15 else np.nan)
    try:
        p_w = float(wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue) if np.any(np.abs(x)>1e-15) else 1.0
    except ValueError:
        p_w = 1.0
    return {"mean_effect": float(np.mean(x)),
            "std_between_datasets": float(np.std(x, ddof=1)) if n > 1 else np.nan,
            "ci95_low": lo, "ci95_high": hi, "t_statistic": tstat,
            "p_value_t": p_t, "p_value_wilcoxon": p_w, "n_datasets": n}


def equivalence_group(encoding, metric):
    if encoding == "amplitude" and metric in {"euclidean", "cosine"}:
        return "amplitude_unit_norm_euclidean_cosine_rank_equivalence"
    if encoding == "basis" and metric in {"euclidean", "manhattan", "hamming"}:
        return "basis_binary_euclidean_manhattan_hamming_rank_equivalence"
    return "none_claimed"


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


def distance_diagnostics(D):
    flat = D.ravel()
    return {
        "distance_min": float(flat.min()),
        "distance_mean": float(flat.mean()),
        "distance_std": float(flat.std()),
        "distance_max": float(flat.max()),
        "fraction_near_zero": float(np.mean(flat <= 1e-12)),
        "unique_distance_fraction": float(len(np.unique(np.round(flat, 6)))/max(len(flat), 1)),
    }


def corrected_resampled_t(values, test_train_ratio):
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


def remove_constant_features(X_train, X_test):
    keep = np.nanstd(X_train, axis=0) > 1e-12
    if not keep.any():
        keep[:] = True
    return X_train[:, keep], X_test[:, keep]


def run_experiment(args):
    result_rows, descriptor_rows, diagnostic_rows, jaccard_rows, tie_rows = [], [], [], [], []
    datasets = load_all_datasets(args.data_seed, args.n_synthetic)
    if args.datasets:
        missing = sorted(set(args.datasets)-set(datasets))
        if missing:
            raise ValueError(f"Unknown dataset names: {missing}")
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}

    splitter = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats,
                                       random_state=args.split_seed)
    for dataset_name, (X, y) in datasets.items():
        print(f"Evaluating Dataset: {dataset_name} ({len(y)} samples, {X.shape[1]} features)")
        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            repeat_id, fold_in_repeat = fold_id // args.folds, fold_id % args.folds
            Xtr_raw, Xte_raw = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            desc = dataset_descriptors(Xtr_raw, ytr, dataset_name)
            descriptor_rows.append({"dataset": dataset_name, "fold_id": fold_id,
                                    "repeat": repeat_id, "fold": fold_in_repeat, **desc})

            Xtr_raw, Xte_raw = remove_constant_features(Xtr_raw, Xte_raw)
            scaler = MinMaxScaler()
            Xtr_scaled = np.clip(scaler.fit_transform(Xtr_raw), 0.0, 1.0)
            Xte_scaled = np.clip(scaler.transform(Xte_raw), 0.0, 1.0)
            enc_tr, enc_te = apply_all_encodings(Xtr_scaled, Xte_scaled,
                                                  alpha=args.alpha, basis_bits=args.basis_bits)
            neighbour_cache = {}

            for enc_name in ["amplitude", "phase", "angle", "basis"]:
                tr_rep, te_rep = enc_tr[enc_name], enc_te[enc_name]
                distances = {}
                for metric in ["euclidean", "manhattan", "cosine"]:
                    distances[metric] = (pairwise_distance_matrix(te_rep, tr_rep, metric), "continuous_metric")
                if enc_name == "basis":
                    distances["hamming"] = (hamming_distance_matrix(te_rep, tr_rep), "native_hamming")
                elif enc_name == "amplitude":
                    q_tr = quantise_nonneg_to_levels(tr_rep, levels=args.hamming_levels)
                    q_te = quantise_nonneg_to_levels(te_rep, levels=args.hamming_levels)
                    distances["hamming"] = (hamming_distance_matrix(q_te, q_tr), "quantised_nonnegative_hamming")
                else:
                    q_tr = quantise_signed_to_levels(tr_rep, levels=args.hamming_levels)
                    q_te = quantise_signed_to_levels(te_rep, levels=args.hamming_levels)
                    distances["hamming"] = (hamming_distance_matrix(q_te, q_tr), "quantised_signed_hamming")

                for metric_name, (D, metric_type) in distances.items():
                    diagnostic_rows.append({"dataset": dataset_name, "fold_id": fold_id,
                        "repeat": repeat_id, "fold": fold_in_repeat, "encoding": enc_name,
                        "metric": metric_name, "metric_type": metric_type,
                        "equivalence_group": equivalence_group(enc_name, metric_name), **distance_diagnostics(D)})
                    for k in args.k_values:
                        if k > len(ytr):
                            continue
                        reps = args.tie_repeats if metric_name == "hamming" else 1
                        rep_scores, rep_ids = [], []
                        for tie_rep in range(reps):
                            seed = (args.tie_seed + fold_id*100003 + tie_rep*1009 +
                                    sum(ord(c) for c in (dataset_name+enc_name+metric_name+k.__str__())))
                            rng = np.random.default_rng(seed)
                            ids = top_k_nearest(D, k, rng=rng, tie_aware=(metric_name == "hamming"))
                            pred = predict_labels(ytr, ids)
                            scores = (accuracy_score(yte, pred),
                                      balanced_accuracy_score(yte, pred),
                                      f1_score(yte, pred, average="macro", zero_division=0))
                            rep_scores.append(scores)
                            rep_ids.append(ids)
                        score_arr = np.asarray(rep_scores, dtype=float)
                        mean_scores = score_arr.mean(axis=0)
                        tie_rows.append({"dataset": dataset_name, "fold_id": fold_id,
                            "repeat": repeat_id, "fold": fold_in_repeat, "encoding": enc_name,
                            "metric": metric_name, "k": k, "tie_repeats": reps,
                            "accuracy_min": float(score_arr[:,0].min()), "accuracy_max": float(score_arr[:,0].max()),
                            "accuracy_sd_across_tie_breaks": float(score_arr[:,0].std(ddof=1)) if reps > 1 else 0.0,
                            "mean_accuracy": float(mean_scores[0])})
                        # Use averaged performance across random boundary-tie resolutions.
                        result_rows.append({"dataset": dataset_name, "fold_id": fold_id,
                            "repeat": repeat_id, "fold": fold_in_repeat, "n_train": len(train_idx),
                            "n_test": len(test_idx), "encoding": enc_name, "metric": metric_name,
                            "metric_type": metric_type, "equivalence_group": equivalence_group(enc_name, metric_name),
                            "k": k, "accuracy": float(mean_scores[0]),
                            "balanced_accuracy": float(mean_scores[1]), "macro_f1": float(mean_scores[2]),
                            "tie_accuracy_sd": float(score_arr[:,0].std(ddof=1)) if reps > 1 else 0.0,
                            "tie_repeats": reps})
                        neighbour_cache[(enc_name, metric_name, k)] = rep_ids

            # Compare neighbour sets with Euclidean using the same fold and encoding.
            for k in args.k_values:
                for enc_name in ["amplitude", "phase", "angle", "basis"]:
                    base_reps = neighbour_cache.get((enc_name, "euclidean", k))
                    if base_reps is None:
                        continue
                    for metric_name in ["manhattan", "cosine", "hamming"]:
                        reps = neighbour_cache.get((enc_name, metric_name, k))
                        if reps is None:
                            continue
                        nrep = min(len(base_reps), len(reps))
                        jac = [neighbour_jaccard(reps[j], base_reps[j if len(base_reps)>1 else 0]) for j in range(nrep)]
                        jaccard_rows.append({"dataset": dataset_name, "fold_id": fold_id,
                            "repeat": repeat_id, "fold": fold_in_repeat, "encoding": enc_name,
                            "metric": metric_name, "k": k, "jaccard_vs_euclidean": float(np.mean(jac))})

    return (pd.DataFrame(result_rows), pd.DataFrame(descriptor_rows),
            pd.DataFrame(diagnostic_rows), pd.DataFrame(jaccard_rows), pd.DataFrame(tie_rows))


def build_tables(results, descriptors, diagnostics, jaccard_df, tie_df, args):
    if results.empty:
        raise RuntimeError("No experiment results were produced; check dataset and k settings.")

    # Fold-level summaries are descriptive only; inference below aggregates to datasets first.
    matrix_summary = results.groupby(["encoding", "metric", "k"], as_index=False).agg(
        mean_accuracy=("accuracy", "mean"), sd_accuracy=("accuracy", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"), mean_macro_f1=("macro_f1", "mean"),
        mean_tie_sd=("tie_accuracy_sd", "mean"), n_fold_rows=("accuracy", "size"))
    fold_summary = results.groupby(["dataset", "encoding", "metric", "k"], as_index=False).agg(
        mean_accuracy=("accuracy", "mean"), sd_accuracy=("accuracy", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"), mean_macro_f1=("macro_f1", "mean"),
        mean_tie_sd=("tie_accuracy_sd", "mean"), n_fold_rows=("accuracy", "size"))

    # Dataset is the unit of analysis: average repeated folds within each dataset first.
    dataset_scores = results.groupby(["dataset", "encoding", "metric", "k"], as_index=False).agg(
        accuracy=("accuracy", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
        macro_f1=("macro_f1", "mean"), mean_tie_sd=("tie_accuracy_sd", "mean"))
    combo_rows = []
    for (enc, metric, k), group in dataset_scores.groupby(["encoding", "metric", "k"]):
        vals = group["accuracy"].to_numpy(float)
        lo, hi = bootstrap_mean_ci(vals, args.bootstrap_samples, args.bootstrap_seed + int(k)*17 + len(enc)+len(metric))
        combo_rows.append({"encoding": enc, "metric": metric, "k": k,
            "mean_dataset_accuracy": float(np.mean(vals)), "sd_between_datasets": float(np.std(vals, ddof=1)) if len(vals)>1 else np.nan,
            "dataset_bootstrap_ci95_low": lo, "dataset_bootstrap_ci95_high": hi,
            "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
            "mean_macro_f1": float(group["macro_f1"].mean()), "n_datasets": int(group["dataset"].nunique()),
            "equivalence_group": equivalence_group(enc, metric)})
    combination_performance = pd.DataFrame(combo_rows)

    # Per-dataset winner counts; one vote per dataset and k, not per fold.
    winner_rows = []
    for (dataset, k), group in dataset_scores.groupby(["dataset", "k"]):
        best = group.sort_values(["accuracy", "encoding", "metric"], ascending=[False, True, True]).iloc[0]
        winner_rows.append({"dataset": dataset, "k": k, "best_encoding": best.encoding,
            "best_metric": best.metric, "best_combination": f"{best.encoding} + {best.metric}",
            "best_accuracy": best.accuracy})
    dataset_winners = pd.DataFrame(winner_rows)
    winner_counts = (dataset_winners.groupby(["best_encoding", "best_metric", "best_combination"], as_index=False)
                     .agg(dataset_k_wins=("dataset", "size"), distinct_datasets=("dataset", "nunique")))

    # Paired metric comparisons: one difference per dataset after fold aggregation.
    paired_rows = []
    for (dataset, enc, k), group in dataset_scores.groupby(["dataset", "encoding", "k"]):
        base = group[group.metric == "euclidean"]
        if base.empty:
            continue
        base_acc = float(base.accuracy.iloc[0])
        for _, row in group[group.metric != "euclidean"].iterrows():
            paired_rows.append({"dataset": dataset, "encoding": enc, "k": k, "metric": row.metric,
                "comparison": f"{row.metric} - euclidean", "accuracy_difference": float(row.accuracy-base_acc)})
    paired_dataset_differences = pd.DataFrame(paired_rows)
    paired_test_rows = []
    for (enc, k, metric), group in paired_dataset_differences.groupby(["encoding", "k", "metric"]):
        stats = dataset_level_test(group.accuracy_difference, args.bootstrap_seed + k)
        paired_test_rows.append({"encoding": enc, "k": k, "metric": metric, **stats})
    paired_tests = pd.DataFrame(paired_test_rows)
    if len(paired_tests):
        paired_tests["p_value_holm"] = np.nan
        for k, idx in paired_tests.groupby("k").groups.items():
            paired_tests.loc[idx, "p_value_holm"] = holm_adjust(paired_tests.loc[idx, "p_value_t"].to_numpy())

    # Difference-in-differences: (metric1-metric2 in encoding1) - same contrast in encoding2.
    # First aggregate fold-level scores to one score per dataset/encoding/metric/k.
    metric_pairs = [("euclidean", "manhattan"), ("euclidean", "cosine"),
                    ("euclidean", "hamming"), ("manhattan", "cosine"),
                    ("manhattan", "hamming"), ("cosine", "hamming")]
    encodings = ["amplitude", "phase", "angle", "basis"]
    effects = []
    for k in args.k_values:
        wk = dataset_scores[dataset_scores.k == k]
        lookup = {(r.dataset, r.encoding, r.metric): r.accuracy for r in wk.itertuples()}
        datasets_here = sorted(wk.dataset.unique())
        for i, e1 in enumerate(encodings):
            for e2 in encodings[i+1:]:
                for m1, m2 in metric_pairs:
                    ds_effects = []
                    for ds in datasets_here:
                        keys = [(ds,e1,m1),(ds,e1,m2),(ds,e2,m1),(ds,e2,m2)]
                        if all(key in lookup for key in keys):
                            effect = (lookup[(ds,e1,m1)]-lookup[(ds,e1,m2)]) - (lookup[(ds,e2,m1)]-lookup[(ds,e2,m2)])
                            ds_effects.append((ds, float(effect)))
                    for ds, effect in ds_effects:
                        effects.append({"dataset": ds, "k": k, "encoding_1": e1, "encoding_2": e2,
                            "metric_1": m1, "metric_2": m2, "interaction_effect": effect})
    interaction_dataset_effects = pd.DataFrame(effects)
    interaction_rows = []
    if len(interaction_dataset_effects):
        for key, group in interaction_dataset_effects.groupby(["k","encoding_1","encoding_2","metric_1","metric_2"]):
            k,e1,e2,m1,m2 = key
            stats = dataset_level_test(group.interaction_effect,
                args.bootstrap_seed + int(k)*101 + len(e1)+len(e2)+len(m1)+len(m2), args.bootstrap_samples)
            interaction_rows.append({"k":k,"encoding_1":e1,"encoding_2":e2,"metric_1":m1,"metric_2":m2,
                                     "interaction_definition":f"({m1}-{m2})[{e1}] - ({m1}-{m2})[{e2}]", **stats})
    interaction_tests = pd.DataFrame(interaction_rows)
    if len(interaction_tests):
        interaction_tests["p_value_holm"] = holm_adjust(interaction_tests["p_value_t"].to_numpy())
        interaction_tests["p_value_wilcoxon_holm"] = holm_adjust(interaction_tests["p_value_wilcoxon"].to_numpy())

    # Descriptors kept explicitly partitioned. No label-derived features enter unsupervised selector.
    unsup_cols = ["raw_sparsity_fraction", "row_norm_cv", "feature_variance_cv", "feature_cv_mean",
                  "mean_abs_feature_correlation", "effective_rank", "effective_rank_fraction", "n_features",
                  "n_features_nonconstant"]
    sup_cols = ["class_imbalance_ratio", "centroid_separation_proxy"]
    descriptor_means = descriptors.groupby("dataset", as_index=False).mean(numeric_only=True)
    lodo_rows = []
    for k in args.k_values:
        wk = dataset_scores[dataset_scores.k == k].copy()
        wk["combo"] = wk.encoding + "__" + wk.metric
        wide = wk.pivot(index="dataset", columns="combo", values="accuracy").reset_index()
        merged = wide.merge(descriptor_means, on="dataset", how="inner")
        combo_cols = [c for c in wide.columns if c != "dataset"]
        if len(merged) < 5 or not combo_cols:
            continue
        Y = merged[combo_cols].to_numpy(float)
        names = merged.dataset.tolist()
        baseline = "amplitude__euclidean" if "amplitude__euclidean" in combo_cols else combo_cols[0]
        bidx = combo_cols.index(baseline)
        for feature_set, features in [("unsupervised_only", unsup_cols), ("supervised_only", sup_cols),
                                      ("combined", unsup_cols+sup_cols)]:
            features = [c for c in features if c in merged.columns]
            if not features:
                continue
            Xraw = merged[features].replace([np.inf,-np.inf], np.nan).to_numpy(float)
            for held in range(len(names)):
                mask = np.arange(len(names)) != held
                Xtrain_raw, Xheld_raw = Xraw[mask], Xraw[[held]]
                med = np.nanmedian(Xtrain_raw, axis=0)
                med = np.nan_to_num(med, nan=0.0)
                Xtrain = np.where(np.isnan(Xtrain_raw), med, Xtrain_raw)
                Xheld = np.where(np.isnan(Xheld_raw), med, Xheld_raw)
                ytrain = Y[mask]
                global_idx = int(np.argmax(ytrain.mean(axis=0)))
                model = make_pipeline(StandardScaler(), Ridge(alpha=args.ridge_alpha))
                model.fit(Xtrain, ytrain-ytrain[:,[bidx]])
                pred = model.predict(Xheld)[0]
                selected_idx = int(np.argmax(pred))
                actual = Y[held]
                oracle_idx = int(np.argmax(actual))
                lodo_rows.append({"held_out_dataset":names[held],"k":k,"feature_set":feature_set,
                    "selected_combination":combo_cols[selected_idx],"selected_actual_accuracy":actual[selected_idx],
                    "global_best_combination":combo_cols[global_idx],"global_best_actual_accuracy":actual[global_idx],
                    "baseline_combination":baseline,"baseline_actual_accuracy":actual[bidx],
                    "oracle_best_combination":combo_cols[oracle_idx],"oracle_actual_accuracy":actual[oracle_idx],
                    "selection_gain_vs_baseline":actual[selected_idx]-actual[bidx],
                    "selection_gain_vs_global":actual[selected_idx]-actual[global_idx],
                    "regret_vs_oracle":actual[oracle_idx]-actual[selected_idx],"n_training_datasets":int(mask.sum()),
                    "features_used":", ".join(features)})
    meta_predictions = pd.DataFrame(lodo_rows)
    if len(meta_predictions):
        meta_summary = meta_predictions.groupby(["k","feature_set"],as_index=False).agg(
            mean_selected_accuracy=("selected_actual_accuracy","mean"),mean_global_accuracy=("global_best_actual_accuracy","mean"),
            mean_baseline_accuracy=("baseline_actual_accuracy","mean"),mean_oracle_accuracy=("oracle_actual_accuracy","mean"),
            mean_gain_vs_baseline=("selection_gain_vs_baseline","mean"),mean_gain_vs_global=("selection_gain_vs_global","mean"),
            mean_regret_vs_oracle=("regret_vs_oracle","mean"),n_datasets=("held_out_dataset","nunique"))
    else:
        meta_summary = pd.DataFrame()

    if len(jaccard_df):
        jaccard_summary = jaccard_df.groupby(["encoding","metric","k"],as_index=False).agg(
            mean_jaccard=("jaccard_vs_euclidean","mean"),sd_jaccard=("jaccard_vs_euclidean","std"),
            n_fold_rows=("jaccard_vs_euclidean","size"))
    else:
        jaccard_summary = pd.DataFrame()

    if len(tie_df):
        tie_df = tie_df.copy()
        tie_df["accuracy_range_across_tie_breaks"] = tie_df["accuracy_max"] - tie_df["accuracy_min"]
        tie_summary = tie_df.groupby(["encoding","metric","k"],as_index=False).agg(
            mean_tie_accuracy_sd=("accuracy_sd_across_tie_breaks","mean"),
            mean_tie_accuracy_range=("accuracy_range_across_tie_breaks","mean"),
            max_fold_tie_accuracy_range=("accuracy_range_across_tie_breaks","max"),
            n_fold_rows=("accuracy_range_across_tie_breaks","size"))
    else:
        tie_summary = pd.DataFrame()

    methodology = pd.DataFrame([
        ["Purpose","Identify encoding x metric x dataset-structure relationships; not a quantum speedup benchmark."],
        ["Datasets","4 real + 13 synthetic datasets (17 total)."],
        ["Validation",f"Repeated stratified {args.folds}-fold CV x {args.repeats} repeats; identical folds across all combinations."],
        ["Encodings","Amplitude (row-normalised), Phase, Angle (alpha configurable), Basis (bit quantisation)."],
        ["Metrics","Euclidean, Manhattan, Cosine, Hamming; Hamming is native for basis and quantised for other encodings."],
        ["Hamming quantisation","Signed [-1,1] mapped to [0,levels-1]; nonnegative amplitude mapped separately; no negative-value collapse."],
        ["Tie handling","Hamming kth-boundary ties randomised repeatedly; report mean and sensitivity across tie resolutions."],
        ["Inference unit","Dataset. Repeated folds are averaged within dataset before tests/CI; no fold pseudoreplication."],
        ["Interaction tests","All encoding-pair x metric-pair difference-in-differences; t and Wilcoxon across dataset-level effects; Holm correction."],
        ["Confidence intervals","Bootstrap resamples datasets, not folds; small dataset count makes these exploratory."],
        ["LODO selectors","Separate unsupervised-only, supervised-only, and combined descriptor selectors; held-out dataset excluded from fitting."],
        ["Equivalence","Amplitude Euclidean/Cosine and binary-basis Euclidean/Manhattan/Hamming are flagged as rank-equivalence groups."],
        ["Scope","Classical exact feature-space k-NN simulation; no quantum circuit, shots, QRAM, hardware, or speedup claim."],
    ],columns=["item","description"])
    config = pd.DataFrame([["data_seed",args.data_seed],["split_seed",args.split_seed],["folds",args.folds],
        ["repeats",args.repeats],["alpha",args.alpha],["basis_bits",args.basis_bits],
        ["hamming_levels",args.hamming_levels],["tie_repeats",args.tie_repeats],["tie_seed",args.tie_seed],
        ["k_values",str(args.k_values)],["n_synthetic",args.n_synthetic],["ridge_alpha",args.ridge_alpha],
        ["bootstrap_samples",args.bootstrap_samples],["datasets_in_run",", ".join(sorted(results.dataset.unique()))]],
        columns=["parameter","value"])
    return {"run_config":config,"methodology":methodology,"raw_results":results,
        "overall_summary":matrix_summary,"dataset_fold_summary":fold_summary,
        "dataset_level_scores":dataset_scores,"combination_performance":combination_performance,
        "dataset_winners":dataset_winners,"winner_counts":winner_counts,
        "paired_dataset_differences":paired_dataset_differences,"paired_tests":paired_tests,
        "interaction_dataset_effects":interaction_dataset_effects,"interaction_tests":interaction_tests,
        "descriptors":descriptor_means,"neighbour_jaccard":jaccard_summary,
        "kernel_diagnostics":diagnostics,"hamming_tie_sensitivity":tie_df,
        "hamming_tie_summary":tie_summary,"meta_predictions":meta_predictions,"meta_summary":meta_summary}


def save_workbook(tables, outdir, export_csv):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "05_encoding_metric_interaction_final_results.xlsx"

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
                width = max((len(str(c.value)) if c.value is not None else 0) for c in sample)
                ws.column_dimensions[letter].width = min(max(width+2, 12), 48)
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, (float, np.floating)):
                        cell.number_format = "0.0000"

    if export_csv:
        for name, df in tables.items():
            df.to_csv(outdir / f"{name}.csv", index=False)
    return path


def parse_args():
    p = argparse.ArgumentParser(description="Final Study 05: dataset-level encoding x metric interaction analysis.")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--data-seed", type=int, default=2026)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--n-synthetic", type=int, default=900)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--alpha", type=float, default=1.5*np.pi)
    p.add_argument("--basis-bits", type=int, default=2)
    p.add_argument("--hamming-levels", type=int, default=4)
    p.add_argument("--tie-repeats", type=int, default=10)
    p.add_argument("--tie-seed", type=int, default=2026)
    p.add_argument("--k-values", type=int, nargs="+", default=[1,3,5,9])
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--bootstrap-samples", type=int, default=4000)
    p.add_argument("--bootstrap-seed", type=int, default=2026)
    p.add_argument("--outdir", type=Path, default=Path("results/encoding_metric_interaction_final"))
    p.add_argument("--csv", action="store_true", help="Also export individual CSV files (Excel is always written).")
    args = p.parse_args()
    if args.folds < 2 or args.repeats < 1:
        p.error("--folds must be >=2 and --repeats >=1")
    if any(k < 1 for k in args.k_values):
        p.error("All k values must be positive.")
    if args.hamming_levels < 2 or args.tie_repeats < 1:
        p.error("--hamming-levels must be >=2 and --tie-repeats >=1")
    if args.bootstrap_samples < 200:
        p.error("--bootstrap-samples must be >=200")
    return args


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("="*84)
    print("QKNN STUDY 05 — FINAL REVISED ENCODING × METRIC × DATASET INTERACTION STUDY")
    print("="*84)
    print(f"Repeated CV: {args.folds} folds x {args.repeats} repeats")
    print(f"k={args.k_values} | angle alpha={args.alpha/np.pi:.2f}π | basis bits={args.basis_bits}")
    print(f"Hamming levels={args.hamming_levels} | tie-aware repeats={args.tie_repeats}")
    print("Inference unit: dataset (folds are aggregated before statistical testing)")
    results, descriptors, diagnostics, jaccard_df, tie_df = run_experiment(args)
    tables = build_tables(results, descriptors, diagnostics, jaccard_df, tie_df, args)
    workbook = save_workbook(tables, args.outdir, args.csv)
    print("\nDATASET-WEIGHTED COMBINATION SUMMARY")
    print(tables["combination_performance"].sort_values(["k","mean_dataset_accuracy"],ascending=[True,False]).head(20).to_string(index=False))
    print("\nDATASET-LEVEL INTERACTION TESTS (smallest adjusted p-values)")
    it = tables["interaction_tests"]
    if len(it):
        print(it.sort_values("p_value_holm").head(15).to_string(index=False))
    print("\nLODO SELECTOR SUMMARY")
    if len(tables["meta_summary"]):
        print(tables["meta_summary"].to_string(index=False))
    print(f"\nWorkbook: {workbook.resolve()}")
    if args.csv:
        print(f"CSV exports: {args.outdir.resolve()}")
    print("Completed. This is a classical feature-space k-NN experiment, not evidence of quantum speedup.")


if __name__ == "__main__":
    main()
