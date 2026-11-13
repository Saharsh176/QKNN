#!/usr/bin/env python3
"""
06_dataset_structure_selection.py
=================================

QKNN research programme — Study 06
Dataset-Structure -> Encoding/Metric Selection

CENTRAL QUESTION
----------------
Can measurable, label-free properties of a dataset predict which of 16
encoding x distance-metric combinations will work well on a previously
unseen dataset?

This is a classical feature-space k-NN experiment. It does NOT simulate
quantum circuits, shots, QRAM, hardware, or quantum speedup.

DESIGN
------
1. 4 representations:
   - amplitude: absolute-valued, L2-normalised feature vectors
   - phase: signed feature values mapped to angles, then represented by
     concatenated cosine/sine coordinates
   - angle: signed feature values mapped to angles, then represented by
     cosine coordinates
   - basis: each feature quantised to a configurable number of bits

2. 4 metrics applied to every representation:
   Euclidean, Manhattan, cosine distance, and Hamming.

3. Hamming uses a consistent ordinal quantisation rule for every encoding.
   For basis, the representation is already discrete. For other encodings,
   values are quantised using training-fold quantile boundaries. This is
   still a methodological limitation: Hamming acts on discretised values,
   not on the continuous representation itself.

4. Ties at the kth distance boundary are handled inclusively: every point
   tied at the kth distance contributes to the vote. This avoids random
   truncation of tied neighbours. The effective neighbour count can exceed k.

5. Repeated stratified CV evaluates each dataset. The dataset—not each fold—
   is the unit used by the meta-selector and its summary statistics.

6. Outer leave-one-dataset-out (LODO) evaluation:
   - descriptors are label-free and calculated from X only;
   - a ridge meta-model is trained on all other datasets' observed
     combination accuracies;
   - it predicts the best combination for the held-out dataset;
   - it is compared against a fixed default (Amplitude + Euclidean) and
     the training-only global-best combination.
   No held-out dataset labels or performance values enter selector fitting.

7. Excel workbook contains separate sheets for all tables. CSV export is
   optional (--csv).

DEPENDENCIES
------------
pip install numpy pandas scipy scikit-learn openpyxl

EXAMPLE
-------
python 06_dataset_structure_selection.py
python 06_dataset_structure_selection.py --folds 5 --repeats 3 --n-synthetic 1200
python 06_dataset_structure_selection.py --datasets iris wine breast_cancer digits
python 06_dataset_structure_selection.py --csv

INTERPRETATION WARNING
----------------------
The dataset collection is deliberately heterogeneous but still modest.
LODO performance over a small collection is exploratory, not proof of
generalisation to arbitrary future datasets. The synthetic datasets are
controlled stress tests, not independent real-world evidence.
"""

from __future__ import annotations

import argparse
import itertools
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.datasets import (
    load_breast_cancer, load_digits, load_iris, load_wine,
    make_blobs, make_circles, make_classification, make_moons,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


ENCODINGS = ("amplitude", "phase", "angle", "basis")
METRICS = ("euclidean", "manhattan", "cosine", "hamming")
COMBINATIONS = tuple((e, m) for e in ENCODINGS for m in METRICS)
DEFAULT_COMBINATION = ("amplitude", "euclidean")


def make_synthetic(seed: int, n: int):
    """Return a diverse, deterministic collection of controlled datasets."""
    rng = np.random.default_rng(seed)
    out = {}

    # Geometry and cluster shape
    X, y = make_blobs(n_samples=n, centers=3, n_features=8,
                      cluster_std=[0.45, 1.0, 1.7], random_state=seed)
    out["anisotropic_blobs"] = (X @ np.array(
        [[3, 1, 0, 0, 0, 0, 0, 0],
         [0, .5, 0, 0, 0, 0, 0, 0],
         [0, 0, 2, .7, 0, 0, 0, 0],
         [0, 0, 0, .4, 0, 0, 0, 0],
         [0, 0, 0, 0, 1.5, .2, 0, 0],
         [0, 0, 0, 0, 0, .6, 0, 0],
         [0, 0, 0, 0, 0, 0, 2, .8],
         [0, 0, 0, 0, 0, 0, 0, .3]]), y)

    X, y = make_moons(n_samples=n, noise=.18, random_state=seed + 1)
    out["moons"] = (X, y)
    X, y = make_circles(n_samples=n, noise=.10, factor=.42, random_state=seed + 2)
    out["circles"] = (X, y)

    # Discrete / ordinal / sparse
    X = rng.integers(0, 2, size=(n, 24)).astype(float)
    y = (X[:, :6].sum(axis=1) >= 3).astype(int)
    out["binary_hamming"] = (X, y)

    X = rng.integers(0, 6, size=(n, 18)).astype(float)
    score = X[:, :5].sum(axis=1) + .4 * X[:, 5:9].sum(axis=1)
    y = (score > np.median(score)).astype(int)
    out["ordinal_discrete"] = (X, y)

    X = rng.exponential(1.0, size=(n, 40))
    mask = rng.random(X.shape) < .82
    X[mask] = 0
    score = X[:, :8].sum(axis=1) - .5 * X[:, 8:16].sum(axis=1)
    y = (score > np.median(score)).astype(int)
    out["sparse_signal"] = (X, y)

    # Directional, magnitude and periodic structure
    angles = rng.uniform(-np.pi, np.pi, size=n)
    radius = rng.normal(1.0, .08, size=n)
    X = np.column_stack([radius*np.cos(angles), radius*np.sin(angles),
                         np.cos(2*angles), np.sin(2*angles)])
    y = ((angles % (2*np.pi)) > np.pi).astype(int)
    out["directional_periodic"] = (X, y)

    X = rng.normal(size=(n, 12))
    scale = np.array([.08, .12, .2, .4, .8, 1.5, 2.5, 4, 6, 8, 10, 12])
    X = X * scale
    y = (X[:, 7] + X[:, 8] - .5*X[:, 0] > 0).astype(int)
    out["anisotropic_scale"] = (X, y)

    X = rng.standard_t(df=2.2, size=(n, 20))
    score = X[:, :5].sum(axis=1)
    y = (score > np.median(score)).astype(int)
    out["heavy_tailed"] = (X, y)

    X = rng.normal(size=(n, 12))
    latent = X[:, :3]
    X[:, 3:9] = latent @ rng.normal(size=(3, 6)) + .03*rng.normal(size=(n, 6))
    y = (latent[:, 0] + latent[:, 1] - latent[:, 2] > 0).astype(int)
    out["correlated_redundant"] = (X, y)

    X, y = make_classification(
        n_samples=n, n_features=60, n_informative=5, n_redundant=10,
        n_repeated=0, n_classes=2, n_clusters_per_class=2,
        class_sep=.75, flip_y=.03, random_state=seed + 3)
    out["high_dim_weak_signal"] = (X, y)

    X = rng.normal(size=(n, 16))
    mag = np.linalg.norm(X[:, :4], axis=1)
    y = (mag > np.median(mag)).astype(int)
    out["radial_magnitude"] = (X, y)

    X = rng.normal(size=(n, 10))
    X[:, 5:] = X[:, :5] + .1*rng.normal(size=(n, 5))
    y = (X[:, 0]**2 + X[:, 1]**2 > 1.5).astype(int)
    out["nonlinear_redundant"] = (X, y)

    X, y = make_blobs(n_samples=n, centers=4, n_features=5,
                      cluster_std=1.5, random_state=seed + 4)
    y = np.where(y == 0, 0, 1)  # merge to a less separable binary task
    out["overlapping_clusters"] = (X, y)

    return out


def load_datasets(seed: int, n_synthetic: int):
    """Load built-in real datasets plus synthetic stress tests."""
    datasets = {}
    for name, loader in (
        ("iris", load_iris), ("wine", load_wine),
        ("breast_cancer", load_breast_cancer), ("digits", load_digits),
    ):
        bunch = loader()
        datasets[name] = (np.asarray(bunch.data, dtype=float),
                          np.asarray(bunch.target))

    datasets.update(make_synthetic(seed, n_synthetic))
    return datasets


def clean_dataset(X, y):
    X = np.asarray(X, dtype=float)
    y = LabelEncoder().fit_transform(np.asarray(y))
    # Remove constant columns; replace non-finite entries using column medians.
    X[~np.isfinite(X)] = np.nan
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(np.isnan(X))
    X[inds] = med[inds[1]]
    keep = np.nanstd(X, axis=0) > 1e-12
    X = X[:, keep]
    if X.shape[1] == 0:
        raise ValueError("Dataset has no non-constant features.")
    return X, y


def describe_dataset(X):
    """Label-free structural descriptors; no class labels are used."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    eps = 1e-12
    absx = np.abs(X)
    norms = np.linalg.norm(X, axis=1)
    centered = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eig = np.linalg.eigvalsh(np.atleast_2d(cov))
    eig = np.clip(eig, 0, None)[::-1]
    p = eig / max(eig.sum(), eps)
    effective_rank = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
    intrinsic_proxy = float((eig.sum()**2) / max(np.sum(eig**2), eps))
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    upper = corr[np.triu_indices_from(corr, k=1)]
    q = np.quantile(absx, [.25, .5, .75, .9, .99])
    return {
        "n_samples": n, "n_features": d, "aspect_ratio_n_over_d": n/max(d, 1),
        "zero_fraction": float(np.mean(absx < 1e-12)),
        "near_zero_fraction": float(np.mean(absx < 1e-3)),
        "mean_abs": float(absx.mean()), "std_abs": float(absx.std()),
        "abs_q25": float(q[0]), "abs_median": float(q[1]),
        "abs_q75": float(q[2]), "abs_q90": float(q[3]), "abs_q99": float(q[4]),
        "mean_row_norm": float(norms.mean()), "std_row_norm": float(norms.std()),
        "row_norm_cv": float(norms.std()/max(norms.mean(), eps)),
        "mean_feature_skew": float(np.nanmean(pd.DataFrame(X).skew(axis=0))),
        "mean_feature_kurtosis": float(np.nanmean(pd.DataFrame(X).kurtosis(axis=0))),
        "mean_abs_correlation": float(np.mean(np.abs(upper))) if len(upper) else 0.0,
        "max_abs_correlation": float(np.max(np.abs(upper))) if len(upper) else 0.0,
        "effective_rank": effective_rank, "participation_ratio": intrinsic_proxy,
        "effective_rank_fraction": effective_rank/max(d, 1),
        "covariance_condition_proxy": float(eig[0]/max(eig[-1], eps)),
        "mean_feature_std": float(np.std(X, axis=0).mean()),
        "feature_scale_ratio": float(np.max(np.std(X, axis=0)) /
                                     max(np.min(np.std(X, axis=0)), eps)),
    }


def fit_transform_encoding(X_train, X_test, encoding, basis_bits):
    """Fit each encoding using training-fold statistics only."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd < 1e-12] = 1.0
    A = (X_train - mu) / sd
    B = (X_test - mu) / sd

    if encoding == "amplitude":
        # Nonnegative amplitudes; L2 normalisation makes Euclidean and cosine
        # rankings mathematically equivalent (up to numerical tolerance).
        A = np.abs(A)
        B = np.abs(B)
        an = np.linalg.norm(A, axis=1, keepdims=True)
        bn = np.linalg.norm(B, axis=1, keepdims=True)
        A = A / np.maximum(an, 1e-12)
        B = B / np.maximum(bn, 1e-12)
        return A, B

    if encoding == "phase":
        # Signed values -> angles; retain both quadratures.
        Aang = np.pi * np.tanh(A)
        Bang = np.pi * np.tanh(B)
        return np.concatenate([np.cos(Aang), np.sin(Aang)], axis=1), \
               np.concatenate([np.cos(Bang), np.sin(Bang)], axis=1)

    if encoding == "angle":
        Aang = 1.5*np.pi * np.tanh(A)
        Bang = 1.5*np.pi * np.tanh(B)
        return np.cos(Aang), np.cos(Bang)

    if encoding == "basis":
        levels = 2 ** basis_bits
        # Training-derived clipping limits prevent test-fold leakage.
        lo = np.quantile(A, .01, axis=0)
        hi = np.quantile(A, .99, axis=0)
        span = np.maximum(hi - lo, 1e-12)
        Aq = np.clip((A - lo) / span, 0, 1)
        Bq = np.clip((B - lo) / span, 0, 1)
        return np.floor(Aq*(levels-1) + 1e-12), \
               np.floor(Bq*(levels-1) + 1e-12)

    raise ValueError(f"Unknown encoding: {encoding}")


def pairwise_distances(A, B, metric, hamming_levels):
    """Vectorised train-to-test distance matrix."""
    if metric == "euclidean":
        a2 = np.sum(A*A, axis=1)[None, :]
        b2 = np.sum(B*B, axis=1)[:, None]
        return np.sqrt(np.maximum(b2 + a2 - 2*B@A.T, 0.0))
    if metric == "manhattan":
        return np.abs(B[:, None, :] - A[None, :, :]).sum(axis=2)
    if metric == "cosine":
        dot = B @ A.T
        den = np.linalg.norm(B, axis=1)[:, None] * np.linalg.norm(A, axis=1)[None, :]
        sim = dot / np.maximum(den, 1e-12)
        return np.clip(1.0 - sim, 0.0, 2.0)
    if metric == "hamming":
        # Basis is already discrete. Other encodings use train-fold quantile
        # bins, learned separately per encoded feature.
        if np.issubdtype(A.dtype, np.floating):
            bins = np.linspace(0, 1, hamming_levels + 1)[1:-1]
            # For basis values already integer-coded but stored as float, use
            # the same direct categorical comparison after rounding.
            if np.nanmax(A) > 1.0 + 1e-9:
                Aq = np.rint(A).astype(int)
                Bq = np.rint(B).astype(int)
            else:
                lo = np.quantile(A, .01, axis=0)
                hi = np.quantile(A, .99, axis=0)
                span = np.maximum(hi-lo, 1e-12)
                Aq = np.digitize(np.clip((A-lo)/span, 0, 1), bins)
                Bq = np.digitize(np.clip((B-lo)/span, 0, 1), bins)
        else:
            Aq, Bq = A.astype(int), B.astype(int)
        return np.mean(Bq[:, None, :] != Aq[None, :, :], axis=2)
    raise ValueError(f"Unknown metric: {metric}")


def predict_tie_aware(D, y_train, k):
    """Include all observations tied at the kth distance boundary."""
    n_test = D.shape[0]
    classes = np.unique(y_train)
    pred = np.empty(n_test, dtype=y_train.dtype)
    effective_k = np.empty(n_test, dtype=int)
    for i in range(n_test):
        row = D[i]
        kth = np.partition(row, min(k-1, len(row)-1))[min(k-1, len(row)-1)]
        idx = np.flatnonzero(row <= kth + 1e-12)
        effective_k[i] = len(idx)
        counts = np.array([(y_train[idx] == c).sum() for c in classes])
        winners = classes[counts == counts.max()]
        if len(winners) == 1:
            pred[i] = winners[0]
        else:
            # Deterministic final tie-break: smallest encoded class.
            pred[i] = np.min(winners)
    return pred, effective_k


def evaluate_one_fold(Xtr, Xte, ytr, yte, k_values, args):
    rows = []
    for enc in ENCODINGS:
        A, B = fit_transform_encoding(Xtr, Xte, enc, args.basis_bits)
        for metric in METRICS:
            D = pairwise_distances(A, B, metric, args.hamming_levels)
            for k in k_values:
                pred, eff = predict_tie_aware(D, ytr, k)
                rows.append({
                    "encoding": enc, "metric": metric, "k": k,
                    "accuracy": accuracy_score(yte, pred),
                    "balanced_accuracy": balanced_accuracy_score(yte, pred),
                    "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
                    "mean_effective_k": float(eff.mean()),
                    "max_effective_k": int(eff.max()),
                })
    return rows


def evaluate_datasets(datasets, args):
    raw_rows = []
    descriptor_rows = []
    split_seed = args.split_seed
    for di, (name, pair) in enumerate(datasets.items(), 1):
        X, y = clean_dataset(*pair)
        if np.min(np.bincount(y)) < args.folds:
            print(f"[SKIP] {name}: smallest class has fewer than {args.folds} samples")
            continue
        desc = describe_dataset(X)
        descriptor_rows.append({"dataset": name, **desc})
        print(f"[{di}/{len(datasets)}] {name}: n={len(y)}, d={X.shape[1]}")
        cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats,
                                     random_state=split_seed + di)
        split_id = 0
        for train_idx, test_idx in cv.split(X, y):
            split_id += 1
            fold_rows = evaluate_one_fold(
                X[train_idx], X[test_idx], y[train_idx], y[test_idx],
                args.k_values, args)
            for r in fold_rows:
                raw_rows.append({
                    "dataset": name, "split": split_id,
                    "repeat": (split_id-1)//args.folds + 1,
                    "fold": (split_id-1)%args.folds + 1, **r,
                })
    return pd.DataFrame(raw_rows), pd.DataFrame(descriptor_rows)


def aggregate_scores(raw):
    metrics = ["accuracy", "balanced_accuracy", "macro_f1",
               "mean_effective_k", "max_effective_k"]
    fold = raw.groupby(["dataset", "encoding", "metric", "k"], as_index=False)[metrics].mean()
    dataset_scores = fold.groupby(
        ["dataset", "encoding", "metric", "k"], as_index=False
    )[metrics].mean()
    overall = dataset_scores.groupby(
        ["encoding", "metric", "k"], as_index=False
    )[metrics].mean()
    overall = overall.sort_values(["k", "accuracy"], ascending=[True, False])
    return fold, dataset_scores, overall


def run_lodo_selector(dataset_scores, descriptors, args):
    """Outer LODO: fit descriptor-to-score model without the held-out dataset."""
    desc_cols = [c for c in descriptors.columns if c != "dataset"]
    merged = dataset_scores.merge(descriptors, on="dataset", how="inner")
    prediction_rows = []
    summary_rows = []

    for k in args.k_values:
        part = merged[merged.k == k].copy()
        names = sorted(part.dataset.unique())
        for heldout in names:
            train = part[part.dataset != heldout]
            test = part[part.dataset == heldout]
            # Fit one descriptor -> accuracy model per encoding-metric pair.
            # This is essential: otherwise identical descriptor rows for all
            # combinations would make every predicted score identical.
            heldout_desc = test[desc_cols].iloc[[0]].to_numpy(dtype=float)
            candidate_rows = []
            for enc, met in COMBINATIONS:
                tr_pair = train[(train.encoding == enc) & (train.metric == met)]
                te_pair = test[(test.encoding == enc) & (test.metric == met)]
                X_train = tr_pair[desc_cols].to_numpy(dtype=float)
                y_train = tr_pair.accuracy.to_numpy(dtype=float)
                mean = X_train.mean(axis=0)
                scale = X_train.std(axis=0)
                scale[scale < 1e-10] = 1.0
                Xs = (X_train-mean)/scale
                Xt = (heldout_desc-mean)/scale
                model = Ridge(alpha=args.ridge_alpha)
                model.fit(Xs, y_train)
                candidate_rows.append({
                    "encoding": enc, "metric": met,
                    "predicted_accuracy": float(model.predict(Xt)[0]),
                    "accuracy": float(te_pair.accuracy.iloc[0]),
                })
            test_candidates = pd.DataFrame(candidate_rows)
            chosen = test_candidates.sort_values(
                ["predicted_accuracy", "encoding", "metric"],
                ascending=[False, True, True]).iloc[0]
            actual_chosen = test[
                (test.encoding == chosen.encoding) & (test.metric == chosen.metric)
            ].iloc[0]

            default = test[
                (test.encoding == DEFAULT_COMBINATION[0]) &
                (test.metric == DEFAULT_COMBINATION[1])
            ].iloc[0]
            train_global = train.groupby(["encoding", "metric"], as_index=False).accuracy.mean()
            global_row = train_global.sort_values(
                ["accuracy", "encoding", "metric"],
                ascending=[False, True, True]).iloc[0]
            global_actual = test[
                (test.encoding == global_row.encoding) & (test.metric == global_row.metric)
            ].iloc[0]
            oracle = test.sort_values("accuracy", ascending=False).iloc[0]

            prediction_rows.append({
                "heldout_dataset": heldout, "k": k,
                "selected_encoding": chosen.encoding, "selected_metric": chosen.metric,
                "selected_actual_accuracy": actual_chosen.accuracy,
                "selected_predicted_accuracy": chosen.predicted_accuracy,
                "fixed_encoding": DEFAULT_COMBINATION[0],
                "fixed_metric": DEFAULT_COMBINATION[1],
                "fixed_actual_accuracy": default.accuracy,
                "training_global_encoding": global_row.encoding,
                "training_global_metric": global_row.metric,
                "training_global_actual_accuracy": global_actual.accuracy,
                "oracle_encoding": oracle.encoding, "oracle_metric": oracle.metric,
                "oracle_accuracy": oracle.accuracy,
            })

        pred_df_k = pd.DataFrame([r for r in prediction_rows if r["k"] == k])
        for col in ("selected_actual_accuracy", "fixed_actual_accuracy",
                    "training_global_actual_accuracy", "oracle_accuracy"):
            vals = pred_df_k[col].to_numpy()
            summary_rows.append({
                "k": k, "strategy": col.replace("_actual_accuracy", "").replace("_accuracy", ""),
                "mean_accuracy": float(np.mean(vals)),
                "std_across_datasets": float(np.std(vals, ddof=1)) if len(vals)>1 else np.nan,
                "n_datasets": len(vals),
            })
        # Paired dataset-level selector vs fixed baseline difference.
        delta = (pred_df_k.selected_actual_accuracy -
                 pred_df_k.fixed_actual_accuracy).to_numpy()
        summary_rows.append({
            "k": k, "strategy": "selector_minus_fixed",
            "mean_accuracy": float(delta.mean()),
            "std_across_datasets": float(delta.std(ddof=1)) if len(delta)>1 else np.nan,
            "n_datasets": len(delta),
        })
    return pd.DataFrame(prediction_rows), pd.DataFrame(summary_rows)


def descriptor_correlations(dataset_scores, descriptors):
    """Exploratory Spearman links between descriptors and combination accuracy."""
    merged = dataset_scores.merge(descriptors, on="dataset", how="inner")
    desc_cols = [c for c in descriptors.columns if c != "dataset"]
    rows = []
    for k in sorted(merged.k.unique()):
        for enc, met in COMBINATIONS:
            sub = merged[(merged.k == k) & (merged.encoding == enc) &
                         (merged.metric == met)]
            for col in desc_cols:
                if sub[col].nunique() < 3:
                    rho, p = np.nan, np.nan
                else:
                    rho, p = spearmanr(sub[col], sub.accuracy)
                rows.append({"k": k, "encoding": enc, "metric": met,
                             "descriptor": col, "spearman_rho": rho,
                             "p_value_unadjusted": p, "n_datasets": len(sub)})
    return pd.DataFrame(rows)


def dataset_winners(dataset_scores):
    idx = dataset_scores.groupby(["dataset", "k"]).accuracy.idxmax()
    winners = dataset_scores.loc[idx].copy()
    winners = winners.rename(columns={"encoding": "winner_encoding",
                                      "metric": "winner_metric",
                                      "accuracy": "winner_accuracy"})
    counts = winners.groupby(["k", "winner_encoding", "winner_metric"],
                             as_index=False).size()
    return winners, counts.sort_values(["k", "size"], ascending=[True, False])


def build_tables(raw, descriptors, fold, dataset_scores, overall,
                 lodo_predictions, lodo_summary, correlations, winners, counts, args):
    methodology = pd.DataFrame([
        ["Research question", "Can label-free dataset structure predict an encoding-metric pair on an unseen dataset?"],
        ["Evaluation unit", "Dataset; CV folds are aggregated before meta-learning and dataset-level comparisons."],
        ["Outer validation", "Leave one dataset out. Held-out dataset is excluded from selector fitting."],
        ["Selector inputs", "Label-free structural descriptors computed from X only."],
        ["Selector model", f"Ridge regression, alpha={args.ridge_alpha}; feature scaling fit within each outer training split."],
        ["Fixed baseline", "Amplitude + Euclidean."],
        ["Training-global baseline", "Best mean combination on outer-training datasets only."],
        ["Oracle", "Best observed combination on held-out dataset; upper bound, not deployable."],
        ["Tie handling", "All observations tied at kth distance are included in voting; effective k can exceed k."],
        ["Hamming", "Training-fold quantisation; Hamming remains a discretised metric for continuous encodings."],
        ["Amplitude equivalence", "L2-normalised nonnegative amplitude vectors make Euclidean and cosine rankings equivalent."],
        ["Quantum scope", "Classical feature-space k-NN only; no circuit, shots, QRAM, hardware or speedup claim."],
        ["Limitations", "Small dataset collection; synthetic tasks are controlled stress tests; LODO results are exploratory."],
    ], columns=["item", "description"])
    config = pd.DataFrame([
        ["folds", args.folds], ["repeats", args.repeats],
        ["n_synthetic", args.n_synthetic], ["data_seed", args.data_seed],
        ["split_seed", args.split_seed], ["basis_bits", args.basis_bits],
        ["hamming_levels", args.hamming_levels], ["k_values", str(args.k_values)],
        ["ridge_alpha", args.ridge_alpha],
        ["datasets_evaluated", ", ".join(sorted(dataset_scores.dataset.unique()))],
    ], columns=["parameter", "value"])
    return {
        "run_config": config, "methodology": methodology,
        "raw_fold_results": raw, "fold_summary": fold,
        "dataset_combination_scores": dataset_scores,
        "overall_combination_summary": overall,
        "dataset_descriptors": descriptors,
        "lodo_predictions": lodo_predictions, "lodo_selector_summary": lodo_summary,
        "descriptor_correlations": correlations,
        "dataset_winners": winners, "winner_counts": counts,
    }


def save_workbook(tables, outdir, export_csv):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "06_dataset_structure_selection_results.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            sheet = name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            ws.freeze_panes = "A2"
            if ws.max_row > 1:
                ws.auto_filter.ref = ws.dimensions
            for cells in ws.columns:
                letter = cells[0].column_letter
                width = max((len(str(c.value)) if c.value is not None else 0)
                            for c in cells[:min(ws.max_row, 500)])
                ws.column_dimensions[letter].width = min(max(width+2, 12), 52)
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, (float, np.floating)):
                        cell.number_format = "0.0000"
    if export_csv:
        for name, df in tables.items():
            df.to_csv(outdir / f"{name}.csv", index=False)
    return path


def parse_args():
    p = argparse.ArgumentParser(description="Study 06: dataset structure to encoding-metric selection.")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Optional subset: iris wine breast_cancer digits plus synthetic dataset names.")
    p.add_argument("--data-seed", type=int, default=2026)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--n-synthetic", type=int, default=900)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--basis-bits", type=int, default=2)
    p.add_argument("--hamming-levels", type=int, default=4)
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 9])
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--outdir", type=Path, default=Path("results/dataset_structure_selection"))
    p.add_argument("--csv", action="store_true", help="Also export individual CSV files.")
    args = p.parse_args()
    if args.folds < 2 or args.repeats < 1:
        p.error("--folds must be >=2 and --repeats >=1")
    if args.n_synthetic < 100:
        p.error("--n-synthetic must be >=100")
    if args.basis_bits < 1 or args.hamming_levels < 2:
        p.error("basis-bits must be >=1 and hamming-levels >=2")
    if any(k < 1 for k in args.k_values):
        p.error("All k values must be positive.")
    return args


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("="*88)
    print("QKNN STUDY 06 — DATASET STRUCTURE -> ENCODING/METRIC SELECTION")
    print("="*88)
    print(f"Repeated stratified CV: {args.folds} folds x {args.repeats} repeats")
    print(f"Encodings: {', '.join(ENCODINGS)} | Metrics: {', '.join(METRICS)}")
    print(f"k={args.k_values} | synthetic n={args.n_synthetic}")
    print("Outer validation: leave-one-dataset-out; selector uses X-only descriptors.")
    datasets = load_datasets(args.data_seed, args.n_synthetic)
    if args.datasets is not None:
        unknown = sorted(set(args.datasets) - set(datasets))
        if unknown:
            raise SystemExit(f"Unknown dataset(s): {unknown}. Available: {sorted(datasets)}")
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}

    raw, descriptors = evaluate_datasets(datasets, args)
    if raw.empty:
        raise SystemExit("No datasets were successfully evaluated.")
    fold, dataset_scores, overall = aggregate_scores(raw)
    lodo_predictions, lodo_summary = run_lodo_selector(dataset_scores, descriptors, args)
    correlations = descriptor_correlations(dataset_scores, descriptors)
    winners, counts = dataset_winners(dataset_scores)
    tables = build_tables(raw, descriptors, fold, dataset_scores, overall,
                          lodo_predictions, lodo_summary, correlations,
                          winners, counts, args)
    workbook = save_workbook(tables, args.outdir, args.csv)

    print("\nTOP OVERALL COMBINATIONS (dataset-weighted)")
    for k in args.k_values:
        print(f"\nk={k}")
        print(overall[overall.k == k].head(5)[
            ["encoding", "metric", "accuracy", "balanced_accuracy", "macro_f1"]
        ].to_string(index=False))

    print("\nLODO SELECTOR SUMMARY")
    print(lodo_summary.to_string(index=False))
    print(f"\nWorkbook: {workbook.resolve()}")
    if args.csv:
        print(f"CSV exports: {args.outdir.resolve()}")
    print("\nCompleted. Treat selector results as exploratory until validated on a")
    print("larger collection of independently sourced real-world datasets.")


if __name__ == "__main__":
    main()
