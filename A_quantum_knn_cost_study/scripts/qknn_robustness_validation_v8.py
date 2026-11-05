#!/usr/bin/env python3
"""QKNN robustness validation — version 8.

This is an ANALYSIS-ONLY follow-up to qknn_shots_study_v7.py.  It reuses the
per-query output produced by v7 and does not rerun or change the quantum
sampling simulation.  The v7 query-level file already contains records for
all four distance estimators (zero, one, avg, diff); v7's association report
only analysed the selected estimator.  v8 evaluates all four.

Analyses
--------
1. Within-(dataset, N, k) Spearman associations between robustness metrics
   and log10 classification-agreement shots, for each estimator/threshold.
2. Pooled OLS models at the primary 0.90 threshold (configurable), controlling
   for dataset, N, k, and log10(delta_s), with query-clustered sandwich standard
   errors. Reports incremental/partial R² over a controls-only model.
3. Dataset-specific versions of the adjusted models at the primary threshold,
   controlling for N, k, and log10(delta_s).
4. Within-stratum redundancy analysis for the two distance-based flip-cost
   metrics.

No pandas or statsmodels dependency is required. Requires numpy and scipy.

Example:
  python qknn_robustness_validation_v8.py \\
    --input results/shots_k_sensitivity_v7/qknn_query_level.csv \\
    --outdir results/robustness_validation_v8
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, t as student_t

ESTIMATORS = ("zero", "one", "avg", "diff")
METRICS = (
    "vote_margin",
    "min_class_flipping_swaps",
    "class_margin_distance_cost",
    "single_competitor_min_flip_penalty",
)
DISTANCE_COST_METRICS = {
    "class_margin_distance_cost",
    "single_competitor_min_flip_penalty",
}
REQUIRED = (
    "dataset", "N", "k", "data_seed", "test_index", "estimator",
    "threshold", "shots_classification_agreement", "delta_s",
    *METRICS,
)


def load_compact_csv(path):
    """Read only needed columns; keep the in-memory representation compact."""
    columns = {name: [] for name in REQUIRED}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [name for name in REQUIRED if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {missing}")
        for row in reader:
            columns["dataset"].append(row["dataset"])
            columns["estimator"].append(row["estimator"])
            for name in ("N", "k", "data_seed", "test_index"):
                columns[name].append(int(float(row[name])))
            for name in ("threshold", "shots_classification_agreement", "delta_s", *METRICS):
                try:
                    columns[name].append(float(row[name]))
                except (TypeError, ValueError):
                    columns[name].append(float("nan"))
    if not columns["dataset"]:
        raise ValueError("Input CSV contains no data rows")
    # Convert numerical columns to numpy arrays; categorical columns stay lists.
    for name in REQUIRED:
        if name not in ("dataset", "estimator"):
            columns[name] = np.asarray(columns[name], dtype=float)
    return columns


def subset_mask(data, estimator, threshold):
    return (
        (np.asarray(data["estimator"]) == estimator)
        & np.isclose(data["threshold"], threshold)
    )


def safe_spearman(x, y):
    if len(x) < 5 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else np.nan


def correlation_analysis(data, estimators, thresholds):
    out = []
    datasets = np.asarray(data["dataset"])
    for estimator in estimators:
        for threshold in thresholds:
            base = subset_mask(data, estimator, threshold)
            for dataset in sorted(set(datasets[base])):
                ds_mask = base & (datasets == dataset)
                for metric in METRICS:
                    rhos, counts = [], []
                    indices = np.flatnonzero(ds_mask)
                    # The original v7 analysis summarized within (dataset,N,k)
                    # correlations, avoiding between-N confounding.
                    groups = defaultdict(list)
                    for idx in indices:
                        groups[(int(data["N"][idx]), int(data["k"][idx]))].append(idx)
                    for ids in groups.values():
                        ids = np.asarray(ids, dtype=int)
                        x = data[metric][ids]
                        shots = data["shots_classification_agreement"][ids]
                        valid = np.isfinite(x) & np.isfinite(shots) & (shots > 0)
                        if metric in DISTANCE_COST_METRICS:
                            valid &= x > 0
                        else:
                            valid &= x >= 0
                        x, shots = x[valid], shots[valid]
                        rho = safe_spearman(x, np.log10(shots))
                        if np.isfinite(rho):
                            rhos.append(rho)
                            counts.append(len(x))
                    out.append({
                        "dataset": dataset,
                        "estimator": estimator,
                        "threshold": threshold,
                        "robustness_metric": metric,
                        "valid_N_k_groups": len(rhos),
                        "median_within_group_spearman_rho": float(np.median(rhos)) if rhos else np.nan,
                        "mean_within_group_spearman_rho": float(np.mean(rhos)) if rhos else np.nan,
                        "total_query_rows_in_valid_groups": int(sum(counts)),
                    })
    return out


def _dummy_columns(values, prefix):
    """Reference-coded categorical dummy matrix (first sorted level omitted)."""
    vals = np.asarray(values)
    levels = sorted(set(vals.tolist()), key=lambda x: str(x))
    if len(levels) <= 1:
        return [], []
    cols, names = [], []
    for level in levels[1:]:
        cols.append((vals == level).astype(float))
        names.append(f"{prefix}[{level}]")
    return cols, names


def build_design(data, ids, metric, include_dataset=True):
    ids = np.asarray(ids, dtype=int)
    xraw = data[metric][ids]
    shots = data["shots_classification_agreement"][ids]
    delta = data["delta_s"][ids]
    valid = np.isfinite(xraw) & np.isfinite(shots) & (shots > 0) & np.isfinite(delta) & (delta > 0)
    if metric in DISTANCE_COST_METRICS:
        valid &= xraw > 0
    else:
        valid &= xraw >= 0
    ids = ids[valid]
    xraw, shots, delta = xraw[valid], shots[valid], delta[valid]
    if len(ids) < 50:
        return None

    x = np.log10(xraw) if metric in DISTANCE_COST_METRICS else np.log1p(xraw)
    y = np.log10(shots)
    log_delta = np.log10(delta)

    columns = [np.ones(len(ids)), x, log_delta]
    names = ["intercept", "robustness_predictor", "log10_delta_s"]
    categorical = [(data["N"][ids].astype(int), "N"), (data["k"][ids].astype(int), "k")]
    if include_dataset:
        categorical.insert(0, (np.asarray(data["dataset"])[ids], "dataset"))
    for values, prefix in categorical:
        cols, colnames = _dummy_columns(values, prefix)
        columns.extend(cols)
        names.extend(colnames)

    X = np.column_stack(columns)
    # Query-level cluster is held fixed across N, k, and thresholds.
    ds = np.asarray(data["dataset"])[ids]
    seeds = data["data_seed"][ids].astype(int)
    tests = data["test_index"][ids].astype(int)
    if include_dataset:
        cluster_keys = [f"{a}:{b}:{c}" for a, b, c in zip(ds, seeds, tests)]
    else:
        cluster_keys = [f"{b}:{c}" for b, c in zip(seeds, tests)]
    return X, y, names, np.asarray(cluster_keys), ids


def fit_cluster_ols(X, y, names, clusters):
    """OLS with CR1 cluster sandwich covariance, implemented in numpy."""
    n, p = X.shape
    if n <= p + 2:
        raise ValueError("not enough observations for model degrees of freedom")
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)
    _, cluster_codes = np.unique(clusters, return_inverse=True)
    G = int(cluster_codes.max()) + 1 if len(cluster_codes) else 0
    if G < 2:
        raise ValueError("fewer than two independent query clusters")
    # Aggregate cluster score vectors in one vectorized pass.  A Python loop
    # over thousands of clusters is prohibitively slow for the 800k-row file.
    row_scores = X * residual[:, None]
    cluster_scores = np.zeros((G, p), dtype=float)
    np.add.at(cluster_scores, cluster_codes, row_scores)
    meat = cluster_scores.T @ cluster_scores
    correction = (G / (G - 1)) * ((n - 1) / max(n - p, 1))
    cov = correction * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    ybar = float(np.mean(y))
    sse = float(residual @ residual)
    sst = float(np.sum((y - ybar) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    j = names.index("robustness_predictor")
    tstat = beta[j] / se[j] if se[j] > 0 else np.nan
    dof = max(G - 1, 1)
    pvalue = float(2 * student_t.sf(abs(tstat), dof)) if np.isfinite(tstat) else np.nan
    crit = float(student_t.ppf(0.975, dof))
    return {
        "n": n, "clusters": G, "coefficient": float(beta[j]),
        "std_error_clustered_CR1": float(se[j]),
        "ci_low_95": float(beta[j] - crit * se[j]),
        "ci_high_95": float(beta[j] + crit * se[j]),
        "t_stat": float(tstat), "p_value_t_approx": pvalue,
        "r_squared": r2, "status": "fit",
    }


def regression_analysis(data, estimators, thresholds, dataset_thresholds=None):
    dataset_thresholds = set(thresholds if dataset_thresholds is None else dataset_thresholds)
    pooled_rows, dataset_rows = [], []
    datasets = np.asarray(data["dataset"])
    for estimator in estimators:
        for threshold in thresholds:
            base_mask = subset_mask(data, estimator, threshold)
            for metric in METRICS:
                ids = np.flatnonzero(base_mask)
                design = build_design(data, ids, metric, include_dataset=True)
                record = {"estimator": estimator, "threshold": threshold, "robustness_metric": metric}
                if design is None:
                    pooled_rows.append({**record, "status": "insufficient_valid_rows"})
                    continue
                X, y, names, clusters, used_ids = design
                try:
                    full = fit_cluster_ols(X, y, names, clusters)
                    # Incremental R²: compare with same controls, excluding robustness.
                    base_X = np.delete(X, names.index("robustness_predictor"), axis=1)
                    base_names = [n for n in names if n != "robustness_predictor"]
                    sse_full = float(np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2))
                    sse_base = float(np.sum((y - base_X @ np.linalg.lstsq(base_X, y, rcond=None)[0]) ** 2))
                    partial_r2 = (sse_base - sse_full) / sse_base if sse_base > 0 else np.nan
                    pooled_rows.append({**record, **full,
                                        "partial_r_squared_over_controls": float(partial_r2),
                                        "formula_description": "log10(shots) ~ transformed_robustness + log10(delta_s) + dataset + N + k"})
                except Exception as exc:
                    pooled_rows.append({**record, "status": f"fit_failed:{type(exc).__name__}:{exc}"})

                # Dataset-specific models; by default only primary threshold
                # 0.90 is fitted, to keep the large query-level analysis tractable.
                if threshold not in dataset_thresholds:
                    continue
                for dataset in sorted(set(datasets[base_mask])):
                    sub_ids = ids[datasets[ids] == dataset]
                    ddesign = build_design(data, sub_ids, metric, include_dataset=False)
                    drec = {**record, "dataset": dataset}
                    if ddesign is None:
                        dataset_rows.append({**drec, "status": "insufficient_valid_rows"})
                        continue
                    dX, dy, dnames, dclusters, _ = ddesign
                    try:
                        dfit = fit_cluster_ols(dX, dy, dnames, dclusters)
                        dataset_rows.append({**drec, **dfit,
                                             "formula_description": "log10(shots) ~ transformed_robustness + log10(delta_s) + N + k"})
                    except Exception as exc:
                        dataset_rows.append({**drec, "status": f"fit_failed:{type(exc).__name__}:{exc}"})
    return pooled_rows, dataset_rows


def redundancy_analysis(data, estimators, thresholds):
    metric_a = "class_margin_distance_cost"
    metric_b = "single_competitor_min_flip_penalty"
    out = []
    datasets = np.asarray(data["dataset"])
    for estimator in estimators:
        for threshold in thresholds:
            base = subset_mask(data, estimator, threshold)
            for dataset in sorted(set(datasets[base])):
                ids = np.flatnonzero(base & (datasets == dataset))
                groups = defaultdict(list)
                for idx in ids:
                    groups[(int(data["N"][idx]), int(data["k"][idx]))].append(idx)
                rhos = []
                exact_equal, valid_count = 0, 0
                for group_ids in groups.values():
                    ix = np.asarray(group_ids, dtype=int)
                    a, b = data[metric_a][ix], data[metric_b][ix]
                    valid = np.isfinite(a) & np.isfinite(b)
                    a, b = a[valid], b[valid]
                    if len(a) < 5:
                        continue
                    exact_equal += int(np.sum(np.isclose(a, b, rtol=1e-10, atol=1e-12)))
                    valid_count += len(a)
                    rho = safe_spearman(a, b)
                    if np.isfinite(rho):
                        rhos.append(rho)
                out.append({
                    "dataset": dataset, "estimator": estimator, "threshold": threshold,
                    "valid_N_k_groups": len(rhos),
                    "median_within_group_spearman_rho": float(np.median(rhos)) if rhos else np.nan,
                    "mean_within_group_spearman_rho": float(np.mean(rhos)) if rhos else np.nan,
                    "fraction_exactly_equal_rows": exact_equal / valid_count if valid_count else np.nan,
                    "valid_rows": valid_count,
                })
    return out


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Estimator-sensitivity and adjusted robustness validation for QKNN v7 results.")
    parser.add_argument("--input", required=True, help="v7 qknn_query_level.csv")
    parser.add_argument("--outdir", default="results/robustness_validation_v8")
    parser.add_argument("--estimators", nargs="+", choices=ESTIMATORS, default=list(ESTIMATORS))
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.8, 0.9, 0.95])
    parser.add_argument("--regression-thresholds", nargs="+", type=float, default=[0.9])
    parser.add_argument("--dataset-regression-thresholds", nargs="+", type=float, default=[0.9],
                        help="thresholds for dataset-specific regressions; default only 0.9")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"input file not found: {input_path}")
    if any(not 0 < x <= 1 for x in args.thresholds + args.regression_thresholds + args.dataset_regression_thresholds):
        parser.error("all thresholds must be in (0, 1]")
    args.thresholds = sorted(set(args.thresholds))
    args.regression_thresholds = sorted(set(args.regression_thresholds))
    args.dataset_regression_thresholds = sorted(set(args.dataset_regression_thresholds))
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading v7 query-level data: {input_path}", flush=True)
    data = load_compact_csv(input_path)
    print(f"Loaded {len(data['dataset']):,} query rows", flush=True)
    print(f"Estimators: {args.estimators}; correlation thresholds: {args.thresholds}", flush=True)
    print(f"Regression thresholds: {args.regression_thresholds}", flush=True)

    correlations = correlation_analysis(data, args.estimators, args.thresholds)
    pooled, by_dataset = regression_analysis(
        data, args.estimators, args.regression_thresholds, args.dataset_regression_thresholds
    )
    redundancy = redundancy_analysis(data, args.estimators, args.thresholds)

    write_csv(outdir / "qknn_v8_robustness_correlations.csv", correlations)
    write_csv(outdir / "qknn_v8_robustness_regression_pooled.csv", pooled)
    write_csv(outdir / "qknn_v8_robustness_regression_by_dataset.csv", by_dataset)
    write_csv(outdir / "qknn_v8_metric_redundancy.csv", redundancy)

    summary = outdir / "qknn_v8_summary.txt"
    with open(summary, "w") as f:
        f.write("QKNN v8 robustness validation summary\n")
        f.write("Analysis-only: reuses v7 query-level data; no new sampling simulation.\n")
        f.write(f"Input: {input_path}\nRows: {len(data['dataset'])}\n")
        f.write(f"Estimators: {', '.join(args.estimators)}\n")
        f.write(f"Correlation thresholds: {args.thresholds}\nRegression thresholds: {args.regression_thresholds}\n")
        f.write(f"Dataset-specific regression thresholds: {args.dataset_regression_thresholds}\n\n")
        f.write("Pooled adjusted regressions (coefficient < 0 indicates higher robustness is associated with fewer shots):\n")
        for row in pooled:
            f.write(f"{row.get('estimator')} | {row.get('threshold')} | {row.get('robustness_metric')} | "
                    f"coef={row.get('coefficient', float('nan'))} | CI=({row.get('ci_low_95', float('nan'))}, {row.get('ci_high_95', float('nan'))}) | "
                    f"partial_R2={row.get('partial_r_squared_over_controls', float('nan'))} | "
                    f"n={row.get('n', 0)} | clusters={row.get('clusters', 0)} | status={row.get('status')}\n")

    print("\nWrote:")
    for name in ("qknn_v8_robustness_correlations.csv", "qknn_v8_robustness_regression_pooled.csv",
                 "qknn_v8_robustness_regression_by_dataset.csv", "qknn_v8_metric_redundancy.csv", "qknn_v8_summary.txt"):
        print(f"  {outdir / name}")
    print("\nThis script does not rerun QKNN sampling. Review the pooled and dataset-specific regression tables before drawing conclusions.")


if __name__ == "__main__":
    main()
