#!/usr/bin/env python3
"""QKNN robustness validation — version 9 (analysis only).

Adds two validations to the V8 analysis, using the existing V7 per-query CSV:

1) Query fixed-effects regression.  Each query is identified by
   (dataset, data_seed, test_index).  This controls for all query attributes
   that are constant across that query's N/k observations.  Models are fitted
   separately for each estimator, threshold and robustness metric:

     log10(shots) ~ transformed_robustness + log10(delta_s)
                    + categorical(N) + categorical(k) + query fixed effects

   Query fixed effects are implemented by within-query demeaning. Standard
   errors use a query-clustered CR1 sandwich estimator.

2) Grouped held-out-query prediction.  Entire query groups are assigned to
   train or test, so observations from one query cannot leak across the split.
   Compares a baseline model (dataset + N + k + log10(delta_s)) with the same
   model augmented by one robustness metric. Reports held-out RMSE, MAE and
   R-squared on log10(shots).

No quantum circuits are executed and no new shots are sampled.
Dependencies: numpy, scipy. No pandas or statsmodels required.

Example:
  python qknn_robustness_validation_v9.py \
    --input results/shots_k_sensitivity_v7/qknn_query_level.csv \
    --outdir results/robustness_validation_v9 \
    --estimators avg --threshold 0.9
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import t as student_t

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
    "threshold", "shots_classification_agreement", "delta_s", *METRICS,
)


def load_compact_csv(path):
    cols = {name: [] for name in REQUIRED}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}")
        for row in reader:
            cols["dataset"].append(row["dataset"])
            cols["estimator"].append(row["estimator"])
            for c in ("N", "k", "data_seed", "test_index"):
                cols[c].append(int(float(row[c])))
            for c in ("threshold", "shots_classification_agreement", "delta_s", *METRICS):
                try:
                    cols[c].append(float(row[c]))
                except (TypeError, ValueError):
                    cols[c].append(float("nan"))
    if not cols["dataset"]:
        raise ValueError("Input CSV contains no data rows")
    for c in REQUIRED:
        if c not in ("dataset", "estimator"):
            cols[c] = np.asarray(cols[c], dtype=float)
    cols["dataset"] = np.asarray(cols["dataset"], dtype=str)
    cols["estimator"] = np.asarray(cols["estimator"], dtype=str)
    return cols


def query_keys(data, ids):
    return np.asarray([
        f"{data['dataset'][i]}:{int(data['data_seed'][i])}:{int(data['test_index'][i])}"
        for i in ids
    ], dtype=str)


def transformed_predictor(raw, metric):
    if metric in DISTANCE_COST_METRICS:
        return np.log10(raw)
    return np.log1p(raw)


def valid_rows(data, ids, metric):
    x = data[metric][ids]
    y = data["shots_classification_agreement"][ids]
    d = data["delta_s"][ids]
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0) & np.isfinite(d) & (d > 0)
    if metric in DISTANCE_COST_METRICS:
        ok &= x > 0
    else:
        ok &= x >= 0
    return ids[ok]


def categorical_columns(values, prefix):
    values = np.asarray(values)
    levels = sorted(set(values.tolist()), key=lambda z: str(z))
    columns, names = [], []
    for level in levels[1:]:
        columns.append((values == level).astype(float))
        names.append(f"{prefix}[{level}]")
    return columns, names


def make_fe_design(data, ids, metric):
    ids = valid_rows(data, ids, metric)
    if len(ids) < 100:
        return None
    xraw = data[metric][ids]
    x = transformed_predictor(xraw, metric)
    y = np.log10(data["shots_classification_agreement"][ids])
    loggap = np.log10(data["delta_s"][ids])
    cols = [x, loggap]
    names = ["robustness_predictor", "log10_delta_s"]
    for vals, prefix in ((data["N"][ids].astype(int), "N"),
                         (data["k"][ids].astype(int), "k")):
        c, n = categorical_columns(vals, prefix)
        cols.extend(c); names.extend(n)
    X = np.column_stack(cols)
    groups, codes = np.unique(query_keys(data, ids), return_inverse=True)
    counts = np.bincount(codes).astype(float)
    # Within-query demeaning absorbs a separate intercept for every query.
    xsum = np.zeros((len(groups), X.shape[1]), dtype=float)
    ysum = np.zeros(len(groups), dtype=float)
    np.add.at(xsum, codes, X)
    np.add.at(ysum, codes, y)
    Xw = X - (xsum / counts[:, None])[codes]
    yw = y - (ysum / counts)[codes]
    # Drop columns with no within-query variation (e.g. a constant feature).
    keep = np.std(Xw, axis=0) > 1e-12
    Xw = Xw[:, keep]
    kept_names = [n for n, flag in zip(names, keep) if flag]
    if "robustness_predictor" not in kept_names:
        return None
    return Xw, yw, kept_names, codes, groups, ids


def cluster_ols(X, y, names, cluster_codes):
    n, p = X.shape
    if n <= p + 2:
        raise ValueError("insufficient residual degrees of freedom")
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    inv = np.linalg.pinv(X.T @ X)
    G = int(cluster_codes.max()) + 1
    if G < 2:
        raise ValueError("fewer than two query clusters")
    scores = np.zeros((G, p), dtype=float)
    np.add.at(scores, cluster_codes, X * resid[:, None])
    meat = scores.T @ scores
    correction = (G / (G - 1)) * ((n - 1) / max(n - p, 1))
    cov = correction * inv @ meat @ inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    j = names.index("robustness_predictor")
    dof = max(G - 1, 1)
    crit = float(student_t.ppf(0.975, dof))
    tstat = beta[j] / se[j] if se[j] > 0 else np.nan
    pval = float(2 * student_t.sf(abs(tstat), dof)) if np.isfinite(tstat) else np.nan
    return beta, {
        "n_rows": n, "n_query_clusters": G,
        "coefficient": float(beta[j]), "clustered_se_CR1": float(se[j]),
        "ci_low_95": float(beta[j] - crit * se[j]),
        "ci_high_95": float(beta[j] + crit * se[j]),
        "t_stat": float(tstat), "p_value_t_approx": pval,
    }


def fixed_effect_analysis(data, estimators, threshold):
    rows = []
    for est in estimators:
        base = np.flatnonzero((data["estimator"] == est) & np.isclose(data["threshold"], threshold))
        for metric in METRICS:
            design = make_fe_design(data, base, metric)
            rec = {"estimator": est, "threshold": threshold, "robustness_metric": metric}
            if design is None:
                rows.append({**rec, "status": "insufficient_valid_rows_or_no_within_query_variation"})
                continue
            X, y, names, codes, groups, ids = design
            try:
                _, fit = cluster_ols(X, y, names, codes)
                # Partial within-query R2: reduction in within-query SSE vs
                # the same within-transformed controls without robustness.
                j = names.index("robustness_predictor")
                base_X = np.delete(X, j, axis=1)
                full_sse = float(np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2))
                base_sse = float(np.sum((y - base_X @ np.linalg.lstsq(base_X, y, rcond=None)[0]) ** 2)) if base_X.shape[1] else float(y @ y)
                partial = (base_sse - full_sse) / base_sse if base_sse > 0 else np.nan
                rows.append({**rec, **fit, "partial_within_r_squared": float(partial),
                             "n_queries": int(len(groups)), "status": "fit",
                             "formula": "query_FE: log10(shots) ~ transformed_robustness + log10(delta_s) + categorical(N) + categorical(k)"})
            except Exception as exc:
                rows.append({**rec, "status": f"fit_failed:{type(exc).__name__}:{exc}"})
    return rows


def prediction_design(data, ids, metric=None, levels=None):
    """Design for out-of-query prediction; no query FE (unseen queries)."""
    datasets = data["dataset"][ids]
    nvals = data["N"][ids].astype(int)
    kvals = data["k"][ids].astype(int)
    if levels is None:
        levels = {
            "dataset": sorted(set(data["dataset"].tolist())),
            "N": sorted(set(data["N"].astype(int).tolist())),
            "k": sorted(set(data["k"].astype(int).tolist())),
        }
    cols = [np.ones(len(ids)), np.log10(data["delta_s"][ids])]
    names = ["intercept", "log10_delta_s"]
    for vals, key in ((datasets, "dataset"), (nvals, "N"), (kvals, "k")):
        levs = levels[key]
        for level in levs[1:]:
            cols.append((vals == level).astype(float))
            names.append(f"{key}[{level}]")
    if metric is not None:
        cols.append(transformed_predictor(data[metric][ids], metric))
        names.append("robustness_predictor")
    return np.column_stack(cols), names, levels


def regression_predict(Xtrain, ytrain, Xtest):
    beta, *_ = np.linalg.lstsq(Xtrain, ytrain, rcond=None)
    return Xtest @ beta


def prediction_metrics(y, pred):
    err = y - pred
    sse = float(err @ err)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return {"rmse_log10_shots": float(np.sqrt(np.mean(err ** 2))),
            "mae_log10_shots": float(np.mean(np.abs(err))),
            "test_r_squared": float(1 - sse / sst) if sst > 0 else np.nan}


def heldout_prediction_analysis(data, estimators, threshold, test_fraction, seed, repeats):
    rows = []
    for est_index, est in enumerate(estimators):
        est_ids = np.flatnonzero((data["estimator"] == est) & np.isclose(data["threshold"], threshold))
        if len(est_ids) == 0:
            continue
        # Split unique query groups, not individual rows: no query leakage.
        keys = query_keys(data, est_ids)
        unique = np.unique(keys)
        if len(unique) < 10:
            rows.append({"estimator": est, "threshold": threshold, "status": "too_few_query_groups"})
            continue
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + 100003 * repeat + est_index)
            shuffled = rng.permutation(unique)
            ntest = max(1, int(round(test_fraction * len(unique))))
            test_groups = set(shuffled[:ntest].tolist())
            test_mask = np.asarray([k in test_groups for k in keys])
            train_ids, test_ids = est_ids[~test_mask], est_ids[test_mask]
            for metric in METRICS:
                tr = valid_rows(data, train_ids, metric)
                te = valid_rows(data, test_ids, metric)
                rec = {"estimator": est, "threshold": threshold, "repeat": repeat + 1, "robustness_metric": metric,
                       "train_query_groups": int(len(np.unique(query_keys(data, tr)))),
                       "test_query_groups": int(len(np.unique(query_keys(data, te)))),
                       "train_rows": int(len(tr)), "test_rows": int(len(te)),
                       "split_seed": seed + 100003 * repeat + est_index, "test_fraction": test_fraction}
                if len(tr) < 100 or len(te) < 30:
                    rows.append({**rec, "status": "insufficient_train_or_test_rows"})
                    continue
                # Use shared category levels so train and test matrices align.
                levels = {"dataset": sorted(set(data["dataset"].tolist())),
                          "N": sorted(set(data["N"].astype(int).tolist())),
                          "k": sorted(set(data["k"].astype(int).tolist()))}
                ytr = np.log10(data["shots_classification_agreement"][tr])
                yte = np.log10(data["shots_classification_agreement"][te])
                try:
                    Xtr0, _, _ = prediction_design(data, tr, None, levels)
                    Xte0, _, _ = prediction_design(data, te, None, levels)
                    Xtr1, _, _ = prediction_design(data, tr, metric, levels)
                    Xte1, _, _ = prediction_design(data, te, metric, levels)
                    p0 = regression_predict(Xtr0, ytr, Xte0)
                    p1 = regression_predict(Xtr1, ytr, Xte1)
                    m0, m1 = prediction_metrics(yte, p0), prediction_metrics(yte, p1)
                    rows.append({**rec, "baseline_rmse_log10_shots": m0["rmse_log10_shots"],
                                 "robustness_rmse_log10_shots": m1["rmse_log10_shots"],
                                 "rmse_reduction_fraction": ((m0["rmse_log10_shots"] - m1["rmse_log10_shots"]) / m0["rmse_log10_shots"] if m0["rmse_log10_shots"] > 0 else np.nan),
                                 "baseline_mae_log10_shots": m0["mae_log10_shots"],
                                 "robustness_mae_log10_shots": m1["mae_log10_shots"],
                                 "baseline_test_r_squared": m0["test_r_squared"],
                                 "robustness_test_r_squared": m1["test_r_squared"],
                                 "status": "fit"})
                except Exception as exc:
                    rows.append({**rec, "status": f"prediction_failed:{type(exc).__name__}:{exc}"})
    return rows


def summarize_prediction_repeats(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("status") == "fit":
            grouped[(row["estimator"], row["threshold"], row["robustness_metric"])].append(row)
    out = []
    for (est, threshold, metric), items in sorted(grouped.items()):
        reductions = np.asarray([x["rmse_reduction_fraction"] for x in items], dtype=float)
        base_rmse = np.asarray([x["baseline_rmse_log10_shots"] for x in items], dtype=float)
        aug_rmse = np.asarray([x["robustness_rmse_log10_shots"] for x in items], dtype=float)
        out.append({"estimator": est, "threshold": threshold, "robustness_metric": metric,
                    "successful_repeats": len(items),
                    "mean_rmse_reduction_fraction": float(np.mean(reductions)),
                    "std_rmse_reduction_fraction": float(np.std(reductions, ddof=1)) if len(reductions) > 1 else 0.0,
                    "mean_baseline_rmse_log10_shots": float(np.mean(base_rmse)),
                    "mean_augmented_rmse_log10_shots": float(np.mean(aug_rmse)),
                    "fraction_repeats_improved": float(np.mean(reductions > 0))})
    return out


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="QKNN V9 query fixed-effects and held-out-query validation (analysis only).")
    p.add_argument("--input", required=True, help="V7 qknn_query_level.csv")
    p.add_argument("--outdir", default="results/robustness_validation_v9")
    p.add_argument("--estimators", nargs="+", choices=("zero", "one", "avg", "diff"), default=["avg"])
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=10, help="independent grouped holdout splits")
    args = p.parse_args()
    if not 0 < args.threshold <= 1:
        p.error("--threshold must be in (0, 1]")
    if not 0.05 <= args.test_fraction <= 0.5:
        p.error("--test-fraction must be between 0.05 and 0.5")
    if args.repeats < 1:
        p.error("--repeats must be at least 1")
    inp = Path(args.input).expanduser().resolve()
    if not inp.is_file():
        p.error(f"input file not found: {inp}")
    outdir = Path(args.outdir).expanduser().resolve(); outdir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {inp}", flush=True)
    data = load_compact_csv(inp)
    print(f"Loaded {len(data['dataset']):,} rows", flush=True)
    print(f"Estimators={args.estimators}; threshold={args.threshold}; held-out fraction={args.test_fraction}", flush=True)
    fe = fixed_effect_analysis(data, args.estimators, args.threshold)
    print("Completed query fixed-effects models", flush=True)
    pred = heldout_prediction_analysis(data, args.estimators, args.threshold, args.test_fraction, args.seed, args.repeats)
    pred_summary = summarize_prediction_repeats(pred)
    print("Completed grouped held-out-query prediction", flush=True)
    write_csv(outdir / "qknn_v9_query_fixed_effects.csv", fe)
    write_csv(outdir / "qknn_v9_heldout_query_prediction.csv", pred)
    write_csv(outdir / "qknn_v9_heldout_query_prediction_summary.csv", pred_summary)
    with open(outdir / "qknn_v9_summary.txt", "w") as f:
        f.write("QKNN V9 statistical validation\n")
        f.write("Analysis-only: no quantum circuits or new sampling.\n")
        f.write(f"Input: {inp}\nRows: {len(data['dataset'])}\n")
        f.write(f"Estimators: {args.estimators}\nThreshold: {args.threshold}\n")
        f.write(f"Held-out query fraction: {args.test_fraction}; split seed: {args.seed}; repeats: {args.repeats}\n\n")
        f.write("Query fixed-effects: negative robustness coefficient means higher robustness is associated with fewer shots within the same query, conditional on delta_s, N and k.\n")
        for r in fe:
            f.write(f"{r.get('estimator')} | {r.get('robustness_metric')} | coef={r.get('coefficient')} | CI=({r.get('ci_low_95')},{r.get('ci_high_95')}) | partial_within_R2={r.get('partial_within_r_squared')} | queries={r.get('n_queries')} | status={r.get('status')}\n")
        f.write("\nHeld-out query prediction: positive RMSE reduction means adding robustness improves prediction on unseen query groups. This is predictive evidence, not causality.\n")
        for r in pred_summary:
            f.write(f"{r.get('estimator')} | {r.get('robustness_metric')} | mean_RMSE_reduction={r.get('mean_rmse_reduction_fraction')} | std={r.get('std_rmse_reduction_fraction')} | improved_fraction={r.get('fraction_repeats_improved')} | repeats={r.get('successful_repeats')}\n")
    print("\nWrote:")
    for name in ("qknn_v9_query_fixed_effects.csv", "qknn_v9_heldout_query_prediction.csv", "qknn_v9_heldout_query_prediction_summary.csv", "qknn_v9_summary.txt"):
        print(outdir / name)
    print("\nInterpretation: this tests within-query association and out-of-query predictive value; neither alone establishes causality.")

if __name__ == "__main__":
    main()
