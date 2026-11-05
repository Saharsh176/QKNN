"""
qknn_shots_study.py
===================
Study 2 (version 6): paired k-sensitivity shot requirements of the authors' quantum k-NN
probability and distance-estimation model.

This version fixes two experimental-design issues:
1. A fixed test set is used for every training-set size N within each data seed.
   Training sets are nested prefixes of one maximum-size training pool.
2. Replicates retain their hierarchy:
       data seed -> shot-sampling trial -> shot count -> metric
   Confidence intervals use a hierarchical bootstrap: resample data seeds as
   independent units, then resample trials within each selected seed. The
   same resampled data-seed indices are used across N, preserving pairing.

The data normalization is still refitted on each N-sized training subset,
matching the authors' training-derived normalization procedure. The raw test
examples remain fixed across N, but their normalized coordinates can therefore
change as the training-derived min/max values change.

IMPORTANT MODEL SCOPE
--------------------
Counts are sampled from the ideal authors' multinomial probabilities. This
models finite-shot sampling only; it does not include hardware noise, gate
errors, or circuit execution overhead. A shot is one circuit execution, and
the training state is assumed to be reloaded for every shot.

RUN
---
    python scripts/validate_against_authors.py
    python scripts/qknn_shots_study.py
    python scripts/qknn_shots_study.py --encoding translation --datasets blobs wine
    python scripts/qknn_shots_study.py --n-test 100 --data-seeds 10 --trials 5
    python scripts/qknn_shots_study.py --k-values 1 3 5 9 --margins --data-seeds 10 --trials 5

Version 6 adds an exhaustive single-competitor class-flip distance penalty. For each query it
searches all subsets of selected neighbours and all single competing classes, adding the
nearest excluded points of that class. The resulting penalty is exact within this attack
family, not over mixed-class replacement attacks.

Outputs are written to <workspace>/results/shots_k_sensitivity by default, preserving previous results/shots files; override with --outdir.
"""

import argparse
import csv
import itertools
import os
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris, load_wine, make_blobs


ESTIMATORS = ("zero", "one", "avg", "diff")
METRICS = ("jaccard", "exact_match", "agree_with_classical", "accuracy")
RESULTS = Path(__file__).resolve().parent.parent / "results"


# --------------------------------------------------------------------------- #
# Data: nested training sets and a fixed test set for each data seed
# --------------------------------------------------------------------------- #
def load_raw(name, n_train, n_test, seed, max_n_train):
    """
    Generate/load one fixed dataset split.

    For a given (name, seed, max_n_train, n_test), all N values use:
      * the same raw test examples;
      * nested training sets, where the N-sized set is a prefix of the
        max_n_train-sized training pool.
    """
    total = max_n_train + n_test

    if name == "blobs":
        X, y = make_blobs(
            n_samples=total,
            centers=3,
            n_features=8,
            cluster_std=2.0,
            random_state=seed,
        )
    else:
        loader = {
            "iris": load_iris,
            "wine": load_wine,
            "digits": load_digits,
        }[name]
        X, y = loader(return_X_y=True)
        idx = np.random.RandomState(seed).permutation(len(X))
        X, y = X[idx], y[idx]

    Xtr_pool = X[:max_n_train]
    ytr_pool = y[:max_n_train]
    Xte = X[max_n_train:max_n_train + n_test]
    yte = y[max_n_train:max_n_train + n_test]

    if len(Xtr_pool) != max_n_train or len(Xte) != n_test:
        raise ValueError(
            f"Could not construct requested split for {name}: "
            f"training pool={len(Xtr_pool)}/{max_n_train}, "
            f"test={len(Xte)}/{n_test}"
        )

    return (
        Xtr_pool[:n_train],
        ytr_pool[:n_train],
        Xte,
        yte,
    )


def authors_normalize(Xtr, Xte):
    """Training-derived normalization used by the authors' data loader."""
    d = Xtr.shape[1]
    sq = np.sqrt(d)
    mid = (Xtr.min(0) + Xtr.max(0)) / 2
    rng = Xtr.max(0) - Xtr.min(0)
    rng = rng.copy()
    rng[rng == 0] = 1.0

    Ztr = (Xtr - mid) / (rng * sq)
    Zte = np.clip(
        (Xte - mid) / (rng * sq),
        -1 / (2 * sq),
        1 / (2 * sq),
    )
    return Ztr, Zte


def dataset_size(name):
    return {
        "blobs": 10**9,
        "iris": 150,
        "wine": 178,
        "digits": 1797,
    }[name]


# --------------------------------------------------------------------------- #
# Authors' probability model, shot sampling, and distance estimators
# --------------------------------------------------------------------------- #
def scalar_products(Ztr, zt, encoding):
    base = (Ztr ** 2).sum(1) - 2 * Ztr @ zt
    return (4 / 3) * base if encoding == "extension" else base - 0.25


def sample_counts(s, S, rng):
    N = len(s)
    p = np.concatenate(((1 + s) / (2 * N), (1 - s) / (2 * N)))
    p = np.clip(p, 0, None)
    p /= p.sum()
    counts = rng.multinomial(S, p)
    return counts[:N], counts[N:]


def estimate_distances(n0, n1, S, tn2, encoding, pseudocounts):
    N = len(n0)
    total = S + pseudocounts * 2 * N
    p0 = (n0 + pseudocounts) / total
    p1 = (n1 + pseudocounts) / total

    offset = 0.0 if encoding == "extension" else 0.25
    coefficient = 0.75 if encoding == "extension" else 1.0

    def distance(sp):
        return np.sqrt(
            np.clip(coefficient * sp + offset + tn2, 0.0, 1.0)
        )

    d0 = distance(2 * N * p0 - 1)
    d1 = distance(1 - 2 * N * p1)
    return {
        "zero": d0,
        "one": d1,
        "avg": 0.5 * (d0 + d1),
        "diff": distance(N * (p0 - p1)),
    }


def top_k(values, k, rng):
    # Tiny random jitter gives reproducible randomized tie-breaking.
    return set(
        np.argsort(values + 1e-12 * rng.random(len(values)))[:k].tolist()
    )


# --------------------------------------------------------------------------- #
# Main experiment: one shot sample is shared across all requested k values
# --------------------------------------------------------------------------- #
def exact_single_competitor_flip_penalty(order_row, squared_distances, labels, k):
    """Minimum added squared-distance cost for a class-changing replacement.

    Enumerates every subset of m currently selected neighbours to remove. For
    each competing class, it adds that class's m closest excluded points.
    Among all such single-competitor replacements that change the k-NN
    prediction, return the minimum sum(d2_added - d2_removed).

    This is exact for the stated single-competitor replacement family. It is
    not a global optimum over replacements mixing multiple competing classes.
    """
    selected = np.asarray(order_row[:k], dtype=int)
    outside = np.asarray(order_row[k:], dtype=int)
    selected_labels = np.asarray(labels[selected])
    classes, counts = np.unique(selected_labels, return_counts=True)
    winner = classes[int(np.argmax(counts))]
    original_prediction = np.bincount(selected_labels).argmax()

    best_penalty = np.inf
    for competitor in np.unique(labels[outside]):
        if competitor == winner:
            continue
        competitor_pool = outside[labels[outside] == competitor]
        competitor_pool = competitor_pool[np.argsort(squared_distances[competitor_pool])]

        for m in range(1, min(k, len(competitor_pool)) + 1):
            added = competitor_pool[:m]
            added_cost = float(np.sum(squared_distances[added]))
            for removed_tuple in itertools.combinations(selected.tolist(), m):
                removed = np.asarray(removed_tuple, dtype=int)
                new_labels = np.concatenate((
                    labels[np.setdiff1d(selected, removed, assume_unique=False)],
                    np.full(m, competitor, dtype=labels.dtype),
                ))
                new_prediction = np.bincount(new_labels).argmax()
                if new_prediction != original_prediction:
                    penalty = added_cost - float(np.sum(squared_distances[removed]))
                    if penalty < best_penalty:
                        best_penalty = penalty

    return float(best_penalty) if np.isfinite(best_penalty) else np.nan


def class_robustness_diagnostics(order_row, distances, labels, k):
    """Measure how many selected-neighbour replacements can change k-NN's vote.

    For each class represented outside the exact top-k set, greedily replace a
    selected member of the currently leading non-competitor class with the
    nearest outside point of the competitor class. The first competitor that
    overtakes the current prediction with the fewest swaps defines the minimum.
    Ties follow numpy.bincount(...).argmax(): the smallest class label wins.
    """
    selected = np.asarray(order_row[:k], dtype=int)
    outside = np.asarray(order_row[k:], dtype=int)
    selected_labels = labels[selected]
    classes, counts = np.unique(selected_labels, return_counts=True)
    winner = classes[int(np.argmax(counts))]
    winner_votes = int(np.max(counts))
    runner_up = int(np.max(counts[classes != winner])) if np.any(classes != winner) else 0
    vote_margin = winner_votes - runner_up

    best_swaps = np.nan
    best_distance_cost = np.nan
    best_competitor = None
    for competitor in np.unique(labels[outside]):
        if competitor == winner:
            continue
        current = {c.item() if hasattr(c, "item") else c: int(np.sum(selected_labels == c))
                   for c in np.unique(labels)}
        available = {c.item() if hasattr(c, "item") else c:
                     list(selected[labels[selected] == c]) for c in np.unique(selected_labels)}
        add_pool = outside[labels[outside] == competitor]
        add_pool = add_pool[np.argsort(distances[add_pool])]
        removed_points = []
        cost = 0.0
        swaps = 0

        def competitor_has_won():
            other = [(v, c) for c, v in current.items() if c != competitor]
            max_other = max((v for v, _ in other), default=0)
            tied_classes = [c for v, c in other if v == max_other]
            return (current.get(competitor, 0) > max_other or
                    (current.get(competitor, 0) == max_other and
                     (not tied_classes or competitor < min(tied_classes))))

        while not competitor_has_won() and swaps < min(k, len(add_pool)):
            # Remove from the strongest opposing class. On a tie, removing
            # any tied leader makes equal progress; choose the one with a
            # selected point farthest from the query (least costly boundary swap).
            max_other = max((v for c, v in current.items() if c != competitor), default=0)
            leaders = [c for c, v in current.items() if c != competitor and v == max_other and v > 0]
            candidates = [(float(distances[idx]), c, idx)
                          for c in leaders for idx in available.get(c, [])]
            if not candidates or swaps >= len(add_pool):
                break
            _, remove_class, remove_idx = max(candidates, key=lambda x: x[0])
            add_idx = int(add_pool[swaps])
            available[remove_class].remove(remove_idx)
            current[remove_class] -= 1
            current[competitor] = current.get(competitor, 0) + 1
            cost += float(distances[add_idx] - distances[remove_idx])
            removed_points.append(int(remove_idx))
            swaps += 1

        if competitor_has_won():
            if (not np.isfinite(best_swaps) or swaps < best_swaps or
                    (swaps == best_swaps and cost < best_distance_cost)):
                best_swaps = int(swaps)
                best_distance_cost = float(cost)
                best_competitor = competitor.item() if hasattr(competitor, "item") else competitor

    exact_flip_penalty = exact_single_competitor_flip_penalty(
        order_row, distances, labels, k
    )

    return {
        "vote_margin": int(vote_margin),
        "min_class_flipping_swaps": best_swaps,
        "class_margin_distance_cost": best_distance_cost,
        "single_competitor_min_flip_penalty": exact_flip_penalty,
        "class_flipping_competitor": best_competitor,
        "classical_prediction": winner.item() if hasattr(winner, "item") else winner,
    }


def run(args):
    rng = np.random.default_rng(args.seed)
    shots = [2**e for e in range(args.min_exp, args.max_exp + 1)]
    k_values = list(args.k_values)
    reps = {}       # (dataset, N, k, estimator) -> (seeds, trials, shots, metrics)
    margins = []
    result_rows = []
    t_start = time.time()

    for name in args.datasets:
        valid_N = sorted({N for N in args.n_grid if N + args.n_test <= dataset_size(name)})
        if not valid_N:
            print(f"  skipping {name}: no N fits with n_test={args.n_test}")
            continue
        if any(N <= max(k_values) for N in valid_N):
            invalid = [N for N in valid_N if N <= max(k_values)]
            raise ValueError(f"Every training size must exceed max(k)={max(k_values)}; invalid sizes: {invalid}")

        max_n_train = max(valid_N)
        print(f"\n{name}: fixed raw test set of {args.n_test}; nested training pool of {max_n_train}; k={k_values}", flush=True)
        # Store a complete seed block for each (N,k,estimator).
        store = {(N, k, est): [] for N in valid_N for k in k_values for est in ESTIMATORS}

        for ds in range(args.data_seeds):
            Xtr_pool, ytr_pool, Xte, yte = load_raw(
                name=name, n_train=max_n_train, n_test=args.n_test,
                seed=1000 + ds, max_n_train=max_n_train,
            )
            T = len(Xte)
            for N in valid_N:
                Xtr, ytr = Xtr_pool[:N], ytr_pool[:N]
                Ztr, Zte = authors_normalize(Xtr, Xte)
                d2 = ((Zte[:, None, :] - Ztr[None, :, :]) ** 2).sum(-1)
                scalar_rows = [scalar_products(Ztr, Zte[i], args.encoding) for i in range(T)]
                target_norm2 = (Zte ** 2).sum(1)
                true_orders = np.argsort(d2, axis=1)
                true_sets = {k: [set(true_orders[i, :k].tolist()) for i in range(T)] for k in k_values}
                true_preds = {}
                for k in k_values:
                    true_preds[k] = [np.bincount(ytr[true_orders[i, :k]]).argmax() for i in range(T)]
                    if args.margins:
                        for i in range(T):
                            order_row = true_orders[i]
                            sorted_d2 = d2[i, order_row]
                            kth, next_k = order_row[k - 1], order_row[k]
                            delta_s = abs(scalar_rows[i][kth] - scalar_rows[i][next_k])
                            diag = class_robustness_diagnostics(order_row, d2[i], ytr, k)
                            margins.append({
                                "dataset": name, "N": N, "k": k,
                                "data_seed": ds, "test_index": i,
                                "gap": float(sorted_d2[k] - sorted_d2[k - 1]),
                                "rel_gap": float((sorted_d2[k] - sorted_d2[k - 1]) / max(sorted_d2[k - 1], 1e-12)),
                                "delta_s": float(delta_s),
                                "delta_p": float(delta_s / (2 * N)),
                                "predicted_factor": float(N / max(delta_s**2, 1e-300)),
                                "classical_accuracy": float(true_preds[k][i] == yte[i]),
                                **diag,
                            })

                # Trial arrays for every k and estimator. The same multinomial
                # counts and estimator ordering are reused for every k, making
                # comparisons paired and reducing Monte-Carlo noise.
                curves = {
                    (k, est): np.zeros((args.trials, len(shots), len(METRICS)), dtype=float)
                    for k in k_values for est in ESTIMATORS
                }
                for si, S in enumerate(shots):
                    for trial in range(args.trials):
                        trial_values = {(k, est): [] for k in k_values for est in ESTIMATORS}
                        for i in range(T):
                            n0, n1 = sample_counts(scalar_rows[i], S, rng)
                            estimates = estimate_distances(
                                n0=n0, n1=n1, S=S, tn2=target_norm2[i],
                                encoding=args.encoding, pseudocounts=args.alpha,
                            )
                            for est in ESTIMATORS:
                                # One tie-break perturbation/order per estimator,
                                # shared by all k via prefixes of this ordering.
                                jitter = 1e-12 * rng.random(N)
                                estimated_order = np.argsort(estimates[est] + jitter)
                                for k in k_values:
                                    found = set(estimated_order[:k].tolist())
                                    intersection = len(found & true_sets[k][i])
                                    jaccard = intersection / (2 * k - intersection)
                                    pred = np.bincount(ytr[estimated_order[:k]]).argmax()
                                    trial_values[(k, est)].append((
                                        jaccard,
                                        float(found == true_sets[k][i]),
                                        float(pred == true_preds[k][i]),
                                        float(pred == yte[i]),
                                    ))
                        for key in curves:
                            curves[key][trial, si] = np.mean(trial_values[key], axis=0)

                for k in k_values:
                    for est in ESTIMATORS:
                        store[(N, k, est)].append(curves[(k, est)])

            print(f"  completed {name} data seed {ds + 1}/{args.data_seeds} ({time.time() - t_start:.0f}s)", flush=True)

        for N in valid_N:
            for k in k_values:
                for est in ESTIMATORS:
                    arr = np.stack(store[(N, k, est)], axis=0)
                    reps[(name, N, k, est)] = arr
                    mean_curve = arr.mean(axis=(0, 1))
                    for si, S in enumerate(shots):
                        result_rows.append({
                            "dataset": name, "N": N, "k": k, "shots": S, "estimator": est,
                            "jaccard": mean_curve[si, 0],
                            "exact_match": mean_curve[si, 1],
                            "agree_with_classical": mean_curve[si, 2],
                            "accuracy": mean_curve[si, 3],
                        })

    return shots, reps, result_rows, margins


# --------------------------------------------------------------------------- #
# Threshold crossing and scaling fits
# --------------------------------------------------------------------------- #
def shots_needed(curve, shots, threshold):
    """First threshold crossing, interpolated linearly in log2(shots)."""
    if curve[0] >= threshold:
        return float(shots[0])

    for i in range(len(shots) - 1):
        y0, y1 = curve[i], curve[i + 1]
        if y0 < threshold <= y1:
            fraction = (threshold - y0) / max(y1 - y0, 1e-12)
            return float(
                2 ** (
                    np.log2(shots[i])
                    + fraction * (np.log2(shots[i + 1]) - np.log2(shots[i]))
                )
            )

    return np.nan


def fit_slope(Ns, shot_values):
    Ns = np.asarray(Ns, dtype=float)
    shot_values = np.asarray(shot_values, dtype=float)
    valid = (
        np.isfinite(shot_values)
        & (shot_values > 0)
        & (Ns > 0)
    )
    if valid.sum() < 3:
        return np.nan
    return float(np.polyfit(
        np.log2(Ns[valid]),
        np.log2(shot_values[valid]),
        1,
    )[0])


def _mean_curve_for_metric(M, metric_col):
    """M shape: (data_seeds, trials, shots, metrics)."""
    return M[:, :, :, metric_col].mean(axis=(0, 1))


def bootstrap_slopes(
    reps, shots, dataset, estimator, metric_col,
    threshold, n_boot, rng,
):
    """
    Hierarchical paired bootstrap.

    Data seeds are the independent outer units. Seeds are resampled jointly
    across N, preserving the nested/paired design. Trials are then resampled
    within each selected seed to include shot-sampling variability.
    """
    Ns = sorted(
        key[1] for key in reps
        if key[0] == dataset and key[2] == estimator
    )
    mats = {
        N: reps[(dataset, N, estimator)]
        for N in Ns
    }

    first = mats[Ns[0]]
    n_seeds, n_trials = first.shape[:2]

    point = np.array([
        shots_needed(
            _mean_curve_for_metric(mats[N], metric_col),
            shots,
            threshold,
        )
        for N in Ns
    ])
    point_slope = fit_slope(Ns, point)

    boot_slopes = []
    for _ in range(n_boot):
        # Same outer seed resample for every N: preserve pairing.
        seed_indices = rng.integers(0, n_seeds, size=n_seeds)
        boot_shots = []

        for N in Ns:
            M = mats[N]
            sampled_seed_curves = []

            for seed_index in seed_indices:
                trial_indices = rng.integers(
                    0, n_trials, size=n_trials
                )
                selected = M[
                    seed_index,
                    trial_indices,
                    :,
                    metric_col,
                ]
                sampled_seed_curves.append(selected.mean(axis=0))

            mean_curve = np.mean(sampled_seed_curves, axis=0)
            boot_shots.append(
                shots_needed(mean_curve, shots, threshold)
            )

        boot_slope = fit_slope(Ns, boot_shots)
        if np.isfinite(boot_slope):
            boot_slopes.append(boot_slope)

    if len(boot_slopes) >= 20:
        ci_low, ci_high = np.percentile(boot_slopes, [2.5, 97.5])
    else:
        ci_low, ci_high = np.nan, np.nan

    return Ns, point, point_slope, float(ci_low), float(ci_high)


def fit_margin_scaling(margins, dataset, k, n_boot=500, seed=1234):
    """Fit delta_s ~ N^-gamma; bootstrap data seeds as paired clusters."""
    selected = [r for r in margins if r["dataset"] == dataset and r["k"] == k]
    Ns = sorted({r["N"] for r in selected})
    seeds = sorted({r["data_seed"] for r in selected})
    per_seed = {}
    for N in Ns:
        for ds in seeds:
            vals = [r["delta_s"] for r in selected if r["N"] == N and r["data_seed"] == ds]
            per_seed[(N, ds)] = float(np.median(vals)) if vals else np.nan
    medians = np.array([np.nanmedian([per_seed[(N, ds)] for ds in seeds]) for N in Ns], dtype=float)

    def gamma_for(seed_sample):
        med = np.array([np.nanmedian([per_seed[(N, ds)] for ds in seed_sample]) for N in Ns])
        valid = (med > 0) & np.isfinite(med)
        if valid.sum() < 3:
            return np.nan
        return float(-np.polyfit(np.log2(np.asarray(Ns)[valid]), np.log2(med[valid]), 1)[0])

    gamma = gamma_for(seeds)
    brng = np.random.default_rng(seed)
    boots = []
    if seeds:
        for _ in range(n_boot):
            sampled = brng.choice(seeds, size=len(seeds), replace=True).tolist()
            value = gamma_for(sampled)
            if np.isfinite(value):
                boots.append(value)
    if len(boots) >= 20:
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo, hi = np.nan, np.nan
    return np.asarray(Ns, dtype=float), medians, gamma, float(lo), float(hi)


def measured_shots_by_N(
    reps, shots, dataset, estimator, metric_col, threshold
):
    Ns = sorted(
        key[1] for key in reps
        if key[0] == dataset and key[2] == estimator
    )
    values = [
        shots_needed(
            _mean_curve_for_metric(
                reps[(dataset, N, estimator)], metric_col
            ),
            shots,
            threshold,
        )
        for N in Ns
    ]
    return np.asarray(Ns, dtype=float), np.asarray(values, dtype=float)


# --------------------------------------------------------------------------- #
# Reporting and output files
# --------------------------------------------------------------------------- #
def write_csv(path, rows):
    if not rows:
        print(f"  not writing empty table: {path}")
        return
    # Some summary tables intentionally combine metric rows with auxiliary
    # rows (e.g. absolute accuracy); use the union of keys for a stable schema.
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def report(shots, reps, rows, margins, args):
    if not rows:
        raise RuntimeError("No experiment rows were generated. Check datasets, N grid, n_test, and dataset-size limits.")
    write_csv("qknn_shots_results.csv", rows)
    if margins:
        write_csv("qknn_margins.csv", margins)

    datasets = sorted({key[0] for key in reps})
    ks = list(args.k_values)
    estimator = args.estimator
    rng = np.random.default_rng(args.seed + 1)
    slope_rows, comparison_rows, theory_rows = [], [], []

    # One comparison record per dataset/k/criterion, with paired hierarchical CIs.
    criteria = (("neighbour_exact_match", 1), ("classification_agreement", 2))
    for name in datasets:
        for k in ks:
            for criterion, metric_col in criteria:
                Ns = sorted(key[1] for key in reps if key[0] == name and key[2] == k and key[3] == estimator)
                values, slope, lo, hi = None, np.nan, np.nan, np.nan
                # Reuse bootstrap_slopes by projecting the k-aware key to its
                # original four-part shape locally.
                local = {(name, N, estimator): reps[(name, N, k, estimator)] for N in Ns}
                Ns_out, values, slope, lo, hi = bootstrap_slopes(
                    local, shots, name, estimator, metric_col,
                    args.threshold, args.bootstrap, rng,
                )
                crossed = int(np.isfinite(values).sum())
                record = {
                    "dataset": name, "k": k, "estimator": estimator,
                    "criterion": criterion, "threshold": args.threshold,
                    "slope": slope, "ci_low": lo, "ci_high": hi,
                    "N_points_crossed": crossed, "N_points_total": len(Ns_out),
                }
                comparison_rows.append(record)
                slope_rows.append(record.copy())
                print(f"{name:7s} k={k:2d} {criterion:25s} b={slope:.3f} [{lo:.3f}, {hi:.3f}] crossings={crossed}/{len(Ns_out)}")

            # Absolute prediction accuracy is reported separately; it is not
            # the same quantity as agreement with exact classical k-NN.
            nmax = max(key[1] for key in reps if key[0] == name and key[2] == k and key[3] == estimator)
            accuracy_curve = _mean_curve_for_metric(reps[(name, nmax, k, estimator)], 3)
            comparison_rows.append({
                "dataset": name, "k": k, "estimator": estimator,
                "criterion": "absolute_accuracy_at_largest_N",
                "threshold": np.nan, "slope": np.nan, "ci_low": np.nan,
                "ci_high": np.nan, "N_points_crossed": np.nan,
                "N_points_total": np.nan, "largest_N": nmax,
                "accuracy_at_max_shots": float(accuracy_curve[-1]),
            })
            if margins:
                diagnostic_rows = [r for r in margins if r["dataset"] == name and r["k"] == k]
                swap_values = [r["min_class_flipping_swaps"] for r in diagnostic_rows
                               if np.isfinite(r["min_class_flipping_swaps"])]
                cost_values = [r["class_margin_distance_cost"] for r in diagnostic_rows
                               if np.isfinite(r["class_margin_distance_cost"])]
                comparison_rows.append({
                    "dataset": name, "k": k, "estimator": estimator,
                    "criterion": "class_robustness_diagnostics",
                    "median_vote_margin": float(np.median([r["vote_margin"] for r in diagnostic_rows])) if diagnostic_rows else np.nan,
                    "median_min_class_flipping_swaps": float(np.median(swap_values)) if swap_values else np.nan,
                    "median_class_margin_distance_cost": float(np.median(cost_values)) if cost_values else np.nan,
                    "median_single_competitor_min_flip_penalty": float(np.nanmedian([
                        r["single_competitor_min_flip_penalty"] for r in diagnostic_rows
                    ])) if any(np.isfinite(r["single_competitor_min_flip_penalty"]) for r in diagnostic_rows) else np.nan,
                    "diagnostic_test_points": len(diagnostic_rows),
                })

    write_csv("qknn_slopes.csv", slope_rows)
    write_csv("qknn_k_comparison.csv", comparison_rows)

    if args.margins and margins:
        # Summarize classical label robustness separately at every N and k.
        # Data-seed medians are computed first so each seed has equal weight.
        robustness_rows = []
        robustness_scaling_rows = []
        for name in datasets:
            for k in ks:
                Ns = sorted({r["N"] for r in margins if r["dataset"] == name and r["k"] == k})
                diagnostic_metrics = (
                    "vote_margin", "min_class_flipping_swaps",
                    "class_margin_distance_cost",
                    "single_competitor_min_flip_penalty",
                )
                seed_medians = {metric: [] for metric in diagnostic_metrics}
                for N in Ns:
                    group = [r for r in margins if r["dataset"] == name and r["k"] == k and r["N"] == N]
                    per_seed = {metric: [] for metric in seed_medians}
                    for ds in sorted({r["data_seed"] for r in group}):
                        seed_group = [r for r in group if r["data_seed"] == ds]
                        for metric in per_seed:
                            vals = np.asarray([r[metric] for r in seed_group], dtype=float)
                            vals = vals[np.isfinite(vals)]
                            if len(vals):
                                per_seed[metric].append(float(np.median(vals)))
                    row = {"dataset": name, "k": k, "N": N, "test_points": len(group)}
                    for metric, values in per_seed.items():
                        row[f"median_{metric}"] = float(np.median(values)) if values else np.nan
                        seed_medians[metric].append(row[f"median_{metric}"])
                    robustness_rows.append(row)

                for metric, vals in seed_medians.items():
                    slope = fit_slope(Ns, vals)
                    robustness_scaling_rows.append({
                        "dataset": name, "k": k, "diagnostic": metric,
                        "N_scaling_exponent": slope,
                        "N_points": int(np.sum(np.isfinite(vals))),
                    })

        write_csv("qknn_class_robustness.csv", robustness_rows)
        write_csv("qknn_class_robustness_scaling.csv", robustness_scaling_rows)

        fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.2), squeeze=False)
        for ax, name in zip(axes[0], datasets):
            for k in ks:
                subset = [r for r in robustness_rows if r["dataset"] == name and r["k"] == k]
                subset.sort(key=lambda r: r["N"])
                x = [r["N"] for r in subset]
                y = [r["median_min_class_flipping_swaps"] for r in subset]
                ax.plot(x, y, marker="o", label=f"k={k}")
            ax.set_xlabel("Training samples N")
            ax.set_ylabel("Median minimum class-flipping swaps")
            ax.set_title(name)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig("qknn_class_robustness_vs_N.png", dpi=200); plt.close(fig)

        # Continuous class-flip penalty is plotted separately because its
        # scale is not comparable to the discrete swap-count diagnostic.
        fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.2), squeeze=False)
        for ax, name in zip(axes[0], datasets):
            for k in ks:
                subset = [r for r in robustness_rows if r["dataset"] == name and r["k"] == k]
                subset.sort(key=lambda r: r["N"])
                x = np.asarray([r["N"] for r in subset], dtype=float)
                y = np.asarray([r["median_single_competitor_min_flip_penalty"] for r in subset], dtype=float)
                valid = np.isfinite(y) & (y > 0)
                if np.any(valid):
                    ax.loglog(x[valid], y[valid], marker="o", label=f"k={k}")
            ax.set_xlabel("Training samples N")
            ax.set_ylabel("Minimum class-flip squared-distance penalty")
            ax.set_title(name)
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig("qknn_class_flip_penalty_vs_N.png", dpi=200); plt.close(fig)

        print("\nClass-robustness scaling (diagnostic exponent vs N):")
        for row in robustness_scaling_rows:
            print(f"{row['dataset']:7s} k={row['k']:2d} {row['diagnostic']:31s} "
                  f"b={row['N_scaling_exponent']:.3f} points={row['N_points']}")

    # Plot exponents as a function of k; do not hide missing threshold crossings.
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.2), squeeze=False)
    for ax, name in zip(axes[0], datasets):
        for criterion, marker in (("neighbour_exact_match", "o-"), ("classification_agreement", "s-")):
            subset = [r for r in comparison_rows if r.get("dataset") == name and r.get("criterion") == criterion]
            subset.sort(key=lambda r: r["k"])
            x = [r["k"] for r in subset if np.isfinite(r["slope"])]
            y = [r["slope"] for r in subset if np.isfinite(r["slope"])]
            if x:
                ax.plot(x, y, marker, label=criterion.replace("_", " "))
        ax.set_xlabel("k")
        ax.set_ylabel("Shot-scaling exponent")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("qknn_k_exponents.png", dpi=200); plt.close(fig)

    if args.margins and margins:
        # For each k, bootstrap gamma using the same seed-cluster logic.
        fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.2), squeeze=False)
        for ax, name in zip(axes[0], datasets):
            for k in ks:
                Ng, delta_med, gamma, gamma_lo, gamma_hi = fit_margin_scaling(
                    margins, name, k, args.bootstrap, args.seed + 100 + k
                )
                predicted = 1 + 2 * gamma
                predicted_lo = 1 + 2 * gamma_lo if np.isfinite(gamma_lo) else np.nan
                predicted_hi = 1 + 2 * gamma_hi if np.isfinite(gamma_hi) else np.nan
                # Compare the distance-margin prediction only with exact set recovery.
                local = {(name, N, estimator): reps[(name, N, k, estimator)]
                         for N in sorted(key[1] for key in reps if key[0] == name and key[2] == k and key[3] == estimator)}
                Ns, measured, measured_b, b_lo, b_hi = bootstrap_slopes(
                    local, shots, name, estimator, 1,
                    args.threshold, args.bootstrap, rng,
                )
                theory_rows.append({
                    "dataset": name, "k": k,
                    "gamma_distance": gamma,
                    "gamma_ci_low": gamma_lo, "gamma_ci_high": gamma_hi,
                    "predicted_exact_neighbour_b": predicted,
                    "predicted_b_ci_low": predicted_lo, "predicted_b_ci_high": predicted_hi,
                    "measured_exact_neighbour_b": measured_b,
                    "measured_b_ci_low": b_lo, "measured_b_ci_high": b_hi,
                    "crossed_N_points": int(np.isfinite(measured).sum()),
                })
                valid = (delta_med > 0) & np.isfinite(delta_med)
                if valid.any():
                    ax.loglog(Ng[valid], delta_med[valid], marker="o", label=f"k={k}")
                print(f"{name:7s} k={k:2d} gamma={gamma:.3f} [{gamma_lo:.3f},{gamma_hi:.3f}] "
                      f"predicted b={predicted:.3f}; exact-neighbour b={measured_b:.3f} [{b_lo:.3f},{b_hi:.3f}]")
        for ax in axes[0]:
            ax.set_xlabel("Training samples N")
            ax.set_ylabel("Median signal margin delta_s")
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig("qknn_k_margin_scaling.png", dpi=200); plt.close(fig)
        write_csv("qknn_margin_theory.csv", theory_rows)

        # Paired k comparison of distance-margin prediction vs measured exponent.
        fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.2), squeeze=False)
        for ax, name in zip(axes[0], datasets):
            subset = [r for r in theory_rows if r["dataset"] == name]
            subset.sort(key=lambda r: r["k"])
            ax.plot([r["k"] for r in subset], [r["predicted_exact_neighbour_b"] for r in subset], "o-", label="1 + 2 gamma")
            ax.plot([r["k"] for r in subset], [r["measured_exact_neighbour_b"] for r in subset], "s-", label="measured exact recovery")
            ax.set_xlabel("k"); ax.set_ylabel("Shot exponent"); ax.set_title(name)
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig("qknn_k_predicted_vs_measured.png", dpi=200); plt.close(fig)

    print("\nAbsolute accuracy at the largest N is in qknn_k_comparison.csv.")
    print("Saved: qknn_shots_results.csv, qknn_slopes.csv, qknn_margins.csv, qknn_k_comparison.csv, qknn_k_exponents.png"
          + (", qknn_margin_theory.csv, qknn_k_margin_scaling.png, qknn_k_predicted_vs_measured.png, "
             "qknn_class_robustness.csv, qknn_class_robustness_scaling.csv, qknn_class_robustness_vs_N.png, "
             "qknn_class_flip_penalty_vs_N.png"
             if args.margins else ""))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Paired finite-shot scaling study for authors' quantum k-NN model.")
    parser.add_argument("--datasets", nargs="+", default=["blobs", "wine", "digits"], choices=["blobs", "iris", "wine", "digits"])
    parser.add_argument("--n-grid", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--k", type=int, default=None, help="legacy single-k option; use --k-values for a comparison")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 9], help="k values compared on the same sampled counts")
    parser.add_argument("--trials", type=int, default=3, help="shot-sampling repetitions per data seed and N")
    parser.add_argument("--data-seeds", type=int, default=3, help="independent dataset splits; bootstrap resamples these as clusters")
    parser.add_argument("--min-exp", type=int, default=6)
    parser.add_argument("--max-exp", type=int, default=44, help="largest shots = 2**this")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.0, help="pseudocount (Laplace smoothing)")
    parser.add_argument("--encoding", choices=["extension", "translation"], default="translation")
    parser.add_argument("--estimator", choices=list(ESTIMATORS), default="avg", help="estimator used for scaling summaries")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--margins", action="store_true", help="also run distance- and class-margin analysis")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default=None, help="default: <workspace>/results/shots_k_sensitivity (keeps old results/shots untouched)")
    args = parser.parse_args()

    if args.k is not None:
        if args.k_values != [1, 3, 5, 9]:
            parser.error("Use either legacy --k or --k-values, not both")
        args.k_values = [args.k]
    args.k_values = sorted(set(args.k_values))
    if not args.k_values or any(k < 1 for k in args.k_values):
        parser.error("--k-values must contain positive integers")
    if args.n_test < 1 or args.trials < 1 or args.data_seeds < 1 or args.bootstrap < 1:
        parser.error("n-test, trials, data-seeds, and bootstrap must be positive")
    if args.min_exp < 0 or args.max_exp < args.min_exp:
        parser.error("shot exponents must satisfy 0 <= min-exp <= max-exp")
    if not 0 < args.threshold <= 1:
        parser.error("threshold must be in (0, 1]")
    if args.alpha < 0:
        parser.error("alpha must be non-negative")
    if not args.n_grid or any(N < 1 for N in args.n_grid):
        parser.error("n-grid must contain positive integers")
    args.n_grid = sorted(set(args.n_grid))

    outdir = Path(args.outdir) if args.outdir else RESULTS / "shots_k_sensitivity"
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(outdir)
    print(f"Writing outputs to {outdir.resolve()}")
    print("Design: fixed raw test set, nested training sets, paired multinomial samples across k, hierarchical paired bootstrap.")
    print(f"Configuration: datasets={args.datasets}, k={args.k_values}, encoding={args.encoding}, estimator={args.estimator}, threshold={args.threshold}")
    shots, reps, rows, margins = run(args)
    report(shots, reps, rows, margins, args)


if __name__ == "__main__":
    main()
