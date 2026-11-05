#!/usr/bin/env python3
"""
controlled_dimension_experiment.py  (matched-grid version; supersedes the unmatched original)
====================================================================================

Controlled-dimension follow-up to qknn_margin_scaling_study.py.

PURPOSE
-------
Test, rather than assume, the chain

    data dimension D
        -> signal-margin scaling delta_s ~ N^(-gamma)
        -> predicted shot exponent b_pred = 1 + 2*gamma
        -> independently measured QKNN shot exponent b_meas

and compare the measured gamma with the separate geometric hypothesis

    gamma ?= 2/D
    b_theory ?= 1 + 4/D.

Two experiments are supported:

1. ambient
   Gaussian data directly in D dimensions.

2. intrinsic
   Gaussian latent data in intrinsic dimension D, embedded by an orthonormal
   linear map into a fixed ambient dimension (default 64). This tests whether
   scaling follows intrinsic rather than ambient dimension.

IMPORTANT
---------
* The geometry stage does NOT use QKNN shot results to fit gamma.
* The shot stage is optional and uses a smaller N grid because it is expensive.
* delta_s is computed at the k/(k+1) boundary in the same "s" units used by
  the translation encoding:
      s_j = ||x_j||^2 - 2 x_j.x_t - 1/4
  so delta_s between two candidates is also their squared-distance gap.
* This is a controlled synthetic experiment, not a proof of gamma = 2/D.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_dataset(N, n_test, intrinsic_dim, ambient_dim, seed, mode):
    """
    Generate Gaussian train/test data.

    ambient mode:
        intrinsic_dim == ambient_dim == D.

    intrinsic mode:
        latent z in R^D is embedded into R^ambient_dim using an orthonormal
        linear map. Small isotropic noise can optionally be added later if
        desired; by default the manifold is exactly D-dimensional.
    """
    rng = np.random.default_rng(seed)
    total = N + n_test

    Z = rng.normal(size=(total, intrinsic_dim))

    if mode == "ambient":
        X = Z
    else:
        # Q has orthonormal columns: Q.T @ Q = I_D.
        A = rng.normal(size=(ambient_dim, intrinsic_dim))
        Q, _ = np.linalg.qr(A)
        X = Z @ Q[:, :intrinsic_dim].T

    return X[:N], X[N:]


def authors_normalize(Xtr, Xte):
    """Same feature-wise normalization used in qknn_shots_study_v2.py."""
    d = Xtr.shape[1]
    sq = np.sqrt(d)
    mid = (Xtr.min(0) + Xtr.max(0)) / 2
    rng = Xtr.max(0) - Xtr.min(0)
    rng[rng == 0] = 1.0

    Ztr = (Xtr - mid) / (rng * sq)
    Zte = np.clip(
        (Xte - mid) / (rng * sq),
        -1 / (2 * sq),
        1 / (2 * sq),
    )
    return Ztr, Zte


# ---------------------------------------------------------------------------
# Geometry: k/(k+1) signal margin
# ---------------------------------------------------------------------------

def boundary_margins(Ztr, Zte, k):
    """
    Return one delta_s per query.

    For translation encoding:
        s_j = ||x_j||^2 - 2 x_j.x_t - 1/4

    For a fixed target, ranking by s_j is the same as ranking by squared
    Euclidean distance because ||x_t||^2 and -1/4 are common offsets.

    delta_s = s_(k+1) - s_(k), with neighbours ordered nearest first.
    """
    train_norm2 = np.sum(Ztr * Ztr, axis=1)
    out = np.empty(len(Zte), dtype=float)

    for i, zt in enumerate(Zte):
        s = train_norm2 - 2.0 * (Ztr @ zt) - 0.25
        # We only need the k-th and (k+1)-th order statistics.
        pair = np.partition(s, [k - 1, k])[[k - 1, k]]
        out[i] = max(float(pair[1] - pair[0]), 0.0)

    return out


def fit_power_law(Ns, ys, min_n=None):
    Ns = np.asarray(Ns, float)
    ys = np.asarray(ys, float)
    ok = np.isfinite(ys) & (ys > 0)
    if min_n is not None:
        ok &= Ns >= min_n
    if ok.sum() < 3:
        return np.nan, np.nan

    slope, intercept = np.polyfit(np.log2(Ns[ok]), np.log2(ys[ok]), 1)
    return slope, intercept


def bootstrap_gamma(gaps_by_n, n_boot, rng, min_n=None):
    """
    Bootstrap queries independently within each N, recompute median(delta_s),
    then refit gamma. This gives uncertainty on the geometry-stage exponent.
    """
    Ns = np.array(sorted(gaps_by_n), float)
    med = np.array([np.median(gaps_by_n[int(N)]) for N in Ns], float)
    slope, _ = fit_power_law(Ns, med, min_n)
    gamma = -slope

    boots = []
    for _ in range(n_boot):
        bmed = []
        for N in Ns.astype(int):
            g = gaps_by_n[N]
            sample = g[rng.integers(0, len(g), size=len(g))]
            bmed.append(np.median(sample))
        s, _ = fit_power_law(Ns, np.array(bmed), min_n)
        if np.isfinite(s):
            boots.append(-s)

    if len(boots) >= 20:
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo = hi = np.nan

    return Ns, med, gamma, lo, hi


def bootstrap_gamma_on_grid(gaps_by_n, n_grid, n_boot, rng):
    """
    Fit gamma only on N values shared with the shot study.

    This prevents the earlier mistake of comparing a shot exponent fitted on
    N=16..1024 with a margin exponent fitted over a different/larger N range.
    """
    common = sorted(set(gaps_by_n).intersection(int(n) for n in n_grid))
    if len(common) < 3:
        return np.nan, np.nan, np.nan, len(common)

    Ns = np.asarray(common, float)
    med = np.asarray([np.median(gaps_by_n[int(N)]) for N in Ns], float)
    slope, _ = fit_power_law(Ns, med)
    gamma = -slope

    boots = []
    for _ in range(n_boot):
        bmed = []
        for N in Ns.astype(int):
            g = gaps_by_n[N]
            sample = g[rng.integers(0, len(g), size=len(g))]
            bmed.append(np.median(sample))
        s, _ = fit_power_law(Ns, np.asarray(bmed))
        if np.isfinite(s):
            boots.append(-s)

    if len(boots) >= 20:
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo = hi = np.nan

    return gamma, lo, hi, len(common)


# ---------------------------------------------------------------------------
# QKNN shot simulation: same probability model as the existing study
# ---------------------------------------------------------------------------

def scalar_products_translation(Ztr, zt):
    return np.sum(Ztr * Ztr, axis=1) - 2.0 * (Ztr @ zt) - 0.25


def sample_counts(s, S, rng):
    N = len(s)
    p = np.concatenate([(1.0 + s) / (2.0 * N),
                        (1.0 - s) / (2.0 * N)])
    p = np.clip(p, 0.0, None)
    p /= p.sum()
    c = rng.multinomial(int(S), p)
    return c[:N], c[N:]


def estimate_avg_translation(n0, n1, S, target_norm2, pseudocount=0.0):
    N = len(n0)
    total = S + pseudocount * 2 * N
    p0 = (n0 + pseudocount) / total
    p1 = (n1 + pseudocount) / total

    # Translation encoding, same formulas as qknn_shots_study_v2.py.
    d0 = np.sqrt(np.clip((2 * N * p0 - 1) + 0.25 + target_norm2, 0.0, 1.0))
    d1 = np.sqrt(np.clip((1 - 2 * N * p1) + 0.25 + target_norm2, 0.0, 1.0))
    return 0.5 * (d0 + d1)


def top_k(values, k, rng):
    jitter = 1e-12 * rng.random(len(values))
    return set(np.argsort(values + jitter)[:k].tolist())


def shots_needed(curve, shots, threshold):
    if curve[0] >= threshold:
        return float(shots[0])

    for i in range(len(shots) - 1):
        y0, y1 = curve[i], curve[i + 1]
        if y0 < threshold <= y1:
            f = (threshold - y0) / max(y1 - y0, 1e-12)
            return 2 ** (
                np.log2(shots[i])
                + f * (np.log2(shots[i + 1]) - np.log2(shots[i]))
            )
    return np.nan


def run_shot_curve(args, D, mode, N, seed_base):
    """
    Independently measure shots needed for Jaccard >= threshold.
    Returns the mean Jaccard curve across data seeds/trials.
    """
    rng = np.random.default_rng(seed_base + 99991 + D * 101 + N)
    shots = np.array([2 ** e for e in range(args.min_exp, args.max_exp + 1)], dtype=np.int64)
    curves = []

    ambient = D if mode == "ambient" else args.ambient_dim

    for ds in range(args.shot_data_seeds):
        Xtr, Xte = make_dataset(
            N, args.shot_n_test, D, ambient,
            seed_base + 10000 * D + 101 * N + ds,
            mode,
        )
        Ztr, Zte = authors_normalize(Xtr, Xte)

        train_norm2 = np.sum(Ztr * Ztr, axis=1)
        target_norm2 = np.sum(Zte * Zte, axis=1)

        true_sets = []
        s_all = []
        for zt in Zte:
            d2 = train_norm2 - 2 * (Ztr @ zt) + np.sum(zt * zt)
            true_sets.append(set(np.argsort(d2)[:args.k].tolist()))
            s_all.append(scalar_products_translation(Ztr, zt))

        for tr in range(args.shot_trials):
            curve = np.zeros(len(shots), float)
            for si, S in enumerate(shots):
                vals = []
                for qi in range(len(Zte)):
                    n0, n1 = sample_counts(s_all[qi], S, rng)
                    d_est = estimate_avg_translation(
                        n0, n1, S, target_norm2[qi], args.alpha
                    )
                    got = top_k(d_est, args.k, rng)
                    inter = len(got & true_sets[qi])
                    vals.append(inter / (2 * args.k - inter))
                curve[si] = np.mean(vals)
            curves.append(curve)

    return shots, np.mean(np.asarray(curves), axis=0)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_geometry(args, mode):
    rows = []
    all_results = {}

    for D in args.dimensions:
        ambient = D if mode == "ambient" else args.ambient_dim
        if mode == "intrinsic" and D > ambient:
            raise ValueError(f"intrinsic D={D} cannot exceed ambient_dim={ambient}")

        gaps_by_n = {}

        # Include the shot-study N values so gamma can be fitted on exactly
        # the same N window as b_measured. Geometry calculations are cheap
        # compared with shot simulation.
        geometry_grid = sorted(set(args.geometry_n_grid) | set(args.shot_n_grid))

        for N in geometry_grid:
            collected = []
            for ds in range(args.geometry_data_seeds):
                Xtr, Xte = make_dataset(
                    N, args.geometry_n_test, D, ambient,
                    args.seed + 100000 * D + 1000 * N + ds,
                    mode,
                )
                Ztr, Zte = authors_normalize(Xtr, Xte)
                collected.extend(boundary_margins(Ztr, Zte, args.k))
            gaps_by_n[N] = np.asarray(collected, float)

        rng = np.random.default_rng(args.seed + D * 17)
        Ns, med, gamma_all, lo_all, hi_all = bootstrap_gamma(
            gaps_by_n, args.bootstrap, rng, min_n=None
        )
        _, _, gamma_tail, lo_tail, hi_tail = bootstrap_gamma(
            gaps_by_n, args.bootstrap, rng, min_n=args.tail_min_n
        )

        gamma_matched, gm_lo, gm_hi, matched_points = bootstrap_gamma_on_grid(
            gaps_by_n,
            args.shot_n_grid,
            args.bootstrap,
            np.random.default_rng(args.seed + D * 1701),
        )

        theory_gamma = 2.0 / D
        theory_b = 1.0 + 4.0 / D
        pred_b_all = 1.0 + 2.0 * gamma_all
        pred_b_tail = 1.0 + 2.0 * gamma_tail
        pred_b_matched = 1.0 + 2.0 * gamma_matched
        pred_b_matched_lo = 1.0 + 2.0 * gm_lo
        pred_b_matched_hi = 1.0 + 2.0 * gm_hi

        all_results[D] = dict(
            Ns=Ns,
            med=med,
            gaps_by_n=gaps_by_n,
            gamma_all=gamma_all,
            gamma_tail=gamma_tail,
            gamma_matched=gamma_matched,
            gamma_matched_ci_low=gm_lo,
            gamma_matched_ci_high=gm_hi,
            predicted_b_matched=pred_b_matched,
            predicted_b_matched_ci_low=pred_b_matched_lo,
            predicted_b_matched_ci_high=pred_b_matched_hi,
        )

        rows.append(dict(
            mode=mode,
            intrinsic_dim=D,
            ambient_dim=ambient,
            gamma_theory_2_over_D=theory_gamma,
            gamma_all=gamma_all,
            gamma_all_ci_low=lo_all,
            gamma_all_ci_high=hi_all,
            gamma_tail=gamma_tail,
            gamma_tail_ci_low=lo_tail,
            gamma_tail_ci_high=hi_tail,
            predicted_b_from_gamma_all=pred_b_all,
            predicted_b_from_gamma_tail=pred_b_tail,
            gamma_shot_window=gamma_matched,
            gamma_shot_window_ci_low=gm_lo,
            gamma_shot_window_ci_high=gm_hi,
            predicted_b_shot_window=pred_b_matched,
            predicted_b_shot_window_ci_low=pred_b_matched_lo,
            predicted_b_shot_window_ci_high=pred_b_matched_hi,
            matched_n_points=matched_points,
            b_theory_1_plus_4_over_D=theory_b,
            tail_min_n=args.tail_min_n,
        ))

        print(
            f"[{mode}] D={D:2d}, ambient={ambient:2d}: "
            f"gamma(all)={gamma_all:.3f} [{lo_all:.3f}, {hi_all:.3f}], "
            f"gamma(N>={args.tail_min_n})={gamma_tail:.3f} [{lo_tail:.3f}, {hi_tail:.3f}], "
            f"gamma(shot-window)={gamma_matched:.3f} [{gm_lo:.3f}, {gm_hi:.3f}], "
            f"1+2gamma={pred_b_matched:.3f} "
            f"[{pred_b_matched_lo:.3f}, {pred_b_matched_hi:.3f}], "
            f"2/D={theory_gamma:.3f}"
        )

    return rows, all_results


def run_shots(args, mode, geometry_rows):
    if args.skip_shots:
        return []

    shot_rows = []
    geom_by_D = {r["intrinsic_dim"]: r for r in geometry_rows}

    for D in args.dimensions:
        measured = []
        used_N = []

        for N in args.shot_n_grid:
            shots, curve = run_shot_curve(args, D, mode, N, args.seed)
            sstar = shots_needed(curve, shots, args.threshold)
            if np.isfinite(sstar):
                used_N.append(N)
                measured.append(sstar)
            print(
                f"[{mode}] shots D={D:2d} N={N:5d}: "
                f"S*={sstar if np.isfinite(sstar) else 'not reached'}"
            )

        used_N = np.asarray(used_N, float)
        measured = np.asarray(measured, float)

        if len(used_N) >= 3:
            b_meas, _ = fit_power_law(used_N, measured)
        else:
            b_meas = np.nan

        g = geom_by_D[D]
        pred_match = g["predicted_b_shot_window"]
        residual = b_meas - pred_match if np.isfinite(b_meas) else np.nan

        shot_rows.append(dict(
            mode=mode,
            intrinsic_dim=D,
            ambient_dim=(D if mode == "ambient" else args.ambient_dim),
            measured_shot_exponent=b_meas,
            gamma_shot_window=g["gamma_shot_window"],
            gamma_shot_window_ci_low=g["gamma_shot_window_ci_low"],
            gamma_shot_window_ci_high=g["gamma_shot_window_ci_high"],
            predicted_b_shot_window=pred_match,
            predicted_b_shot_window_ci_low=g["predicted_b_shot_window_ci_low"],
            predicted_b_shot_window_ci_high=g["predicted_b_shot_window_ci_high"],
            measured_minus_predicted=residual,
            predicted_b_from_gamma_all=g["predicted_b_from_gamma_all"],
            predicted_b_from_gamma_tail=g["predicted_b_from_gamma_tail"],
            b_theory_1_plus_4_over_D=g["b_theory_1_plus_4_over_D"],
            shot_points=len(used_N),
        ))

        print(
            f"    => b_measured={b_meas:.3f}, "
            f"b_pred(matched)={pred_match:.3f} "
            f"[{g['predicted_b_shot_window_ci_low']:.3f}, "
            f"{g['predicted_b_shot_window_ci_high']:.3f}], "
            f"residual={residual:+.3f}, "
            f"1+4/D={g['b_theory_1_plus_4_over_D']:.3f}"
        )

    return shot_rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_geometry(mode, results, args, outdir):
    # One figure: delta_s vs N for all D.
    plt.figure(figsize=(7.2, 5.2))
    for D in args.dimensions:
        r = results[D]
        plt.loglog(r["Ns"], r["med"], "o-", label=f"D={D}, gamma={r['gamma_all']:.2f}")
    plt.xlabel("Training samples N")
    plt.ylabel("Median delta_s at k/(k+1) boundary")
    plt.title(f"Controlled dimension: signal-margin scaling ({mode})")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"controlled_{mode}_signal_gap_vs_N.png", dpi=200)
    plt.close()

    # One figure: measured gamma vs 2/D.
    Ds = np.asarray(args.dimensions, float)
    gam = np.asarray([results[D]["gamma_all"] for D in args.dimensions])
    tail = np.asarray([results[D]["gamma_tail"] for D in args.dimensions])

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(Ds, 2.0 / Ds, "o--", label="hypothesis: 2/D")
    plt.plot(Ds, gam, "o-", label="measured gamma: all N")
    plt.plot(Ds, tail, "s-", label=f"measured gamma: N >= {args.tail_min_n}")
    plt.xlabel("Intrinsic dimension D")
    plt.ylabel("Margin exponent gamma")
    plt.title(f"Does gamma follow 2/D? ({mode})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"controlled_{mode}_gamma_vs_dimension.png", dpi=200)
    plt.close()


def plot_shot_summary(mode, shot_rows, outdir):
    if not shot_rows:
        return

    D = np.asarray([r["intrinsic_dim"] for r in shot_rows], float)
    bm = np.asarray([r["measured_shot_exponent"] for r in shot_rows], float)
    bp = np.asarray([r["predicted_b_shot_window"] for r in shot_rows], float)
    blo = np.asarray([r["predicted_b_shot_window_ci_low"] for r in shot_rows], float)
    bhi = np.asarray([r["predicted_b_shot_window_ci_high"] for r in shot_rows], float)
    bt = np.asarray([r["b_theory_1_plus_4_over_D"] for r in shot_rows], float)

    # Dimension summary with bootstrap interval on the matched-range prediction.
    plt.figure(figsize=(6.8, 4.9))
    plt.plot(D, bm, "o-", label="measured QKNN b")
    plt.errorbar(
        D, bp,
        yerr=np.vstack([bp - blo, bhi - bp]),
        fmt="s--", capsize=4,
        label="predicted 1 + 2 gamma (matched N)"
    )
    plt.plot(D, bt, "d:", label="hypothesis 1 + 4/D")
    plt.xlabel("Intrinsic dimension D")
    plt.ylabel("Shot-scaling exponent b")
    plt.title(f"Matched-range shot-scaling test ({mode})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"controlled_{mode}_shot_exponents_matched.png", dpi=200)
    plt.close()

    # Cleanest validation plot: predicted exponent vs independently measured one.
    ok = np.isfinite(bp) & np.isfinite(bm)
    if ok.any():
        lo = min(bp[ok].min(), bm[ok].min()) - 0.15
        hi = max(bp[ok].max(), bm[ok].max()) + 0.15

        plt.figure(figsize=(5.8, 5.4))
        plt.plot([lo, hi], [lo, hi], "--", label="ideal: measured = predicted")
        plt.errorbar(
            bp[ok], bm[ok],
            xerr=np.vstack([bp[ok] - blo[ok], bhi[ok] - bp[ok]]),
            fmt="o", capsize=4
        )
        for d, x, y in zip(D[ok], bp[ok], bm[ok]):
            plt.annotate(f"D={int(d)}", (x, y), xytext=(6, 5),
                         textcoords="offset points")
        plt.xlabel("Predicted exponent 1 + 2 gamma (same N window)")
        plt.ylabel("Measured QKNN shot exponent b")
        plt.title(f"Margin prediction vs measured shot scaling ({mode})")
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"controlled_{mode}_predicted_vs_measured_b.png", dpi=200)
        plt.close()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", choices=["ambient", "intrinsic", "both"], default="both")
    ap.add_argument("--dimensions", nargs="+", type=int, default=[2, 4, 8, 16])
    ap.add_argument("--ambient-dim", type=int, default=64)
    ap.add_argument("--k", type=int, default=5)

    # Geometry stage: deliberately goes farther in N than shot simulation.
    ap.add_argument(
        "--geometry-n-grid", nargs="+", type=int,
        default=[64, 128, 256, 512, 1024, 2048, 4096, 8192]
    )
    ap.add_argument("--geometry-n-test", type=int, default=200)
    ap.add_argument("--geometry-data-seeds", type=int, default=8)
    ap.add_argument("--tail-min-n", type=int, default=512)
    ap.add_argument("--bootstrap", type=int, default=1000)

    # Independent QKNN shot stage.
    ap.add_argument(
        "--shot-n-grid", nargs="+", type=int,
        default=[16, 32, 64, 128, 256, 512, 1024]
    )
    ap.add_argument("--shot-n-test", type=int, default=50)
    ap.add_argument("--shot-data-seeds", type=int, default=3)
    ap.add_argument("--shot-trials", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--min-exp", type=int, default=8)
    ap.add_argument("--max-exp", type=int, default=44)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--skip-shots", action="store_true")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None, help="default: <workspace>/results/controlled_dimension")

    args = ap.parse_args()
    outdir = Path(args.outdir) if args.outdir else Path(__file__).resolve().parent.parent / "results" / "controlled_dimension"
    outdir.mkdir(parents=True, exist_ok=True)

    modes = ["ambient", "intrinsic"] if args.mode == "both" else [args.mode]

    for mode in modes:
        print("\n" + "=" * 72)
        print(f"CONTROLLED DIMENSION EXPERIMENT: {mode.upper()}")
        print("=" * 72)

        geometry_rows, results = run_geometry(args, mode)
        write_csv(outdir / f"controlled_{mode}_geometry.csv", geometry_rows)
        plot_geometry(mode, results, args, outdir)

        shot_rows = run_shots(args, mode, geometry_rows)
        write_csv(outdir / f"controlled_{mode}_shots.csv", shot_rows)
        plot_shot_summary(mode, shot_rows, outdir)

    print(f"\nSaved outputs in: {outdir.resolve()}")
    print("Recommended interpretation order:")
    print("  1. Treat gamma = 2/D as a hypothesis, not an established law.")
    print("  2. Compare b_measured with 1+2*gamma fitted on the SAME N grid.")
    print("  3. Use the matched-range bootstrap CI and measured-minus-predicted residual.")
    print("  4. Orthonormal intrinsic embeddings are a distance-preserving sanity check only.")


if __name__ == "__main__":
    main()
