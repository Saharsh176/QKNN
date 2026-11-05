"""
qknn_cost_model.py
==================
Study 1: end-to-end cost comparison of the Zardini et al. quantum k-NN against
classical k-NN for ONE prediction, under different data-loading assumptions.

WHAT IS NEW IN THIS VERSION
---------------------------
* It no longer extrapolates a single fitted exponent to N = 10^12 as if it
  were a law. The measured exponent holds only over the measured N range.
  Four clearly labelled scenarios are shown instead (see SCENARIOS below).
* It adds the quantity that actually decides the outcome:
      c  = shots per training sample  (= S / N)          what the method needs
      c* = affordable shots per sample                    what it could afford
  and prints how many times too large c is. The break-even condition is c < c*.
* The statement "never wins" is reported per loading model and per baseline,
  with the digits-style exception (flat shot curve) visible and labelled fragile.

WHAT IS COUNTED (operations for ONE prediction; NOT seconds)
------------------------------------------------------------
Classical brute force : N*F multiply-adds + N to pick the k smallest.
Classical approximate : ann_evals * log2(N) distance checks of F operations each
                        (graph-search style; it can miss neighbours).
Quantum (this method) : q * S * (loading cost per shot) + N classical steps to
                        turn the counts of all N index states into distances.
  Every shot reloads the whole training state (measurement destroys it).
  Loading models:
    generic : ~N*F gates per shot (arbitrary state, no QRAM).
    reduced : r * N*F per shot (cheaper approximate loading; r is a free knob).
    qram    : ~log2(N*F) per shot (IDEAL QRAM, query depth only; very optimistic).
    stream  : c_sk * (N*F/eps) * log2(N*F) per shot (quantum oracle sketching;
              assumption-heavy order-of-magnitude model, one query per shot).
  q = slowdown of one quantum operation versus one classical operation
  (default 1 = the most generous case).

SCENARIOS FOR HOW SHOTS GROW ABOVE THE LARGEST MEASURED N
---------------------------------------------------------
S(N) is anchored at the largest measured N and continued as S ~ N^b:
  measured : b fitted over all measured N. Finite-range exponent; for exact
             neighbours it is PESSIMISTIC (the exponent falls towards 1 + 4/D).
  local    : b fitted on the last few measured points only (noisy).
  b = 1    : shots proportional to N. For exact neighbours this is the
             favourable bound (the exponent should not fall below 1).
  flat     : b = 0. Extreme, only plausible if the decision margin grows with N;
             the digits vote curve looks like this but it is noisy. FRAGILE.
EVERYTHING ABOVE THE LARGEST MEASURED N IS EXTRAPOLATION.

RUN
---
    python scripts/qknn_shots_study.py --margins        # first: writes results/shots/qknn_shots_results.csv
    python scripts/qknn_cost_model.py --dataset blobs --criterion vote
    python scripts/qknn_cost_model.py --dataset blobs --criterion neighbours
Outputs go to results/cost_model/ (override with --outdir; read a different shots file with --csv).
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES = {"blobs": 8, "wine": 13, "digits": 64, "iris": 4}
METRIC = {"vote": "agree_with_classical", "neighbours": "jaccard"}


# --------------------------------------------------------------------------- #
# Read measured shots and fit growth
# --------------------------------------------------------------------------- #
def shots_needed(pts, thr):
    pts = sorted(pts)
    if pts and pts[0][1] >= thr:
        return float(pts[0][0])
    for (s0, j0), (s1, j1) in zip(pts, pts[1:]):
        if j0 < thr <= j1:
            f = (thr - j0) / max(j1 - j0, 1e-12)
            return 2 ** (np.log2(s0) + f * (np.log2(s1) - np.log2(s0)))
    return np.nan


def load_measured(path, dataset, estimator, criterion, thr):
    by_n = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["dataset"] == dataset and r["estimator"] == estimator:
                by_n.setdefault(int(r["N"]), []).append(
                    (int(r["shots"]), float(r[METRIC[criterion]])))
    Ns, S = [], []
    for N in sorted(by_n):
        s = shots_needed(by_n[N], thr)
        if not np.isnan(s):
            Ns.append(N)
            S.append(s)
    if len(Ns) < 3:
        raise SystemExit("Need at least 3 measured points; rerun the shot study "
                         "with a larger --max-exp or other settings.")
    return np.array(Ns, float), np.array(S, float)


def fit_exponent(Ns, S):
    return np.polyfit(np.log2(Ns), np.log2(S), 1)[0]


# --------------------------------------------------------------------------- #
# Cost functions (elementary operations for ONE prediction)
# --------------------------------------------------------------------------- #
def cost_brute(N, F):
    return N * F + N


def cost_ann(N, F, evals):
    return evals * np.log2(np.maximum(N, 2.0)) * F


def per_shot_cost(N, F, model, r, eps, c_sk):
    N = np.asarray(N, float)
    if model == "generic":
        return N * F
    if model == "reduced":
        return r * N * F
    if model == "qram":
        return np.log2(N * F)
    if model == "stream":
        return c_sk * (N * F / eps) * np.log2(N * F)
    raise ValueError(model)


def cost_quantum(N, F, S, model, q, r, eps, c_sk):
    return q * S * per_shot_cost(N, F, model, r, eps, c_sk) + N


def affordable_c(N, F, classical, model, q, r, eps, c_sk):
    """Largest shots-per-sample c* = S*/N that still beats `classical`.
    Condition: q*S*per_shot + N < classical.  Returns nan if impossible."""
    room = classical - N
    out = np.where(room > 0, room / (q * per_shot_cost(N, F, model, r, eps, c_sk) * N), np.nan)
    return out


def first_win(N, quantum, classical):
    wins = quantum < classical
    if not wins.any():
        return None
    if wins[-1]:
        idx = np.where(~wins)[0]
        start = 0 if len(idx) == 0 else idx[-1] + 1
        return N[start] if start < len(N) else None
    return "not sustained"


def fmt_n(x):
    if x is None:
        return "never (to grid end)"
    if isinstance(x, str):
        return x
    return f"{x:,.3g}"


def fmt_c(x):
    return "impossible" if x is None or np.isnan(x) else f"{x:,.3g}"


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="default: <workspace>/results/shots/qknn_shots_results.csv")
    ap.add_argument("--outdir", default=None, help="default: <workspace>/results/cost_model")
    ap.add_argument("--dataset", default="blobs", choices=list(FEATURES))
    ap.add_argument("--criterion", default="vote", choices=["vote", "neighbours"])
    ap.add_argument("--estimator", default="avg", choices=["avg", "zero", "one", "diff"])
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--features", type=int, default=None, help="F; default by dataset")
    ap.add_argument("--q", type=float, default=1.0, help="quantum op slowdown vs classical")
    ap.add_argument("--r", type=float, default=0.01, help="'reduced' loading factor")
    ap.add_argument("--ann-evals", type=float, default=64.0)
    ap.add_argument("--stream-eps", type=float, default=0.1, help="oracle sketch accuracy")
    ap.add_argument("--stream-const", type=float, default=1.0, help="sample-count constant")
    ap.add_argument("--n-max", type=float, default=1e12)
    ap.add_argument("--local-points", type=int, default=3,
                    help="points used for the local exponent")
    ap.add_argument("--extra-exponents", nargs="*", type=float, default=[],
                    help="additional shot-growth exponents to show")
    args = ap.parse_args()
    RESULTS = Path(__file__).resolve().parent.parent / "results"
    csv_path = os.path.abspath(args.csv) if args.csv else str(RESULTS / "shots" / "qknn_shots_results.csv")
    outdir = Path(args.outdir) if args.outdir else RESULTS / "cost_model"
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(outdir)
    print(f"Reading {csv_path}\nWriting outputs to {outdir.resolve()}")

    F = args.features or FEATURES[args.dataset]
    Nm, Sm = load_measured(csv_path, args.dataset, args.estimator, args.criterion,
                           args.threshold)
    N_ref, S_ref = Nm[-1], Sm[-1]
    b_fit = fit_exponent(Nm, Sm)
    n_loc = min(args.local_points, len(Nm))
    b_loc = fit_exponent(Nm[-n_loc:], Sm[-n_loc:])
    scenarios = [("measured (finite-range)", b_fit), (f"local (last {n_loc} pts)", b_loc),
                 ("b = 1 (favourable)", 1.0), ("flat, b = 0 (extreme)", 0.0)]
    scenarios += [(f"b = {b:g}", b) for b in args.extra_exponents]
    models = [("generic", "no QRAM (generic loading)"),
              ("reduced", f"reduced loading r={args.r:g}"),
              ("qram", "ideal QRAM (query depth)"),
              ("stream", f"streaming (eps={args.stream_eps:g})")]
    kw = dict(q=args.q, r=args.r, eps=args.stream_eps, c_sk=args.stream_const)

    print(f"dataset={args.dataset}  F={F}  criterion={args.criterion}  "
          f"threshold={args.threshold}  estimator={args.estimator}  q={args.q:g}")
    print("\nMeasured shots and shots per training sample (c = S/N):")
    for N_, S_ in zip(Nm, Sm):
        print(f"  N={int(N_):5d}   S={S_:>12,.0f}   c={S_ / N_:>10,.1f}")
    print(f"fitted exponent over all points b = {b_fit:.2f};  local (last {n_loc}) = {b_loc:.2f}")
    print(f"anchor: S({N_ref:.0f}) = {S_ref:,.0f}")

    print("\nMemory (what must be held while the circuit runs):")
    for n_ex in (1e3, 1e6, 1e9):
        qubits = 2 + int(np.ceil(np.log2(n_ex))) + int(np.ceil(np.log2(2 * F + 4)))
        print(f"  N = {n_ex:.0e}: this circuit {qubits} qubits; classical brute force "
              f"{n_ex * F:.1e} numbers")

    # ---- affordable shots per sample at the largest measured N ---------------
    c_meas = S_ref / N_ref
    cb, ca = cost_brute(N_ref, F), cost_ann(N_ref, F, args.ann_evals)
    print(f"\nAt N = {N_ref:.0f}: the method needs c = {c_meas:,.1f} shots per training sample.")
    print("Shots per sample it could AFFORD to still do less work (c*), and how many times too many:")
    aff_rows = []
    for m, label in models:
        cs_b = float(affordable_c(N_ref, F, cb, m, **kw))
        cs_a = float(affordable_c(N_ref, F, ca, m, **kw))
        rb = c_meas / cs_b if cs_b == cs_b else None
        ra = c_meas / cs_a if cs_a == cs_a else None
        print(f"  {label:28s} vs brute: c* = {fmt_c(cs_b):>10s}"
              f"  (needs {'-' if rb is None else f'{rb:,.3g}x'} fewer shots)"
              f"   | vs approximate: c* = {fmt_c(cs_a):>10s}"
              f"{'' if ra is not None else '  (readout of N distances alone costs more)'}")
        aff_rows.append(dict(model=m, N=int(N_ref), measured_c=c_meas, c_star_vs_brute=cs_b,
                             factor_vs_brute=rb if rb is not None else "", c_star_vs_approx=cs_a))

    # ---- break-even over a big grid, per scenario ---------------------------
    N = np.logspace(1, np.log10(args.n_max), 221)
    c_brute, c_ann = cost_brute(N, F), cost_ann(N, F, args.ann_evals)
    print("\nFirst N where the quantum method does less total work, and stays ahead "
          "to the end of the grid (EXTRAPOLATED above the measured range):")
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.2 * len(scenarios), 4.6), squeeze=False)
    rows = []
    for ax, (sname, b) in zip(axes[0], scenarios):
        S = S_ref * (N / N_ref) ** b
        ax.loglog(N, c_brute, "k-", lw=2, label="classical brute force")
        ax.loglog(N, c_ann, "k--", lw=2, label="classical approximate")
        for (m, label), col in zip(models, ["tab:red", "tab:orange", "tab:blue", "tab:purple"]):
            cq = cost_quantum(N, F, S, m, **kw)
            ax.loglog(N, cq, color=col, label=f"quantum, {label}")
            wb, wa = first_win(N, cq, c_brute), first_win(N, cq, c_ann)
            print(f"  [{sname:24s} b={b:5.2f}] {label:28s} | vs brute: {fmt_n(wb):>20s}"
                  f" | vs approximate: {fmt_n(wa):>20s}")
            rows.append(dict(scenario=sname, exponent=round(b, 3), model=m,
                             break_even_vs_brute=fmt_n(wb), break_even_vs_approx=fmt_n(wa)))
        ax.axvline(N_ref, color="gray", ls=":")
        ax.set_title(f"{sname}, S ~ N^{b:.2f}", fontsize=9)
        ax.set_xlabel("Training samples N")
        ax.set_ylabel("Operations per prediction")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6, loc="upper left")
    plt.tight_layout()
    tag = f"{args.dataset}_{args.criterion}"
    plt.savefig(f"qknn_cost_curves_{tag}.png", dpi=200)
    plt.close()

    # ---- shots per sample: needed versus affordable --------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.loglog(Nm, Sm / Nm, "ko", label="measured c = S/N")
    Nx = np.logspace(np.log10(N_ref), np.log10(args.n_max), 100)
    for (sname, b), ls in zip(scenarios, ["-", "-.", "--", ":", ":"]):
        ax.loglog(Nx, (S_ref / N_ref) * (Nx / N_ref) ** (b - 1), ls, label=f"need: {sname}")
    Nall = np.logspace(1, np.log10(args.n_max), 200)
    cs = affordable_c(Nall, F, cost_brute(Nall, F), "qram", **kw)
    ax.loglog(Nall, cs, "b-", lw=2, label="could afford: ideal QRAM vs brute force")
    cs2 = affordable_c(Nall, F, cost_brute(Nall, F), "reduced", **kw)
    ax.loglog(Nall, cs2, "r-", lw=1.5, label=f"could afford: reduced loading r={args.r:g} vs brute force")
    ax.set_xlabel("Training samples N")
    ax.set_ylabel("Shots per training sample")
    ax.set_title(f"Shots needed per sample against what could be afforded ({tag})", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5)
    plt.tight_layout()
    plt.savefig(f"qknn_shots_per_sample_{tag}.png", dpi=200)
    plt.close()

    # ---- how cheap would loading need to be ---------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for sname, b in scenarios:
        S = S_ref * (N / N_ref) ** b
        for cname, c, ls in (("brute", c_brute, "-"), ("approx", c_ann, "--")):
            r_need = (c - N) / (args.q * S * N * F)
            ok = r_need > 0
            if ok.any():
                ax.loglog(N[ok], r_need[ok], ls, label=f"{sname} vs {cname}")
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_xlabel("Training samples N")
    ax.set_ylabel("Needed loading cost / generic loading cost (r*)")
    ax.set_title("How cheap would loading have to be to break even?")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(f"qknn_loading_needed_{tag}.png", dpi=200)
    plt.close()

    for name, data in ((f"qknn_cost_breakeven_{tag}.csv", rows), (f"qknn_affordable_shots_{tag}.csv", aff_rows)):
        with open(name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    print(f"\nSaved: qknn_cost_curves_{tag}.png, qknn_shots_per_sample_{tag}.png, "
          f"qknn_loading_needed_{tag}.png, qknn_cost_breakeven_{tag}.csv, "
          f"qknn_affordable_shots_{tag}.csv")
    print("Reminder: operation counts, not seconds. Everything above the largest measured N "
          "is extrapolated; 'flat' and 'local' scenarios are fragile.")


if __name__ == "__main__":
    main()
