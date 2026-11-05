"""
Quantum fidelity kernels: label-aware vs correlation-based vs random CNOT-topology selection.
(v6: adds a tunable single-qubit bandwidth beta, a low-alpha grid, a fair TUNED protocol shared with
 the classical baseline, an end-to-end leakage test, non-planted control datasets, and richer
 recovery diagnostics.)

FEATURE MAP (1 repetition, closed form verified against Qiskit)
    H, RZ(2*beta*x_q) on every qubit, then for each chosen edge (i,j):
    CX(i,j) RZ(2*alpha*(pi-x_i)(pi-x_j)) on j, CX(i,j).
    beta  = single-qubit bandwidth (beta<1 widens the kernel),  alpha = interaction strength.

PROTOCOL (all sections)
  * Independent TRAIN and TEST sets. The scaler, correlation matrix, topology selection and ALL
    hyper-parameters (alpha, beta, RBF gamma) use TRAIN data only. Test labels are used only to
    score. `leakage_test()` verifies this end-to-end (it permutes test labels and checks that every
    selected topology/hyper-parameter is unchanged).
  * Every method, quantum or classical, sees the SAME train/test split.
  * TUNED protocol (section TUNED/CONTROLS = the numbers to quote): each method picks its own
    hyper-parameters by inner 3-fold CV accuracy on TRAIN (12-point grid for quantum (alpha,beta)
    and for RBF gamma). Classifier settings are identical (kNN k=5, SVM C=1) for all methods.
  * GRID section = descriptive landscape at FIXED hyper-parameters. Do NOT pick the best row by
    looking at test results and quote it; that is selection on the test set.
  * Statistics: paired per-dataset differences, 95% t-intervals, W/L/T. No multiplicity correction.

SECTIONS (toggle in CONFIG)
  GRID      fixed (alpha,beta) landscape incl. low alpha, edge-count curve
  TUNED     planted-interaction synthetic data, tuned protocol, recovery diagnostics
  CONTROLS  no planted structure: breast cancer, wine, random smooth function (tuned protocol)
  SWEEPS    one-factor sweeps (noise, decoys, train size, interaction strength, #pairs)
  NOISY     exact density-matrix simulation, matched draws, gate audit over all draws

WHAT THE RESULTS CAN AND CANNOT SHOW
  * The planted dataset contains u0*u1 + u2*u3 by construction. It tests whether a label-aware
    strategy can exploit interactions that exist, NOT whether useful interactions are found in
    arbitrary data. The CONTROLS section is the (partial) check for that.
  * Edge recovery is a stricter criterion than accuracy: substitute edges can classify well, and
    exact recovery can still lose to bandwidth/finite-sample effects. Read recovery as a diagnostic.
  * The closed-form-vs-Qiskit check validates the IDEAL kernel only, not the noise model.
  * Noise = depolarizing on transpiled basis gates only (no readout error, T1/T2, crosstalk),
    all-to-all connectivity (no routing). "uniform": basis {h,rz,cx}, 1q p2/2 on h,rz, 2q p2 on cx.
    "hardware_like": basis {rz,sx,cx}, rz error-free, 1q p2/10 on sx, 2q p2 on cx.

Usage:  python qkernel_rigorous_v6.py          full run (tens of minutes; see toggles / N_JOBS)
        python qkernel_rigorous_v6.py quick    smoke test (~2 min)
Install: pip install qiskit qiskit-aer scikit-learn scipy pandas numpy
"""
import itertools
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

warnings.filterwarnings("ignore")

# ----------------------------- CONFIG ---------------------------------------
N = 6
DEFAULT = dict(n_train=100, n_test=200, flip=0.03, decoy=0.15, w_inter=1.0, m_pairs=2)
N_JOBS = 1                                  # >1 parallelises datasets (fork-based; fine on Linux/Colab)

RUN_GRID, RUN_TUNED, RUN_CONTROLS, RUN_SWEEPS, RUN_NOISY = True, True, True, True, True

GRID_ALPHAS = [0.05, 0.10, 0.15, 0.25, 0.35, 0.50]
GRID_BETAS = [0.5, 1.0]
N_DATASETS_GRID = 30
N_RANDOM = 10
K_MAX = 5
CURVE_POINT = (0.25, 1.0)                   # (alpha, beta) for the edge-count curve and sweeps

TUNE_ALPHAS = [0.05, 0.15, 0.30, 0.50]
TUNE_BETAS = [0.25, 0.50, 1.00]
N_DATASETS_TUNED = 30
N_RANDOM_TUNED = 8
CONTROLS = ["breast_cancer", "wine", "random_fn"]
N_DATASETS_CONTROL = 20
SPLIT_SIZES = {"breast_cancer": (100, 200), "wine": (60, 100), "random_fn": (100, 200)}

N_DATASETS_SWEEP, N_RANDOM_SWEEP = 20, 10
SWEEPS = {"flip": [0.0, 0.03, 0.10, 0.20],
          "decoy": [0.05, 0.15, 0.50, 1.00],
          "n_train": [30, 60, 100, 200],
          "w_inter": [0.25, 0.50, 0.75, 1.00],
          "m_pairs": [1, 2, 3]}

NOISY_ALPHA, NOISY_BETA = 0.25, 1.0
NOISE_VARIANTS = ["hardware_like"]          # add "uniform" for the second noise model
NOISE_P2 = [0.02, 0.08]
N_SEL_NOISY, N_TRAIN_NOISY, N_TEST_NOISY = 100, 30, 40
N_DATASETS_NOISY, N_RANDOM_NOISY = 6, 3
OPT_LEVEL = 1
SEED = 0

if "quick" in sys.argv:
    GRID_ALPHAS, GRID_BETAS, N_DATASETS_GRID, N_RANDOM = [0.1, 0.25], [1.0], 3, 3
    TUNE_ALPHAS, TUNE_BETAS, N_DATASETS_TUNED, N_RANDOM_TUNED = [0.15, 0.3], [0.5, 1.0], 3, 2
    N_DATASETS_CONTROL = 2
    N_DATASETS_SWEEP, N_RANDOM_SWEEP, SWEEPS = 2, 3, {"flip": [0.0, 0.2], "m_pairs": [1, 3]}
    NOISE_VARIANTS, NOISE_P2 = ["uniform", "hardware_like"], [0.04]
    N_DATASETS_NOISY, N_RANDOM_NOISY, N_TRAIN_NOISY, N_TEST_NOISY = 2, 1, 12, 12
# ----------------------------------------------------------------------------
ALL_EDGES = list(itertools.combinations(range(N), 2))
_BITS = (np.arange(2 ** N)[:, None] >> np.arange(N)) & 1
_Z = 1 - 2 * _BITS
BASES = {"uniform": ["h", "rz", "cx"], "hardware_like": ["rz", "sx", "cx"]}


def _call(t):
    fn, a = t
    return fn(*a)


def pmap(fn, arglist):
    if N_JOBS <= 1:
        return [fn(*a) for a in arglist]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(N_JOBS) as ex:
        return list(ex.map(_call, [(fn, a) for a in arglist]))


# ============================ data ==========================================
def planted_pairs(m):
    return [(2 * i, 2 * i + 1) for i in range(m)]


def make_data(n, rng, s):
    """label = sign( w*I + (1-w)*M ) with I = sum u_a u_b / (sqrt(m) pi^2/12), M = sum u_a / (sqrt(m) pi/sqrt(12)),
    u = x - pi/2; fixed POPULATION scaling and threshold 0 (no dependence on the batch), then label flips.
    Features >= 2m are noisy copies of planted features (correlated decoys)."""
    m = s["m_pairs"]
    x = rng.uniform(0, np.pi, (n, N))
    src = [1, 0, 3, 2, 5, 4][:2 * m]
    for f in range(2 * m, N):
        x[:, f] = np.clip(x[:, src[(f - 2 * m) % (2 * m)]] + rng.normal(0, s["decoy"], n), 0, np.pi)
    u = x - np.pi / 2
    inter = sum(u[:, a] * u[:, b] for a, b in planted_pairs(m)) / (np.sqrt(m) * np.pi ** 2 / 12)
    main = sum(u[:, a] for a, _ in planted_pairs(m)) / (np.sqrt(m) * np.pi / np.sqrt(12))
    sc = s["w_inter"] * inter + (1 - s["w_inter"]) * main
    y = np.where(sc > 0, 1, -1)
    y[rng.random(n) < s["flip"]] *= -1
    return x, y


_REAL = {}


def real_xy(name):
    if name not in _REAL:
        if name == "breast_cancer":
            d = load_breast_cancer()
            _REAL[name] = (d.data[:, [0, 1, 4, 5, 6, 8]], np.where(d.target == 0, 1, -1))
        else:
            d = load_wine()
            _REAL[name] = (d.data[:, :6], np.where(d.target == 0, 1, -1))
    return _REAL[name]


def get_split(kind, ds, sidx, s):
    """Returns scaled (Xtr, ytr, Xte, yte). Scaler is fit on TRAIN only; test is clipped to [0, pi]."""
    rng = np.random.default_rng([SEED, sidx, ds])
    if kind == "synthetic":
        Xtr_r, ytr = make_data(s["n_train"], rng, s)
        Xte_r, yte = make_data(s["n_test"], rng, s)
    elif kind == "random_fn":
        ntr, nte = SPLIT_SIZES[kind]
        W, a, b = rng.normal(0, 1, (5, N)), rng.normal(0, 1, 5), rng.uniform(0, 2 * np.pi, 5)
        f = lambda X: np.cos(X @ W.T + b) @ a
        thr = np.median(f(rng.uniform(0, np.pi, (4000, N))))
        Xtr_r, Xte_r = rng.uniform(0, np.pi, (ntr, N)), rng.uniform(0, np.pi, (nte, N))
        ytr, yte = np.where(f(Xtr_r) > thr, 1, -1), np.where(f(Xte_r) > thr, 1, -1)
        for y in (ytr, yte):
            y[rng.random(len(y)) < DEFAULT["flip"]] *= -1
    else:
        X, y = real_xy(kind)
        ntr, nte = SPLIT_SIZES[kind]
        while True:
            idx = rng.permutation(len(y))
            tr, te = idx[:ntr], idx[ntr:ntr + nte]
            if len(set(y[tr])) == 2 and len(set(y[te])) == 2:
                break
        Xtr_r, ytr, Xte_r, yte = X[tr], y[tr], X[te], y[te]
    sc = MinMaxScaler((0, np.pi)).fit(Xtr_r)
    return sc.transform(Xtr_r), ytr, np.clip(sc.transform(Xte_r), 0, np.pi), yte


# ============================ kernel ========================================
def _states(X, edges, h):
    alpha, beta = h
    ph = beta * (X @ _Z.T)
    for i, j in edges:
        ph = ph + alpha * np.outer((np.pi - X[:, i]) * (np.pi - X[:, j]), _Z[:, i] * _Z[:, j])
    return np.exp(-1j * ph) / np.sqrt(2 ** N)


def kernel(XA, XB, edges, h):
    return np.abs(_states(XA, edges, h).conj() @ _states(XB, edges, h).T) ** 2


def feature_map(x, edges, h):
    alpha, beta = h
    qc = QuantumCircuit(N)
    for q in range(N):
        qc.h(q)
        qc.rz(2.0 * beta * x[q], q)
    for i, j in edges:
        qc.cx(i, j)
        qc.rz(2.0 * alpha * (np.pi - x[i]) * (np.pi - x[j]), j)
        qc.cx(i, j)
    return qc


# ============================ metrics =======================================
def centered_kta(K, y):
    n = len(K)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc, Yc = H @ K @ H, H @ np.outer(y, y).astype(float) @ H
    return float(np.sum(Kc * Yc) / (np.linalg.norm(Kc) * np.linalg.norm(Yc) + 1e-12))


def fit_predict_acc(Ktr, ytr, Kte, yte):
    Ktr, Kte = np.clip(Ktr, 0, 1), np.clip(Kte, 0, 1)
    p_knn = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(1 - Ktr, ytr).predict(1 - Kte)
    p_svm = SVC(kernel="precomputed", C=1.0).fit(Ktr, ytr).predict(Kte)
    return float((p_knn == yte).mean()), float((p_svm == yte).mean())


def inner_cv_acc(K, y):
    """3-fold CV on TRAIN only, mean of kNN and SVM accuracy. Identical for every method."""
    K = np.clip(K, 0, 1)
    accs = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=0).split(K, y):
        a, b = fit_predict_acc(K[np.ix_(tr, tr)], y[tr], K[np.ix_(te, tr)], y[te])
        accs.append((a + b) / 2)
    return float(np.mean(accs))


def evaluate(Xtr, ytr, Xte, yte, edges, h):
    Ktr, Kte = kernel(Xtr, Xtr, edges, h), kernel(Xte, Xtr, edges, h)
    knn, svm = fit_predict_acc(Ktr, ytr, Kte, yte)
    return dict(knn=knn, svm=svm, kta=centered_kta(kernel(Xte, Xte, edges, h), yte))


def edge_metrics(edges, oracle):
    """recov = fraction of planted edges selected; exact = all planted edges and no others;
    cover = fraction of planted pairs connected through the selected graph (substitute paths count)."""
    if oracle is None or edges is None:
        return dict(recov=np.nan, exact=np.nan, cover=np.nan)
    parent = list(range(N))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        parent[find(i)] = find(j)
    return dict(recov=len(set(edges) & set(oracle)) / len(oracle),
                exact=float(set(edges) == set(oracle)),
                cover=float(np.mean([find(a) == find(b) for a, b in oracle])))


# ============================ selection =====================================
def greedy(Xtr, ytr, h, k_max, objective):
    """Add the edge maximising a TRAIN objective: 'kta', or 'cv' (inner-CV accuracy, KTA tie-break)."""
    def obj(edges):
        K = kernel(Xtr, Xtr, edges, h)
        kta = centered_kta(K, ytr)
        return kta if objective == "kta" else (inner_cv_acc(K, ytr), kta)

    chosen, path = [], [[]]
    for _ in range(k_max):
        cands = [e for e in ALL_EDGES if e not in chosen]
        vals = [obj(chosen + [e]) for e in cands]
        chosen = chosen + [cands[max(range(len(cands)), key=lambda i: vals[i])]]
        path.append(list(chosen))
    return path


def corr_rank(X, descending):
    c = np.abs(np.corrcoef(X.T))
    return sorted(ALL_EDGES, key=lambda e: c[e], reverse=descending)


def tune(Xtr, ytr, grid, topo_fn):
    """Pick (alpha,beta) by inner-CV accuracy on TRAIN (KTA tie-break); topo_fn(h) selects edges on TRAIN."""
    best = None
    for h in grid:
        e = topo_fn(h)
        K = kernel(Xtr, Xtr, e, h)
        score = (inner_cv_acc(K, ytr), centered_kta(K, ytr))
        if best is None or score > best[0]:
            best = (score, h, e)
    return best[1], best[2]


def rbf_rows(Xtr, ytr, Xte, yte, n_grid):
    """Classical baseline: default gamma, and gamma tuned by the SAME inner CV on TRAIN (n_grid points)."""
    g0 = 1.0 / (N * Xtr.var())
    grid = g0 * np.geomspace(1 / 16, 16, n_grid)
    g_best = max(grid, key=lambda g: inner_cv_acc(rbf_kernel(Xtr, gamma=g), ytr))
    out = []
    for name, g in (("rbf_default", g0), ("rbf_tuned", g_best)):
        knn, svm = fit_predict_acc(rbf_kernel(Xtr, gamma=g), ytr, rbf_kernel(Xte, Xtr, gamma=g), yte)
        out.append(dict(method=name, k=0, alpha=np.nan, beta=np.nan, gamma=float(g), edges="rbf",
                        knn=knn, svm=svm, kta=centered_kta(rbf_kernel(Xte, gamma=g), yte),
                        recov=np.nan, exact=np.nan, cover=np.nan))
    return out


# ============================ GRID (fixed hyper-parameters) =================
def run_dataset_grid(ds, sidx, s, points, curve, n_random):
    Xtr, ytr, Xte, yte = get_split("synthetic", ds, sidx, s)
    m, oracle = s["m_pairs"], planted_pairs(s["m_pairs"])
    erng = np.random.default_rng([SEED, 999, sidx, ds])
    ks = list(range(1, K_MAX + 1)) if curve else [m]
    rand_sets = {k: [[ALL_EDGES[i] for i in erng.choice(len(ALL_EDGES), k, replace=False)]
                     for _ in range(n_random)] for k in ks}
    hi, lo = corr_rank(Xtr, True), corr_rank(Xtr, False)
    rows = []
    for h in points:
        g_kta, g_cv = greedy(Xtr, ytr, h, max(ks), "kta"), greedy(Xtr, ytr, h, max(ks), "cv")

        def add(method, k, edges):
            r = evaluate(Xtr, ytr, Xte, yte, edges, h)
            r.update(ds=ds, method=method, k=k, alpha=h[0], beta=h[1], gamma=np.nan, edges=str(edges),
                     **edge_metrics(edges, oracle))
            rows.append(r)

        add("none", 0, [])
        add("oracle", m, oracle)
        add("chain", N - 1, [(i, i + 1) for i in range(N - 1)])
        for k in ks:
            add("greedy_kta", k, g_kta[k])
            add("greedy_cvacc", k, g_cv[k])
            add("corr_top", k, hi[:k])
            add("corr_bottom", k, lo[:k])
            for e in rand_sets[k]:
                add("random", k, e)
    for r in rbf_rows(Xtr, ytr, Xte, yte, 12):
        r.update(ds=ds)
        rows.append(r)
    return rows


# ============================ TUNED (fair protocol) =========================
def run_dataset_tuned(ds, sidx, kind, s, n_random, perm_test=False):
    Xtr, ytr, Xte, yte = get_split(kind, ds, sidx, s)
    if perm_test:                                    # leakage test: scramble test labels
        yte = np.random.default_rng([SEED, 4242, ds]).permutation(yte)
    synthetic = kind == "synthetic"
    k = s["m_pairs"] if synthetic else 2
    oracle = planted_pairs(s["m_pairs"]) if synthetic else None
    erng = np.random.default_rng([SEED, 999, sidx, ds])
    grid = [(a, b) for a in TUNE_ALPHAS for b in TUNE_BETAS]
    beta_only = [(0.0, b) for b in TUNE_BETAS]
    hi, lo = corr_rank(Xtr, True)[:k], corr_rank(Xtr, False)[:k]
    chain = [(i, i + 1) for i in range(N - 1)]
    rows = []

    def done(method, h, edges):
        r = evaluate(Xtr, ytr, Xte, yte, edges, h)
        r.update(ds=ds, method=method, k=len(edges), alpha=h[0], beta=h[1], gamma=np.nan,
                 edges=str(edges), **edge_metrics(edges, oracle))
        rows.append(r)

    h, e = tune(Xtr, ytr, beta_only, lambda h: [])
    done("none", h, e)
    h, e = tune(Xtr, ytr, grid, lambda h: chain)
    done("chain", h, e)
    if synthetic:
        h, e = tune(Xtr, ytr, grid, lambda h: oracle)
        done("oracle", h, e)
    h, e = tune(Xtr, ytr, grid, lambda h: hi)
    done("corr_top", h, e)
    for _ in range(n_random):
        rs = [ALL_EDGES[i] for i in erng.choice(len(ALL_EDGES), k, replace=False)]
        h, e = tune(Xtr, ytr, grid, lambda h: rs)
        done("random", h, e)
    h, e = tune(Xtr, ytr, grid, lambda h: greedy(Xtr, ytr, h, k, "kta")[k])
    done("greedy_kta", h, e)
    h, e = tune(Xtr, ytr, grid, lambda h: greedy(Xtr, ytr, h, k, "cv")[k])
    done("greedy_cvacc", h, e)
    for r in rbf_rows(Xtr, ytr, Xte, yte, len(grid)):
        r.update(ds=ds)
        rows.append(r)
    return rows


# ============================ statistics / reporting ========================
def ci(a, b=None):
    d = (a - b if b is not None else a).dropna()
    n = len(d)
    if n < 2:
        return np.nan, np.nan, np.nan, 0, 0, 0
    mean = d.mean()
    hw = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    w, l = int((d > 0).sum()), int((d < 0).sum())
    return mean, mean - hw, mean + hw, w, l, n - w - l


def fmt(t, wlt=True):
    star = "*" if (t[1] > 0 or t[2] < 0) else " "
    s = f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]{star}"
    return s + (f"  W/L/T {t[3]}/{t[4]}/{t[5]}" if wlt else "")


def pm(df, method, col="knn", k=None):
    d = df[df.method == method]
    if k is not None:
        d = d[d.k == k]
    return d.groupby("ds")[col].mean()


def report_tuned(df, title, has_oracle):
    print(f"\n===== {title}: {df.ds.nunique()} datasets; hyper-parameters tuned on TRAIN only (inner 3-fold CV) =====")
    order = ["none", "chain"] + (["oracle"] if has_oracle else []) + \
            ["corr_top", "random", "greedy_kta", "greedy_cvacc", "rbf_default", "rbf_tuned"]
    extra = "  exact  cover" if has_oracle else ""
    print(f"  {'method':<13}{'edges':>5}  {'kNN':>13}  {'SVM':>13}  {'testKTA':>13}  {'alpha':>5} {'beta':>5}{extra}")
    for meth in order:
        d = df[df.method == meth]
        c = d.groupby("ds")[["knn", "svm", "kta", "recov", "exact", "cover"]].mean()
        ex = f"  {c.exact.mean():.2f}   {c.cover.mean():.2f}" if has_oracle and c.exact.notna().any() else ""
        a = f"{d.alpha.median():.2f}" if d.alpha.notna().any() else "  - "
        b = f"{d.beta.median():.2f}" if d.beta.notna().any() else "  - "
        print(f"  {meth:<13}{int(d.k.median()):>5}  {c.knn.mean():.3f}+/-{c.knn.std():.3f}  "
              f"{c.svm.mean():.3f}+/-{c.svm.std():.3f}  {c.kta.mean():.3f}+/-{c.kta.std():.3f}  {a:>5} {b:>5}{ex}")
    pairs = [("greedy_kta - random", "greedy_kta", "random"), ("greedy_cvacc - random", "greedy_cvacc", "random"),
             ("corr_top - random", "corr_top", "random"), ("greedy_kta - corr_top", "greedy_kta", "corr_top"),
             ("greedy_kta - none (beta tuned)", "greedy_kta", "none"), ("greedy_kta - chain", "greedy_kta", "chain"),
             ("none - rbf_tuned", "none", "rbf_tuned"), ("chain - rbf_tuned", "chain", "rbf_tuned"),
             ("greedy_kta - rbf_tuned", "greedy_kta", "rbf_tuned")]
    if has_oracle:
        pairs += [("oracle - none (beta tuned)", "oracle", "none"), ("oracle - greedy_kta", "oracle", "greedy_kta"),
                  ("oracle - rbf_tuned", "oracle", "rbf_tuned")]
    for col in ("knn", "svm"):
        print(f"\n  paired differences, test {col.upper()}: mean [95% CI] (* = CI excludes 0)")
        for label, a, b in pairs:
            print(f"    {label:<32}{fmt(ci(pm(df, a, col), pm(df, b, col)))}")
    if has_oracle:
        print("\n  recovery diagnostics (interpret with care: substitute edges can classify well):")
        orc = pm(df, "oracle", "knn")
        for meth in ("greedy_kta", "greedy_cvacc"):
            d = df[df.method == meth].set_index("ds")
            ex = d.exact == 1
            print(f"    {meth}: exact recovery {ex.mean():.2f} | mean edge recovery {d.recov.mean():.2f} | "
                  f"planted pairs connected {d.cover.mean():.2f}")
            if ex.sum() >= 2 and (~ex).sum() >= 2:
                gap = (orc.reindex(d.index) - d.knn)
                print(f"      kNN exact {d.knn[ex].mean():.3f} (n={ex.sum()}) vs not exact {d.knn[~ex].mean():.3f} "
                      f"(n={(~ex).sum()}); gap to oracle {gap[ex].mean():+.3f} vs {gap[~ex].mean():+.3f}")
            r = stats.spearmanr(d.recov, d.knn)[0]
            print(f"      Spearman(edge recovery, test kNN) across datasets: {r:+.2f}")


def kof(meth, m):
    return 0 if meth == "none" else (N - 1 if meth == "chain" else m)


def report_grid(df, m):
    ref = pm(df, "rbf_tuned", "knn")
    q = df[df.alpha.notna()]
    print(f"\n===== GRID (fixed hyper-parameters; DESCRIPTIVE ONLY; {df.ds.nunique()} datasets; test kNN) =====")
    print(f"  rbf_tuned reference kNN = {ref.mean():.3f}")
    print(f"  {'alpha':>5} {'beta':>5} | {'none':>5} {'oracle':>6} {'greedy':>6} {'random':>6} {'chain':>5} | "
          f"{'oracle-none':<27}{'greedy_kta-random':<27}{'corr_top-random':<27}{'greedy_kta-chain':<27}")
    for (a, b), d in q.groupby(["alpha", "beta"]):
        g = lambda meth: d[(d.method == meth) & (d.k == kof(meth, m))].groupby("ds").knn.mean()
        print(f"  {a:>5.2f} {b:>5.2f} | {g('none').mean():.3f} {g('oracle').mean():.3f}  {g('greedy_kta').mean():.3f}  "
              f"{g('random').mean():.3f}  {g('chain').mean():.3f} | "
              f"{fmt(ci(g('oracle'), g('none')), False):<27}{fmt(ci(g('greedy_kta'), g('random')), False):<27}"
              f"{fmt(ci(g('corr_top'), g('random')), False):<27}{fmt(ci(g('greedy_kta'), g('chain')), False):<27}")
    print("  NOTE: do not pick the best row by test accuracy and quote it; use the TUNED section for that.")


def report_curve(df, m):
    print(f"\n  edge-count curve at (alpha,beta)={CURVE_POINT}, mean test kNN over {df.ds.nunique()} datasets:")
    print(f"    k   greedy_kta  greedy_cv  corr_top  random   | chain(k={N-1}) {pm(df,'chain','knn').mean():.3f}"
          f"  none {pm(df,'none','knn').mean():.3f}  rbf_tuned {pm(df,'rbf_tuned','knn').mean():.3f}")
    for k in range(1, K_MAX + 1):
        v = [pm(df, mt, "knn", k).mean() for mt in ("greedy_kta", "greedy_cvacc", "corr_top", "random")]
        print(f"    {k}   {v[0]:.3f}       {v[1]:.3f}      {v[2]:.3f}     {v[3]:.3f}")


def run_sweeps():
    print(f"\n===== SWEEPS at (alpha,beta)={CURVE_POINT}, {N_DATASETS_SWEEP} datasets/setting (test kNN, paired diffs, 95% CI) =====")
    print(f"  {'setting':<14}{'oracle-none':<28}{'greedy_kta-random':<28}{'greedy_cv-random':<28}{'corr_top-random':<28}recov")
    out = []
    for fi, (factor, values) in enumerate(SWEEPS.items()):
        for vi, v in enumerate(values):
            s = dict(DEFAULT)
            s[factor] = v
            sidx = 10 * (fi + 1) + vi + 1
            rows = sum(pmap(run_dataset_grid, [(ds, sidx, s, [CURVE_POINT], False, N_RANDOM_SWEEP)
                                               for ds in range(N_DATASETS_SWEEP)]), [])
            df = pd.DataFrame(rows)
            df["factor"], df["value"] = factor, v
            out.append(df)
            m = s["m_pairs"]
            g = lambda meth, c="knn": pm(df[df.alpha.notna()], meth, c, kof(meth, m))
            print(f"  {factor + '=' + str(v):<14}{fmt(ci(g('oracle'), g('none')), False):<28}"
                  f"{fmt(ci(g('greedy_kta'), g('random')), False):<28}{fmt(ci(g('greedy_cvacc'), g('random')), False):<28}"
                  f"{fmt(ci(g('corr_top'), g('random')), False):<28}{g('greedy_kta', 'recov').mean():.2f}", flush=True)
    return pd.concat(out, ignore_index=True)


# ============================ checks ========================================
def sanity_check():
    rng = np.random.default_rng(1)
    Xt = rng.uniform(0, np.pi, (4, N))
    h, e = (0.5, 0.7), [(0, 1), (2, 3), (1, 4)]
    S = np.array([Statevector.from_instruction(feature_map(x, e, h)).data for x in Xt])
    err = np.abs(np.abs(S.conj() @ S.T) ** 2 - kernel(Xt, Xt, e, h)).max()
    print(f"sanity: closed-form vs Qiskit statevector kernel (beta!=1, overlapping edges): max abs err {err:.1e}")
    print("        (validates the IDEAL kernel only; the noise model is a modelling assumption, not validated.)")


def leakage_test():
    s = dict(DEFAULT)
    a = run_dataset_tuned(0, 0, "synthetic", s, 2, False)
    b = run_dataset_tuned(0, 0, "synthetic", s, 2, True)
    key = lambda rows: [(r["method"], r["edges"], str(r["alpha"]), str(r["beta"]), str(r["gamma"])) for r in rows]
    same = key(a) == key(b)
    changed = any(abs(x["knn"] - y["knn"]) > 0 for x, y in zip(a, b))
    print(f"leakage test (test labels permuted): selected topologies/hyper-parameters identical = {same}; "
          f"test accuracies changed = {changed}  -> {'PASS' if same and changed else 'FAIL'}")
    if not (same and changed):
        raise SystemExit("leakage test failed")


# ============================ NOISY =========================================
def noise_model(p2, variant):
    nm = NoiseModel()
    if variant == "uniform":
        nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 2, 1), ["h", "rz"])
    else:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["sx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return nm


def overlap_circuits(Xtr, Xte, edges, h, basis):
    ftr = [feature_map(x, edges, h) for x in Xtr]
    fte = [feature_map(x, edges, h) for x in Xte]
    jobs, circs = [], []
    for i, j in itertools.combinations(range(len(Xtr)), 2):
        circs.append(ftr[i].compose(ftr[j].inverse()))
        jobs.append(("tr", i, j))
    for i in range(len(Xte)):
        for j in range(len(Xtr)):
            circs.append(fte[i].compose(ftr[j].inverse()))
            jobs.append(("te", i, j))
    circs = transpile(circs, basis_gates=basis, optimization_level=OPT_LEVEL, seed_transpiler=SEED)
    for qc in circs:
        qc.save_probabilities()
    return circs, jobs


def noisy_kernels(circs, jobs, ntr, nte, p2, variant):
    sim = AerSimulator(method="density_matrix", noise_model=noise_model(p2, variant))
    Ktr, Kte = np.eye(ntr), np.zeros((nte, ntr))
    for s in range(0, len(circs), 300):
        res = sim.run(circs[s:s + 300]).result()
        for k in range(len(circs[s:s + 300])):
            kind, i, j = jobs[s + k]
            v = float(res.data(k)["probabilities"][0])
            if kind == "tr":
                Ktr[i, j] = Ktr[j, i] = v
            else:
                Kte[i, j] = v
    return Ktr, Kte


def gate_audit(edges, h, basis):
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, np.pi, (2, N))
    qc = feature_map(x, edges, h).compose(feature_map(y, edges, h).inverse())
    out = {}
    for lvl in (0, 1):
        t = transpile(qc, basis_gates=basis, optimization_level=lvl, seed_transpiler=1)
        ops = t.count_ops()
        out[lvl] = (ops.get("cx", 0), sum(ops.values()), t.depth())
    return out


def run_noisy():
    s, h, m = DEFAULT, (NOISY_ALPHA, NOISY_BETA), DEFAULT["m_pairs"]
    ntr, nte = N_TRAIN_NOISY, N_TEST_NOISY
    print(f"\n===== NOISY alpha={h[0]} beta={h[1]}; {N_DATASETS_NOISY} matched draws, train={ntr} test={nte} "
          f"({ntr * (ntr - 1) // 2 + ntr * nte} overlap circuits/topology/variant), opt_level={OPT_LEVEL} =====")
    rows, audit = [], []
    t0 = time.time()
    for ds in range(N_DATASETS_NOISY):
        rng = np.random.default_rng([SEED, 7, ds])
        erng = np.random.default_rng([SEED, 8, ds])
        Xsel_r, ysel = make_data(N_SEL_NOISY, rng, s)
        Xtr_r, ytr = make_data(ntr, rng, s)
        Xte_r, yte = make_data(nte, rng, s)
        sc = MinMaxScaler((0, np.pi)).fit(Xsel_r)
        Xsel = sc.transform(Xsel_r)
        Xtr, Xte = (np.clip(sc.transform(a), 0, np.pi) for a in (Xtr_r, Xte_r))
        topo = {"none": [], "oracle": planted_pairs(m),
                "greedy_kta": greedy(Xsel, ysel, h, m, "kta")[m],
                "greedy_cvacc": greedy(Xsel, ysel, h, m, "cv")[m],
                "corr_top": corr_rank(Xsel, True)[:m],
                "chain": [(i, i + 1) for i in range(N - 1)]}
        for r in range(N_RANDOM_NOISY):
            topo[f"random_{r}"] = [ALL_EDGES[i] for i in erng.choice(len(ALL_EDGES), m, replace=False)]
        for name, e in topo.items():
            Ktr_i, Kte_i = kernel(Xtr, Xtr, e, h), kernel(Xte, Xtr, e, h)
            knn, svm = fit_predict_acc(Ktr_i, ytr, Kte_i, yte)
            rows.append(dict(ds=ds, variant="ideal", p2=0.0, topology=name, knn=knn, svm=svm, kerr=0.0))
            for variant in NOISE_VARIANTS:
                a = gate_audit(e, h, BASES[variant])
                audit.append((variant, "random" if name.startswith("random_") else name, len(e),
                              a[0][0], a[1][0], a[1][1], a[1][2]))
                circs, jobs = overlap_circuits(Xtr, Xte, e, h, BASES[variant])
                for p2 in NOISE_P2:
                    Ktr_n, Kte_n = noisy_kernels(circs, jobs, ntr, nte, p2, variant)
                    knn, svm = fit_predict_acc(Ktr_n, ytr, Kte_n, yte)
                    iu = np.triu_indices(ntr, 1)
                    kerr = float(np.mean(np.concatenate([np.abs(Ktr_n - Ktr_i)[iu], np.abs(Kte_n - Kte_i).ravel()])))
                    rows.append(dict(ds=ds, variant=variant, p2=p2, topology=name, knn=knn, svm=svm, kerr=kerr))
        print(f"  draw {ds + 1}/{N_DATASETS_NOISY} done [{time.time() - t0:.0f}s]", flush=True)

    ad = pd.DataFrame(audit, columns=["variant", "topology", "edges", "cx_L0", "cx_L1", "gates_L1", "depth_L1"])
    print("\n  --- GATE AUDIT over ALL draws (one representative overlap circuit per topology; mean [min-max]) ---")
    print(f"  {'variant':<14}{'topology':<14}{'edges':>5}{'cx L0':>8}{'cx L1':>16}{'gates L1':>18}{'depth L1':>18}")
    for (v, t), d in ad.groupby(["variant", "topology"], sort=False):
        f = lambda c: f"{d[c].mean():.1f} [{d[c].min()}-{d[c].max()}]"
        print(f"  {v:<14}{t:<14}{int(d.edges.iloc[0]):>5}{d.cx_L0.mean():>8.1f}{f('cx_L1'):>16}{f('gates_L1'):>18}{f('depth_L1'):>18}")
    print("  (L0 = no optimisation, L1 = level 1: adjacent mirrored CX pairs cancel. No coupling map: routing/SWAP overhead NOT included.)")

    df = pd.DataFrame(rows)
    df["group"] = np.where(df.topology.str.startswith("random_"), "random", df.topology)
    ideal = df[df.variant == "ideal"].pivot_table(index="ds", columns="group", values="knn")
    cn = {"none": 0, "oracle": 2 * m, "greedy_kta": 2 * m, "greedy_cvacc": 2 * m, "corr_top": 2 * m,
          "random": 2 * m, "chain": 2 * (N - 1)}
    for variant in NOISE_VARIANTS:
        for p2 in NOISE_P2:
            sub = df[(df.variant == variant) & (df.p2 == p2)]
            pk, ps, ke = (sub.pivot_table(index="ds", columns="group", values=c) for c in ("knn", "svm", "kerr"))
            print(f"\n  --- noise={variant}, p_cx={p2} ({pk.shape[0]} matched draws) ---")
            print(f"  {'topology':<14}{'cnots':>6}  {'kNN':>13}  {'SVM':>13}  {'kernel err':>10}  {'kNN noisy-ideal':>16}")
            for g in cn:
                print(f"  {g:<14}{cn[g]:>6}  {pk[g].mean():.3f}+/-{pk[g].std():.3f}  {ps[g].mean():.3f}+/-{ps[g].std():.3f}  "
                      f"{ke[g].mean():>10.4f}  {(pk[g] - ideal[g]).mean():>+16.3f}")
            print("  paired differences (test kNN | SVM), mean [95% CI]; * = CI excludes 0:")
            for label, a, b in [("greedy_kta - random", "greedy_kta", "random"), ("greedy_cvacc - random", "greedy_cvacc", "random"),
                                ("corr_top - random", "corr_top", "random"), ("oracle - none", "oracle", "none"),
                                ("greedy_kta - chain", "greedy_kta", "chain")]:
                print(f"    {label:<24}kNN {fmt(ci(pk[a], pk[b]), False):<28} SVM {fmt(ci(ps[a], ps[b]), False)}")
    print(f"\n  NOTE: with {N_DATASETS_NOISY} draws intervals are wide; increase N_DATASETS_NOISY before claiming anything.")
    return df


# ============================ main ==========================================
def main():
    t0 = time.time()
    print(__doc__.split("WHAT THE RESULTS CAN AND CANNOT SHOW")[1].split("Usage:")[0].rstrip())
    sanity_check()
    leakage_test()
    m = DEFAULT["m_pairs"]

    if RUN_GRID:
        pts = [(a, b) for a in GRID_ALPHAS for b in GRID_BETAS]
        rows = sum(pmap(run_dataset_grid, [(ds, 0, DEFAULT, pts, False, N_RANDOM) for ds in range(N_DATASETS_GRID)]), [])
        dfg = pd.DataFrame(rows)
        report_grid(dfg, m)
        dfg.to_csv("v6_grid.csv", index=False)
        rows = sum(pmap(run_dataset_grid, [(ds, 0, DEFAULT, [CURVE_POINT], True, N_RANDOM)
                                           for ds in range(N_DATASETS_GRID)]), [])
        dfc = pd.DataFrame(rows)
        report_curve(dfc, m)
        dfc.to_csv("v6_curve.csv", index=False)
        print(f"  [grid done, {time.time() - t0:.0f}s]", flush=True)

    if RUN_TUNED:
        rows = sum(pmap(run_dataset_tuned, [(ds, 0, "synthetic", DEFAULT, N_RANDOM_TUNED)
                                            for ds in range(N_DATASETS_TUNED)]), [])
        dft = pd.DataFrame(rows)
        report_tuned(dft, "TUNED, planted-interaction synthetic data", True)
        dft.to_csv("v6_tuned.csv", index=False)
        print(f"  [tuned done, {time.time() - t0:.0f}s]", flush=True)

    if RUN_CONTROLS:
        frames = []
        for ci_, kind in enumerate(CONTROLS):
            rows = sum(pmap(run_dataset_tuned, [(ds, 50 + ci_, kind, DEFAULT, N_RANDOM_TUNED)
                                                for ds in range(N_DATASETS_CONTROL)]), [])
            d = pd.DataFrame(rows)
            d["dataset"] = kind
            frames.append(d)
            report_tuned(d, f"CONTROL '{kind}' (no planted structure; edge budget k=2)", False)
        pd.concat(frames).to_csv("v6_controls.csv", index=False)
        print(f"  [controls done, {time.time() - t0:.0f}s]", flush=True)

    if RUN_SWEEPS:
        run_sweeps().to_csv("v6_sweeps.csv", index=False)
        print(f"  [sweeps done, {time.time() - t0:.0f}s]", flush=True)

    if RUN_NOISY:
        run_noisy().to_csv("v6_noisy.csv", index=False)

    print(f"\nDone in {time.time() - t0:.0f}s. CSVs: v6_grid/curve/tuned/controls/sweeps/noisy.csv")
    print("GUARDRAILS: (1) quote TUNED/CONTROLS numbers, not best GRID rows; (2) planted data favour the oracle by "
          "construction; (3) recovery is a diagnostic, not a goal; (4) closed-form check validates ideal kernels only; "
          "(5) no multiplicity correction; (6) no routing/readout/coherent errors in the noisy stage.")


if __name__ == "__main__":
    main()
