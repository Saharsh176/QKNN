#!/usr/bin/env python3
"""
07_quantum_knn_implementation_v4_1.py
=====================================

Study 07.4 (v4.1) - circuit-level quantum k-NN, repeated-split experiment.

v4.1 fixes (on top of v4)
-------------------------
A. Quantile binning no longer collapses on binary / few-valued features.
   Bin edges are now built from the distinct training values, so a 0/1
   feature still gets one threshold (0.5). Unused edges are set to +inf.
B. Shot efficiency no longer relies only on "prediction agreement with exact",
   which is unfair for tie-heavy encodings (Basis). New column:
   shots_for_accuracy_within_tolerance = first shot level where the accuracy
   gap to exact stays within --accuracy-tolerance at that and all higher levels.
   Precision/agreement now also require all higher levels to pass.
   Agreement is flagged as unreliable when exact ties are common.
C. Interval columns are named *_split_t_interval95_halfwidth and the config
   states that overlapping splits make them descriptive, not formal CIs.
D. Default tie repeats raised to 50.

What changed compared with the 07.3 pilot
-----------------------------------------
1. REPEATED SPLITS. Every dataset is evaluated on many different stratified
   train/test splits (default 10; use 20 for the final run). Results are
   reported as mean +/- 95% confidence interval, and as a paired difference
   against classical k-NN on the same splits.

2. LARGER SAMPLES. Defaults are 100 training and 40 test points per split
   (the pilot used 50 / 15).

3. REPRESENTATION QUALITY AND SHOT EFFICIENCY ARE REPORTED SEPARATELY.
   * summary_exact   : how good is the encoding itself (no shot noise)?
   * summary_shots   : how does the estimated result behave at each shot count?
   * shot_efficiency : how many shots are needed to recover the exact neighbours?

4. TIE-AWARE HEADLINE METRIC. When many candidates have identical similarity
   (always true for Basis encoding), a single accuracy number depends on an
   arbitrary tie-break. The headline metric is therefore accuracy averaged over
   many tie-break seeds ("accuracy_tie_averaged"), with min/max also reported.

5. TIE-ROBUST NEIGHBOUR RECOVERY. The pilot compared neighbour sets to ONE
   arbitrary exact set. That is unfair when ties exist. v4 uses
   "neighbour_precision_vs_exact": the fraction of chosen neighbours whose
   exact fidelity is at least the exact k-th fidelity (i.e. a valid exact
   neighbour under some tie-break).

6. BASIS-ENCODING VARIANTS. Basis fidelity between two states is either 1 (same
   bit string) or 0 (different), so basis k-NN can only "see" exact matches.
   v4 lets you test more bits per feature (basis_2bit, basis_3bit) and
   quantile binning (suffix _q) and reports collisions and exact-match rates.
   Note: more bits reduce collisions but also reduce exact matches, so this
   does not remove the core limitation. See pairwise_summary to check.

7. FAIRER PREPROCESSING. The scaler and the variance-based feature cap are now
   fitted on the training split only, and the cap is applied AFTER min-max
   scaling (before, it ran on raw data and favoured large-scale features).

8. FASTER SHOT SIMULATION WITHOUT LOSING THE CIRCUIT EVIDENCE.
   * estimator "swap_test_circuit": real Aer SWAP-test circuits.
   * estimator "swap_test_binomial": draws counts from the exact SWAP-test
     outcome distribution, P(ancilla=0) = (1+F)/2. This is statistically the
     same experiment for an ideal simulator, just much faster.
   Real circuits are used for the first --circuit-splits splits, and a sheet
   (estimator_crosscheck) compares the two estimators on those splits.

9. Basis states are prepared with X gates (not a generic state-preparation
   block), which is the honest, cheap circuit for a bit string.

Important interpretation
------------------------
* Aer is a classical simulator, not quantum hardware.
* No QRAM, no hardware speedup, no quantum advantage is claimed.
* Angle and phase maps used here are mathematically fidelity-equivalent
  (both give F = cos^2(delta_theta / 2) per feature).
* The raw SWAP estimator is not clipped; a clipped value is exported only as a
  diagnostic in the validation sheet.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    make_classification, make_blobs,
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, pairwise_distances,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import StatePreparation
    from qiskit_aer import AerSimulator
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Qiskit and Qiskit Aer are required. Install with:\n"
        "  pip install qiskit qiskit-aer scikit-learn pandas openpyxl scipy\n"
        f"Import error: {exc}"
    )

warnings.filterwarnings("ignore", message=".*y_pred contains classes not in y_true.*")

MASK31 = 2 ** 31 - 10


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]


def load_dataset(name: str, random_state: int) -> Dataset:
    if name == "iris":
        d = load_iris()
        return Dataset(name, d.data, d.target, list(d.feature_names))
    if name == "wine":
        d = load_wine()
        return Dataset(name, d.data, d.target, list(d.feature_names))
    if name == "breast_cancer":
        d = load_breast_cancer()
        return Dataset(name, d.data, d.target, list(d.feature_names))
    if name == "digits":
        d = load_digits()
        return Dataset(name, d.data, d.target, [f"pixel_{i}" for i in range(d.data.shape[1])])
    if name == "synthetic_binarized":
        X, y = make_classification(
            n_samples=500, n_features=8, n_informative=5, n_redundant=1,
            n_classes=2, class_sep=1.1, random_state=random_state,
        )
        return Dataset(name, X, y, [f"x{i}" for i in range(X.shape[1])])
    if name == "synthetic_blobs":
        X, y = make_blobs(
            n_samples=500, n_features=6, centers=3, cluster_std=2.0,
            random_state=random_state,
        )
        return Dataset(name, X, y, [f"x{i}" for i in range(X.shape[1])])
    raise ValueError(f"Unknown dataset: {name}")


def dataset_split(
    ds: Dataset, max_train: int, max_test: int, max_features: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Stratified split. Scaling and feature cap are fitted on TRAIN only."""
    X = np.asarray(ds.X, dtype=float)
    y = LabelEncoder().fit_transform(ds.y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    if max_train > 0 and len(X_train) > max_train:
        _, X_train, _, y_train = train_test_split(
            X_train, y_train, test_size=max_train, random_state=(seed + 1) % MASK31,
            stratify=y_train,
        )
    if max_test > 0 and len(X_test) > max_test:
        _, X_test, _, y_test = train_test_split(
            X_test, y_test, test_size=max_test, random_state=(seed + 2) % MASK31,
            stratify=y_test,
        )
    scaler = MinMaxScaler().fit(X_train)
    X_train = np.clip(scaler.transform(X_train), 0.0, 1.0)
    X_test = np.clip(scaler.transform(X_test), 0.0, 1.0)

    names = list(ds.feature_names)
    if max_features > 0 and X_train.shape[1] > max_features:
        # Label-free cap: keep the highest-variance features AFTER scaling.
        keep = np.sort(np.argsort(np.var(X_train, axis=0))[::-1][:max_features])
        X_train, X_test = X_train[:, keep], X_test[:, keep]
        names = [names[i] for i in keep]
    return X_train, X_test, y_train, y_test, names


# --------------------------------------------------------------------------
# Encodings
# --------------------------------------------------------------------------
FIXED_ENCODINGS = ("angle", "phase", "amplitude", "amplitude_centered")
BASIS_RE = re.compile(r"^basis(?:_(\d+)bit)?(_q)?$")
MAX_DENSE_QUBITS = 16


def parse_basis(enc: str) -> Optional[Tuple[int, bool]]:
    """'basis' -> (1, False); 'basis_2bit_q' -> (2, True); else None."""
    m = BASIS_RE.match(enc)
    if not m:
        return None
    return (int(m.group(1)) if m.group(1) else 1), bool(m.group(2))


def validate_encoding(enc: str) -> None:
    if enc in FIXED_ENCODINGS:
        return
    pb = parse_basis(enc)
    if pb and 1 <= pb[0] <= 4:
        return
    raise SystemExit(
        f"Unknown encoding '{enc}'. Use: {', '.join(FIXED_ENCODINGS)}, "
        "basis, basis_2bit, basis_3bit (optional suffix _q for quantile bins)."
    )


def fit_basis_edges(X_train: np.ndarray, bits: int, quantile: bool) -> np.ndarray:
    """Bin edges per feature, shape (n_features, 2**bits - 1). Fitted on train."""
    levels = 2 ** bits
    probs = np.linspace(0.0, 1.0, levels + 1)[1:-1]
    if not quantile:
        return np.tile(probs, (X_train.shape[1], 1))  # equal width on [0, 1]
    edges = np.full((X_train.shape[1], levels - 1), np.inf)
    for j in range(X_train.shape[1]):
        col = X_train[:, j]
        u = np.unique(col)
        if len(u) <= levels:
            e = (u[:-1] + u[1:]) / 2.0          # split between distinct values
        else:
            e = np.unique(np.quantile(col, probs))
            e = e[e > u[0]]                     # an edge at the minimum would be useless
        edges[j, :len(e)] = e
    return edges


def local_state(encoding: str, x: float) -> np.ndarray:
    theta = math.pi * float(np.clip(x, 0.0, 1.0))
    if encoding == "angle":
        return np.array([math.cos(theta / 2), math.sin(theta / 2)], dtype=complex)
    if encoding == "phase":
        return np.array([1.0, np.exp(1j * theta)], dtype=complex) / math.sqrt(2.0)
    raise ValueError(encoding)


def encode_state(encoding: str, x: np.ndarray, edges: Optional[np.ndarray] = None) -> np.ndarray:
    """Encode one row as a statevector with a consistent tensor ordering."""
    x = np.asarray(x, dtype=float)

    if encoding in ("amplitude", "amplitude_centered"):
        v = np.clip(x, 0.0, None) if encoding == "amplitude" else x - 0.5
        n = 1 << int(math.ceil(math.log2(max(2, len(v)))))
        vec = np.zeros(n, dtype=complex)
        vec[:len(v)] = v
        norm = np.linalg.norm(vec)
        if norm < 1e-14:
            vec[0] = 1.0
        else:
            vec /= norm
        return vec

    pb = parse_basis(encoding)
    if pb is not None:
        bits, _ = pb
        if bits * len(x) > MAX_DENSE_QUBITS:
            raise SystemExit(
                f"{encoding} with {len(x)} features needs {bits * len(x)} qubits per "
                f"register; the dense simulation limit is {MAX_DENSE_QUBITS}. "
                "Lower --max-features or the number of bits."
            )
        levels = (x[:, None] >= edges).sum(axis=1)
        idx = 0
        for lv in levels:
            idx = (idx << bits) | int(lv)
        vec = np.zeros(1 << (bits * len(x)), dtype=complex)
        vec[idx] = 1.0
        return vec

    state = np.array([1.0 + 0j])
    for value in x:
        state = np.kron(state, local_state(encoding, float(value)))
    return state / np.linalg.norm(state)


# --------------------------------------------------------------------------
# SWAP test
# --------------------------------------------------------------------------
def basis_index(state: np.ndarray) -> Optional[int]:
    """If the state is a computational-basis state, return its index."""
    i = int(np.argmax(np.abs(state)))
    return i if abs(abs(state[i]) - 1.0) < 1e-12 else None


def prepare_register(qc: QuantumCircuit, state: np.ndarray, qubits: List[int]) -> None:
    idx = basis_index(state)
    if idx is not None:
        for j, q in enumerate(qubits):          # Qiskit is little-endian
            if (idx >> j) & 1:
                qc.x(q)
    else:
        qc.append(StatePreparation(state), qubits)


def build_swap_test(state_a: np.ndarray, state_b: np.ndarray) -> QuantumCircuit:
    if len(state_a) != len(state_b):
        raise ValueError("State dimensions do not match")
    dim = len(state_a)
    if dim < 2 or (dim & (dim - 1)) != 0:
        raise ValueError("State dimension must be a power of two >= 2")
    n = int(round(math.log2(dim)))
    qc = QuantumCircuit(1 + 2 * n, 1)
    anc = 0
    qa = list(range(1, 1 + n))
    qb = list(range(1 + n, 1 + 2 * n))
    qc.h(anc)
    prepare_register(qc, state_a, qa)
    prepare_register(qc, state_b, qb)
    for i in range(n):
        qc.cswap(anc, qa[i], qb[i])
    qc.h(anc)
    qc.measure(anc, 0)
    return qc


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def counts_to_estimate(counts: Dict[str, int]) -> Tuple[float, int, int]:
    zeros, ones = int(counts.get("0", 0)), int(counts.get("1", 0))
    total = zeros + ones
    if total == 0:
        raise RuntimeError(f"No measurement counts returned: {counts}")
    return 2.0 * zeros / total - 1.0, zeros, ones


def run_swap_estimate(backend, state_a, state_b, shots, seed, optimization_level=0):
    qc = build_swap_test(state_a, state_b)
    tqc = transpile(qc, backend, optimization_level=optimization_level, seed_transpiler=seed)
    result = backend.run(tqc, shots=shots, seed_simulator=seed).result()
    return counts_to_estimate(result.get_counts(0))


def estimate_with_circuits(backend, compiled_by_query, shots: int, seed_parts: tuple) -> np.ndarray:
    """Run real Aer SWAP-test circuits for every query/train pair."""
    n_te, n_tr = len(compiled_by_query), len(compiled_by_query[0])
    est = np.empty((n_te, n_tr), dtype=float)
    for qi, circuits in enumerate(compiled_by_query):
        result = backend.run(circuits, shots=shots,
                             seed_simulator=stable_seed(*seed_parts, qi)).result()
        for ti in range(n_tr):
            est[qi, ti], _, _ = counts_to_estimate(result.get_counts(ti))
    return est


def estimate_binomial(exact: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Same estimator, sampled from the ideal SWAP-test outcome distribution."""
    p0 = np.clip((1.0 + exact) / 2.0, 0.0, 1.0)
    zeros = rng.binomial(shots, p0)
    return 2.0 * zeros / shots - 1.0


# --------------------------------------------------------------------------
# k-NN with explicit tie handling
# --------------------------------------------------------------------------
def exact_k_indices(scores: np.ndarray, k: int, tie_order: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """Select exactly k highest scores; boundary ties broken by tie_order."""
    scores = np.asarray(scores, dtype=float)
    k_eff = min(int(k), len(scores))
    order = np.lexsort((tie_order, -scores))
    chosen = order[:k_eff]
    kth = scores[chosen[-1]]
    boundary = int(np.sum(np.isclose(scores, kth, rtol=0.0, atol=1e-12)))
    return chosen, boundary, float(kth)


def predict_from_neighbours(similarities, y_train, k, tie_order):
    idx, boundary_count, kth_score = exact_k_indices(similarities, k, tie_order)
    labels, counts = np.unique(y_train[idx], return_counts=True)
    pred = labels[np.flatnonzero(counts == counts.max())[0]]  # smallest label on a vote tie
    return pred, idx, boundary_count, kth_score


def evaluate_tie_sensitivity(score_matrix, y_train, y_test, k, base_seed, repeats) -> Dict[str, float]:
    """Repeat exact-k selection across independent tie-break seeds."""
    predictions, neighbour_sets = [], []
    for rep in range(repeats):
        rep_pred, rep_sets = [], []
        for qi, scores in enumerate(score_matrix):
            tie_order = np.random.default_rng(
                stable_seed(base_seed, "tie_sensitivity", rep, qi)
            ).permutation(len(y_train))
            pred, idx, _, _ = predict_from_neighbours(scores, y_train, k, tie_order)
            rep_pred.append(pred)
            rep_sets.append(set(idx.tolist()))
        predictions.append(np.asarray(rep_pred))
        neighbour_sets.append(rep_sets)

    pa = np.asarray(predictions)
    acc = np.asarray([accuracy_score(y_test, pa[r]) for r in range(repeats)])

    disagreement = []
    for qi in range(len(y_test)):
        _, counts = np.unique(pa[:, qi], return_counts=True)
        disagreement.append(1.0 - float(counts.max()) / repeats)

    jaccards = []
    for qi in range(len(y_test)):
        for a in range(repeats):
            for b in range(a + 1, repeats):
                sa, sb = neighbour_sets[a][qi], neighbour_sets[b][qi]
                jaccards.append(len(sa & sb) / max(1, len(sa | sb)))

    return {
        "accuracy_tie_averaged": float(np.mean(acc)),
        "accuracy_tie_std": float(np.std(acc, ddof=1)) if repeats > 1 else 0.0,
        "accuracy_tie_min": float(np.min(acc)),
        "accuracy_tie_max": float(np.max(acc)),
        "mean_prediction_disagreement": float(np.mean(disagreement)),
        "mean_neighbour_jaccard_across_tie_seeds": float(np.mean(jaccards)) if jaccards else 1.0,
    }


def evaluate_scores(
    scores: np.ndarray, y_train, y_test, k: int, tie_orders: np.ndarray,
    tie_seed: int, tie_repeats: int,
    classical_preds: Optional[np.ndarray] = None,
    exact_ref: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
):
    """All metrics for one similarity matrix and one k."""
    preds, idxs, boundaries, kths = [], [], [], []
    for qi, row in enumerate(scores):
        pred, idx, boundary, kth = predict_from_neighbours(row, y_train, k, tie_orders[qi])
        preds.append(pred)
        idxs.append(idx)
        boundaries.append(boundary)
        kths.append(kth)
    preds = np.asarray(preds)
    out: Dict[str, Any] = {
        "accuracy_single_tiebreak": accuracy_score(y_test, preds),
        "balanced_accuracy": balanced_accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro", zero_division=0),
        "mean_boundary_tie_count": float(np.mean(boundaries)),
        "fraction_queries_with_boundary_tie": float(np.mean(np.asarray(boundaries) > 1)),
    }
    out.update(evaluate_tie_sensitivity(scores, y_train, y_test, k, tie_seed, tie_repeats))
    if classical_preds is not None:
        out["prediction_agreement_with_classical"] = float(np.mean(preds == classical_preds))
    if exact_ref is not None:
        ex_preds, ex_matrix, ex_kths = exact_ref
        out["prediction_agreement_with_exact_encoding"] = float(np.mean(preds == ex_preds))
        out["neighbour_precision_vs_exact"] = float(np.mean([
            np.mean(ex_matrix[qi, idxs[qi]] >= ex_kths[qi] - 1e-12) for qi in range(len(idxs))
        ]))
    return out, preds, np.asarray(kths)


# --------------------------------------------------------------------------
# Circuit validation (known fidelities)
# --------------------------------------------------------------------------
def run_circuit_validation(backend, shot_levels, repeats, seed) -> pd.DataFrame:
    z = np.array([1.0, 0.0], dtype=complex)
    orth = np.array([0.0, 1.0], dtype=complex)
    target_f = 0.37
    nontrivial = np.array([math.sqrt(target_f), math.sqrt(1 - target_f)], dtype=complex)
    cases = [("identical_F1", z, z, 1.0), ("orthogonal_F0", z, orth, 0.0),
             ("nontrivial_F037", z, nontrivial, target_f)]
    rows = []
    for case_name, a, b, expected in cases:
        for shots in shot_levels:
            for rep in range(repeats):
                s = stable_seed(seed, "validation", case_name, shots, rep)
                est, zeros, ones = run_swap_estimate(backend, a, b, shots, s)
                rows.append({
                    "case": case_name, "expected_fidelity": expected, "shots": shots,
                    "repeat": rep, "estimate_raw": est,
                    "estimate_clipped_diagnostic": float(np.clip(est, 0, 1)),
                    "zeros": zeros, "ones": ones, "absolute_error": abs(est - expected),
                    "theoretical_std": math.sqrt(max(0.0, 1 - expected ** 2) / shots),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def ci_half(values) -> float:
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    return float(student_t.ppf(0.975, n - 1) * np.std(x, ddof=1) / math.sqrt(n))


def agg_mean_ci(df: pd.DataFrame, keys: List[str], cols: List[str],
                ci_cols: Optional[List[str]] = None) -> pd.DataFrame:
    g = df.groupby(keys, dropna=False)
    out = g[cols].mean().add_suffix("_mean")
    ci_cols = cols if ci_cols is None else ci_cols
    if ci_cols:
        out = out.join(g[ci_cols].agg(ci_half).add_suffix("_split_t_interval95_halfwidth"))
    out["n_splits"] = g["split"].nunique()
    return out.reset_index()


def first_shots_reaching(group: pd.DataFrame, col: str, thr: float, within: bool = False) -> float:
    """First shot level where the criterion holds at that AND all higher levels."""
    g = group.sort_values("shots")
    passed = (g[col].abs() <= thr if within else g[col] >= thr).to_numpy()
    shots = g["shots"].to_numpy()
    for i in range(len(g)):
        if passed[i:].all():
            return float(shots[i])
    return float("nan")


def build_summaries(results: pd.DataFrame, args) -> Dict[str, pd.DataFrame]:
    results = results.copy()
    results["estimator_family"] = np.where(
        results["estimator"].str.startswith("swap_test"), "swap_test", "exact")

    # --- exact / representation quality, paired against classical -----------
    ex = results[results["estimator_family"] == "exact"].copy()
    classical = ex[ex["encoding"] == "classical_euclidean"][
        ["dataset", "split", "k", "accuracy_tie_averaged"]
    ].rename(columns={"accuracy_tie_averaged": "classical_accuracy"})
    ex = ex.merge(classical, on=["dataset", "split", "k"], how="left")
    ex["accuracy_diff_vs_classical"] = ex["accuracy_tie_averaged"] - ex["classical_accuracy"]
    ex.loc[ex["encoding"] == "classical_euclidean", "accuracy_diff_vs_classical"] = np.nan
    summary_exact = agg_mean_ci(
        ex, ["dataset", "encoding", "k"],
        ["accuracy_tie_averaged", "accuracy_diff_vs_classical"],
    )
    extra = ex.groupby(["dataset", "encoding", "k"], dropna=False)[[
        "accuracy_tie_min", "accuracy_tie_max", "mean_prediction_disagreement",
        "fraction_queries_with_boundary_tie", "mean_boundary_tie_count",
    ]].mean().add_suffix("_mean").reset_index()
    summary_exact = summary_exact.merge(extra, on=["dataset", "encoding", "k"])

    # --- shot behaviour -----------------------------------------------------
    sh = results[results["estimator_family"] == "swap_test"].copy()
    sh["is_circuit"] = (sh["estimator"] == "swap_test_circuit").astype(float)
    exact_acc = ex[["dataset", "encoding", "split", "k", "accuracy_tie_averaged"]].rename(
        columns={"accuracy_tie_averaged": "exact_accuracy_tie_averaged"})
    sh = sh.merge(exact_acc, on=["dataset", "encoding", "split", "k"], how="left")
    sh["accuracy_gap_to_exact"] = sh["accuracy_tie_averaged"] - sh["exact_accuracy_tie_averaged"]
    summary_shots = agg_mean_ci(
        sh, ["dataset", "encoding", "k", "shots"],
        ["accuracy_tie_averaged", "accuracy_gap_to_exact",
         "neighbour_precision_vs_exact", "prediction_agreement_with_exact_encoding"],
    )
    extra = sh.groupby(["dataset", "encoding", "k", "shots"], dropna=False)[[
        "fraction_queries_with_boundary_tie", "mean_prediction_disagreement", "is_circuit",
    ]].mean().rename(columns={"is_circuit": "share_of_splits_using_real_circuits"}) \
        .add_suffix("_mean").reset_index()
    summary_shots = summary_shots.merge(extra, on=["dataset", "encoding", "k", "shots"])

    # --- shot efficiency ----------------------------------------------------
    eff_rows = []
    thr = args.efficiency_threshold
    tol = args.accuracy_tolerance
    tie_frac = summary_exact.set_index(["dataset", "encoding", "k"])[
        "fraction_queries_with_boundary_tie_mean"].to_dict()
    for (ds, enc, k), g in summary_shots.groupby(["dataset", "encoding", "k"]):
        g = g.sort_values("shots")
        last = g.iloc[-1]
        many_ties = tie_frac.get((ds, enc, k), 0.0) > 0.5
        eff_rows.append({
            "dataset": ds, "encoding": enc, "k": k, "threshold": thr,
            "accuracy_tolerance": tol,
            "shots_for_accuracy_within_tolerance": first_shots_reaching(
                g, "accuracy_gap_to_exact_mean", tol, within=True),
            "prediction_agreement_caveat": (
                "unreliable: many exact ties, agreement compares to one arbitrary tie-break"
                if many_ties else "ok"),
            "shots_for_neighbour_precision": first_shots_reaching(
                g, "neighbour_precision_vs_exact_mean", thr),
            "shots_for_prediction_agreement": first_shots_reaching(
                g, "prediction_agreement_with_exact_encoding_mean", thr),
            "max_shots_tested": int(last["shots"]),
            "neighbour_precision_at_max_shots": last["neighbour_precision_vs_exact_mean"],
            "prediction_agreement_at_max_shots": last["prediction_agreement_with_exact_encoding_mean"],
            "note": "NaN = criterion not met at every level from that point up to the max tested",
        })
    shot_efficiency = pd.DataFrame(eff_rows)

    # --- tie sensitivity ----------------------------------------------------
    tie_sens = results.groupby(["dataset", "encoding", "estimator_family", "shots", "k"],
                               dropna=False)[[
        "accuracy_tie_averaged", "accuracy_tie_min", "accuracy_tie_max", "accuracy_tie_std",
        "mean_prediction_disagreement", "mean_neighbour_jaccard_across_tie_seeds",
        "fraction_queries_with_boundary_tie", "mean_boundary_tie_count",
    ]].mean().add_suffix("_mean_over_splits").reset_index()

    return {
        "summary_exact": summary_exact,
        "summary_shots": summary_shots,
        "shot_efficiency": shot_efficiency,
        "tie_sensitivity": tie_sens,
    }


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
def run_experiment(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    backend = AerSimulator()
    try:
        import qiskit
        import qiskit_aer
        versions = f"qiskit {qiskit.__version__}, qiskit-aer {qiskit_aer.__version__}"
    except Exception:
        versions = "unknown"

    config = pd.DataFrame([{
        "datasets": ",".join(args.datasets), "encodings": ",".join(args.encodings),
        "k_values": ",".join(map(str, args.k_values)),
        "shot_levels": ",".join(map(str, args.shot_levels)),
        "n_splits": args.n_splits, "circuit_splits": args.circuit_splits,
        "max_circuit_qubits_per_register": args.max_circuit_qubits,
        "max_train": args.max_train, "max_test": args.max_test,
        "max_features": args.max_features, "seed": args.seed,
        "validation_repeats": args.validation_repeats, "tie_repeats": args.tie_repeats,
        "efficiency_threshold": args.efficiency_threshold,
        "accuracy_tolerance": args.accuracy_tolerance,
        "interval_interpretation": "95% split-level t-intervals describe variability across repeated, overlapping partitions; not formal independent-sample confidence intervals",
        "shot_efficiency_rule": "first tested shot level meeting the criterion, with all higher tested levels also meeting it",
        "tie_policy": "exact_k_seeded_random_boundary_tie_break; headline = mean over tie seeds",
        "simulator": "Qiskit AerSimulator", "hardware": False, "versions": versions,
    }])
    method = pd.DataFrame([
        ["angle", "Per-feature Ry(pi*x)|0>", "Tensor product of one-qubit angle states"],
        ["phase", "Per-feature Rz(pi*x)|+>", "Same fidelity geometry as angle (not independent)"],
        ["amplitude", "MinMax(x), zero-pad, L2 normalise", "log2(n) qubits; fidelity = squared cosine similarity"],
        ["amplitude_centered", "(MinMax(x) - 0.5), pad, L2 normalise", "Optional variant with signed amplitudes"],
        ["basis", "x >= 0.5 -> |1>, else |0>", "1 bit/feature; fidelity is 1 for identical bit strings, else 0"],
        ["basis_Nbit", "N bits per feature, equal-width bins", "More bits: fewer collisions, fewer exact matches"],
        ["suffix _q", "Quantile bins fitted on training data", "Balanced bins; reduces collisions on skewed features"],
        ["SWAP test", "Ancilla H, controlled-SWAPs, H, measure", "F_hat = 2*P_hat(0) - 1; raw estimate not clipped"],
        ["swap_test_circuit", "Real Aer circuits", "Used for the first circuit_splits splits (and small registers)"],
        ["swap_test_binomial", "Counts drawn from Binomial(shots, (1+F)/2)", "Statistically equivalent for ideal simulation; fast"],
        ["tie policy", "Exactly k selected; seeded random boundary tie-break", "Headline accuracy is averaged over tie seeds"],
        ["neighbour_precision_vs_exact", "Share of chosen neighbours with exact F >= exact k-th F", "Tie-robust neighbour recovery"],
        ["splits", "Repeated stratified random splits", "Mean +/- 95% t-interval across splits; paired vs classical"],
        ["preprocessing", "MinMax and feature cap fitted on train only", "Cap uses variance after scaling"],
        ["comparison", "Classical Euclidean k-NN", "Same splits, same k, same tie handling"],
    ], columns=["component", "definition", "interpretation"])

    validation = run_circuit_validation(
        backend, args.shot_levels, args.validation_repeats, args.seed)
    val_summary = validation.groupby(["case", "expected_fidelity", "shots"]).agg(
        mean_abs_error=("absolute_error", "mean"),
        theoretical_std=("theoretical_std", "first"),
    ).reset_index()
    val_summary["expected_mean_abs_error_gaussian_approx"] = 0.7979 * val_summary["theoretical_std"]

    results, pair_rows, resource_rows, dataset_rows = [], [], [], []
    equivalence_rows, cross_rows = [], []

    for dataset_name in args.datasets:
        print(f"\nDataset: {dataset_name}", flush=True)
        ds = load_dataset(dataset_name, args.seed)

        for split_idx in range(args.n_splits):
            split_seed = stable_seed(args.seed, dataset_name, "split", split_idx) % MASK31
            Xtr, Xte, ytr, yte, feature_names = dataset_split(
                ds, args.max_train, args.max_test, args.max_features, split_seed)
            n_tr, n_te = len(Xtr), len(Xte)
            dataset_rows.append({
                "dataset": dataset_name, "split": split_idx, "split_seed": split_seed,
                "n_total": len(ds.X), "n_train": n_tr, "n_test": n_te,
                "n_features_used": Xtr.shape[1], "n_classes": len(np.unique(ytr)),
                "feature_names": ", ".join(feature_names),
            })
            print(f"  split {split_idx + 1}/{args.n_splits}  (train {n_tr}, test {n_te})", flush=True)

            tie_orders = np.stack([
                np.random.default_rng(stable_seed(args.seed, dataset_name, split_idx, "tie", qi)
                                      ).permutation(n_tr)
                for qi in range(n_te)
            ])

            def make_row(enc, estimator, shots, k, out, seconds=np.nan):
                return {"dataset": dataset_name, "split": split_idx, "split_seed": split_seed,
                        "encoding": enc, "estimator": estimator, "shots": shots, "k": k,
                        "seconds": seconds, **out}

            # Classical reference
            csim = -pairwise_distances(Xte, Xtr, metric="euclidean")
            classical_preds: Dict[int, np.ndarray] = {}
            for k in args.k_values:
                out, preds, _ = evaluate_scores(
                    csim, ytr, yte, k, tie_orders,
                    stable_seed(args.seed, dataset_name, split_idx, "classical"),
                    args.tie_repeats)
                classical_preds[k] = preds
                results.append(make_row("classical_euclidean", "classical_exact", 0, k, out))

            exact_cache: Dict[str, np.ndarray] = {}
            for enc in args.encodings:
                pb = parse_basis(enc)
                edges = fit_basis_edges(Xtr, pb[0], pb[1]) if pb else None
                Str = np.vstack([encode_state(enc, x, edges) for x in Xtr])
                Ste = np.vstack([encode_state(enc, x, edges) for x in Xte])
                n_qubits = int(round(math.log2(Str.shape[1])))
                exact = np.clip(np.abs(np.conj(Ste) @ Str.T) ** 2, 0.0, 1.0)
                exact_cache[enc] = exact

                if split_idx == 0:
                    tqc = transpile(build_swap_test(Ste[0], Str[0]), backend,
                                    optimization_level=0, seed_transpiler=args.seed)
                    ops = tqc.count_ops()
                    resource_rows.append({
                        "dataset": dataset_name, "encoding": enc,
                        "state_dimension": Str.shape[1],
                        "state_qubits_per_register": n_qubits,
                        "swap_test_total_qubits": tqc.num_qubits,
                        "transpiled_depth": tqc.depth(),
                        "transpiled_gate_count": sum(ops.values()),
                        "transpiled_operations": str(dict(ops)),
                        "resource_note": "Representative pair circuit; simulator counts, not hardware cost",
                    })

                # Pairwise / collision diagnostics
                srt = np.sort(exact, axis=1)
                if parse_basis(enc):
                    unique_states = len({int(np.argmax(np.abs(s))) for s in Str})
                else:
                    unique_states = np.unique(
                        np.round(np.hstack([Str.real, Str.imag]), 9), axis=0).shape[0]
                match = exact > 1 - 1e-9
                pair_rows.append({
                    "dataset": dataset_name, "split": split_idx, "encoding": enc,
                    "qubits_per_register": n_qubits,
                    "mean_pairwise_fidelity": float(np.mean(exact)),
                    "std_pairwise_fidelity": float(np.std(exact)),
                    "mean_query_max_fidelity": float(np.mean(srt[:, -1])),
                    "mean_query_min_fidelity": float(np.mean(srt[:, 0])),
                    "mean_top1_top2_gap": float(np.mean(srt[:, -1] - srt[:, -2])) if n_tr > 1 else np.nan,
                    "unique_train_states": unique_states,
                    "train_collision_rate": 1.0 - unique_states / n_tr,
                    "fraction_queries_with_exact_match": float(np.mean(match.any(axis=1))),
                    "mean_exact_matches_per_query": float(np.mean(match.sum(axis=1))),
                })

                # Exact-statevector results
                exact_ref: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
                for k in args.k_values:
                    out, preds, kths = evaluate_scores(
                        exact, ytr, yte, k, tie_orders,
                        stable_seed(args.seed, dataset_name, split_idx, enc, "exact"),
                        args.tie_repeats, classical_preds=classical_preds[k])
                    out["prediction_agreement_with_exact_encoding"] = 1.0
                    out["neighbour_precision_vs_exact"] = 1.0
                    exact_ref[k] = (preds, exact, kths)
                    results.append(make_row(enc, "exact_statevector", 0, k, out))

                # Finite-shot SWAP-test results
                use_circuit = (split_idx < args.circuit_splits
                               and n_qubits <= args.max_circuit_qubits)
                compiled_by_query = []
                if use_circuit:
                    for qi in range(n_te):
                        circuits = [build_swap_test(Ste[qi], Str[ti]) for ti in range(n_tr)]
                        compiled_by_query.append(transpile(
                            circuits, backend, optimization_level=0,
                            seed_transpiler=stable_seed(args.seed, dataset_name, split_idx, enc, qi)))

                for shots in args.shot_levels:
                    t0 = time.perf_counter()
                    if use_circuit:
                        est = estimate_with_circuits(
                            backend, compiled_by_query, shots,
                            (args.seed, dataset_name, split_idx, enc, shots))
                        label = "swap_test_circuit"
                    else:
                        rng = np.random.default_rng(stable_seed(
                            args.seed, dataset_name, split_idx, enc, shots, "binomial"))
                        est = estimate_binomial(exact, shots, rng)
                        label = "swap_test_binomial"
                    elapsed = time.perf_counter() - t0

                    if use_circuit:
                        rng = np.random.default_rng(stable_seed(
                            args.seed, dataset_name, split_idx, enc, shots, "crosscheck"))
                        bino = estimate_binomial(exact, shots, rng)
                        cross_rows.append({
                            "dataset": dataset_name, "split": split_idx, "encoding": enc,
                            "shots": shots,
                            "mean_abs_error_circuit": float(np.mean(np.abs(est - exact))),
                            "mean_abs_error_binomial": float(np.mean(np.abs(bino - exact))),
                            "mean_theoretical_std": float(np.mean(np.sqrt((1 - exact ** 2) / shots))),
                            "circuit_seconds": elapsed,
                        })

                    for k in args.k_values:
                        out, _, _ = evaluate_scores(
                            est, ytr, yte, k, tie_orders,
                            stable_seed(args.seed, dataset_name, split_idx, enc, label, shots),
                            args.tie_repeats, classical_preds=classical_preds[k],
                            exact_ref=exact_ref[k])
                        results.append(make_row(enc, label, shots, k, out, elapsed))

            if "angle" in exact_cache and "phase" in exact_cache:
                equivalence_rows.append({
                    "dataset": dataset_name, "split": split_idx,
                    "encoding_a": "angle", "encoding_b": "phase",
                    "max_abs_pairwise_fidelity_difference": float(
                        np.max(np.abs(exact_cache["angle"] - exact_cache["phase"]))),
                    "interpretation": "Expected numerical equivalence for these specific maps",
                })

    results_df = pd.DataFrame(results)
    summaries = build_summaries(results_df, args)
    pair_df = pd.DataFrame(pair_rows)
    pair_summary = pair_df.groupby(["dataset", "encoding"]).mean(numeric_only=True) \
        .drop(columns=["split"]).add_suffix("_mean").reset_index()

    return {
        "run_config": config,
        "methodology": method,
        "summary_exact": summaries["summary_exact"],
        "summary_shots": summaries["summary_shots"],
        "shot_efficiency": summaries["shot_efficiency"],
        "tie_sensitivity": summaries["tie_sensitivity"],
        "pairwise_summary": pair_summary,
        "encoding_equivalence": pd.DataFrame(equivalence_rows),
        "circuit_validation": validation,
        "validation_summary": val_summary,
        "estimator_crosscheck": pd.DataFrame(cross_rows),
        "circuit_resources": pd.DataFrame(resource_rows),
        "dataset_metadata": pd.DataFrame(dataset_rows),
        "results_by_split": results_df,
        "pairwise_by_split": pair_df,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Study 07.4: repeated-split circuit-level quantum k-NN")
    p.add_argument("--datasets", nargs="+",
                   default=["iris", "wine", "breast_cancer", "synthetic_binarized"],
                   choices=["iris", "wine", "breast_cancer", "digits",
                            "synthetic_binarized", "synthetic_blobs"])
    p.add_argument("--encodings", nargs="+",
                   default=["angle", "phase", "amplitude", "basis", "basis_q", "basis_2bit", "basis_2bit_q"],
                   help="angle phase amplitude amplitude_centered basis basis_2bit basis_3bit "
                        "(add _q for quantile bins, e.g. basis_2bit_q)")
    p.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5])
    p.add_argument("--shot-levels", nargs="+", type=int, default=[128, 512, 2048, 8192])
    p.add_argument("--n-splits", type=int, default=10, help="Repeated stratified splits (use 20 for the final run)")
    p.add_argument("--circuit-splits", type=int, default=1,
                   help="How many splits use real Aer circuits; the rest use the binomial estimator")
    p.add_argument("--max-circuit-qubits", type=int, default=5,
                   help="Largest register size (qubits) simulated as real circuits")
    p.add_argument("--max-train", type=int, default=100)
    p.add_argument("--max-test", type=int, default=40)
    p.add_argument("--max-features", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-repeats", type=int, default=10)
    p.add_argument("--tie-repeats", type=int, default=50)
    p.add_argument("--efficiency-threshold", type=float, default=0.95,
                   help="Target neighbour precision / prediction agreement for shot_efficiency")
    p.add_argument("--accuracy-tolerance", type=float, default=0.02,
                   help="Allowed accuracy gap to exact (absolute) for shots_for_accuracy_within_tolerance")
    p.add_argument("--quick", action="store_true", help="Tiny smoke-test preset")
    p.add_argument("--outdir", type=str, default="results/quantum_knn_study_07_4_1")
    p.add_argument("--workbook", type=str, default="07_quantum_knn_results_v4_1.xlsx")
    args = p.parse_args()
    if args.quick:
        args.datasets = ["iris"]
        args.n_splits = 2
        args.shot_levels = [128, 512]
        args.tie_repeats = 5
        args.max_train, args.max_test = 40, 15
        args.validation_repeats = 3
    return args


def main() -> None:
    args = parse_args()
    if any(k < 1 for k in args.k_values):
        raise SystemExit("All k values must be >= 1")
    if any(s < 1 for s in args.shot_levels):
        raise SystemExit("All shot levels must be >= 1")
    if args.tie_repeats < 2:
        raise SystemExit("--tie-repeats must be >= 2")
    if args.n_splits < 1:
        raise SystemExit("--n-splits must be >= 1")
    for enc in args.encodings:
        validate_encoding(enc)
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 78)
    print("QKNN STUDY 07.4 (v4.1) - REPEATED SPLITS, SHOT EFFICIENCY, TIE-AWARE METRICS")
    print("=" * 78)
    print(f"Splits: {args.n_splits} (real circuits on first {args.circuit_splits}); "
          f"train<= {args.max_train}, test<= {args.max_test}")
    print("Reminder: Aer is a classical simulator, not quantum hardware.")

    sheets = run_experiment(args)
    output_path = os.path.join(args.outdir, args.workbook)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            name = sheet_name[:31]
            frame.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for column_cells in ws.columns:
                max_len = max((len(str(c.value)) if c.value is not None else 0)
                              for c in column_cells[:200])
                ws.column_dimensions[column_cells[0].column_letter].width = min(max(12, max_len + 2), 48)

    print("\n" + "=" * 78)
    print("COMPLETE")
    print(f"Workbook: {os.path.abspath(output_path)}")
    print("Worksheets: " + ", ".join(sheets.keys()))


if __name__ == "__main__":
    main()