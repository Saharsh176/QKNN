"""
validate_against_authors.py
===========================
Checks that the probability model used in qknn_shots_study.py matches the
authors' implementation (github.com/ZarHenry96/euclidean-quantum-k-nn,
algorithm/qknn.py).

TWO CHECKS
----------
1. CIRCUIT CHECK. The authors' amplitude-building loop is ported line by line
   to NumPy (same indices, same encodings). Their three gates (H, CNOT, H on
   the first two qubits) are then applied to a dense state vector, and the
   resulting probabilities are compared with the closed form
        P(j,0) = (1 + s_j) / (2N)       P(j,1) = (1 - s_j) / (2N)
   used by the shot study. No Qiskit is needed.

2. LOG CHECK. Uses the authors' qknn_log.txt. It detects whether the log is a finite-shot sample (it is: S = 8332) and tests that
   P(j,0) + P(j,1) = 1/N and that the printed distances follow from the
   authors' distance formula, which also reveals which encoding the log used.

RUN
---
    python validate_against_authors.py --log qknn_log.txt
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Dense state-vector version of the authors' circuit
# --------------------------------------------------------------------------- #
def authors_amplitudes(train, target, encoding):
    """Port of the amplitude-building code in build_qknn_circuit (algorithm/qknn.py).
    train: (N, d) normalised training data; target: (d,) normalised target."""
    N, d = train.shape
    index_qubits_num = math.ceil(math.log2(N))
    features_qubits_num = math.ceil(math.log2(2 * d + (3 if encoding == "extension" else 4)))
    init_qubits_num = 1 + index_qubits_num + features_qubits_num
    amplitudes = np.zeros(2 ** init_qubits_num)
    base = math.sqrt(1 / (2 * N))
    if encoding == "extension":
        mult, off, trans = math.sqrt(4 / 3), 0, 0
    else:
        mult, off, trans = 1, 1, 0.5
    norms = []
    tnorm = np.linalg.norm(target)
    for j in range(N):
        norms.append(np.linalg.norm(train[j]))
        for i in range(2 * d):
            amplitudes[2 * j + (2 ** (index_qubits_num + 1)) * i] = base * mult * train[j][i % d]
        amplitudes[2 * j + (2 ** (index_qubits_num + 1)) * (2 * d)] = base * mult * norms[-1]
        if encoding == "translation":
            amplitudes[2 * j + (2 ** (index_qubits_num + 1)) * (2 * d + off)] = base * trans
        amplitudes[2 * j + (2 ** (index_qubits_num + 1)) * (2 * d + off + 1)] = 0
        amplitudes[2 * j + (2 ** (index_qubits_num + 1)) * (2 * d + off + 2)] = base * math.sqrt(
            1 - (3 * mult ** 2 * norms[-1] ** 2 + trans ** 2))
    for j in range(N):
        for i in range(2 * d):
            amplitudes[1 + 2 * j + (2 ** (index_qubits_num + 1)) * i] = base * mult * (-target[i % d])
        amplitudes[1 + 2 * j + (2 ** (index_qubits_num + 1)) * (2 * d)] = base * mult * norms[j]
        if encoding == "translation":
            amplitudes[1 + 2 * j + (2 ** (index_qubits_num + 1)) * (2 * d + off)] = base * (-trans)
        amplitudes[1 + 2 * j + (2 ** (index_qubits_num + 1)) * (2 * d + off + 1)] = base * math.sqrt(
            1 - (2 * mult ** 2 * tnorm ** 2 + mult ** 2 * norms[j] ** 2 + trans ** 2))
        amplitudes[1 + 2 * j + (2 ** (index_qubits_num + 1)) * (2 * d + off + 2)] = 0
    return amplitudes, index_qubits_num, init_qubits_num


def apply_1q(state, gate, qubit, n):
    """Apply a 2x2 gate to `qubit` (little-endian: qubit 0 is the lowest bit)."""
    st = state.reshape([2] * n)  # axis k corresponds to qubit n-1-k
    axis = n - 1 - qubit
    st = np.moveaxis(np.tensordot(gate, np.moveaxis(st, axis, 0), axes=(1, 0)), 0, axis)
    return st.reshape(-1)


def apply_cnot(state, control, target, n):
    st = state.reshape([2] * n).copy()
    ca, ta = n - 1 - control, n - 1 - target
    idx = [slice(None)] * n
    idx[ca] = 1
    sub = st[tuple(idx)]
    # after fixing the control axis, target axis index shifts if it comes after
    t_axis = ta - 1 if ta > ca else ta
    st[tuple(idx)] = np.flip(sub, axis=t_axis)
    return st.reshape(-1)


def circuit_probabilities(train, target, encoding):
    amps, iq, init_q = authors_amplitudes(train, target, encoding)
    assert np.isclose((amps ** 2).sum(), 1.0), "amplitudes not normalised"
    n = 1 + init_q  # qubit 0 = ancilla, qubits 1.. = initialised register
    state = np.zeros(2 ** n)
    state[0::2] = amps  # ancilla (lowest bit) starts in |0>
    H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    state = apply_1q(state, H, 0, n)
    state = apply_cnot(state, 0, 1, n)
    state = apply_1q(state, H, 0, n)
    prob = state ** 2
    N = len(train)
    P = np.zeros((N, 2))
    # full index = ancilla + 2 * (init index); init index = bit0 (qubit 1) + 2*j + ...
    for full in range(len(prob)):
        anc = full & 1
        init = full >> 1
        j = (init >> 1) & ((1 << iq) - 1)
        if j < N:
            P[j, anc] += prob[full]
    return P


def closed_form(train, target, encoding):
    N = len(train)
    xx = (train ** 2).sum(1)
    xt = train @ target
    if encoding == "extension":
        s = (4 / 3) * (xx - 2 * xt)
    else:
        s = (xx - 2 * xt) - 0.25
    return np.stack([(1 + s) / (2 * N), (1 - s) / (2 * N)], axis=1)


def circuit_check():
    rng = np.random.default_rng(1)
    print("CHECK 1: dense state-vector version of the authors' circuit vs closed form")
    worst = 0.0
    for encoding in ("extension", "translation"):
        for N, d in [(3, 2), (5, 3), (7, 4)]:
            lim = 1 / (2 * math.sqrt(d))
            train = rng.uniform(-lim, lim, size=(N, d))
            target = rng.uniform(-lim, lim, size=d)
            P_circ = circuit_probabilities(train, target, encoding)
            P_form = closed_form(train, target, encoding)
            err = np.abs(P_circ - P_form).max()
            worst = max(worst, err)
            print(f"  {encoding:11s} N={N} d={d}  max |difference| = {err:.2e}")
    assert worst < 1e-12, "closed form does not match the circuit"
    print("  PASSED: P(j,0) = (1+s)/(2N), P(j,1) = (1-s)/(2N) with")
    print("          s = (4/3)(|x_j|^2 - 2 x_j.x_t)        (extension)")
    print("          s = |x_j|^2 - 2 x_j.x_t - 1/4         (translation)\n")


# --------------------------------------------------------------------------- #
# 2. Check against the authors' log file
# --------------------------------------------------------------------------- #
def infer_shots(P, max_shots=5_000_000, tol=1e-4):
    """If the log holds finite-shot frequencies, every probability is count/S for an integer S.
    Returns the smallest such S (or None if the log looks like exact probabilities)."""
    chunk = 200_000
    flat = P.ravel()
    for start in range(100, max_shots, chunk):
        S = np.arange(start, min(start + chunk, max_shots))[:, None]
        err = np.abs(flat[None, :] * S - np.round(flat[None, :] * S)).max(1)
        hit = np.where(err < tol)[0]
        if len(hit):
            return int(S[hit[0], 0])
    return None


def log_check(path):
    text = open(path).read()
    pj = {}
    for m in re.finditer(r"P\((\d+),0\) = ([\d.]+)\s+P\(\d+,1\) = ([\d.]+)", text):
        pj[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    dist = {}
    for m in re.finditer(r"index state (\d+) \(binary: \d+\)\s+'zero' estimate = ([\d.]+)", text):
        dist[int(m.group(1))] = float(m.group(2))
    real = [j for j in pj if j in dist]
    N = len(real)
    P = np.array([pj[j] for j in real])
    print(f"CHECK 2: authors' log ({path}), {N} real index states")

    
    # 2a. Normalisation and index-state uniformity
    total = P.sum()
    sums = P.sum(1)

    print(f"  total probability over real index states = {total:.10f} (expected 1)")
    assert abs(total - 1) < 1e-6, "total probability does not sum to 1"

    S = infer_shots(P)

    if S is None:
        print("  Could not identify a common integer shot denominator.")
        print("  This does NOT establish that the probabilities are exact.")
        print("  Skipping the finite-shot uniformity test.")
    else:
        counts = np.rint(P * S).astype(int)

        # Check whether rounding the displayed probabilities produces
        # a complete set of S measurement counts.
        count_total = int(counts.sum())
        assert count_total == S, (
            f"Rounded counts sum to {count_total}, not {S}"
        )

        nj = counts.sum(axis=1)
        expected = S / N
        shot_std = math.sqrt((1 / N) * (1 - 1 / N) / S)
        z = (nj - expected) / math.sqrt(
            S * (1 / N) * (1 - 1 / N)
        )

        chi2 = float(((nj - expected) ** 2 / expected).sum())

        msg = f"chi-square = {chi2:.2f} (df = {N - 1})"
        try:
            from scipy.stats import chi2 as chi2_distribution
            p_value = 1 - chi2_distribution.cdf(chi2, N - 1)
            msg += f", p = {p_value:.3f}"
        except ImportError:
            pass

        print(f"  inferred finite-shot sample: S = {S}")
        print(
            f"  P(j,0)+P(j,1): min {sums.min():.4f}, "
            f"max {sums.max():.4f}, expected 1/N = {1/N:.4f}"
        )
        print(f"  expected shot-noise std = {shot_std:.5f}")
        print(
            f"  index-state counts = {nj.tolist()}, "
            f"expected {expected:.1f} each"
        )
        print(f"  max |z| = {np.abs(z).max():.2f}; {msg}")

        assert np.abs(z).max() < 4.0, (
            "Index-state frequencies are unusually far from uniform"
        )

        print("  -> Consistent with a uniform index superposition.")


    # 2b. distance formula (deterministic given the printed probabilities, so tight tolerance is fine)
    sp = {j: 2 * N * pj[j][0] - 1 for j in real}
    j0 = min(real, key=lambda j: dist[j])
    verdict = {}
    for encoding, c in (("extension", 0.75), ("translation", 1.0)):
        off = 0.0 if encoding == "extension" else 0.25
        tn2 = dist[j0] ** 2 - (c * sp[j0] + off)
        ok = 0 <= tn2 <= 0.25
        pred = {j: math.sqrt(max(min(c * sp[j] + off + tn2, 1.0), 0.0)) for j in real}
        err = max(abs(pred[j] - dist[j]) for j in real)
        verdict[encoding] = ok and err < 1e-6
        print(f"  hypothesis '{encoding}': implied |target|^2 = {tn2:.6f} "
              f"({'valid' if ok else 'IMPOSSIBLE, must be in [0, 0.25]'}), max distance error = {err:.1e}")
    assert sum(verdict.values()) == 1, "exactly one encoding should reproduce the logged distances"
    enc = [k for k, v in verdict.items() if v][0]
    print(f"  -> PASSED: the log was produced with the '{enc}' encoding and the authors' distance formula.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None,
                    help="authors' log; default: ./qknn_log.txt or <workspace>/results/authors_log/qknn_log.txt")
    args = ap.parse_args()
    circuit_check()
    here = Path(__file__).resolve().parent.parent / "results" / "authors_log" / "qknn_log.txt"
    cands = [Path(args.log)] if args.log else [Path("qknn_log.txt"), here]
    log = next((p for p in cands if p.exists()), None)
    if log is None:
        print("CHECK 2 skipped: no authors' log found. Put it at", here)
    else:
        log_check(str(log))