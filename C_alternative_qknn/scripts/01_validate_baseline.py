"""
01_validate_baseline.py
=======================

Project C — Alternative QKNN
Stage 1: validate the original Zardini et al. QKNN probability model.

Purpose
-------
This script establishes a trusted baseline before alternative encodings or
similarity estimators are introduced.

It performs three checks:

1. Circuit-to-formula validation
   Reconstructs the authors' amplitude-building procedure and applies the
   ancilla H-CNOT-H sequence using a dense NumPy state vector. It compares
   the resulting measurement probabilities against the closed-form model.

2. Dataset/preprocessing sanity checks
   Loads the same dataset families used in the earlier study and checks the
   training-derived normalization used by the authors' simulation pipeline.

3. Probability and ranking sanity checks
   Checks probability normalization, non-negativity, and agreement between
   rankings induced by the exact model's distance estimates and the formula's
   underlying squared-distance ordering.

Scope and limitations
---------------------
- This is a state-vector validation, not a Qiskit hardware execution.
- It does not simulate finite-shot sampling, gate noise, or runtime.
- The dense state vector is only used for small validation cases.
- The authors' translation/extension amplitude construction is reproduced
  from the previously validated `validate_against_authors.py`.
- Dataset checks validate preprocessing and probability-model consistency;
  they do not claim that QKNN is more accurate than classical KNN.

Run from this directory:
    python scripts/01_validate_baseline.py

Optional:
    python scripts/01_validate_baseline.py --datasets blobs wine digits
    python scripts/01_validate_baseline.py --output-dir results/baseline_validation
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.datasets import load_digits, load_iris, load_wine, make_blobs


# ---------------------------------------------------------------------------
# Dataset loading and authors' normalization
# ---------------------------------------------------------------------------

DATASET_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "digits": load_digits,
}


def load_dataset_split(
    name: str,
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic train/test split.

    For the synthetic blobs dataset, generate a fresh deterministic sample.
    For sklearn datasets, use a seeded permutation.
    """
    if name == "blobs":
        X, y = make_blobs(
            n_samples=n_train + n_test,
            centers=3,
            n_features=8,
            cluster_std=2.0,
            random_state=seed,
        )
    elif name in DATASET_LOADERS:
        X, y = DATASET_LOADERS[name](return_X_y=True)
        if n_train + n_test > len(X):
            raise ValueError(
                f"{name}: requested {n_train + n_test} samples, "
                f"but dataset contains only {len(X)}."
            )
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(X))[: n_train + n_test]
        X, y = X[indices], y[indices]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def authors_normalize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Training-derived feature normalization used in the earlier study.

    Z = (X - midpoint) / (range * sqrt(d))

    Test values are clipped to the interval used by the authors'
    state-preparation construction.
    """
    d = X_train.shape[1]
    scale = math.sqrt(d)

    minimum = X_train.min(axis=0)
    maximum = X_train.max(axis=0)
    midpoint = (minimum + maximum) / 2.0
    feature_range = maximum - minimum
    feature_range = feature_range.copy()
    feature_range[feature_range == 0.0] = 1.0

    Z_train = (X_train - midpoint) / (feature_range * scale)
    Z_test = (X_test - midpoint) / (feature_range * scale)
    limit = 1.0 / (2.0 * scale)
    Z_test = np.clip(Z_test, -limit, limit)

    return Z_train, Z_test


# ---------------------------------------------------------------------------
# Authors' amplitude construction and circuit
# ---------------------------------------------------------------------------

def authors_amplitudes(
    train: np.ndarray,
    target: np.ndarray,
    encoding: str,
) -> tuple[np.ndarray, int, int]:
    """Reproduce the authors' amplitude-building loop.

    Parameters
    ----------
    train:
        Normalized training matrix, shape (N, d).
    target:
        Normalized query vector, shape (d,).
    encoding:
        "translation" or "extension".
    """
    if encoding not in {"translation", "extension"}:
        raise ValueError("encoding must be 'translation' or 'extension'")

    N, d = train.shape
    index_qubits = math.ceil(math.log2(N))
    feature_qubits = math.ceil(
        math.log2(2 * d + (3 if encoding == "extension" else 4))
    )
    initial_qubits = 1 + index_qubits + feature_qubits

    amplitudes = np.zeros(2**initial_qubits, dtype=float)
    base = math.sqrt(1.0 / (2.0 * N))

    if encoding == "extension":
        multiplier, offset, translation = math.sqrt(4.0 / 3.0), 0, 0.0
    else:
        multiplier, offset, translation = 1.0, 1, 0.5

    train_norms = np.linalg.norm(train, axis=1)
    target_norm = np.linalg.norm(target)

    # Training-data branch.
    for j in range(N):
        for i in range(2 * d):
            amplitudes[
                2 * j + (2 ** (index_qubits + 1)) * i
            ] = base * multiplier * train[j, i % d]

        amplitudes[
            2 * j + (2 ** (index_qubits + 1)) * (2 * d)
        ] = base * multiplier * train_norms[j]

        if encoding == "translation":
            amplitudes[
                2 * j + (2 ** (index_qubits + 1)) * (2 * d + offset)
            ] = base * translation

        amplitudes[
            2 * j + (2 ** (index_qubits + 1)) * (2 * d + offset + 1)
        ] = 0.0

        residual = 1.0 - (
            3.0 * multiplier**2 * train_norms[j] ** 2 + translation**2
        )
        if residual < -1e-10:
            raise ValueError(
                f"Invalid training amplitude residual {residual:.3e}; "
                "check preprocessing or encoding."
            )
        amplitudes[
            2 * j + (2 ** (index_qubits + 1)) * (2 * d + offset + 2)
        ] = base * math.sqrt(max(0.0, residual))

    # Query-data branch.
    for j in range(N):
        for i in range(2 * d):
            amplitudes[
                1 + 2 * j + (2 ** (index_qubits + 1)) * i
            ] = base * multiplier * (-target[i % d])

        amplitudes[
            1 + 2 * j + (2 ** (index_qubits + 1)) * (2 * d)
        ] = base * multiplier * train_norms[j]

        if encoding == "translation":
            amplitudes[
                1 + 2 * j + (2 ** (index_qubits + 1)) * (2 * d + offset)
            ] = base * (-translation)

        residual = 1.0 - (
            2.0 * multiplier**2 * target_norm**2
            + multiplier**2 * train_norms[j] ** 2
            + translation**2
        )
        if residual < -1e-10:
            raise ValueError(
                f"Invalid query amplitude residual {residual:.3e}; "
                "check preprocessing or encoding."
            )
        amplitudes[
            1 + 2 * j + (2 ** (index_qubits + 1)) * (2 * d + offset + 1)
        ] = base * math.sqrt(max(0.0, residual))

        amplitudes[
            1 + 2 * j + (2 ** (index_qubits + 1)) * (2 * d + offset + 2)
        ] = 0.0

    return amplitudes, index_qubits, initial_qubits


def apply_single_qubit_gate(
    state: np.ndarray,
    gate: np.ndarray,
    qubit: int,
    n_qubits: int,
) -> np.ndarray:
    """Apply a 2x2 gate using little-endian qubit indexing."""
    tensor = state.reshape([2] * n_qubits)
    axis = n_qubits - 1 - qubit
    tensor = np.moveaxis(
        np.tensordot(
            gate,
            np.moveaxis(tensor, axis, 0),
            axes=(1, 0),
        ),
        0,
        axis,
    )
    return tensor.reshape(-1)


def apply_cnot(
    state: np.ndarray,
    control: int,
    target: int,
    n_qubits: int,
) -> np.ndarray:
    """Apply CNOT using little-endian qubit indexing."""
    tensor = state.reshape([2] * n_qubits).copy()
    control_axis = n_qubits - 1 - control
    target_axis = n_qubits - 1 - target

    index = [slice(None)] * n_qubits
    index[control_axis] = 1
    sub_tensor = tensor[tuple(index)]

    target_axis_after_fixing = (
        target_axis - 1 if target_axis > control_axis else target_axis
    )
    tensor[tuple(index)] = np.flip(
        sub_tensor,
        axis=target_axis_after_fixing,
    )
    return tensor.reshape(-1)


def circuit_probabilities(
    train: np.ndarray,
    target: np.ndarray,
    encoding: str,
) -> np.ndarray:
    """Simulate the authors' H-CNOT-H ancilla circuit.

    Returns P with shape (N, 2), where columns correspond to ancilla outcomes
    0 and 1.
    """
    amplitudes, index_qubits, initial_qubits = authors_amplitudes(
        train, target, encoding
    )

    norm = float(np.sum(amplitudes**2))
    if not np.isclose(norm, 1.0, atol=1e-10):
        raise AssertionError(f"Initial state norm is {norm:.12f}, not 1.")

    n_qubits = 1 + initial_qubits
    state = np.zeros(2**n_qubits, dtype=float)
    state[0::2] = amplitudes

    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)
    state = apply_single_qubit_gate(state, hadamard, 0, n_qubits)
    state = apply_cnot(state, 0, 1, n_qubits)
    state = apply_single_qubit_gate(state, hadamard, 0, n_qubits)

    probabilities = state**2
    N = len(train)
    result = np.zeros((N, 2), dtype=float)

    for full_index, probability in enumerate(probabilities):
        ancilla = full_index & 1
        initial_index = full_index >> 1
        training_index = (
            initial_index >> 1
        ) & ((1 << index_qubits) - 1)

        if training_index < N:
            result[training_index, ancilla] += probability

    return result


def closed_form_probabilities(
    train: np.ndarray,
    target: np.ndarray,
    encoding: str,
) -> np.ndarray:
    """Closed-form outcome probabilities from the authors' model."""
    N = len(train)
    train_norm_squared = np.sum(train**2, axis=1)
    train_query_dot = train @ target

    base = train_norm_squared - 2.0 * train_query_dot

    if encoding == "extension":
        signal = (4.0 / 3.0) * base
    elif encoding == "translation":
        signal = base - 0.25
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    return np.column_stack(
        ((1.0 + signal) / (2.0 * N), (1.0 - signal) / (2.0 * N))
    )


# ---------------------------------------------------------------------------
# Derived quantities and validation routines
# ---------------------------------------------------------------------------

def exact_squared_distances(
    train: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Classical squared Euclidean distances in normalized feature space."""
    return np.sum((train - target) ** 2, axis=1)


def distance_estimates_from_probabilities(
    probabilities: np.ndarray,
    target: np.ndarray,
    encoding: str,
) -> np.ndarray:
    """Recover the authors' distance estimate from ideal P(j,0).

    This is the zero-estimator formula used in the earlier simulation.
    """
    N = probabilities.shape[0]
    signal = 2.0 * N * probabilities[:, 0] - 1.0
    target_norm_squared = float(target @ target)

    if encoding == "extension":
        distance_squared = 0.75 * signal + target_norm_squared
    elif encoding == "translation":
        distance_squared = signal + 0.25 + target_norm_squared
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    return np.sqrt(np.clip(distance_squared, 0.0, None))


def ranking_agreement(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Jaccard overlap of the two top-k index sets."""
    top_a = set(np.argsort(a)[:k].tolist())
    top_b = set(np.argsort(b)[:k].tolist())
    union = top_a | top_b
    return len(top_a & top_b) / len(union) if union else 1.0


def validate_random_circuits(
    seeds: Iterable[int],
    cases: Iterable[tuple[int, int]],
    encodings: Iterable[str],
) -> list[dict]:
    """Compare dense circuit probabilities with closed-form probabilities."""
    rows = []

    for encoding in encodings:
        for seed in seeds:
            rng = np.random.default_rng(seed)

            for N, d in cases:
                # The authors' normalized coordinates lie in this range.
                limit = 1.0 / (2.0 * math.sqrt(d))
                train = rng.uniform(-limit, limit, size=(N, d))
                target = rng.uniform(-limit, limit, size=d)

                circuit_p = circuit_probabilities(train, target, encoding)
                formula_p = closed_form_probabilities(train, target, encoding)

                max_error = float(np.max(np.abs(circuit_p - formula_p)))
                total_probability_error = float(
                    np.max(np.abs(circuit_p.sum(axis=1) - 1.0 / N))
                )
                minimum_probability = float(np.min(circuit_p))

                rows.append({
                    "check": "circuit_vs_closed_form",
                    "encoding": encoding,
                    "seed": seed,
                    "N": N,
                    "features": d,
                    "max_probability_error": max_error,
                    "max_index_mass_error": total_probability_error,
                    "minimum_probability": minimum_probability,
                    "passed": (
                        max_error < 1e-12
                        and total_probability_error < 1e-12
                        and minimum_probability >= -1e-12
                    ),
                })

    return rows


def validate_dataset_cases(
    datasets: Iterable[str],
    seed: int,
    n_train: int,
    n_test: int,
    encoding: str,
) -> list[dict]:
    """Check normalization, ideal probabilities, and ranking consistency."""
    rows = []

    for name in datasets:
        X_train, y_train, X_test, y_test = load_dataset_split(
            name, n_train, n_test, seed
        )
        Z_train, Z_test = authors_normalize(X_train, X_test)

        train_bound = 1.0 / (2.0 * math.sqrt(Z_train.shape[1]))
        train_max_abs = float(np.max(np.abs(Z_train)))

        for test_index, target in enumerate(Z_test):
            probabilities = closed_form_probabilities(
                Z_train, target, encoding
            )
            exact_distances = np.sqrt(
                exact_squared_distances(Z_train, target)
            )
            estimated_distances = distance_estimates_from_probabilities(
                probabilities, target, encoding
            )

            probability_sum = float(probabilities.sum())
            min_probability = float(probabilities.min())
            max_distance_error = float(
                np.max(np.abs(exact_distances - estimated_distances))
            )
            top_k = min(5, len(Z_train))
            top_k_jaccard = ranking_agreement(
                exact_distances, estimated_distances, top_k
            )

            rows.append({
                "check": "dataset_probability_and_distance",
                "dataset": name,
                "encoding": encoding,
                "seed": seed,
                "test_index": test_index,
                "N": len(Z_train),
                "features": Z_train.shape[1],
                "train_max_abs": train_max_abs,
                "allowed_train_bound": train_bound,
                "probability_sum": probability_sum,
                "minimum_probability": min_probability,
                "max_distance_error": max_distance_error,
                "top_k": top_k,
                "top_k_jaccard": top_k_jaccard,
                "test_label": int(y_test[test_index]),
                "passed": (
                    np.isclose(probability_sum, 1.0, atol=1e-10)
                    and min_probability >= -1e-10
                    and max_distance_error < 1e-8
                    and top_k_jaccard == 1.0
                    and train_max_abs <= train_bound + 1e-10
                ),
            })

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(
        key for row in rows for key in row.keys()
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the original QKNN translation/extension model."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["blobs", "iris", "wine", "digits"],
        choices=["blobs", "iris", "wine", "digits"],
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument(
        "--encodings",
        nargs="+",
        default=["translation", "extension"],
        choices=["translation", "extension"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results" / "baseline_validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows: list[dict] = []

    # Small dimensions keep dense state-vector simulation inexpensive.
    circuit_cases = [(3, 2), (5, 3), (7, 4), (8, 5)]
    circuit_rows = validate_random_circuits(
        seeds=[1, 7, 19],
        cases=circuit_cases,
        encodings=args.encodings,
    )
    all_rows.extend(circuit_rows)

    for row in circuit_rows:
        print(
            f"[CIRCUIT] {row['encoding']:11s} "
            f"N={row['N']:2d} d={row['features']:2d} "
            f"max_error={row['max_probability_error']:.3e} "
            f"{'PASS' if row['passed'] else 'FAIL'}"
        )

    # Dataset size caps: the fixed sklearn datasets cannot supply arbitrary
    # training/test counts without replacement.
    dataset_rows = []
    for name in args.datasets:
        requested_train = args.n_train
        requested_test = args.n_test
        if name in {"iris", "wine", "digits"}:
            total = {
                "iris": 150,
                "wine": 178,
                "digits": 1797,
            }[name]
            if requested_train + requested_test > total:
                raise ValueError(
                    f"{name}: n_train+n_test exceeds dataset size {total}."
                )

        for encoding in args.encodings:
            current = validate_dataset_cases(
                datasets=[name],
                seed=args.seed,
                n_train=requested_train,
                n_test=requested_test,
                encoding=encoding,
            )
            dataset_rows.extend(current)
            all_rows.extend(current)

    for row in dataset_rows:
        if row["test_index"] == 0:
            print(
                f"[DATASET] {row['dataset']:7s} {row['encoding']:11s} "
                f"N={row['N']:4d} d={row['features']:2d} "
                f"distance_error={row['max_distance_error']:.3e} "
                f"top-{row['top_k']} Jaccard={row['top_k_jaccard']:.3f} "
                f"{'PASS' if row['passed'] else 'FAIL'}"
            )

    output_path = args.output_dir / "baseline_validation.csv"
    write_csv(output_path, all_rows)

    failures = [row for row in all_rows if not row["passed"]]
    print("\n" + "=" * 68)
    print(f"Total checks: {len(all_rows)}")
    print(f"Passed:       {len(all_rows) - len(failures)}")
    print(f"Failed:       {len(failures)}")
    print(f"CSV saved:    {output_path}")

    if failures:
        raise SystemExit(
            "Baseline validation failed. Inspect the CSV before proceeding."
        )

    print("BASELINE VALIDATION PASSED.")


if __name__ == "__main__":
    main()
