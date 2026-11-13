#!/usr/bin/env python3
"""
Study 08 — Quantum k-NN State Geometry & Neighbourhood Preservation
===================================================================

Purpose
-------
A focused follow-up to Study 07. It tests whether encoding geometry, rather
than measurement precision alone, explains QKNN performance differences.

Experiments
-----------
1. Amplitude encoding: MinMax [0,1] baseline, fixed-0.5 centering, and
   training-mean centering (the mean is fitted on TRAIN ONLY).
2. Angle encoding reference: tensor-product Ry(pi*x) states.
3. Basis encoding diagnostics: 1-bit and 2-bit discretization, exact-state
   fidelity, collision rates, and tie rates.
4. Classical geometry diagnostics: Euclidean and cosine neighbour rankings
   on the same preprocessed vectors.
5. For every representation: classification accuracy, neighbour Jaccard,
   Spearman ranking correlation, fidelity boundary gaps, and tie sensitivity.
6. Ideal finite-shot SWAP measurement: binomial sampling from
   P(ancilla=0)=(1+F)/2, shot-level accuracy and neighbour recovery.

Important methodological notes
------------------------------
- This script evaluates ideal state geometry and ideal SWAP measurement
  statistics. It does NOT execute Aer circuits or model hardware noise.
- Amplitude encoding normalizes every vector to unit length. Consequently,
  vector magnitude is not retained in the encoded state; the script records
  pre-normalization norms to diagnose this information loss.
- The fixed-0.5 and training-mean centering operations are applied after
  training-fitted MinMax scaling. Test data are clipped to [0,1] using the
  training-fitted scaler.
- Dataset feature selection is deliberately simple: the first --max-features
  columns are used, matching the small-feature regime of Study 07. Conclusions
  should not be generalized to all features without a follow-up.
- Repeated stratified splits overlap; split-level intervals are descriptive,
  not independent-population confidence intervals.

Dependencies
------------
pip install numpy pandas scipy scikit-learn openpyxl

Example
-------
python 08_quantum_knn_geometry_study.py --quick
python 08_quantum_knn_geometry_study.py --n-splits 20 --tie-repeats 30

Output
------
One Excel workbook with separate worksheets, plus a small run log.
"""

from __future__ import annotations
import argparse
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, f_oneway
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(description="Study 08: QKNN state geometry and neighbourhood preservation")
    p.add_argument('--datasets', nargs='+', default=['iris', 'wine', 'breast_cancer'])
    p.add_argument('--n-splits', type=int, default=20)
    p.add_argument('--max-train', type=int, default=100)
    p.add_argument('--max-test', type=int, default=40)
    p.add_argument('--max-features', type=int, default=5)
    p.add_argument('--k-values', nargs='+', type=int, default=[1, 3, 5, 9])
    p.add_argument('--shot-levels', nargs='+', type=int, default=[128, 512, 2048, 8192])
    p.add_argument('--shot-repeats', type=int, default=8)
    p.add_argument('--tie-repeats', type=int, default=30)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--outdir', type=str, default='results/study_08_geometry')
    p.add_argument('--name', type=str, default='study_08_geometry_results.xlsx')
    p.add_argument('--quick', action='store_true', help='Small smoke run: Iris, 2 splits, smaller samples and budgets')
    return p.parse_args()


def load_dataset(name: str):
    if name == 'iris':
        d = load_iris(); return d.data, d.target, d.feature_names
    if name == 'wine':
        d = load_wine(); return d.data, d.target, d.feature_names
    if name == 'breast_cancer':
        d = load_breast_cancer(); return d.data, d.target, d.feature_names
    if name == 'synthetic_binarized':
        X, y = make_classification(n_samples=500, n_features=10, n_informative=6,
            n_redundant=1, n_repeated=0, n_classes=2, class_sep=1.1,
            random_state=1701)
        # Genuine binary dataset: threshold each feature at its training-agnostic
        # global median solely to define the synthetic dataset itself.
        X = (X > np.median(X, axis=0)).astype(float)
        return X, y, [f'bit_feature_{i}' for i in range(X.shape[1])]
    raise ValueError(f"Unknown dataset '{name}'. Choose iris, wine, breast_cancer, synthetic_binarized.")


def stratified_cap(X, y, max_n, seed):
    if max_n is None or len(y) <= max_n:
        return X, y
    # Keep class proportions when reducing a split.
    X_keep, _, y_keep, _ = train_test_split(X, y, train_size=max_n,
        stratify=y, random_state=seed)
    return X_keep, y_keep


def unit(v):
    n = np.linalg.norm(v)
    if n < EPS:
        return None
    return v / n


def amp_fidelity_matrix(A, B):
    """Squared real inner products for normalized amplitude vectors."""
    return np.clip((A @ B.T) ** 2, 0.0, 1.0)


def angle_fidelity_matrix(A, B):
    """Tensor-product Ry(pi*x) fidelity: product_j cos^2(pi*(x-y)/2)."""
    diff = A[:, None, :] - B[None, :, :]
    terms = np.cos(np.pi * diff / 2.0) ** 2
    return np.clip(np.prod(terms, axis=2), 0.0, 1.0)


def phase_fidelity_matrix(A, B):
    """For the chosen Rz(pi*x)|+> map, this equals the angle map above."""
    return angle_fidelity_matrix(A, B)


def basis_codes(X, bits=1):
    bins = 2 ** bits
    q = np.floor(np.clip(X, 0.0, 1.0 - 1e-12) * bins).astype(int)
    q = np.clip(q, 0, bins - 1)
    return q


def basis_fidelity_matrix(CA, CB):
    # Basis states have fidelity 1 only for an identical full bitstring.
    return (CA[:, None, :] == CB[None, :, :]).all(axis=2).astype(float)


def cosine_similarity_matrix(A, B):
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), EPS)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), EPS)
    return np.clip(An @ Bn.T, -1.0, 1.0)


def euclidean_similarity_matrix(A, B):
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    return -d


def topk_indices(scores, k, rng):
    """Select exactly k; randomize only the tie at the kth boundary."""
    n = scores.shape[0]
    k = min(k, n)
    order = np.argsort(-scores, kind='mergesort')
    cutoff = scores[order[k - 1]]
    above = np.flatnonzero(scores > cutoff + 1e-12)
    tied = np.flatnonzero(np.abs(scores - cutoff) <= 1e-12)
    need = k - len(above)
    if need > 0 and len(tied) > need:
        chosen = rng.choice(tied, size=need, replace=False)
    else:
        chosen = tied[:need]
    return np.concatenate([above, chosen]).astype(int)


def predict_from_scores(scores, y_train, k, seed):
    rng = np.random.default_rng(seed)
    preds, selected = [], []
    for row in scores:
        idx = topk_indices(row, k, rng)
        vals, counts = np.unique(y_train[idx], return_counts=True)
        winners = vals[counts == counts.max()]
        pred = rng.choice(winners)
        preds.append(pred); selected.append(idx)
    return np.asarray(preds), selected


def jaccard(a, b):
    sa, sb = set(map(int, a)), set(map(int, b))
    return len(sa & sb) / max(1, len(sa | sb))


def ranking_corr(a, b):
    if len(a) < 3 or np.std(a) < EPS or np.std(b) < EPS:
        return np.nan
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else np.nan


def tie_count(scores, k):
    out = []
    for row in scores:
        vals = np.sort(row)[::-1]
        if len(vals) <= k:
            out.append(1)
        else:
            out.append(int(np.sum(np.abs(row - vals[k-1]) <= 1e-12)))
    return np.asarray(out)


def fidelity_gap(scores, k):
    if scores.shape[1] <= k:
        return np.full(scores.shape[0], np.nan)
    s = np.sort(scores, axis=1)[:, ::-1]
    return s[:, k-1] - s[:, k]


def encode_variants(Xtr_scaled, Xte_scaled):
    variants = {}
    # Amplitude baseline: min-max scaled, then amplitude normalization.
    variants['amplitude_raw'] = (Xtr_scaled.copy(), Xte_scaled.copy(), 'amplitude')
    # Fixed offset used in the earlier exploratory test.
    variants['amplitude_center_0.5'] = (Xtr_scaled - 0.5, Xte_scaled - 0.5, 'amplitude')
    # Correctly fitted only on training data.
    mu = Xtr_scaled.mean(axis=0)
    variants['amplitude_center_train_mean'] = (Xtr_scaled - mu, Xte_scaled - mu, 'amplitude')
    variants['angle'] = (Xtr_scaled.copy(), Xte_scaled.copy(), 'angle')
    variants['phase'] = (Xtr_scaled.copy(), Xte_scaled.copy(), 'phase')
    variants['basis_1bit'] = (basis_codes(Xtr_scaled, 1), basis_codes(Xte_scaled, 1), 'basis')
    variants['basis_2bit'] = (basis_codes(Xtr_scaled, 2), basis_codes(Xte_scaled, 2), 'basis')
    return variants, mu


def make_similarity(kind, A, B):
    if kind == 'amplitude':
        def norm_rows(X):
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            # Zero vectors cannot be amplitude encoded. Deterministic fallback:
            # replace a zero row by the first computational basis vector.
            out = X.copy().astype(float)
            zero = norms[:, 0] < EPS
            out[~zero] /= norms[~zero]
            out[zero] = 0.0; out[zero, 0] = 1.0
            return out, norms[:, 0], zero
        An, nA, zA = norm_rows(A); Bn, nB, zB = norm_rows(B)
        return amp_fidelity_matrix(Bn, An), nA, nB, zA, zB
    if kind in ('angle', 'phase'):
        F = angle_fidelity_matrix(B, A) if kind == 'angle' else phase_fidelity_matrix(B, A)
        return F, np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1), np.zeros(len(A), bool), np.zeros(len(B), bool)
    if kind == 'basis':
        F = basis_fidelity_matrix(B, A)
        return F, np.full(len(A), np.nan), np.full(len(B), np.nan), np.zeros(len(A), bool), np.zeros(len(B), bool)
    raise ValueError(kind)


def amplitude_geometry_diagnostics(dataset, split_id, variant, A, B, F, k_values):
    """Compare amplitude fidelity with the classical geometry it induces."""
    def normalize_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        out = X.astype(float).copy()
        zero = norms[:, 0] < EPS
        out[~zero] /= norms[~zero]
        out[zero] = 0.0
        out[zero, 0] = 1.0
        return out
    An, Bn = normalize_rows(A), normalize_rows(B)
    signed_cos = np.clip(Bn @ An.T, -1.0, 1.0)
    squared_cos = signed_cos ** 2
    abs_cos = np.abs(signed_cos)
    neg_unit_euc = -np.linalg.norm(Bn[:, None, :] - An[None, :, :], axis=2)
    rows = []
    for k in k_values:
        for metric_name, M in [('signed_cosine', signed_cos),
                               ('squared_cosine', squared_cos),
                               ('absolute_cosine', abs_cos),
                               ('euclidean_on_unit_vectors', neg_unit_euc)]:
            overlaps = [jaccard(np.argsort(-F[q])[:k], np.argsort(-M[q])[:k])
                        for q in range(F.shape[0])]
            corrs = [ranking_corr(F[q], M[q]) for q in range(F.shape[0])]
            rows.append({'dataset': dataset, 'split': split_id, 'variant': variant, 'k': k,
                'comparison_geometry': metric_name,
                'topk_jaccard_vs_amplitude_fidelity': float(np.mean(overlaps)),
                'spearman_vs_amplitude_fidelity': float(np.nanmean(corrs)) if np.any(np.isfinite(corrs)) else np.nan,
                'mean_signed_cosine': float(np.mean(signed_cos)),
                'fraction_negative_cosine_pairs': float(np.mean(signed_cos < 0)),
                'amplitude_fidelity_equals_squared_cosine_max_error': float(np.max(np.abs(F - squared_cos)))})
    return rows


def norm_class_diagnostics(dataset, split_id, variant, Xtr_variant, Xte_variant, ytr, yte):
    """Quantify whether pre-normalization vector length varies by class."""
    rows = []
    for part, Xv, yv in [('train', Xtr_variant, ytr), ('test', Xte_variant, yte)]:
        norms = np.linalg.norm(Xv, axis=1)
        groups = [norms[yv == c] for c in np.unique(yv)]
        overall = float(np.mean(norms)) if len(norms) else np.nan
        ss_total = float(np.sum((norms - overall) ** 2)) if len(norms) else np.nan
        ss_between = 0.0
        for c in np.unique(yv):
            g = norms[yv == c]
            if len(g): ss_between += len(g) * (float(np.mean(g)) - overall) ** 2
        eta2 = ss_between / ss_total if ss_total > EPS else 0.0
        try:
            pval = float(f_oneway(*groups).pvalue) if len(groups) >= 2 and all(len(g) > 1 for g in groups) else np.nan
        except Exception:
            pval = np.nan
        rows.append({'dataset':dataset, 'split':split_id, 'variant':variant, 'partition':part,
            'mean_pre_normalization_norm':overall, 'std_pre_normalization_norm':float(np.std(norms)),
            'norm_eta_squared_by_class':float(eta2), 'one_way_anova_p_value':pval,
            'interpretation_note':'Class-associated norm variation is potentially discarded by amplitude normalization; descriptive diagnostic only.'})
    return rows


def summarize_variant(dataset, split_id, variant, kind, F, ytr, yte,
                      classical_euc, classical_cos, k_values, tie_repeats, seed,
                      norms_tr, norms_te, zero_tr, zero_te):
    rows, query_rows, pred_cache = [], [], {}
    for k in k_values:
        accs, jacs, disagreements, tie_sizes = [], [], [], []
        all_pred = []
        for rep in range(tie_repeats):
            pred, selected = predict_from_scores(F, ytr, k, seed + rep * 1009 + k)
            all_pred.append(pred)
            accs.append(float(np.mean(pred == yte)))
            js = [jaccard(selected[q], np.argsort(-classical_euc[q])[:k]) for q in range(len(yte))]
            jacs.append(float(np.mean(js)))
            tie_sizes.extend(tie_count(F, k).tolist())
        pred_stack = np.asarray(all_pred)
        mean_pred = []
        for q in range(len(yte)):
            vals, cnt = np.unique(pred_stack[:, q], return_counts=True)
            mean_pred.append(vals[np.argmax(cnt)])
        disagreement = float(np.mean([len(np.unique(pred_stack[:, q])) > 1 for q in range(len(yte))]))
        gaps = fidelity_gap(F, k)
        row = {
            'dataset': dataset, 'split': split_id, 'variant': variant, 'encoding_family': kind, 'k': k,
            'accuracy_mean_over_tie_seeds': float(np.mean(accs)),
            'accuracy_sd_over_tie_seeds': float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            'accuracy_min': float(np.min(accs)), 'accuracy_max': float(np.max(accs)),
            'neighbour_jaccard_vs_euclidean': float(np.mean(jacs)),
            'prediction_disagreement_fraction': disagreement,
            'mean_k_boundary_fidelity_gap': float(np.nanmean(gaps)),
            'median_k_boundary_fidelity_gap': float(np.nanmedian(gaps)),
            'queries_with_zero_boundary_gap_fraction': float(np.mean(gaps <= 1e-12)),
            'mean_number_of_candidates_tied_at_k_boundary': float(np.mean(tie_count(F, k))),
            'mean_training_vector_norm_before_amplitude_normalization': (float(np.nanmean(norms_tr)) if np.any(np.isfinite(norms_tr)) else np.nan),
            'mean_test_vector_norm_before_amplitude_normalization': (float(np.nanmean(norms_te)) if np.any(np.isfinite(norms_te)) else np.nan),
            'zero_training_vectors_fraction': float(np.mean(zero_tr)),
            'zero_test_vectors_fraction': float(np.mean(zero_te)),
        }
        rows.append(row)
        # Per-query diagnostics, using the modal prediction across tie seeds.
        for q in range(len(yte)):
            qcorr = ranking_corr(F[q], classical_euc[q])
            query_rows.append({
                'dataset': dataset, 'split': split_id, 'variant': variant, 'k': k, 'query_index': q,
                'true_label': int(yte[q]), 'modal_prediction': int(mean_pred[q]),
                'correct': bool(mean_pred[q] == yte[q]),
                'fidelity_boundary_gap': float(gaps[q]) if np.isfinite(gaps[q]) else np.nan,
                'tie_count_at_k_boundary': int(tie_count(F, k)[q]),
                'spearman_fidelity_vs_negative_euclidean': qcorr,
                'neighbour_jaccard_vs_euclidean': jaccard(
                    np.argsort(-F[q])[:k], np.argsort(-classical_euc[q])[:k]),
                'cosine_topk_jaccard_vs_euclidean': jaccard(
                    np.argsort(-classical_cos[q])[:k], np.argsort(-classical_euc[q])[:k]),
            })
    return rows, query_rows


def shot_experiment(dataset, split_id, variant, F, ytr, yte, classical_euc,
                    k_values, shot_levels, repeats, seed):
    rows = []
    # Ideal SWAP test: P(0)=(1+F)/2; estimate F_hat=2*X/shots - 1.
    p0 = np.clip((1.0 + F) / 2.0, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    for shots in shot_levels:
        for k in k_values:
            acc, jac = [], []
            for rep in range(repeats):
                counts = rng.binomial(shots, p0)
                Fest = np.clip(2.0 * counts / shots - 1.0, 0.0, 1.0)
                pred, selected = predict_from_scores(Fest, ytr, k, seed + rep * 7919 + shots + k)
                acc.append(float(np.mean(pred == yte)))
                jac.append(float(np.mean([jaccard(selected[q], np.argsort(-classical_euc[q])[:k])
                                          for q in range(len(yte))])))
            rows.append({
                'dataset': dataset, 'split': split_id, 'variant': variant, 'k': k, 'shots': shots,
                'accuracy_mean': float(np.mean(acc)), 'accuracy_sd': float(np.std(acc, ddof=1)) if len(acc)>1 else 0.0,
                'neighbour_jaccard_mean': float(np.mean(jac)), 'shot_repeats': repeats,
                'measurement_model': 'ideal_binomial_swap_test',
            })
    return rows


def main():
    args = parse_args()
    if args.quick:
        args.datasets = ['iris']; args.n_splits = 2
        args.max_train = min(args.max_train, 45); args.max_test = min(args.max_test, 18)
        args.k_values = [1, 3]; args.shot_levels = [128, 512]
        args.shot_repeats = 3; args.tie_repeats = 5
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, args.name)
    rng_master = np.random.default_rng(args.seed)

    summary_rows, query_rows, shot_rows, split_rows = [], [], [], []
    metadata_rows, equivalence_rows, hypothesis_rows = [], [], []
    amp_geometry_rows, norm_class_rows = [], []
    started = time.time()

    for ds_i, ds_name in enumerate(args.datasets):
        X, y, feature_names = load_dataset(ds_name)
        X = np.asarray(X, dtype=float); y = LabelEncoder().fit_transform(y)
        # Preserve the Study 07 small-feature cap, without test-informed selection.
        X = X[:, :min(args.max_features, X.shape[1])]
        n_classes = len(np.unique(y))
        print(f"\n[{ds_name}] n={len(y)}, d={X.shape[1]}, classes={n_classes}", flush=True)
        splitter = StratifiedShuffleSplit(n_splits=args.n_splits, test_size=0.25,
                                          random_state=args.seed + ds_i * 101)
        for split_id, (itr, ite) in enumerate(splitter.split(X, y), start=1):
            Xtr, Xte, ytr, yte = X[itr], X[ite], y[itr], y[ite]
            Xtr, ytr = stratified_cap(Xtr, ytr, args.max_train, args.seed + split_id)
            Xte, yte = stratified_cap(Xte, yte, args.max_test, args.seed + split_id + 10000)
            scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(Xte)
            Xtr_s = np.clip(Xtr_s, 0.0, 1.0); Xte_s = np.clip(Xte_s, 0.0, 1.0)
            # Classical references use exactly the same scaled feature matrix.
            euc = euclidean_similarity_matrix(Xte_s, Xtr_s)
            cos = cosine_similarity_matrix(Xte_s, Xtr_s)
            variants, train_mean = encode_variants(Xtr_s, Xte_s)
            split_rows.append({'dataset': ds_name, 'split': split_id, 'n_train': len(ytr),
                'n_test': len(yte), 'n_features': Xtr_s.shape[1], 'n_classes': n_classes,
                'train_mean_vector': str(np.round(train_mean, 6).tolist()),
                'train_only_scaling': True})

            cache = {}
            for v_i, (variant, (A, B, kind)) in enumerate(variants.items()):
                F, norms_tr, norms_te, zero_tr, zero_te = make_similarity(kind, A, B)
                cache[variant] = F
                r, qr = summarize_variant(ds_name, split_id, variant, kind, F, ytr, yte,
                    euc, cos, args.k_values, args.tie_repeats,
                    args.seed + ds_i*100000 + split_id*1000 + v_i*31,
                    norms_tr, norms_te, zero_tr, zero_te)
                summary_rows.extend(r); query_rows.extend(qr)
                if variant in ('amplitude_raw', 'amplitude_center_0.5', 'amplitude_center_train_mean', 'angle'):
                    shot_rows.extend(shot_experiment(ds_name, split_id, variant, F, ytr, yte,
                        euc, args.k_values, args.shot_levels, args.shot_repeats,
                        args.seed + ds_i*200000 + split_id*3000 + v_i*97))
                unique_train_states = len(np.unique(A, axis=0)) if kind == 'basis' else np.nan
                metadata_rows.append({'dataset': ds_name, 'split': split_id, 'variant': variant,
                    'encoding_family': kind, 'unique_training_states': unique_train_states,
                    'training_state_collision_fraction': (1 - unique_train_states/len(A)) if kind == 'basis' else np.nan,
                    'train_zero_vector_count': int(np.sum(zero_tr)), 'test_zero_vector_count': int(np.sum(zero_te)),
                    'mean_train_vector_norm_pre_normalization': (float(np.nanmean(norms_tr)) if np.any(np.isfinite(norms_tr)) else np.nan),
                    'mean_test_vector_norm_pre_normalization': (float(np.nanmean(norms_te)) if np.any(np.isfinite(norms_te)) else np.nan)})
                if kind == 'amplitude':
                    amp_geometry_rows.extend(amplitude_geometry_diagnostics(ds_name, split_id, variant, A, B, F, args.k_values))
                    norm_class_rows.extend(norm_class_diagnostics(ds_name, split_id, variant, A, B, ytr, yte))

            # Angle/Phase equivalence diagnostic, per split.
            if 'angle' in cache and 'phase' in cache:
                equivalence_rows.append({'dataset': ds_name, 'split': split_id,
                    'max_abs_fidelity_difference_angle_phase': float(np.max(np.abs(cache['angle']-cache['phase']))),
                    'mean_abs_fidelity_difference_angle_phase': float(np.mean(np.abs(cache['angle']-cache['phase']))),
                    'interpretation': 'Expected equivalence for the specific product-state maps implemented here.'})

            # Classical cosine-vs-Euclidean neighbourhood overlap, for context.
            for k in args.k_values:
                overlap = np.mean([jaccard(np.argsort(-cos[q])[:k], np.argsort(-euc[q])[:k])
                                   for q in range(len(yte))])
                hypothesis_rows.append({'dataset': ds_name, 'split': split_id, 'hypothesis': 'classical_cosine_vs_euclidean',
                    'k': k, 'metric_value': float(overlap),
                    'interpretation': 'Low overlap indicates that cosine and Euclidean define materially different neighbourhoods.'})

        print(f"  completed {args.n_splits} split(s) for {ds_name}", flush=True)

    summary = pd.DataFrame(summary_rows)
    queries = pd.DataFrame(query_rows)
    shots = pd.DataFrame(shot_rows)
    metadata = pd.DataFrame(metadata_rows)
    splits = pd.DataFrame(split_rows)
    equivalence = pd.DataFrame(equivalence_rows)
    hypotheses = pd.DataFrame(hypothesis_rows)
    amp_geometry = pd.DataFrame(amp_geometry_rows)
    norm_class = pd.DataFrame(norm_class_rows)

    # Aggregate by dataset/variant/k. These are means over overlapping splits;
    # label them descriptively rather than presenting as population CIs.
    agg = summary.groupby(['dataset','variant','encoding_family','k'], dropna=False).agg(
        accuracy_mean=('accuracy_mean_over_tie_seeds','mean'),
        accuracy_sd_across_splits=('accuracy_mean_over_tie_seeds','std'),
        neighbour_jaccard_mean=('neighbour_jaccard_vs_euclidean','mean'),
        boundary_gap_mean=('mean_k_boundary_fidelity_gap','mean'),
        boundary_tie_fraction=('queries_with_zero_boundary_gap_fraction','mean'),
        tie_prediction_disagreement=('prediction_disagreement_fraction','mean'),
    ).reset_index()
    agg['interval_note'] = 'Descriptive split variation; repeated stratified splits overlap.'

    shot_agg = shots.groupby(['dataset','variant','k','shots'], dropna=False).agg(
        accuracy_mean=('accuracy_mean','mean'), accuracy_sd_across_splits=('accuracy_mean','std'),
        neighbour_jaccard_mean=('neighbour_jaccard_mean','mean')).reset_index()
    shot_agg['interval_note'] = 'Ideal binomial SWAP sampling; descriptive split variation.'

    # Sustained shot threshold: first level that meets both thresholds at itself
    # and every higher tested shot level. Thresholds are intentionally explicit.
    efficiency_rows = []
    for (ds, var, k), g in shot_agg.groupby(['dataset','variant','k']):
        g = g.sort_values('shots')
        qualifying = (g['accuracy_mean'] >= 0.90) & (g['neighbour_jaccard_mean'] >= 0.90)
        sustained = []
        for i in range(len(g)):
            sustained.append(bool(qualifying.iloc[i:].all()))
        chosen = int(g.iloc[np.argmax(sustained)]['shots']) if any(sustained) else np.nan
        efficiency_rows.append({'dataset':ds,'variant':var,'k':k,
            'minimum_sustained_shots_for_accuracy_and_jaccard_ge_0.90':chosen,
            'threshold_accuracy':0.90,'threshold_neighbour_jaccard':0.90,
            'criterion':'Must meet both thresholds at this and every higher tested shot level.'})
    efficiency = pd.DataFrame(efficiency_rows)

    config = pd.DataFrame([
        ['study','08 — Quantum k-NN State Geometry & Neighbourhood Preservation'],
        ['datasets', ', '.join(args.datasets)], ['n_splits', args.n_splits],
        ['max_train', args.max_train], ['max_test', args.max_test], ['max_features', args.max_features],
        ['k_values', str(args.k_values)], ['shot_levels', str(args.shot_levels)],
        ['shot_repeats', args.shot_repeats], ['tie_repeats', args.tie_repeats], ['seed', args.seed],
        ['preprocessing','MinMax scaler fitted on training data only; test values clipped to [0,1].'],
        ['amplitude_centering','Fixed 0.5 and training mean (training mean computed from scaled training data only).'],
        ['shot_model','Ideal binomial SWAP-test sampling: P(0)=(1+F)/2; not Aer execution or hardware noise.'],
        ['feature_policy','First max_features columns; no test-informed feature selection.'],
        ['uncertainty','Repeated stratified splits overlap; split variation is descriptive, not independent-sample CI.'],
        ['zero_vector_policy','Zero amplitude vectors are deterministically mapped to |0...0>; counts reported.'],
    ], columns=['setting','value'])

    methodology = pd.DataFrame([
        ['Amplitude fidelity','F(x,y)=|<x/||x||, y/||y||>|^2. Vector magnitude is discarded by normalization.'],
        ['Fixed centering','Subtract 0.5 from each MinMax-scaled feature, then amplitude-normalize.'],
        ['Training-mean centering','Subtract the feature-wise mean of scaled training data from train and test vectors.'],
        ['Angle fidelity','For product Ry(pi*x_j)|0> maps, F=product_j cos^2(pi*(x_j-y_j)/2).'],
        ['Phase fidelity','For product Rz(pi*x_j)|+> maps, fidelity equals the implemented angle fidelity.'],
        ['Basis fidelity','Exact computational-basis-state match: 1 if all quantized feature codes match, else 0.'],
        ['Neighbour Jaccard','Jaccard overlap between the encoding top-k and classical Euclidean top-k neighbours.'],
        ['Ranking correlation','Per-query Spearman correlation between encoding fidelity and negative Euclidean distance.'],
        ['Boundary gap','kth-largest fidelity minus (k+1)th-largest fidelity.'],
        ['Shot model','Binomial ancilla-zero counts with p0=(1+F)/2; F_hat=2*count/shots-1, clipped to [0,1].'],
        ['Amplitude geometry diagnostic','For real amplitude states, fidelity is squared cosine similarity; the workbook compares it with signed cosine, absolute cosine and Euclidean geometry on unit vectors.'],
        ['Norm/class diagnostic','Reports class-associated variation in pre-normalization vector length (eta-squared and one-way ANOVA); this length is discarded by amplitude normalization.'],
        ['Interpretation','A larger fidelity gap can reduce measurement ambiguity, but does not guarantee a better neighbour ranking or classification.'],
    ], columns=['quantity','definition'])

    # A concise workbook-level results digest.
    digest_rows = []
    for (ds, var, k), g in agg.groupby(['dataset','variant','k']):
        digest_rows.append({'dataset':ds,'variant':var,'k':k,
            'accuracy_mean':g['accuracy_mean'].iloc[0],
            'neighbour_jaccard_mean':g['neighbour_jaccard_mean'].iloc[0],
            'boundary_gap_mean':g['boundary_gap_mean'].iloc[0],
            'boundary_tie_fraction':g['boundary_tie_fraction'].iloc[0],
            'tie_prediction_disagreement':g['tie_prediction_disagreement'].iloc[0]})
    digest = pd.DataFrame(digest_rows)

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        config.to_excel(writer, sheet_name='run_config', index=False)
        methodology.to_excel(writer, sheet_name='methodology', index=False)
        digest.to_excel(writer, sheet_name='results_digest', index=False)
        agg.to_excel(writer, sheet_name='summary_by_dataset', index=False)
        summary.to_excel(writer, sheet_name='results_by_split', index=False)
        queries.to_excel(writer, sheet_name='query_diagnostics', index=False)
        shot_agg.to_excel(writer, sheet_name='shot_summary', index=False)
        shots.to_excel(writer, sheet_name='shot_results_by_split', index=False)
        efficiency.to_excel(writer, sheet_name='shot_efficiency', index=False)
        metadata.to_excel(writer, sheet_name='state_diagnostics', index=False)
        equivalence.to_excel(writer, sheet_name='angle_phase_check', index=False)
        hypotheses.to_excel(writer, sheet_name='classical_metric_check', index=False)
        amp_geometry.to_excel(writer, sheet_name='amplitude_geometry', index=False)
        norm_class.to_excel(writer, sheet_name='norm_class_diagnostics', index=False)
        splits.to_excel(writer, sheet_name='split_metadata', index=False)

        # Readability: freeze headers, filters, sensible widths.
        for ws in writer.book.worksheets:
            ws.freeze_panes = 'A2'
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                letter = col[0].column_letter
                max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col[:250])
                ws.column_dimensions[letter].width = min(max(max_len + 2, 12), 48)

    elapsed = time.time() - started
    log_path = os.path.join(args.outdir, 'study_08_run_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Study 08 completed in {elapsed:.1f} seconds\n")
        f.write(f"Workbook: {out_path}\n")
        f.write(f"Datasets: {args.datasets}\nSplits: {args.n_splits}\n")
        f.write('WARNING: ideal binomial SWAP sampling only; no Aer circuits or hardware noise.\n')
        f.write('Repeated stratified splits overlap; reported split variation is descriptive.\n')
    print(f"\nDONE. Workbook: {out_path}")
    print(f"Run log: {log_path}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
