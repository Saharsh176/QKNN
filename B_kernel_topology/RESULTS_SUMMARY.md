# Project B: CNOT-topology selection for fidelity quantum kernels (v6 run, status: FINISHED, null result)

Script: `qkernel_rigorous_v6.py`; full run with outputs: `simulations2_v6_run_with_results.ipynb` (3047 s).
Protocol: train/test split, topology + hyper-parameters (alpha, beta, RBF gamma) chosen on TRAIN only
(inner 3-fold CV), same splits for the classical RBF baseline, leakage test PASSED, closed-form kernel
matches Qiskit to 3e-15 (ideal kernel only).

## Headline
Once the single-qubit bandwidth beta is tuned, entanglement gives no measurable benefit, and the
"label-aware CNOT selection" gains seen earlier at beta = 1 disappear.

| TUNED, planted-interaction data, test kNN (30 datasets) | value |
|---|---|
| none (beta tuned -> 0.5) | 0.814 |
| oracle (planted edges)   | 0.809 |
| greedy-KTA, 2 edges      | 0.804 |
| random, 2 edges          | 0.806 |
| chain, 5 edges           | 0.804 |
| RBF tuned                | 0.812 |
| oracle - none            | -0.004 [-0.015, +0.006] |
| greedy_kta - random      | -0.002 [-0.012, +0.008] |

GRID (descriptive): at beta = 1.0 oracle - none = +0.011 ... +0.082 (peak alpha = 0.25), but "none" itself is
only 0.648 there versus 0.814 at beta = 0.5. At beta = 0.5, oracle - none is ~0 for alpha <= 0.25 and
negative for alpha >= 0.35 (-0.012, -0.029). The earlier benefit was the interaction term compensating
for an over-narrow single-qubit bandwidth, not entanglement capturing structure.

CONTROLS (tuned, no planted structure): breast cancer greedy_kta - random +0.009 [+0.003,+0.016] but
greedy_kta - rbf_tuned +0.006 [-0.002,+0.014]; wine greedy_kta - random -0.010 [-0.018,-0.001];
random smooth function +0.018 [-0.005,+0.041]. No consistent benefit.

NOISY (6 matched draws, hardware-like noise): intervals too wide for any claim. Gate audit: disjoint-edge
topologies compile to 4 CX / depth 9; the 5-edge chain to ~18 CX / depth 33 (no routing included).

## What can honestly be claimed
* Correlation-based pruning is no better than random (and worse when correlated decoys exist).
* Label-aware (KTA/CV) selection helps only when beta is mis-tuned; not after tuning.
* No quantum variant beats a tuned RBF. Edge recovery of planted pairs is a diagnostic, not a result.
* Useful methodological lesson: always tune the base bandwidth before attributing gains to entanglement.

## If revisited
Only with data whose structure is aligned with the feature map (e.g. labels generated from the quantum
feature map itself, or genuinely high-dimensional data where classical kernels fail), and with tuned beta.
