# Research workspace

Two projects. **A is the recommended main line; B is finished (null result) and becomes a short chapter/appendix.**

```
research_workspace/
  A_quantum_knn_cost_study/        Zardini et al. Euclidean quantum k-NN: shots vs N, margin theory, end-to-end cost
    scripts/
      validate_against_authors.py        checks the probability model against the authors' circuit/log
      qknn_shots_study.py                shots needed vs N (all 4 estimators), bootstrap CI, --margins = margin theory
      controlled_dimension_experiment.py tests gamma = 2/D and b = 1 + 4/D (ambient vs intrinsic dimension, matched N grid)
      qknn_cost_model.py                 operation-count break-even vs classical k-NN (needs shots results)
    results/                              ALL outputs land here (shots/, controlled_dimension/, cost_model/, authors_log/)
  B_kernel_topology/                Quantum-kernel CNOT-topology pruning (FINISHED)
    qkernel_rigorous_v6.py, simulations2_v6_run_with_results.ipynb, RESULTS_SUMMARY.md
  _archive_superseded/              safe to delete once you have a backup (see table)
```

## Keep / archive
| File | Decision | Why |
|---|---|---|
| qknn_margin_scaling_study.py | KEEP, renamed `qknn_shots_study.py` | strict superset of v2 (adds delta_s, gamma fit, theory plots) |
| qknn_shots_study_v2.py | archive | superseded by the above |
| controlled_dimension_experiment_matched.py | KEEP, renamed `controlled_dimension_experiment.py` | fits gamma on the same N range as the shot study |
| controlled_dimension_experiment.py (original) | archive | has the N-range mismatch the matched version fixes |
| qknn_cost_model_2.py | KEEP, renamed `qknn_cost_model.py` | reports c vs c* instead of extrapolating one exponent |
| qknn_cost_model.py (v1) | archive | superseded; still cites the invalid v1 shot study |
| validate_against_authors.py | KEEP | validation step; needs the authors' `qknn_log.txt` (not uploaded) |
| qkernel_rigorous_v6.py + simulations2.ipynb | KEEP (project B) | the only protocol with tuned beta, leakage test, controls |
| Simulations.ipynb | archive | history of v1-v5 incl. an OOM crash at 16 qubits; superseded by v6 |
| qkernel_topology_benchmark.py, _v3, qkernel_greedy_synthetic.py, _v5 | archive | superseded |

## Run order for A (from the workspace root)
1. `python A_quantum_knn_cost_study/scripts/validate_against_authors.py` (put the authors' log in `results/authors_log/`)
2. `python A_quantum_knn_cost_study/scripts/qknn_shots_study.py --margins --n-test 100 --data-seeds 5 --trials 5`
3. `python A_quantum_knn_cost_study/scripts/controlled_dimension_experiment.py --mode both` (add `--skip-shots` for the cheap geometry stage)
4. `python A_quantum_knn_cost_study/scripts/qknn_cost_model.py --dataset blobs --criterion vote` (repeat for wine, digits, neighbours)

Scripts resolve `results/` from their own location, so the working directory does not matter. Override with `--outdir`.

## Do not quote
v1 shot-study numbers (wrong probability model); v6 GRID rows chosen by test accuracy; any project-B result at beta = 1.
