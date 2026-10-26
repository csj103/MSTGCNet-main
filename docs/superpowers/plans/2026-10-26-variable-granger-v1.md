# Variable Granger V1 implementation plan

Goal: implement the user's E0--E5 specification in the supplied source directory,
using remote V12 commit 5382d528ff93d03f5a30e3b34563c4e70803f5e0.
The directory was a clean GitHub source archive; Git was initialized and the
remote baseline index attached without changing any working files.
All work stays on exp/variable-granger-v1; no push, no DTSGAD changes.

Architecture: retain the original Model implementation as E0. Add a variable
model selected through Model's constructor, with shared scalar temporal encoding
and identity embeddings. Reuse router DFT, trend, noise, Top-K, sparse dispatch,
balance loss and expert patch attention/FFN design. E1 uses the original fixed
embedding binary adjacency. E2 uses input Q/K. E3 uses a fixed historical prior.
E4/E5 add the prior to Q/K logits; E5 jointly permutes prior rows and columns.
Patch-major nodes have index patch*V+variable. Prior rows are targets. Prior bias
is historical only and limited by overlap with the fitted lag support. Same-patch
dynamic edges are associations, not instantaneous causal claims.

The detailed user specification authorizes implementation and CPU verification;
execute inline through the following checkpoints without extra approval gates.

- [x] 1. Read model, entry point, loader, preprocessing, scoring, threshold code;
  generate the existing fine_grained split; run existing focused baseline tests.
- [x] 2. Write failing tests in tests/test_variable_granger.py for all six modes,
  variable isolation, patch roundtrip/padding, adjacency direction, gradients,
  zero-prior equivalence, reproducibility and E0 compatibility.
- [x] 3. Add models/variable_granger.py and layers/GrangerPrior.py. Adapt only
  routing-feature extraction, sparse weight broadcasting and Model dispatch in
  models/MSTGCNet.py. Add validated CLI/fingerprint settings in run.py.
- [x] 4. Write synthetic directional VAR tests; implement
  scripts/build_granger_prior.py using train.csv/train_meta.csv only, conditional
  OLS nested model F-tests, per-window multiple-testing correction, stability
  aggregated equally across segments and source flights. Audit sampling and
  explicitly declare row-lag versus regular-time semantics. Save ordered matrix,
  source hashes, metadata, per-window/flight stability, edges and heatmap.
- [x] 5. Add scripts/check_variable_granger.py to verify all-mode CPU forward and
  backward, parameter counts, timings, real normal-window small training for
  E1--E4, validation reconstruction, routing and graph snapshots.
- [x] 6. Keep the loss, causal_last scoring and ATSSD unchanged. Extend evaluation
  reporting with machine-readable metrics, per-event delay and benchmark/graph
  artifacts. Add a Windows CMD experiment launcher for E0--E5.
- [x] 7. Run focused new tests, original tests, baseline bit-exact comparison,
  real prior extraction, all-mode smoke and small normal training. Review data
  leakage, graph direction, Q/K gradients, bounded memory and documentation.
- [x] 8. Deliver docs/variable_granger_v1.md with exact commands, measured results,
  limitations and unverified GPU/full-training status. Leave work on experiment
  branch, never push or modify the remote baseline.

Validation commands (Windows CMD):

```bat
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
python -m pytest tests/test_variable_granger.py tests/test_granger_prior.py -q
python scripts/build_granger_prior.py --help
python scripts/check_variable_granger.py --help
```

Major constraints: L=96, V=10, d_var=16 or 32, 3 GMoE layers, 4 experts/layer,
original patch sizes, top_k=3, original balance/loss/threshold behavior. Neither
smoke losses nor validation reconstruction establish anomaly-detection gains.

Final evidence: 57 full-suite tests passed; all-mode CPU smoke and E1--E4 bounded normal training completed. GPU and full anomaly evaluation remain explicitly unverified. See docs/variable_granger_v1.md.
