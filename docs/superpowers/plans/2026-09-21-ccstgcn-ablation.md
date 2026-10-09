# CC-STGCN graph propagation ablation

Approved scope: option A. Retain patching, temporal attention, FFN, routing,
training loss, data split and ATSSD. Reuse the original abl_GCN branch; do not
repair unrelated baseline issues or change parameter initialization.

Implementation plan (inline execution):
- Add regression tests for an explicit --without_graph alias, original baseline
  setting compatibility, distinct no_graph output names, graph independence,
  retained temporal/FFN gradients and end-to-end reconstruction backward.
- Observe the new CLI tests fail before implementing the alias.
- Extend run.py only: constrain abl_GCN to 0/1, alias --without_graph to 1,
  append no_graph to ablation settings only, and print the selected experiment.
- Document baseline and ablation commands, retraining, retained parameters,
  and the limits of comparison against the original graph implementation.
- Run the focused tests, existing suite and compare baseline outputs against
  Git HEAD using identical weights/inputs. Review the final diff.

No full experiment training, automatic commits or existing result replacement.

Validation completed:
- New tests initially reported 2 expected failures and 1 pass; after implementation
  all 3 pass. The complete tests directory reports 22 passed.
- Default resolved arguments and setting match Git HEAD exactly; identical seed
  and input produce bitwise-equal baseline reconstruction and auxiliary loss.
- Independent read-only review found no actionable issues; git diff --check passes.
