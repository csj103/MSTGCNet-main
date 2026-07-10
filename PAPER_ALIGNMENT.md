# Paper Alignment Status

This project is a paper-faithful reconstruction, not the authors' complete
implementation. The public repository still contains placeholder `pass`
implementations for the core GMoE and graph modules.

## Implemented From Explicit Paper Equations

- Eq. (1)-(3): RevIN, learned Conv1D scale, positional encoding, and residual
  GMoE blocks.
- Eq. (6)-(12): channel-wise top-k Fourier reconstruction, multi-kernel trend
  extraction, noisy sparse top-k routing, and weighted expert aggregation.
- Eq. (13)-(23): variable projection, zero padding, patching, shared intra-patch
  MHA, learned patch embeddings, causal top-k binary graph, normalized graph
  convolution, unpatching, reconstruction projection, and inverse RevIN.
- Eq. (24)-(31): per-window squared reconstruction loss, straight-through
  binary-gate balance loss, and ATSSD with `L=96`, `alpha=0.01`.
- Table IV: optimizer and all listed learning/structure defaults.

## Not Uniquely Specified

- Eq. (8) does not specify the output shape or reduction used by `L1`.
- Eq. (10) maps a `dm x L` tensor to `K` weights without defining the temporal
  reduction; this implementation projects channels to one value and learns a
  linear map over all `L` timestamps.
- Eq. (15) does not disclose the internal MHA projection. This implementation
  preserves node identity by reshaping `Pin` to `[B*m*Ns,Lp,1]`, projecting
  each scalar timestamp to `dm`, applying shared intra-patch temporal MHA, and
  projecting back to `[B,m*Ns,Lp]` before graph convolution.
- Fig. 7 confirms first-block patch sizes `[8, 12, 16, 32]`. The configured
  second and third blocks use `[6, 8, 12, 16]` and `[2, 6, 8, 12]` as an
  explicit coarse-to-fine reproduction assumption; the paper does not state
  those two assignments directly.
- The activation in Eq. (21), balance coefficient `lambda`, padding modes, and
  several initialization details are not reported.
- The current implementation has about 0.26M trainable parameters, while
  Table VII reports 3.2451M. The missing complete expert implementation and
  parameter accounting cannot be recovered from the paper alone.
- The exact ALFA flight split and resampling procedure are not released. After
  excluding `no_ground_truth`, train/validation counts cannot equal Table III.
- Point adjustment is absent from the paper's evaluation section, although the
  released experiment scaffold applies it before reporting metrics.

Exact numerical reproduction therefore requires the authors' completed source,
processed split, or clarification of the items above.
