# Follow-up experiments: diagnosing the ConvLSTM Non-vegetated collapse

The original comparison (`../ca_validate.py` vs. `../train_convlstm.py`)
found CA-Markov out-performing ConvLSTM on the Non-vegetated class despite
ConvLSTM's higher overall accuracy. These four scripts test three candidate
explanations — a label-downsampling artifact, a resolution mismatch between
the two models, and unweighted-loss class imbalance — and quantify
statistical significance. Run in order; each depends on the previous one's
output (see each script's own docstring for exact dependencies). All read
from `LC_DATA_DIR` (defaults to `../data`, same convention as the parent
`analysis/` scripts).

```
python exp0_diagnose_label_downsampling.py
python exp1_ca_markov_coarse_grid.py
python exp2_train_convlstm_fixed.py
python exp3_significance_tests.py
```

## What each one does, and what it found

**`exp0_diagnose_label_downsampling.py`** — Compares three ways of
downsampling the native-resolution 2024 classification to ConvLSTM's
300x233 training grid (nearest-neighbor subsampling, block-wise majority
vote, and reclassifying a bilinear-downsampled NDVI). *Finding:* the
nearest-neighbor method `train_convlstm.py` actually uses is not the worst
of the three for Non-vegetated pixel retention — if anything it retains the
most (6.10% share vs. 4.15%/3.50% for the alternatives). Label-downsampling
method alone does not explain the collapse.

**`exp1_ca_markov_coarse_grid.py`** — Re-runs the same CA-Markov transition-
matrix-plus-neighborhood-allocation logic as `../ca_validate.py`, but on
majority-vote-downsampled classifications at ConvLSTM's own 300x233 grid
instead of the native analysis grid. *Finding:* CA-Markov's Non-vegetated
Producer's Accuracy collapses too at this resolution (44.2% at native
resolution -> 8.4% here) — the resolution mismatch between the two models
in the original comparison was a real confound, not just a theoretical one.

**`exp2_train_convlstm_fixed.py`** — Retrains ConvLSTM under 4 configurations
(baseline / +corrected label downsampling / +inverse-frequency class-weighted
loss / +both), 5 seeds each, identical architecture and epoch budget to the
original for comparability. *Finding:*

| Config | OA | Kappa | PA[Non-vegetated] |
|---|---|---|---|
| baseline (original) | 75.16±0.48% | 0.445±0.027 | 2.58±3.06% |
| + fix_labels only | 82.40±0.67% | 0.613±0.018 | 8.72±1.76% |
| + fix_weight only | 63.82±1.91% | 0.374±0.024 | 52.35±3.04% |
| + fix_labels + fix_weight | 74.21±2.39% | 0.520±0.031 | 59.56±3.42% |

The baseline collapse replicates robustly across seeds (not a one-off run).
Class-weighting the loss, combined with correct label downsampling, resolves
most of the collapse (PA recovers to ~60%) while Kappa *improves* over the
unweighted baseline — evidence the original collapse was substantially a
fixable training artifact rather than an inherent data/architecture limit.

**`exp3_significance_tests.py`** — Bootstrap 95% CIs and McNemar's paired
test comparing CA-Markov (coarse-grid, from exp1) against ConvLSTM, on the
subset of pixels where both pipelines' ground truth agrees. *Finding:* using
the **original, unfixed** ConvLSTM prediction, ConvLSTM's overall OA is
significantly higher than CA-Markov's at this matched resolution (85.64% vs.
75.39%, non-overlapping CIs), but CA-Markov remains significantly better on
Non-vegetated specifically (McNemar p = 5.55e-16 restricted to Non-vegetated
ground truth) — statistically backing the collapse claim.

**Known gap:** exp3 only tests the *original* unfixed ConvLSTM against
CA-Markov, because exp2 saves only aggregate per-config metrics, not
per-pixel prediction arrays for a specific seed. A significance test of
CA-Markov vs. the *fixed* ConvLSTM (the comparison that now matters most,
given exp2's results) has not yet been run.
