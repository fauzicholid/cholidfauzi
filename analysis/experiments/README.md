# Follow-up experiments: diagnosing the ConvLSTM Non-vegetated collapse

The original comparison (`../ca_validate.py` vs. `../train_convlstm.py`)
found CA-Markov out-performing ConvLSTM on the Non-vegetated class despite
ConvLSTM's higher overall accuracy. These five scripts test three candidate
explanations — a label-downsampling artifact, a resolution mismatch between
the two models, and unweighted-loss class imbalance — and quantify
statistical significance, including for the corrected model. Run in order;
each depends on the previous one's output (see each script's own docstring
for exact dependencies). All read from `LC_DATA_DIR` (defaults to `../data`,
same convention as the parent `analysis/` scripts).

```
python exp0_diagnose_label_downsampling.py
python exp1_ca_markov_coarse_grid.py
python exp2_train_convlstm_fixed.py
python exp3_significance_tests.py
python exp4_fixed_convlstm_vs_camarkov.py
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

**`exp4_fixed_convlstm_vs_camarkov.py`** — Closes the gap left by exp3:
retrains ConvLSTM under the winning combined configuration from exp2
(corrected label downsampling + class-weighted loss) across the same 5
seeds, saves each seed's per-pixel 2024 prediction, and repeats exp3's
bootstrap-CI + McNemar procedure against CA-Markov's coarse-grid prediction
(from exp1) for every seed. *Finding:* decisive and consistent —

| Seed | OA — CA-Markov | OA — Fixed ConvLSTM | PA[Non-veg] — CA-Markov | PA[Non-veg] — Fixed ConvLSTM | McNemar p (Non-veg) |
|---|---|---|---|---|---|
| 0 | 74.37% | 75.13% | 4.71% | 62.40% | ≈0 |
| 1 | 74.38% | 79.51% | 4.70% | 64.69% | ≈0 |
| 2 | 74.38% | 81.27% | 4.71% | 56.42% | ≈0 |
| 3 | 74.38% | 74.64% | 4.71% | 63.29% | ≈0 |
| 4 | 74.38% | 76.85% | 4.71% | 66.53% | ≈0 |

The fixed ConvLSTM exceeds CA-Markov's Non-vegetated PA in 5/5 seeds,
significantly so in 5/5 (McNemar restricted to Non-vegetated ground truth,
p ≈ 0 every time). Overall OA also favors the fixed ConvLSTM in 5/5 seeds,
significant in 4/5 (seed 3 was a near-tie on OA, p = 0.29, though its
Non-vegetated-restricted difference was still significant). CA-Markov's
PA[Non-veg] here (~4.7%) differs from exp1's own figure (8.4%) because this
test restricts to the stricter subset of pixels where both pipelines' ground
truth agrees, matching exp3's methodology — not a discrepancy.

Once resolution is matched and the loss is class-weighted, ConvLSTM does not
just recover from its initial collapse — it significantly and consistently
outperforms CA-Markov on the exact class the whole comparison turns on. The
original "CA-Markov is more defensible" conclusion does not survive this
diagnostic chain.
