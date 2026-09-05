"""EXPERIMENT 3 — Statistical significance: bootstrap CIs + McNemar's test.

Requires exp1 (ca_markov_coarse_grid.py) to have been run first, so that
CA-Markov's prediction lives on the SAME 300x233 grid as ConvLSTM's
prediction (convlstm_pred2024.npy from the original train_convlstm.py run,
or a fixed version from exp2). Comparing two models pixel-by-pixel with
McNemar's test is only valid when they were evaluated on literally the same
pixels -- which is exactly why exp1 had to happen first.

Produces, for OA and for Non-vegetated PA specifically:
  - percentile bootstrap 95% CI (piksel-level resample, 2000 iterations)
  - McNemar's test (paired, since both models are scored on the same pixels)
"""
import numpy as np
import json
import os
from scipy.stats import chi2

DATA_DIR = os.environ.get("LC_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
N_CLASSES = 4
NAMES = ["Air", "Non-vegetasi", "Veg. jarang", "Veg. rapat"]
N_BOOT = 2000
RNG = np.random.default_rng(0)

# ---- Load CA-Markov (coarse grid, from exp1) ----
ca_pred = np.load(f"{DATA_DIR}/ca_markov_sim2024_coarse.npy")
ca_actual = np.load(f"{DATA_DIR}/ca_markov_coarse_actual2024.npy")
ca_mask = np.load(f"{DATA_DIR}/ca_markov_coarse_mask.npy")

# ---- Load ConvLSTM (original run, same 300x233 grid) ----
cl_pred = np.load(f"{DATA_DIR}/convlstm_pred2024.npy")
cl_actual = np.load(f"{DATA_DIR}/convlstm_actual2024_ds.npy")
cl_mask = np.load(f"{DATA_DIR}/convlstm_test_mask.npy")

# Sanity check: shapes must match, and ideally actual/mask should agree
# between the two pipelines (same year, same grid). If they don't exactly
# agree (e.g. different valid-pixel definitions), restrict to intersection.
assert ca_pred.shape == cl_pred.shape, (
    f"Grid mismatch: CA-Markov {ca_pred.shape} vs ConvLSTM {cl_pred.shape}. "
    f"Did you run exp1 with the exact same TARGET_SIZE as train_convlstm.py?"
)
common_mask = ca_mask & cl_mask & (ca_actual == cl_actual)  # only compare where both agree on ground truth
n_disagree_actual = int((ca_mask & cl_mask & (ca_actual != cl_actual)).sum())
if n_disagree_actual > 0:
    print(f"WARNING: {n_disagree_actual} pixels have different 'actual' labels between "
          f"the CA-Markov and ConvLSTM pipelines (likely due to different label-"
          f"downsampling methods -- majority-vote vs NEAREST). Restricting comparison "
          f"to the {common_mask.sum()} pixels where both pipelines agree on ground truth.")

ys, xs = np.where(common_mask)
n = len(ys)
print(f"Comparing on {n} common, ground-truth-agreeing pixels.\n")

actual = ca_actual[ys, xs]
pred_ca = ca_pred[ys, xs]
pred_cl = cl_pred[ys, xs]

correct_ca = (pred_ca == actual)
correct_cl = (pred_cl == actual)


def oa(correct):
    return correct.mean()


def pa_for_class(actual, pred, cls):
    is_cls = actual == cls
    if is_cls.sum() == 0:
        return float("nan")
    return (pred[is_cls] == cls).mean()


# ---- Bootstrap CIs ----
def bootstrap_ci(stat_fn, n_boot=N_BOOT):
    idx_all = np.arange(n)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.choice(idx_all, size=n, replace=True)
        stats[b] = stat_fn(idx)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return stats.mean(), lo, hi


print("=" * 78)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS (2000 resamples)")
print("=" * 78)

oa_ca_mean, oa_ca_lo, oa_ca_hi = bootstrap_ci(lambda idx: oa(correct_ca[idx]))
oa_cl_mean, oa_cl_lo, oa_cl_hi = bootstrap_ci(lambda idx: oa(correct_cl[idx]))
print(f"CA-Markov  OA: {oa_ca_mean*100:.2f}%  [95% CI {oa_ca_lo*100:.2f}% - {oa_ca_hi*100:.2f}%]")
print(f"ConvLSTM   OA: {oa_cl_mean*100:.2f}%  [95% CI {oa_cl_lo*100:.2f}% - {oa_cl_hi*100:.2f}%]")
overlap = not (oa_ca_hi < oa_cl_lo or oa_cl_hi < oa_ca_lo)
print(f"  -> CIs {'OVERLAP (difference may not be significant)' if overlap else 'DO NOT overlap (difference likely significant)'}")

print()
for cls in range(N_CLASSES):
    def pa_ca(idx, cls=cls):
        return pa_for_class(actual[idx], pred_ca[idx], cls)
    def pa_cl(idx, cls=cls):
        return pa_for_class(actual[idx], pred_cl[idx], cls)
    m_ca, lo_ca, hi_ca = bootstrap_ci(pa_ca)
    m_cl, lo_cl, hi_cl = bootstrap_ci(pa_cl)
    print(f"PA[{NAMES[cls]:14s}]  CA-Markov: {m_ca*100:5.2f}% [{lo_ca*100:5.2f}-{hi_ca*100:5.2f}%]   "
          f"ConvLSTM: {m_cl*100:5.2f}% [{lo_cl*100:5.2f}-{hi_cl*100:5.2f}%]")

# ---- McNemar's test (paired, on OA-level correctness) ----
print("\n" + "=" * 78)
print("McNEMAR'S TEST (paired) -- is one model's correctness pattern significantly")
print("different from the other's, pixel-by-pixel?")
print("=" * 78)

# 2x2 contingency: CA correct/incorrect x ConvLSTM correct/incorrect
b = int(((correct_ca) & (~correct_cl)).sum())   # CA right, ConvLSTM wrong
c = int(((~correct_ca) & (correct_cl)).sum())   # CA wrong, ConvLSTM right
print(f"Pixels where CA-Markov correct, ConvLSTM wrong (b): {b}")
print(f"Pixels where CA-Markov wrong, ConvLSTM correct (c): {c}")

if b + c == 0:
    print("b + c = 0, McNemar's test undefined (models agree on every disagreement pixel).")
else:
    # continuity-corrected McNemar statistic
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(stat, df=1)
    print(f"McNemar chi2 (continuity-corrected) = {stat:.3f}, p-value = {p_value:.4g}")
    print(f"  -> {'SIGNIFICANT difference (p < 0.05)' if p_value < 0.05 else 'NOT significant at alpha=0.05'}")

# Also run McNemar's test restricted to Non-vegetated ground-truth pixels only,
# since that's the class the whole paper's argument hinges on.
is_nonveg = actual == 1
if is_nonveg.sum() > 0:
    b_nv = int(((correct_ca) & (~correct_cl) & is_nonveg).sum())
    c_nv = int(((~correct_ca) & (correct_cl) & is_nonveg).sum())
    print(f"\n--- Restricted to Non-vegetated ground truth (n={int(is_nonveg.sum())}) ---")
    print(f"CA-Markov correct, ConvLSTM wrong (b): {b_nv}")
    print(f"CA-Markov wrong, ConvLSTM correct (c): {c_nv}")
    if b_nv + c_nv > 0:
        stat_nv = (abs(b_nv - c_nv) - 1) ** 2 / (b_nv + c_nv)
        p_nv = 1 - chi2.cdf(stat_nv, df=1)
        print(f"McNemar chi2 = {stat_nv:.3f}, p-value = {p_nv:.4g}")
        print(f"  -> {'SIGNIFICANT' if p_nv < 0.05 else 'NOT significant'} -- this is the number to "
              f"report in the paper to back up the Non-vegetated-collapse claim statistically.")

results = dict(
    n_common_pixels=n,
    oa_ca=dict(mean=oa_ca_mean, ci95=[oa_ca_lo, oa_ca_hi]),
    oa_convlstm=dict(mean=oa_cl_mean, ci95=[oa_cl_lo, oa_cl_hi]),
    mcnemar_overall=dict(b=b, c=c),
)
json.dump(results, open(f"{DATA_DIR}/exp3_significance_results.json", "w"), indent=2)
print(f"\nSaved exp3_significance_results.json")
