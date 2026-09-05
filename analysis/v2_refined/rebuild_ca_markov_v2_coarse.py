"""Re-runs CA-Markov a second time on the refined (NDVI+NDMI) classifications,
at the SAME 300x233 coarse grid train_convlstm_v2.py/v3.py train and validate
on. Mirrors ../experiments/exp1_ca_markov_coarse_grid.py exactly, just
swapping sequence_raw.pkl for sequence_raw_v2.pkl (refined labels).

Why this script exists: comparing refined CA-Markov's native-resolution
validation (rebuild_ca_markov_v2.py) directly against refined ConvLSTM's
necessarily coarse-grid validation (train_convlstm_v3.py) would repeat the
exact resolution confound Section III-H/IV-F of the paper diagnosed for the
original classification -- without ever checking whether that confound
recurs under the refined labels too. It does: Non-vegetated Producer's
Accuracy collapses from 46.3% (native) to 10.2% (coarse), the same pattern
found for the original classification (44.2% -> 8.4%). At that shared,
resolution-matched grid, refined ConvLSTM v3 still exceeds refined
CA-Markov on every metric -- and by more than the unmatched comparison had
suggested, so this fair re-run strengthens rather than undermines the
paper's ConvLSTM-Kappa claim.
"""
import os
import numpy as np
import pickle
import json
from scipy import stats as sp_stats
from scipy.linalg import fractional_matrix_power
from scipy.ndimage import uniform_filter

DATA_DIR = os.environ.get("LC_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
TARGET_SIZE = (300, 233)  # (W, H) -- must match train_convlstm_v2/v3.py exactly
NAMES = ["Air", "Non-vegetasi", "Veg. jarang", "Veg. rapat"]
N_CLASSES = 4

raw = pickle.load(open(f"{DATA_DIR}/sequence_raw_v2.pkl", "rb"))


def downsample_majority_vote(cls, valid, size):
    src_h, src_w = cls.shape
    dst_w, dst_h = size
    y_edges = np.linspace(0, src_h, dst_h + 1).round().astype(int)
    x_edges = np.linspace(0, src_w, dst_w + 1).round().astype(int)
    out = np.full((dst_h, dst_w), -1, dtype=np.int8)
    for iy in range(dst_h):
        y0, y1 = y_edges[iy], max(y_edges[iy] + 1, y_edges[iy + 1])
        for ix in range(dst_w):
            x0, x1 = x_edges[ix], max(x_edges[ix] + 1, x_edges[ix + 1])
            block = cls[y0:y1, x0:x1]
            block_valid = valid[y0:y1, x0:x1]
            vals = block[block_valid & (block >= 0)]
            if vals.size == 0:
                continue
            out[iy, ix] = int(sp_stats.mode(vals, keepdims=False).mode)
    return out


print("Downsampling REFINED 2019, 2021, 2024 classifications to 300x233 via majority vote...")
cls_coarse = {}
valid_coarse = {}
for y in (2019, 2021, 2024):
    cls_native = raw[y]["cls"]
    valid_native = raw[y]["valid"]
    c = downsample_majority_vote(cls_native, valid_native, TARGET_SIZE)
    cls_coarse[y] = c
    valid_coarse[y] = c >= 0
    dist = [int((c == i).sum()) for i in range(N_CLASSES)]
    print(f"  {y}: coarse-grid class counts = {dist}")

base_valid = valid_coarse[2019] & valid_coarse[2021]
c19, c21 = cls_coarse[2019], cls_coarse[2021]

counts19 = np.array([int(((c19 == i) & base_valid).sum()) for i in range(N_CLASSES)])
trans = np.zeros((N_CLASSES, N_CLASSES))
for i in range(N_CLASSES):
    mask_i = base_valid & (c19 == i)
    n_i = mask_i.sum()
    if n_i == 0:
        trans[i, i] = 1.0
        continue
    for j in range(N_CLASSES):
        trans[i, j] = ((c21 == j) & mask_i).sum() / n_i
print(f"\n2-year transition matrix (2019->2021, coarse grid, refined labels):\n{trans}")

P3 = fractional_matrix_power(trans, 1.5).real
P3 = np.clip(P3, 0, None)
P3 = P3 / P3.sum(axis=1, keepdims=True)
print(f"\nP3 = P21^1.5 (projected 3 years forward, coarse grid):\n{P3}")

cls21 = cls_coarse[2021]
counts = np.array([int(((cls21 == i) & base_valid).sum()) for i in range(N_CLASSES)])
print(f"\n2021 counts (coarse grid, within common-valid mask): {counts}")

quota = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        if i != j:
            quota[i, j] = int(round(counts[i] * P3[i, j]))
for i in range(N_CLASSES):
    off_diag_sum = sum(quota[i, j] for j in range(N_CLASSES) if j != i)
    quota[i, i] = counts[i] - off_diag_sum
print(f"Quota matrix (coarse grid):\n{quota}")

K = 7
density = np.zeros((N_CLASSES,) + cls21.shape, dtype=np.float32)
valid_f = base_valid.astype(np.float32)
valid_density = uniform_filter(valid_f, size=K, mode="constant", cval=0.0)
valid_density[valid_density == 0] = 1.0
for c in range(N_CLASSES):
    onehot = ((cls21 == c) & base_valid).astype(np.float32)
    density[c] = uniform_filter(onehot, size=K, mode="constant", cval=0.0) / valid_density

rng = np.random.default_rng(7)
sim2024_coarse = cls21.copy()
for i in range(N_CLASSES):
    ys, xs = np.where(base_valid & (cls21 == i))
    n_i = len(ys)
    if n_i == 0:
        continue
    targets = [j for j in range(N_CLASSES) if j != i and quota[i, j] > 0]
    if not targets:
        continue
    scores = np.stack([density[j][ys, xs] for j in targets], axis=1).astype(np.float32)
    scores += rng.uniform(0, 1e-3, size=scores.shape).astype(np.float32)
    flat_scores = scores.ravel()
    order = np.argsort(-flat_scores)
    n_targets = len(targets)
    assigned = np.zeros(n_i, dtype=bool)
    remaining = {j: quota[i, j] for j in targets}
    n_assigned, total_quota = 0, sum(remaining.values())
    pix_idx, tgt_idx = order // n_targets, order % n_targets
    for k in range(len(order)):
        if n_assigned >= total_quota:
            break
        p = pix_idx[k]
        if assigned[p]:
            continue
        j = targets[tgt_idx[k]]
        if remaining[j] <= 0:
            continue
        sim2024_coarse[ys[p], xs[p]] = j
        assigned[p] = True
        remaining[j] -= 1
        n_assigned += 1

np.save(f"{DATA_DIR}/ca_markov_v2_sim2024_coarse.npy", sim2024_coarse)

actual_coarse = cls_coarse[2024]
mask = base_valid & (actual_coarse >= 0) & (sim2024_coarse >= 0)
CM = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
for a in range(N_CLASSES):
    for p in range(N_CLASSES):
        CM[a, p] = int(((actual_coarse == a) & (sim2024_coarse == p) & mask).sum())

total = CM.sum()
OA = np.trace(CM) / total
row_sums, col_sums = CM.sum(axis=1), CM.sum(axis=0)
pe = (row_sums * col_sums).sum() / (total ** 2)
kappa = (OA - pe) / (1 - pe) if pe < 1 else float("nan")
PA = np.divide(np.diag(CM), row_sums, out=np.zeros(N_CLASSES), where=row_sums > 0)
UA = np.divide(np.diag(CM), col_sums, out=np.zeros(N_CLASSES), where=col_sums > 0)

print("\n" + "=" * 78)
print("REFINED CA-MARKOV @ COARSE 300x233 GRID -- results (compare directly")
print("against Table 17's pooled ConvLSTM v3 metrics, since both are now on")
print("the SAME grid and the SAME refined NDVI+NDMI labels)")
print("=" * 78)
print(f"Confusion matrix:\n{CM}")
print(f"n = {total}")
print(f"Overall Accuracy: {OA*100:.2f}%   Kappa: {kappa:.4f}")
print(f"PA per class {NAMES}: {[f'{v*100:.1f}%' for v in PA]}")
print(f"UA per class {NAMES}: {[f'{v*100:.1f}%' for v in UA]}")

json.dump(dict(OA=float(OA), kappa=float(kappa), PA=PA.tolist(), UA=UA.tolist(),
               n=int(total), CM=CM.tolist(), grid="300x233_majority_vote_refined"),
          open(f"{DATA_DIR}/ca_markov_v2_coarse_metrics.json", "w"), indent=2)
print("\nSaved: ca_markov_v2_sim2024_coarse.npy, ca_markov_v2_coarse_metrics.json")
