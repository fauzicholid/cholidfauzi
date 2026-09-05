"""Rebuilds the CA-Markov transition matrix, 2025 projection, and
out-of-sample validation (../ca_markov.py / ../ca_validate.py) in full on
the refined NDVI+NDMI classification (build_sequence_dataset_v2.py),
using the exact same capping methodology as the original pipeline: the
Non-vegetated<->Water cap is applied to the ANNUALIZED rate for the main
5-year matrix but to the RAW 2-year rate for the validation matrix -- the
cap point differs between the two on purpose, matching how the original,
pre-refinement pipeline was published.
"""
import time, pickle, json, os
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.ndimage import uniform_filter

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

DATA_DIR = os.environ.get("LC_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
N_CLASSES = 4
NAMES = ["Air", "Non-vegetasi", "Veg. jarang", "Veg. rapat"]

log("Loading v2 sequence...")
raw = pickle.load(open(f"{DATA_DIR}/sequence_raw_v2.pkl", "rb"))
valid24 = np.load(f"{DATA_DIR}/valid.npy")
cls24 = raw[2024]["cls"]  # == cls24_improved.npy
cls19 = raw[2019]["cls"]
cls21 = raw[2021]["cls"]

def transition_matrix(cls_from, cls_to, base_valid):
    """Raw (uncapped) transition matrix -- capping is applied separately,
    at the point in the pipeline matching the paper's established methodology
    (annualized rate for the main matrix, raw 2-year rate for the validation
    matrix -- see the cap_matrix() calls below)."""
    counts_from = np.array([int(((cls_from == i) & base_valid).sum()) for i in range(N_CLASSES)])
    T = np.zeros((N_CLASSES, N_CLASSES))
    for i in range(N_CLASSES):
        mask_i = base_valid & (cls_from == i)
        n_i = mask_i.sum()
        if n_i == 0:
            T[i, i] = 1.0
            continue
        for j in range(N_CLASSES):
            T[i, j] = ((cls_to == j) & mask_i).sum() / n_i
    return T, counts_from

def cap_matrix(T, i, j, cap):
    """Cap T[i,j] at `cap`, redistributing the excess probability mass to
    T[i,i] (self-persistence) -- the disclosed domain correction for the
    Non-vegetated<->Water wet-paddy/bare-land spectral confusion."""
    T = T.copy()
    raw_rate = T[i, j]
    if T[i, j] > cap:
        excess = T[i, j] - cap
        T[i, j] = cap
        T[i, i] += excess
    return T, raw_rate

# ==== MAIN PROJECTION: 2019 -> 2024 (5-year), annualize, project 2025 ====
log("\n=== 5-year transition matrix (2019->2024) ===")
base_valid_5y = valid24 & (cls19 >= 0) & (cls24 >= 0)
log(f"n = {int(base_valid_5y.sum())}")
T5, counts19 = transition_matrix(cls19, cls24, base_valid_5y)
log(f"T5 (5-year, raw/uncapped):\n{T5}")

P1_raw = fractional_matrix_power(T5, 1/5).real
P1_raw = np.clip(P1_raw, 0, None)
P1_raw = P1_raw / P1_raw.sum(axis=1, keepdims=True)
log(f"P1_raw (annualized, uncapped):\n{P1_raw}")
P1, raw_rate_annual = cap_matrix(P1_raw, 1, 0, 0.01)
log(f"Raw annualized Non-vegetasi->Air rate: {raw_rate_annual*100:.2f}%/year (capped to 1%/year)")
log(f"P1 (annualized, capped):\n{P1}")

log("\n=== CA spatial allocation for 2025 (v2) ===")
counts24 = np.array([int((cls24 == i).sum()) for i in range(N_CLASSES)])
quota = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        if i != j:
            quota[i, j] = int(round(counts24[i] * P1[i, j]))
for i in range(N_CLASSES):
    off_diag_sum = sum(quota[i, j] for j in range(N_CLASSES) if j != i)
    quota[i, i] = counts24[i] - off_diag_sum
log(f"Quota matrix:\n{quota}")

K = 7
density = np.zeros((N_CLASSES,) + cls24.shape, dtype=np.float32)
valid_f = valid24.astype(np.float32)
valid_density = uniform_filter(valid_f, size=K, mode="constant", cval=0.0)
valid_density[valid_density == 0] = 1.0
for c in range(N_CLASSES):
    onehot = ((cls24 == c) & valid24).astype(np.float32)
    density[c] = uniform_filter(onehot, size=K, mode="constant", cval=0.0) / valid_density

rng = np.random.default_rng(42)
proj2025 = cls24.copy()
for i in range(N_CLASSES):
    ys, xs = np.where(valid24 & (cls24 == i))
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
        proj2025[ys[p], xs[p]] = j
        assigned[p] = True
        remaining[j] -= 1
        n_assigned += 1
    log(f"  class {NAMES[i]}: assigned {n_assigned}/{total_quota} (pool {n_i})")

np.save(f"{DATA_DIR}/proj2025_v2.npy", proj2025)
px_area_km2 = 35522.3 / valid24.sum()  # matches paper's stated total AOI area
counts2025 = np.array([int((proj2025 == i).sum()) for i in range(N_CLASSES)])
log("\n2024 vs 2025 (v2) areas (km^2):")
for i, name in enumerate(NAMES):
    a24 = counts24[i] * px_area_km2
    a25 = counts2025[i] * px_area_km2
    log(f"  {name:15s} 2024={a24:9.1f}  2025={a25:9.1f}  delta={a25-a24:+8.1f} ({100*(a25-a24)/a24:+.1f}%)")

# ==== VALIDATION: 2019 -> 2021 (2-year), project 3 years forward to simulate 2024 ====
log("\n\n=== Validation: 2-year transition matrix (2019->2021) ===")
base_valid_2y = valid24 & (cls19 >= 0) & (cls21 >= 0)
log(f"n = {int(base_valid_2y.sum())}")
T2_raw, counts19b = transition_matrix(cls19, cls21, base_valid_2y)
log(f"T2 (2-year, raw/uncapped):\n{T2_raw}")
T2, raw_rate_2y = cap_matrix(T2_raw, 1, 0, 0.02)
log(f"Raw Non-vegetasi->Air rate: {raw_rate_2y*100:.2f}%/2yr (capped to 2%/2yr)")
log(f"T2 (2-year, capped):\n{T2}")

P3 = fractional_matrix_power(T2, 1.5).real
P3 = np.clip(P3, 0, None)
P3 = P3 / P3.sum(axis=1, keepdims=True)
log(f"P3 = T2^1.5 (projected 3yr forward):\n{P3}")

log("\nCA allocation: simulate 2024 from 2021 using P3...")
base_valid21 = valid24 & (cls21 >= 0)
counts21 = np.array([int(((cls21 == i) & base_valid21).sum()) for i in range(N_CLASSES)])
quota_v = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        if i != j:
            quota_v[i, j] = int(round(counts21[i] * P3[i, j]))
for i in range(N_CLASSES):
    off_diag_sum = sum(quota_v[i, j] for j in range(N_CLASSES) if j != i)
    quota_v[i, i] = counts21[i] - off_diag_sum

density_v = np.zeros((N_CLASSES,) + cls21.shape, dtype=np.float32)
valid_f21 = base_valid21.astype(np.float32)
valid_density21 = uniform_filter(valid_f21, size=K, mode="constant", cval=0.0)
valid_density21[valid_density21 == 0] = 1.0
for c in range(N_CLASSES):
    onehot = ((cls21 == c) & base_valid21).astype(np.float32)
    density_v[c] = uniform_filter(onehot, size=K, mode="constant", cval=0.0) / valid_density21

rng2 = np.random.default_rng(7)
sim2024 = cls21.copy()
for i in range(N_CLASSES):
    ys, xs = np.where(base_valid21 & (cls21 == i))
    n_i = len(ys)
    if n_i == 0:
        continue
    targets = [j for j in range(N_CLASSES) if j != i and quota_v[i, j] > 0]
    if not targets:
        continue
    scores = np.stack([density_v[j][ys, xs] for j in targets], axis=1).astype(np.float32)
    scores += rng2.uniform(0, 1e-3, size=scores.shape).astype(np.float32)
    flat_scores = scores.ravel()
    order = np.argsort(-flat_scores)
    n_targets = len(targets)
    assigned = np.zeros(n_i, dtype=bool)
    remaining = {j: quota_v[i, j] for j in targets}
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
        sim2024[ys[p], xs[p]] = j
        assigned[p] = True
        remaining[j] -= 1
        n_assigned += 1

np.save(f"{DATA_DIR}/sim2024_v2.npy", sim2024)

log("\nValidating sim2024 (v2) against actual cls24 (v2)...")
mask = valid24 & (cls24 >= 0) & (sim2024 >= 0)
CM = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
for a in range(N_CLASSES):
    for p in range(N_CLASSES):
        CM[a, p] = int(((cls24 == a) & (sim2024 == p) & mask).sum())
total = CM.sum()
OA = np.trace(CM) / total
row_sums, col_sums = CM.sum(axis=1), CM.sum(axis=0)
pe = (row_sums * col_sums).sum() / (total ** 2)
kappa = (OA - pe) / (1 - pe) if pe < 1 else float("nan")
PA = np.divide(np.diag(CM), row_sums, out=np.zeros(N_CLASSES), where=row_sums > 0)
UA = np.divide(np.diag(CM), col_sums, out=np.zeros(N_CLASSES), where=col_sums > 0)
log(f"Confusion matrix:\n{CM}")
log(f"n={total}  OA={OA*100:.2f}%  Kappa={kappa:.4f}")
log(f"PA: {[f'{v*100:.1f}%' for v in PA]}")
log(f"UA: {[f'{v*100:.1f}%' for v in UA]}")

# FoM (Pontius): hits = correctly predicted change; misses = actual change not predicted (or predicted wrong);
# false alarms = predicted change where actual persisted.
actual_change = (cls21 != cls24) & mask
sim_change = (cls21 != sim2024) & mask
hits = int((actual_change & sim_change & (sim2024 == cls24)).sum())
misses = int((actual_change & ~(sim_change & (sim2024 == cls24))).sum())
false_alarms = int((~actual_change & sim_change).sum())
fom = hits / (hits + misses + false_alarms) if (hits+misses+false_alarms) > 0 else float("nan")
log(f"FoM: hits={hits} misses={misses} false_alarms={false_alarms} FoM={fom*100:.2f}%")
log(f"Reference change %: {100*actual_change.sum()/mask.sum():.2f}%  Simulated change %: {100*sim_change.sum()/mask.sum():.2f}%")

json.dump(dict(
    T5_raw=T5.tolist(), P1_raw=P1_raw.tolist(), raw_rate_annual=raw_rate_annual, P1=P1.tolist(),
    T2_raw=T2_raw.tolist(), T2_capped=T2.tolist(), raw_rate_2y=raw_rate_2y, P3=P3.tolist(),
    counts24=counts24.tolist(), counts2025=counts2025.tolist(), px_area_km2=px_area_km2,
    validation=dict(OA=float(OA), kappa=float(kappa), PA=PA.tolist(), UA=UA.tolist(),
                     CM=CM.tolist(), n=int(total), fom=float(fom), hits=hits, misses=misses,
                     false_alarms=false_alarms,
                     ref_change_pct=float(100*actual_change.sum()/mask.sum()),
                     sim_change_pct=float(100*sim_change.sum()/mask.sum())),
), open(f"{DATA_DIR}/ca_markov_v2_results.json", "w"), indent=2)
log("\nSaved ca_markov_v2_results.json, proj2025_v2.npy, sim2024_v2.npy")
