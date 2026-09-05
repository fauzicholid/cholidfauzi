"""Cellular Automata-Markov Chain (CA-Markov) spatial allocation: applies the
annualized, corrected transition matrix P1_adjusted.npy to the 2024
classification to project 2025 land cover.
"""
import time
import numpy as np
from scipy.ndimage import uniform_filter
import os

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

DATA_DIR = os.environ.get("LC_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

cls24 = np.load(f"{DATA_DIR}/cls24_4.npy")
valid24 = np.load(f"{DATA_DIR}/valid.npy")
P1 = np.load(f"{DATA_DIR}/P1_adjusted.npy")
n_classes = 4
names = ["Air", "Non-vegetasi", "Veg. jarang", "Veg. rapat"]

counts = np.array([int((cls24 == i).sum()) for i in range(n_classes)])
log(f"2024 counts: {counts}")

quota = np.zeros((n_classes, n_classes), dtype=np.int64)
for i in range(n_classes):
    for j in range(n_classes):
        if i != j:
            quota[i, j] = int(round(counts[i] * P1[i, j]))
for i in range(n_classes):
    off_diag_sum = sum(quota[i, j] for j in range(n_classes) if j != i)
    quota[i, i] = counts[i] - off_diag_sum
log(f"Quota matrix:\n{quota}")

K = 7
density = np.zeros((n_classes,) + cls24.shape, dtype=np.float32)
valid_f = valid24.astype(np.float32)
valid_density = uniform_filter(valid_f, size=K, mode="constant", cval=0.0)
valid_density[valid_density == 0] = 1.0
for c in range(n_classes):
    onehot = ((cls24 == c) & valid24).astype(np.float32)
    density[c] = uniform_filter(onehot, size=K, mode="constant", cval=0.0) / valid_density
log("density computed")

rng = np.random.default_rng(42)
proj2025 = cls24.copy()

for i in range(n_classes):
    ys, xs = np.where(valid24 & (cls24 == i))
    n_i = len(ys)
    if n_i == 0:
        continue
    targets = [j for j in range(n_classes) if j != i and quota[i, j] > 0]
    if not targets:
        log(f"class {names[i]}: no outgoing quota")
        continue

    # vectorized score matrix: (n_i, n_targets)
    scores = np.stack([density[j][ys, xs] for j in targets], axis=1).astype(np.float32)
    scores += rng.uniform(0, 1e-3, size=scores.shape).astype(np.float32)

    flat_scores = scores.ravel()
    order = np.argsort(-flat_scores)  # descending
    n_targets = len(targets)

    assigned = np.zeros(n_i, dtype=bool)
    remaining = {j: quota[i, j] for j in targets}
    n_assigned = 0
    total_quota = sum(remaining.values())

    pix_idx = order // n_targets
    tgt_idx = order % n_targets

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

    log(f"class {names[i]}: assigned {n_assigned} of quota {total_quota} (pool {n_i})")

np.save(f"{DATA_DIR}/proj2025.npy", proj2025)

final_counts = np.array([int((proj2025 == i).sum()) for i in range(n_classes)])
log(f"2024 counts: {counts}")
log(f"2025 projected counts: {final_counts}")
log(f"delta: {final_counts - counts}")
log("DONE")
