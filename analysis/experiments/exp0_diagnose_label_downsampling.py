"""EXPERIMENT 0 — Diagnostic (no retraining needed, run this first).

Quantifies how much class distribution is distorted by the current
Image.NEAREST downsampling of the discrete 2024 label map (analysis grid,
e.g. 2447x3000) to the ConvLSTM training grid (300x233), versus two
alternatives:
  (A) CURRENT: classify at analysis grid, then NEAREST-subsample the labels.
  (B) MAJORITY-VOTE: classify at analysis grid, then take the mode class per
      block when downsampling (proper categorical downsampling).
  (C) RECLASSIFY-AFTER-DOWNSAMPLE: downsample the *continuous* NDVI with
      BILINEAR (consistent with how ndvi_seq is already built), THEN apply
      the NDVI thresholds to the downsampled, averaged NDVI. This is
      arguably the most physically sensible option, since it means "the
      average vegetation signal in this coarse cell", not "one lucky/unlucky
      point sample" or "whichever class had more raw pixels" (which can differ
      from what the coarse-cell's true dominant land cover actually is,
      especially for cells that are a NDVI-gradient boundary).

Run this BEFORE touching the training script. It only needs sequence_raw.pkl
(already produced by build_sequence_dataset.py) and reports class counts for
year 2024 under all three approaches, plus year-by-year for the full sequence.
"""
import numpy as np
import pickle
import os
from scipy import stats as sp_stats
from PIL import Image

DATA_DIR = os.environ.get("LC_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
TARGET_SIZE = (300, 233)  # (W, H) -- must match train_convlstm.py
NAMES = ["Air", "Non-vegetasi", "Veg. jarang", "Veg. rapat"]

raw = pickle.load(open(f"{DATA_DIR}/sequence_raw.pkl", "rb"))
W, H = TARGET_SIZE


def classify4(ndvi, valid):
    c = np.full(ndvi.shape, -1, dtype=np.int8)
    c[valid & (ndvi < 0.05)] = 0
    c[valid & (ndvi >= 0.05) & (ndvi < 0.25)] = 1
    c[valid & (ndvi >= 0.25) & (ndvi < 0.55)] = 2
    c[valid & (ndvi >= 0.55)] = 3
    return c


def downsample_nearest_labels(cls, size):
    img = Image.fromarray(cls.astype(np.uint8), mode="L")
    return np.array(img.resize(size, Image.NEAREST), dtype=np.int8)


def downsample_majority_vote(cls, valid, size, n_classes=4):
    """Block-wise mode. Uses block_reduce-style reshaping; requires the
    source shape to be (approximately) evenly divisible by the block size.
    Falls back to per-block looping at the (small) target resolution, which
    is fine since target is only 300x233 = ~70k blocks.
    """
    src_h, src_w = cls.shape
    dst_w, dst_h = size
    # block edges (float, then rounded) so this works for non-integer ratios too
    y_edges = np.linspace(0, src_h, dst_h + 1).round().astype(int)
    x_edges = np.linspace(0, src_w, dst_w + 1).round().astype(int)
    out = np.full((dst_h, dst_w), -1, dtype=np.int8)
    for iy in range(dst_h):
        y0, y1 = y_edges[iy], max(y_edges[iy] + 1, y_edges[iy + 1])
        for ix in range(dst_w):
            x0, x1 = x_edges[ix], max(x_edges[ix] + 1, x_edges[ix + 1])
            block = cls[y0:y1, x0:x1]
            block_valid = valid[y0:y1, x0:x1] if valid is not None else (block >= 0)
            vals = block[block_valid & (block >= 0)]
            if vals.size == 0:
                continue
            out[iy, ix] = int(sp_stats.mode(vals, keepdims=False).mode)
    return out


def downsample_reclassify(ndvi, valid, size):
    """Downsample continuous NDVI with BILINEAR (matches ndvi_seq construction
    in train_convlstm.py), downsample the valid-fraction mask, then classify
    the coarse NDVI directly."""
    ndvi_img = Image.fromarray(np.nan_to_num(ndvi, nan=0.0).astype(np.float32), mode="F")
    ndvi_coarse = np.array(ndvi_img.resize(size, Image.BILINEAR), dtype=np.float32)
    valid_img = Image.fromarray(valid.astype(np.float32), mode="F")
    valid_frac = np.array(valid_img.resize(size, Image.BOX), dtype=np.float32)
    valid_coarse = valid_frac > 0.5
    return classify4(ndvi_coarse, valid_coarse)


def class_counts(cls, n=4):
    return [int((cls == i).sum()) for i in range(n)]


print("=" * 78)
print("Class distribution comparison: NEAREST (current) vs MAJORITY-VOTE vs RECLASSIFY-AFTER-DOWNSAMPLE")
print("=" * 78)

rows = []
for y in YEARS:
    cls_native = raw[y]["cls"]  # analysis-grid resolution, already classified
    valid_native = raw[y]["valid"]
    ndvi_adj = raw[y]["ndvi_adj"]

    counts_native = class_counts(cls_native)

    cls_A = downsample_nearest_labels(cls_native, TARGET_SIZE)
    counts_A = class_counts(cls_A)

    cls_B = downsample_majority_vote(cls_native, valid_native, TARGET_SIZE)
    counts_B = class_counts(cls_B)

    cls_C = downsample_reclassify(ndvi_adj, valid_native, TARGET_SIZE)
    counts_C = class_counts(cls_C)

    rows.append((y, counts_native, counts_A, counts_B, counts_C))

    print(f"\n--- Year {y} ---")
    print(f"  Native (analysis grid) counts        : {counts_native}")
    tot_native = sum(counts_native)
    print(f"  Native (analysis grid) share          : {[f'{c/tot_native*100:.2f}%' for c in counts_native]}")

    for label, counts in [("A) NEAREST (current)", counts_A),
                           ("B) MAJORITY-VOTE", counts_B),
                           ("C) RECLASSIFY-after-bilinear-downsample", counts_C)]:
        tot = sum(counts)
        share = [f"{c/tot*100:.2f}%" if tot else "n/a" for c in counts]
        # relative change in Non-vegetated (class 1) share vs native
        native_share_1 = counts_native[1] / tot_native if tot_native else 0
        this_share_1 = counts[1] / tot if tot else 0
        rel_change = (this_share_1 - native_share_1) / native_share_1 * 100 if native_share_1 else float("nan")
        print(f"  {label:42s}: counts={counts}  share={share}  "
              f"[Non-veg. share change vs native: {rel_change:+.1f}%]")

print("\n" + "=" * 78)
print("KEY DIAGNOSTIC: 2024 Non-vegetated (class 1) pixel count at training resolution")
print("=" * 78)
y2024 = rows[-1]
_, native, A, B, C = y2024
print(f"  Native analysis-grid pixels  : {native[1]:>10,d}")
print(f"  (A) NEAREST [current method] : {A[1]:>10,d}  <- this is what train_convlstm.py actually trains/tests on")
print(f"  (B) MAJORITY-VOTE            : {B[1]:>10,d}")
print(f"  (C) RECLASSIFY-after-bilinear: {C[1]:>10,d}")
print()
print("If (A) is drastically lower than (B) and (C), the ConvLSTM collapse on")
print("Non-vegetated is very likely a label-downsampling artifact, NOT purely")
print("a class-imbalance / architecture problem -- proceed to exp1 and exp2.")
