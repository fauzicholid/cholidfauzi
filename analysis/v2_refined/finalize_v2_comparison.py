"""Computes 2025 province-wide area projections for the refined ConvLSTM v2
configuration (full inverse-frequency weighting) across all 5 seeds, and
runs an exploratory coarse-grid three-way check of this study's own refined
2024 classification against the independent reference layer, at the same
300x233 grid ConvLSTM trains on. (The corresponding v3/sqrt-dampened areas
were computed with a short one-off snippet during the original session
rather than a saved script; see analysis/v2_refined/README.md for those
numbers.)
"""
import json, pickle, os
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from PIL import Image

DATA_DIR = os.environ.get("LC_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
REF_FILE = os.environ.get("LC_REF_FILE", os.path.join(DATA_DIR, "independent_reference.geojson"))
NAMES = ["Air", "Non-vegetasi", "Veg. jarang", "Veg. rapat"]
N_CLASSES = 4
CROSSWALK = {
    "Badan Air": 0, "Tambak": 0, "Rawa": 0, "Belukar Rawa": 0,
    "Tanah Terbuka": 1, "Pemukiman": 1, "Bandara / Pelabuhan": 1, "Pertambangan": 1,
    "Sawah": 2, "Pertanian Lahan Kering": 2, "Pertanian Lahan Kering Campur": 2, "Belukar": 2,
    "Hutan Lahan Kering Primer": 3, "Hutan Lahan Kering Sekunder": 3, "Hutan Tanaman": 3,
    "Perkebunan": 3, "Hutan Mangrove Sekunder": 3,
}

# ---- 1) ConvLSTM v2 2025 province-wide area projection (mean across 5 seeds) ----
transform24, crs24 = pickle.load(open(f"{DATA_DIR}/geo.pkl", "rb"))
valid24 = np.load(f"{DATA_DIR}/valid.npy")
px_area_km2 = 35522.3 / valid24.sum()
valid_common = np.load(f"{DATA_DIR}/seq_valid_common_v2.npy")  # (233,300)

# upscale each seed's coarse 300x233 pred2025 to full grid via nearest, restricted to valid_common,
# then just report class SHARE within valid_common at coarse resolution (avoids resampling artifacts;
# areas reported at coarse-grid px count x px_area equivalent).
coarse_valid_count = int(valid_common.sum())
coarse_px_km2 = 35522.3 / coarse_valid_count  # approx km^2 per coarse training pixel, within AOI

seed_area_tables = []
for seed in range(5):
    pred = np.load(f"{DATA_DIR}/convlstm_v2_pred2025_seed{seed}.npy")
    counts = [int(((pred == c) & valid_common).sum()) for c in range(N_CLASSES)]
    areas = [c * coarse_px_km2 for c in counts]
    seed_area_tables.append(areas)
seed_area_tables = np.array(seed_area_tables)
print("ConvLSTM v2 2025 projected areas (km^2) per seed:")
for c, name in enumerate(NAMES):
    print(f"  {name:15s} " + "  ".join(f"{v:8.1f}" for v in seed_area_tables[:, c]) +
          f"   mean={seed_area_tables[:,c].mean():8.1f}  sd={seed_area_tables[:,c].std():6.1f}")

# CA-Markov v2 2025 areas (already computed)
ca_res = json.load(open(f"{DATA_DIR}/ca_markov_v2_results.json"))
counts2025_ca = ca_res["counts2025"]
areas_ca = [c * px_area_km2 for c in counts2025_ca]
print("\nCA-Markov v2 2025 projected areas (km^2):", dict(zip(NAMES, [round(a,1) for a in areas_ca])))
print("ConvLSTM v2 2025 mean areas (km^2):", dict(zip(NAMES, [round(v,1) for v in seed_area_tables.mean(axis=0)])))

# ---- 2) Independent ground-truth check for BOTH models at the SAME coarse (300x233) resolution ----
print("\n" + "="*90)
print("Independent ground-truth check: CA-Markov v2 vs ConvLSTM v2, both at coarse 300x233 grid, vs independent reference")
print("="*90)

gdf = gpd.read_file(REF_FILE).to_crs(crs24)
shapes = [(geom, CROSSWALK[leg]) for geom, leg in zip(gdf.geometry, gdf["Legenda"])]
ref_raster_native = rasterize(shapes, out_shape=valid24.shape, transform=transform24, fill=-1, dtype="int16")

# downsample reference to the coarse 300x233 grid via majority vote (consistent with earlier
# analysis/experiments/exp1/exp4 approach)
from scipy import stats as sp_stats
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

ref_coarse = downsample_majority_vote(ref_raster_native, valid24 & (ref_raster_native >= 0), (300, 233))
ref_valid_coarse = ref_coarse >= 0

# CA-Markov v2's actual-2024 (v2) also needs to be at this coarse grid for a fair three-way comparison
cls24_v2 = np.load(f"{DATA_DIR}/cls24_improved.npy")
cls24_coarse = downsample_majority_vote(cls24_v2, valid24 & (cls24_v2 >= 0), (300, 233))

def confusion_and_metrics(actual, pred, mask, n_classes=N_CLASSES):
    CM = np.zeros((n_classes, n_classes), dtype=np.int64)
    for a in range(n_classes):
        for p in range(n_classes):
            CM[a, p] = int(((actual == a) & (pred == p) & mask).sum())
    total = CM.sum()
    OA = np.trace(CM) / total if total > 0 else float("nan")
    row_sums, col_sums = CM.sum(axis=1), CM.sum(axis=0)
    pe = (row_sums * col_sums).sum() / (total ** 2) if total > 0 else float("nan")
    kappa = (OA - pe) / (1 - pe) if pe < 1 else float("nan")
    PA = np.divide(np.diag(CM), row_sums, out=np.zeros(n_classes), where=row_sums > 0)
    return dict(CM=CM.tolist(), total=int(total), OA=float(OA), kappa=float(kappa), PA=PA.tolist())

mask_cls = ref_valid_coarse & (cls24_coarse >= 0)
res_cls24_coarse = confusion_and_metrics(ref_coarse, cls24_coarse, mask_cls)
print(f"\nThis study's own 2024 classification (v2) at coarse grid vs independent reference:")
print(f"  n={res_cls24_coarse['total']}  OA={res_cls24_coarse['OA']*100:.2f}%  Kappa={res_cls24_coarse['kappa']:.4f}")
print(f"  PA: {[f'{v*100:.1f}%' for v in res_cls24_coarse['PA']]}")

json.dump(dict(
    convlstm_v2_2025_areas_per_seed=seed_area_tables.tolist(),
    convlstm_v2_2025_areas_mean=seed_area_tables.mean(axis=0).tolist(),
    convlstm_v2_2025_areas_std=seed_area_tables.std(axis=0).tolist(),
    ca_markov_v2_2025_areas=areas_ca,
    own_cls24_v2_vs_reference_coarse=res_cls24_coarse,
), open(f"{DATA_DIR}/finalize_v2_comparison_results.json", "w"), indent=2)
print("\nSaved finalize_v2_comparison_results.json")
