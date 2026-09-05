"""Fits an NDMI threshold to correct the specific error the independent
validation (validate_against_reference.py) diagnosed: NDVI>=0.55 pixels
labeled Dense vegetation that the independent reference calls agriculture
or shrub (a single greenness index cannot separate closed-canopy forest
from healthy, high-NDVI cropland/plantation).

Uses a spatial checkerboard train/test split against the independent
reference (never fitting and evaluating the threshold on the same pixels),
fits the threshold on the train half only, and evaluates the resulting
classifier honestly on the held-out test half.
"""
import time, pickle, json, os
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

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

log("Loading grids...")
transform24, crs24 = pickle.load(open(f"{DATA_DIR}/geo.pkl", "rb"))
cls24_baseline = np.load(f"{DATA_DIR}/cls24_4.npy")   # existing pure-NDVI classification
valid24 = np.load(f"{DATA_DIR}/valid.npy")
ndvi = np.load(f"{DATA_DIR}/ndvi.npy")
ndmi = np.load(f"{DATA_DIR}/ndmi.npy")
shape = cls24_baseline.shape

log("Rasterizing reference layer...")
gdf = gpd.read_file(REF_FILE).to_crs(crs24)
shapes = [(geom, CROSSWALK[leg]) for geom, leg in zip(gdf.geometry, gdf["Legenda"])]
ref_raster = rasterize(shapes, out_shape=shape, transform=transform24, fill=-1, dtype="int16")
np.save(f"{DATA_DIR}/ref_raster.npy", ref_raster)

# ---- Spatial checkerboard train/test split (20x20 pixel blocks, ~ 1.7km blocks) ----
K = 20
rows = np.arange(shape[0])[:, None] // K
cols = np.arange(shape[1])[None, :] // K
checker = (rows + cols) % 2
train_mask_geo = (checker == 0)
test_mask_geo = (checker == 1)

have_ndmi = ~np.isnan(ndmi)
base_mask = valid24 & (cls24_baseline >= 0) & (ref_raster >= 0) & have_ndmi
train_mask = base_mask & train_mask_geo
test_mask = base_mask & test_mask_geo
log(f"Usable pixels: {base_mask.sum()}  train={train_mask.sum()}  test={test_mask.sum()}")

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
    UA = np.divide(np.diag(CM), col_sums, out=np.zeros(n_classes), where=col_sums > 0)
    return dict(CM=CM.tolist(), total=int(total), OA=float(OA), kappa=float(kappa),
                PA=PA.tolist(), UA=UA.tolist())

# ---- Baseline (pure NDVI) performance on the SAME test split, for fair comparison ----
log("\n=== BASELINE (pure NDVI threshold), evaluated on TEST split only ===")
res_base_test = confusion_and_metrics(ref_raster, cls24_baseline, test_mask)
log(f"OA={res_base_test['OA']*100:.2f}%  Kappa={res_base_test['kappa']:.4f}")
log(f"PA {NAMES}: {[f'{v*100:.1f}%' for v in res_base_test['PA']]}")

# ---- Fit NDMI threshold on TRAIN split: among candidate-Dense pixels (NDVI>=0.55), ----
# ---- find the NDMI cut that best separates reference-Dense from reference-Sparse. ----
log("\n=== Fitting NDMI threshold on TRAIN split (NDVI>=0.55 candidates only) ===")
candidate_train = train_mask & (ndvi >= 0.55) & np.isin(ref_raster, [2, 3])
ndmi_train = ndmi[candidate_train]
ref_train = ref_raster[candidate_train]
log(f"Candidate (NDVI>=0.55, ref in {{Sparse,Dense}}) train pixels: {candidate_train.sum()} "
    f"(ref Sparse={np.sum(ref_train==2)}, ref Dense={np.sum(ref_train==3)})")

best_thr, best_score = None, -1
for thr in np.linspace(np.nanpercentile(ndmi_train, 1), np.nanpercentile(ndmi_train, 99), 200):
    pred = np.where(ndmi_train >= thr, 3, 2)
    # balanced accuracy across the two classes (robust to class imbalance in the candidate pool)
    acc_dense = (pred[ref_train == 3] == 3).mean() if (ref_train == 3).sum() > 0 else 0
    acc_sparse = (pred[ref_train == 2] == 2).mean() if (ref_train == 2).sum() > 0 else 0
    score = (acc_dense + acc_sparse) / 2
    if score > best_score:
        best_score, best_thr = score, thr
log(f"Best NDMI threshold (train, balanced accuracy): {best_thr:.4f} (train balanced acc={best_score*100:.2f}%)")

# median NDMI by class on train, for context
for c, name in [(2, "Sparse"), (3, "Dense")]:
    vals = ndmi_train[ref_train == c]
    if len(vals):
        log(f"  train NDMI median for ref-{name}: {np.median(vals):.4f} (n={len(vals)})")

# ---- Apply the fitted rule to build the improved classification (whole grid) ----
log("\n=== Applying improved NDVI+NDMI rule to build revised classification ===")
cls24_improved = cls24_baseline.copy()
reassign = valid24 & (cls24_baseline == 3) & have_ndmi & (ndmi < best_thr)
cls24_improved[reassign] = 2
log(f"Reassigned {int(reassign.sum())} pixels from Dense -> Sparse "
    f"({100*reassign.sum()/(cls24_baseline==3).sum():.1f}% of originally-Dense pixels)")
np.save(f"{DATA_DIR}/cls24_improved.npy", cls24_improved)

log("\n=== IMPROVED classifier, evaluated on TEST split only (same pixels as baseline test) ===")
res_imp_test = confusion_and_metrics(ref_raster, cls24_improved, test_mask)
log(f"OA={res_imp_test['OA']*100:.2f}%  Kappa={res_imp_test['kappa']:.4f}")
log(f"PA {NAMES}: {[f'{v*100:.1f}%' for v in res_imp_test['PA']]}")
log(f"UA {NAMES}: {[f'{v*100:.1f}%' for v in res_imp_test['UA']]}")
log(f"Confusion matrix (rows=reference, cols=improved classification):\n{np.array(res_imp_test['CM'])}")

log("\n=== For reference: IMPROVED classifier evaluated on FULL AOI (train+test, informational only) ===")
res_imp_full = confusion_and_metrics(ref_raster, cls24_improved, base_mask)
log(f"OA={res_imp_full['OA']*100:.2f}%  Kappa={res_imp_full['kappa']:.4f}")

# ---- Updated province-wide area totals under the improved classification ----
px_area_km2 = abs(transform24.a * transform24.e) / 1e6
log(f"\nPixel area: {px_area_km2:.4f} km^2")
log("Province-wide class areas, BASELINE vs IMPROVED (within AOI valid.npy):")
for c, name in enumerate(NAMES):
    a_base = int((valid24 & (cls24_baseline == c)).sum()) * px_area_km2
    a_imp = int((valid24 & (cls24_improved == c)).sum()) * px_area_km2
    log(f"  {name:15s} baseline={a_base:10.1f} km^2   improved={a_imp:10.1f} km^2   delta={a_imp-a_base:+8.1f} km^2")

json.dump(dict(
    best_ndmi_threshold=float(best_thr),
    train_balanced_acc=float(best_score),
    test_baseline=res_base_test,
    test_improved=res_imp_test,
    full_improved=res_imp_full,
    n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
    n_reassigned_dense_to_sparse=int(reassign.sum()),
), open(f"{DATA_DIR}/improve_classification_results.json", "w"), indent=2)
log("\nSaved improve_classification_results.json, cls24_improved.npy, ref_raster.npy")
