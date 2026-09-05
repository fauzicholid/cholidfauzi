"""Independent ground-truth validation: rasterizes an externally sourced,
17-class government land-cover vector layer directly onto this study's own
analysis grid and compares it, pixel-exact, against the pure-NDVI 2024
classification (cls24_4.npy) that every other validation in this project is
otherwise self-referential against.

The reference layer itself is not redistributed with this repository (it is
a third-party government land-cover product); point LC_REF_FILE at a local
copy with a "Legenda" attribute using the 17-class taxonomy named in
CROSSWALK below, or adjust CROSSWALK to match a different reference layer's
attribute schema.
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

# Crosswalk from the 17 "Legenda" classes (Indonesian national land-cover
# taxonomy) to this study's 4 NDVI-based classes. Documented rationale:
#   0 Air (Water): unambiguous water bodies + aquaculture (permanently
#     inundated, low NDVI).
#   1 Non-vegetasi: built-up / bare / extractive land uses (low NDVI).
#   2 Veg. jarang (Sparse vegetation): row-crop agriculture and shrub
#     (moderate NDVI).
#   3 Veg. rapat (Dense vegetation): closed-canopy forest and tree-crop
#     plantation (high NDVI).
# AMBIGUOUS, flagged explicitly rather than silently decided:
#   - "Rawa" (swamp) and "Belukar Rawa" (swamp shrub): genuinely mixed
#     water/vegetation; assigned to Air for the primary crosswalk (since
#     both are officially "wetland" water-associated classes) but reported
#     separately in a sensitivity check that excludes them entirely.
#   - "Perkebunan" (tree-crop plantation, e.g. oil palm/rubber/tea):
#     canopy density varies by crop/age; assigned to Veg. rapat (closed-
#     canopy default) but also reported in the sensitivity check.
CROSSWALK = {
    "Badan Air": 0,
    "Tambak": 0,
    "Rawa": 0,               # ambiguous
    "Belukar Rawa": 0,       # ambiguous
    "Tanah Terbuka": 1,
    "Pemukiman": 1,
    "Bandara / Pelabuhan": 1,
    "Pertambangan": 1,
    "Sawah": 2,
    "Pertanian Lahan Kering": 2,
    "Pertanian Lahan Kering Campur": 2,
    "Belukar": 2,
    "Hutan Lahan Kering Primer": 3,
    "Hutan Lahan Kering Sekunder": 3,
    "Hutan Tanaman": 3,
    "Perkebunan": 3,         # ambiguous
    "Hutan Mangrove Sekunder": 3,
}
AMBIGUOUS = {"Rawa", "Belukar Rawa", "Perkebunan"}

log("Loading reference layer...")
gdf = gpd.read_file(REF_FILE)
log(f"Reference layer: {len(gdf)} features, CRS={gdf.crs}, classes={sorted(gdf['Legenda'].unique())}")
missing = set(gdf["Legenda"].unique()) - set(CROSSWALK.keys())
if missing:
    raise SystemExit(f"Unmapped Legenda classes: {missing}")

log("Loading own 2024 classification grid...")
transform24, crs24 = pickle.load(open(f"{DATA_DIR}/geo.pkl", "rb"))
cls24 = np.load(f"{DATA_DIR}/cls24_4.npy")
valid24 = np.load(f"{DATA_DIR}/valid.npy")
shape = cls24.shape
log(f"Grid shape: {shape}, CRS={crs24}")

log("Reprojecting reference layer to analysis-grid CRS...")
gdf_proj = gdf.to_crs(crs24)

log("Rasterizing reference layer onto the analysis grid (primary crosswalk)...")
shapes_main = [(geom, CROSSWALK[leg]) for geom, leg in zip(gdf_proj.geometry, gdf_proj["Legenda"])]
ref_raster = rasterize(shapes_main, out_shape=shape, transform=transform24, fill=-1, dtype="int16")

log("Rasterizing a second layer marking which pixels fall in an AMBIGUOUS reference class...")
shapes_ambig = [(geom, 1) for geom, leg in zip(gdf_proj.geometry, gdf_proj["Legenda"]) if leg in AMBIGUOUS]
ambig_raster = rasterize(shapes_ambig, out_shape=shape, transform=transform24, fill=0, dtype="uint8") if shapes_ambig else np.zeros(shape, dtype="uint8")

ref_covered = (ref_raster >= 0)
log(f"Reference layer covers {ref_covered.sum()} / {valid24.sum()} of this study's AOI-valid pixels "
    f"({100*ref_covered[valid24].mean():.1f}% of AOI-valid pixels have reference coverage)")

# ---- Primary comparison: full crosswalk, no exclusions ----
mask = valid24 & (cls24 >= 0) & ref_covered
n = int(mask.sum())
log(f"Primary comparison n = {n} pixels")

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
                PA=PA.tolist(), UA=UA.tolist(), row_sums=row_sums.tolist(), col_sums=col_sums.tolist())

result_primary = confusion_and_metrics(ref_raster, cls24, mask)
log(f"PRIMARY (all reference classes, ambiguous ones included in crosswalk):")
log(f"  n={result_primary['total']}  OA={result_primary['OA']*100:.2f}%  Kappa={result_primary['kappa']:.4f}")
log(f"  PA per class {NAMES}: {[f'{v*100:.1f}%' for v in result_primary['PA']]}")
log(f"  UA per class {NAMES}: {[f'{v*100:.1f}%' for v in result_primary['UA']]}")
log(f"  Confusion matrix (rows=reference/actual, cols=this study's 2024 classification):\n{np.array(result_primary['CM'])}")

# ---- Sensitivity comparison: exclude ambiguous reference classes entirely ----
mask_strict = mask & (ambig_raster == 0)
n_strict = int(mask_strict.sum())
log(f"\nSensitivity comparison (excluding ambiguous Rawa/Belukar Rawa/Perkebunan pixels): n = {n_strict} "
    f"({100*(1 - n_strict/n):.1f}% of the primary comparison's pixels excluded)")
result_strict = confusion_and_metrics(ref_raster, cls24, mask_strict)
log(f"  OA={result_strict['OA']*100:.2f}%  Kappa={result_strict['kappa']:.4f}")
log(f"  PA per class {NAMES}: {[f'{v*100:.1f}%' for v in result_strict['PA']]}")
log(f"  UA per class {NAMES}: {[f'{v*100:.1f}%' for v in result_strict['UA']]}")

# ---- Reference layer's own class-area breakdown for context ----
px_area_km2 = None
try:
    px_area_km2 = abs(transform24.a * transform24.e) / 1e6  # m^2 per pixel from affine
except Exception as e:
    log(f"  (could not compute pixel area: {e})")

json.dump(dict(primary=result_primary, strict=result_strict,
               ref_coverage_frac=float(ref_covered[valid24].mean()),
               px_area_km2=px_area_km2,
               crosswalk=CROSSWALK, ambiguous_classes=list(AMBIGUOUS)),
          open(f"{DATA_DIR}/bappeda_validation_results.json", "w"), indent=2)
log("\nSaved bappeda_validation_results.json")
