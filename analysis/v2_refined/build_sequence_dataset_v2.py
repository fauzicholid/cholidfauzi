"""Rebuild the multi-year sequence using the improved NDVI+NDMI classifier
(improve_classification.py) instead of the pure-NDVI classifier from
../build_sequence_dataset.py.

For each year, reprojects that year's own-grid B08 onto the 2024 reference
grid (mirroring how NDVI is already reprojected), pairs it with a B11
mosaic fetched and cloud-masked the same way as B02/B03/B04/B08 and
reprojected directly onto the 2024 grid (saved as {year}_B11_on2024grid.npy,
2024 itself as b11.npy), computes NDMI on the 2024 grid, and classifies
using the NDMI threshold fit on the 2024 train split against the
independent reference layer (improve_classification.py) -- applied
uniformly across years rather than re-fit per year, since independent
ground truth exists for only one year.
"""
import os
import numpy as np, pickle
from rasterio.warp import reproject, Resampling

DATA_DIR = os.environ.get("LC_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
NDMI_THR = 0.2680  # fit on 2024 train split, see improve_classification.py

transform24, crs24 = pickle.load(open(f"{DATA_DIR}/geo.pkl", "rb"))
valid24 = np.load(f"{DATA_DIR}/valid.npy")
shape24 = valid24.shape

def classify4_improved(ndvi, ndmi, valid):
    c = np.full(ndvi.shape, -1, dtype=np.int8)
    c[valid & (ndvi < 0.05)] = 0
    c[valid & (ndvi >= 0.05) & (ndvi < 0.25)] = 1
    c[valid & (ndvi >= 0.25) & (ndvi < 0.55)] = 2
    cand_dense = valid & (ndvi >= 0.55)
    have_ndmi = ~np.isnan(ndmi)
    c[cand_dense & have_ndmi & (ndmi >= NDMI_THR)] = 3
    c[cand_dense & have_ndmi & (ndmi < NDMI_THR)] = 2
    c[cand_dense & ~have_ndmi] = 3  # fallback: keep old rule where NDMI missing
    return c

def load_year(year):
    if year == 2024:
        ndvi = np.load(f"{DATA_DIR}/ndvi.npy")
        brightness = np.load(f"{DATA_DIR}/brightness.npy")
        ndmi = np.load(f"{DATA_DIR}/ndmi.npy")
        return ndvi, brightness, valid24, ndmi

    b02 = np.load(f"{DATA_DIR}/{year}_B02.npy"); b03 = np.load(f"{DATA_DIR}/{year}_B03.npy")
    b04 = np.load(f"{DATA_DIR}/{year}_B04.npy"); b08 = np.load(f"{DATA_DIR}/{year}_B08.npy")
    transform_y, crs_y = pickle.load(open(f"{DATA_DIR}/{year}_geo.pkl", "rb"))
    valid_y = (b02 > 0) & (b03 > 0) & (b04 > 0) & (b08 > 0)
    d = b08 + b04
    ndvi_y = np.zeros_like(b08)
    ndvi_y[valid_y] = (b08[valid_y] - b04[valid_y]) / np.where(d[valid_y] == 0, 1, d[valid_y])
    brightness_y = (b02 + b03 + b04) / 3

    ndvi_aligned = np.full(shape24, np.nan, dtype=np.float32)
    reproject(source=ndvi_y, destination=ndvi_aligned,
              src_transform=transform_y, src_crs=crs_y, dst_transform=transform24, dst_crs=crs24,
              resampling=Resampling.bilinear, src_nodata=None, dst_nodata=np.nan)
    bright_aligned = np.full(shape24, np.nan, dtype=np.float32)
    reproject(source=brightness_y, destination=bright_aligned,
              src_transform=transform_y, src_crs=crs_y, dst_transform=transform24, dst_crs=crs24,
              resampling=Resampling.bilinear, src_nodata=None, dst_nodata=np.nan)
    valid_aligned = np.zeros(shape24, dtype=np.uint8)
    reproject(source=valid_y.astype(np.uint8), destination=valid_aligned,
              src_transform=transform_y, src_crs=crs_y, dst_transform=transform24, dst_crs=crs24,
              resampling=Resampling.nearest)
    valid_aligned = valid_aligned.astype(bool)

    # B08 on the 2024 grid, to pair with B11 (already fetched directly onto this grid)
    b08_aligned = np.full(shape24, np.nan, dtype=np.float32)
    b08_f = b08.astype(np.float32)
    b08_f[~valid_y] = np.nan
    reproject(source=b08_f, destination=b08_aligned,
              src_transform=transform_y, src_crs=crs_y, dst_transform=transform24, dst_crs=crs24,
              resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)

    b11_on_grid = np.load(f"{DATA_DIR}/{year}_B11_on2024grid.npy")
    dsum = b08_aligned + b11_on_grid
    ndmi_y = np.full(shape24, np.nan, dtype=np.float32)
    ok = ~np.isnan(b08_aligned) & ~np.isnan(b11_on_grid)
    ndmi_y[ok] = (b08_aligned[ok] - b11_on_grid[ok]) / np.where(dsum[ok] == 0, 1, dsum[ok])

    return ndvi_aligned, bright_aligned, valid_aligned, ndmi_y

ndvi_2024, bright_2024, valid_2024, _ = load_year(2024)

raw = {}
for y in YEARS:
    ndvi_y, bright_y, valid_y, ndmi_y = load_year(y)
    raw[y] = dict(ndvi=ndvi_y, bright=bright_y, valid=valid_y, ndmi=ndmi_y)
    n_ndmi = int((~np.isnan(ndmi_y) & valid_y).sum())
    print(f"{y}: loaded, valid frac (own) = {valid_y.mean():.3f}, NDMI coverage within valid = "
          f"{n_ndmi/max(1,valid_y.sum()):.3f}", flush=True)

med_ref = np.nanmedian(ndvi_2024[valid_2024])
for y in YEARS:
    if y == 2024:
        raw[y]["ndvi_adj"] = raw[y]["ndvi"]
        continue
    common = valid_2024 & raw[y]["valid"]
    med_y = np.nanmedian(raw[y]["ndvi"][common])
    shift = med_ref - med_y
    raw[y]["ndvi_adj"] = raw[y]["ndvi"] + shift
    print(f"{y}: NDVI median={med_y:.4f} shift={shift:+.4f}", flush=True)

# ---- NDMI needs an analogous cross-date shift: same reproducible offset ----
# ---- pattern already documented for NDVI shows up in NDMI too (worst in ----
# ---- 2019 and 2023, the same two years with the largest NDVI offsets). ----
print("\nChecking cross-date NDMI consistency (median over common-valid, NDMI-available pixels):", flush=True)
ndmi_2024 = raw[2024]["ndmi"]
common_2024 = valid_2024 & ~np.isnan(ndmi_2024)
med_ndmi_ref = np.nanmedian(ndmi_2024[common_2024])
print(f"  2024 NDMI median (reference) = {med_ndmi_ref:.4f}", flush=True)
for y in YEARS:
    if y == 2024:
        continue
    common = valid_2024 & raw[y]["valid"] & ~np.isnan(raw[y]["ndmi"]) & ~np.isnan(ndmi_2024)
    if common.sum() == 0:
        continue
    med_y = np.nanmedian(raw[y]["ndmi"][common])
    print(f"  {y}: NDMI median={med_y:.4f} (diff from 2024: {med_y - med_ndmi_ref:+.4f}), n={int(common.sum())}", flush=True)

# ---- Apply an analogous median-shift to NDMI (same rationale as NDVI's shift) ----
for y in YEARS:
    if y == 2024:
        raw[y]["ndmi_adj"] = raw[y]["ndmi"]
        continue
    common = valid_2024 & raw[y]["valid"] & ~np.isnan(raw[y]["ndmi"]) & ~np.isnan(ndmi_2024)
    med_y = np.nanmedian(raw[y]["ndmi"][common]) if common.sum() > 0 else med_ndmi_ref
    shift = med_ndmi_ref - med_y
    raw[y]["ndmi_adj"] = raw[y]["ndmi"] + shift
    print(f"{y}: NDMI median={med_y:.4f} shift={shift:+.4f}", flush=True)

for y in YEARS:
    raw[y]["cls"] = classify4_improved(raw[y]["ndvi_adj"], raw[y]["ndmi_adj"], raw[y]["valid"])
    dist = [int((raw[y]["cls"] == i).sum()) for i in range(4)]
    print(f"{y}: improved class dist (own valid, incl out-of-AOI) = {dist}", flush=True)

pickle.dump(raw, open(f"{DATA_DIR}/sequence_raw_v2.pkl", "wb"))
print("saved sequence_raw_v2.pkl", flush=True)
