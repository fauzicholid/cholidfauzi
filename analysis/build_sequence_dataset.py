"""Load, radiometrically normalize, and NDVI-classify 8 years (2017-2024) of
Sentinel-2 band mosaics onto the 2024 reference grid; saves sequence_raw.pkl
for use by train_convlstm.py.
"""
import numpy as np, pickle
from rasterio.warp import reproject, Resampling
import os

DATA_DIR = os.environ.get("LC_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

transform24, crs24 = pickle.load(open(f"{DATA_DIR}/geo.pkl", "rb"))
valid24 = np.load(f"{DATA_DIR}/valid.npy")
shape24 = valid24.shape

def load_year(year):
    if year == 2024:
        # already have final AOI-clipped decimated arrays
        b02 = np.load(f"{DATA_DIR}/b02.npy"); b03 = np.load(f"{DATA_DIR}/b03.npy")
        b04 = np.load(f"{DATA_DIR}/b04.npy"); b08 = np.load(f"{DATA_DIR}/b08.npy")
        valid = valid24
        ndvi = np.load(f"{DATA_DIR}/ndvi.npy")
        brightness = np.load(f"{DATA_DIR}/brightness.npy")
        return ndvi, brightness, valid

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
    return ndvi_aligned, bright_aligned, valid_aligned

# reference for radiometric normalization = 2024 median (within common valid area)
ndvi_2024, bright_2024, valid_2024 = load_year(2024)

raw = {}
for y in YEARS:
    ndvi_y, bright_y, valid_y = load_year(y)
    raw[y] = dict(ndvi=ndvi_y, bright=bright_y, valid=valid_y)
    print(f"{y}: loaded, valid frac (own) = {valid_y.mean():.3f}")

# normalize each year's NDVI median to match 2024's median, using pixels valid in both
med_ref = np.median(ndvi_2024[valid_2024])
for y in YEARS:
    if y == 2024:
        raw[y]["ndvi_adj"] = raw[y]["ndvi"]
        continue
    common = valid_2024 & raw[y]["valid"]
    med_y = np.median(raw[y]["ndvi"][common])
    shift = med_ref - med_y
    raw[y]["ndvi_adj"] = raw[y]["ndvi"] + shift
    print(f"{y}: median={med_y:.4f} shift={shift:+.4f}")

def classify4(ndvi, valid):
    c = np.full(ndvi.shape, -1, dtype=np.int8)
    c[valid & (ndvi < 0.05)] = 0
    c[valid & (ndvi >= 0.05) & (ndvi < 0.25)] = 1
    c[valid & (ndvi >= 0.25) & (ndvi < 0.55)] = 2
    c[valid & (ndvi >= 0.55)] = 3
    return c

for y in YEARS:
    raw[y]["cls"] = classify4(raw[y]["ndvi_adj"], raw[y]["valid"])
    dist = [int((raw[y]["cls"] == i).sum()) for i in range(4)]
    print(f"{y}: class dist (own valid, incl out-of-AOI) = {dist}")

pickle.dump(raw, open(f"{DATA_DIR}/sequence_raw.pkl", "wb"))
print("saved sequence_raw.pkl")
