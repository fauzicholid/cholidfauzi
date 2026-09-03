#!/usr/bin/env python3
"""Generate a PLACEHOLDER RTRW-style pola ruang (land-use plan) polygon layer
and a matching synthetic satellite raster, so the CNN pipeline
(prepare_training_patches.py -> train_cnn.py -> predict_lulc.py) can be
exercised end-to-end with real West Java coordinates and real RTRW class
names, without depending on the actual government dataset.

This is NOT the official RTRW Provinsi Jawa Barat pola ruang dataset. The
real one (Perda RTRW Jabar No. 22/2010, updated 2029 revision) is published
by the West Java provincial government and is not reachable from this
sandboxed environment's network. Sources to get the real data:
    - https://opendata.jabarprov.go.id  (search "pola ruang" / "RTRW")
    - https://satupeta.jabarprov.go.id
    - https://tanahair.indonesia.go.id/sdi (Ina-Geoportal / ATR-BPN)

What this script produces instead is a small grid of polygons, tagged with
real RTRW pola-ruang class names, over a ~3 km x 3 km window near Bandung
that is verified to sit inside preprocessing/aoi/west_java_boundary.geojson,
plus a synthetic 4-band raster (random values standing in for Sentinel-2
B02/B03/B04/B08) covering the same window at 10 m resolution. Swap both out
for the real RTRW layer and a real stack_bands.py output once you have them
-- everything downstream (prepare_training_patches.py --label-field
kelas_pr, train_cnn.py, predict_lulc.py) works unchanged either way.

Usage:
    python generate_placeholder_rtrw.py --output-dir modeling/sample_data
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

# Real RTRW pola-ruang class names (kawasan lindung / budidaya categories used
# in West Java's spatial plan), reused here only as class labels.
POLA_RUANG_CLASSES = [
    "Kawasan Hutan Lindung",
    "Kawasan Pertanian Lahan Basah",
    "Kawasan Pertanian Lahan Kering",
    "Kawasan Perkebunan",
    "Kawasan Permukiman Perkotaan",
    "Kawasan Permukiman Perdesaan",
    "Kawasan Industri",
    "Badan Air",
]

CENTER_LON, CENTER_LAT = 107.6191, -6.9175  # near Bandung, inside West Java
HALF_WINDOW_DEG = 0.015  # ~3.3 km across
GRID_CELLS = 4  # 4x4 grid of pola-ruang polygons
PIXEL_SIZE_M = 10.0  # mimics Sentinel-2 10 m bands
UTM_CRS = "EPSG:32748"  # UTM zone 48S, covers West Java
BAND_NAMES = ["B02", "B03", "B04", "B08"]


def build_label_polygons() -> gpd.GeoDataFrame:
    boundary = gpd.read_file(Path(__file__).resolve().parents[2] / "preprocessing" / "aoi" / "west_java_boundary.geojson")
    window = box(
        CENTER_LON - HALF_WINDOW_DEG, CENTER_LAT - HALF_WINDOW_DEG,
        CENTER_LON + HALF_WINDOW_DEG, CENTER_LAT + HALF_WINDOW_DEG,
    )
    if not boundary.geometry.iloc[0].contains(window):
        raise RuntimeError("Placeholder window is not fully inside the West Java boundary; adjust CENTER_LON/CENTER_LAT")

    minx, miny, maxx, maxy = window.bounds
    step_x = (maxx - minx) / GRID_CELLS
    step_y = (maxy - miny) / GRID_CELLS

    rng = np.random.default_rng(42)
    rows = []
    for i in range(GRID_CELLS):
        for j in range(GRID_CELLS):
            cell = box(minx + i * step_x, miny + j * step_y, minx + (i + 1) * step_x, miny + (j + 1) * step_y)
            cls = POLA_RUANG_CLASSES[rng.integers(0, len(POLA_RUANG_CLASSES))]
            rows.append({"kelas_pr": cls, "kode_pr": POLA_RUANG_CLASSES.index(cls) + 1, "geometry": cell})

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf.attrs_note = "PLACEHOLDER data for pipeline testing -- not the official RTRW Jabar dataset"
    return gdf


def build_synthetic_raster(output_path: Path):
    gdf = gpd.GeoDataFrame(geometry=[box(
        CENTER_LON - HALF_WINDOW_DEG, CENTER_LAT - HALF_WINDOW_DEG,
        CENTER_LON + HALF_WINDOW_DEG, CENTER_LAT + HALF_WINDOW_DEG,
    )], crs="EPSG:4326").to_crs(UTM_CRS)
    minx, miny, maxx, maxy = gdf.total_bounds

    width = int((maxx - minx) / PIXEL_SIZE_M)
    height = int((maxy - miny) / PIXEL_SIZE_M)
    transform = from_origin(minx, maxy, PIXEL_SIZE_M, PIXEL_SIZE_M)

    rng = np.random.default_rng(7)
    # Rough, NOT radiometrically meaningful stand-ins for surface reflectance
    # (real Sentinel-2 L2A values would come from stack_bands.py instead).
    image = rng.integers(400, 3000, size=(len(BAND_NAMES), height, width)).astype(np.uint16)

    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": len(BAND_NAMES),
        "dtype": "uint16", "crs": UTM_CRS, "transform": transform, "nodata": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(image)
        dst.descriptions = tuple(BAND_NAMES)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent, help="Directory to write the placeholder files to")
    return parser.parse_args()


def main():
    args = parse_args()
    labels_path = args.output_dir / "rtrw_polaruang_placeholder.geojson"
    raster_path = args.output_dir / "synthetic_sentinel2_stack.tif"

    gdf = build_label_polygons()
    gdf.to_file(labels_path, driver="GeoJSON")
    print(f"Wrote {len(gdf)} placeholder pola-ruang polygons -> {labels_path}")

    build_synthetic_raster(raster_path)
    print(f"Wrote synthetic 4-band raster -> {raster_path}")


if __name__ == "__main__":
    main()
