#!/usr/bin/env python3
"""Mosaic Sentinel-2 band tiles and clip the result to an area of interest (AOI).

Usage:
    python sentinel2_mosaic_clip.py \\
        --input-dir /path/to/sentinel2_tiles \\
        --aoi /path/to/aoi.geojson \\
        --output-dir /path/to/output \\
        --bands B02 B03 B04 B08

Input directory is expected to hold Sentinel-2 band raster tiles (.jp2 or .tif),
one or more tiles per band, with the band code (e.g. "B04") somewhere in the
filename -- the default naming produced by Copernicus Sentinel-2 SAFE products
(e.g. "T33UVP_20230601T101031_B04_10m.jp2"). Tiles for the same band are
mosaicked together, then clipped to the AOI geometry.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge

DEFAULT_BANDS = ["B02", "B03", "B04", "B08"]


def find_band_files(input_dir: Path, band_code: str) -> list[Path]:
    matches = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".jp2", ".tif", ".tiff")
        and band_code in p.stem
    )
    if not matches:
        raise FileNotFoundError(
            f"No raster tiles matching band '{band_code}' found under {input_dir}"
        )
    return matches


def mosaic_band(paths: list[Path]):
    datasets = [rasterio.open(p) for p in paths]
    try:
        mosaic_array, mosaic_transform = merge(datasets)
        crs = datasets[0].crs
        nodata = datasets[0].nodata
    finally:
        for ds in datasets:
            ds.close()
    return mosaic_array, mosaic_transform, crs, nodata


def clip_to_aoi(mosaic_array: np.ndarray, mosaic_transform, crs, nodata, aoi_path: Path):
    aoi = gpd.read_file(aoi_path)
    if aoi.crs is None:
        raise ValueError(f"AOI file {aoi_path} has no CRS defined")
    if aoi.crs != crs:
        aoi = aoi.to_crs(crs)
    geometries = list(aoi.geometry)

    height, width = mosaic_array.shape[-2], mosaic_array.shape[-1]
    count = mosaic_array.shape[0] if mosaic_array.ndim == 3 else 1
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": mosaic_array.dtype,
        "crs": crs,
        "transform": mosaic_transform,
        "nodata": nodata,
    }

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(mosaic_array)
        with memfile.open() as dataset:
            clipped_array, clipped_transform = mask(dataset, geometries, crop=True, nodata=nodata)

    return clipped_array, clipped_transform, profile


def save_raster(output_path: Path, array: np.ndarray, transform, profile: dict):
    profile = dict(profile)
    profile.update(
        height=array.shape[-2],
        width=array.shape[-1],
        transform=transform,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array)


def process_band(input_dir: Path, band_code: str, aoi_path: Path, output_dir: Path) -> Path:
    paths = find_band_files(input_dir, band_code)
    mosaic_array, mosaic_transform, crs, nodata = mosaic_band(paths)
    clipped_array, clipped_transform, profile = clip_to_aoi(
        mosaic_array, mosaic_transform, crs, nodata, aoi_path
    )
    output_path = output_dir / f"{band_code}_mosaic_clipped.tif"
    save_raster(output_path, clipped_array, clipped_transform, profile)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing Sentinel-2 band tiles")
    parser.add_argument("--aoi", required=True, type=Path, help="Vector file (GeoJSON/Shapefile) defining the AOI")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write mosaicked/clipped bands to")
    parser.add_argument("--bands", nargs="+", default=DEFAULT_BANDS, help=f"Band codes to process (default: {DEFAULT_BANDS})")
    return parser.parse_args()


def main():
    args = parse_args()
    for band_code in args.bands:
        output_path = process_band(args.input_dir, band_code, args.aoi, args.output_dir)
        print(f"{band_code}: wrote {output_path}")


if __name__ == "__main__":
    main()
