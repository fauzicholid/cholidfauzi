#!/usr/bin/env python3
"""Mosaic Sentinel-2 band tiles, mask clouds, apply atmospheric correction,
and clip the result to an area of interest (AOI).

Usage:
    python sentinel2_mosaic_clip.py \\
        --input-dir /path/to/sentinel2_tiles \\
        --aoi preprocessing/aoi/west_java_boundary.geojson \\
        --output-dir /path/to/output \\
        --bands B02 B03 B04 B08

Input directory is expected to hold Sentinel-2 band raster tiles (.jp2 or .tif),
one or more tiles per band, with the band code (e.g. "B04") somewhere in the
filename -- the default naming produced by Copernicus Sentinel-2 SAFE products
(e.g. "T33UVP_20230601T101031_B04_10m.jp2"). Tiles for the same band are
mosaicked together, cloud-masked, atmospherically corrected, and clipped to
the AOI geometry.

Cloud masking uses the Scene Classification Layer (SCL) band shipped with
Sentinel-2 Level-2A products: pixels classified as no-data, saturated,
cloud shadow, cloud, or cirrus are set to nodata. It is skipped automatically
if no SCL band is found under --input-dir (e.g. Level-1C-only input), or can
be disabled with --no-cloud-mask.

Atmospheric correction defaults to Dark Object Subtraction (DOS), a simple
per-band haze-removal approximation suitable when full radiative-transfer
correction (e.g. Sen2Cor) hasn't already been applied upstream. If the input
is already Level-2A (surface reflectance), pass --atmos-correction none.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject

DEFAULT_BANDS = ["B02", "B03", "B04", "B08"]
SCL_BAND_CODE = "SCL"
# SCL classes: 0 no-data, 1 saturated/defective, 3 cloud shadow,
# 8/9 cloud medium/high probability, 10 thin cirrus.
DEFAULT_CLOUD_MASK_CLASSES = [0, 1, 3, 8, 9, 10]


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


def resample_to_match(src_array: np.ndarray, src_transform, src_crs, dst_shape, dst_transform, dst_crs) -> np.ndarray:
    """Reproject/resample a single-band array onto another band's grid (nearest-neighbor)."""
    if src_array.ndim == 3:
        src_array = src_array[0]
    dst_array = np.zeros(dst_shape, dtype=src_array.dtype)
    reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    return dst_array


def mask_clouds(array: np.ndarray, scl_array: np.ndarray, mask_classes: list[int], nodata) -> np.ndarray:
    """Set pixels whose SCL class is in mask_classes to nodata."""
    if scl_array.ndim == 3:
        scl_array = scl_array[0]
    cloud_mask = np.isin(scl_array, mask_classes)
    masked = array.copy()
    fill_value = nodata if nodata is not None else 0
    masked[:, cloud_mask] = fill_value
    return masked


def dark_object_subtraction(array: np.ndarray, nodata, dark_percentile: float = 1.0) -> np.ndarray:
    """Approximate atmospheric correction: subtract each band's dark-object value
    (a low percentile of its valid pixel values) from every pixel, clipped at 0.
    """
    dtype = array.dtype
    corrected = array.astype(np.float32)
    for i in range(corrected.shape[0]):
        band = corrected[i]
        valid = band[band != nodata] if nodata is not None else band.ravel()
        valid = valid[valid > 0]
        if valid.size == 0:
            continue
        dark_object = np.percentile(valid, dark_percentile)
        corrected[i] = np.clip(band - dark_object, 0, None)
    return corrected.astype(dtype)


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


def process_band(
    input_dir: Path,
    band_code: str,
    aoi_path: Path,
    output_dir: Path,
    apply_cloud_mask: bool = True,
    cloud_mask_classes: list[int] = DEFAULT_CLOUD_MASK_CLASSES,
    atmos_correction: str = "dos",
) -> Path:
    paths = find_band_files(input_dir, band_code)
    mosaic_array, mosaic_transform, crs, nodata = mosaic_band(paths)

    if apply_cloud_mask and band_code != SCL_BAND_CODE:
        try:
            scl_paths = find_band_files(input_dir, SCL_BAND_CODE)
        except FileNotFoundError:
            print(f"  {band_code}: no SCL band found under {input_dir}; skipping cloud mask")
        else:
            scl_array, scl_transform, scl_crs, _ = mosaic_band(scl_paths)
            scl_resampled = resample_to_match(
                scl_array, scl_transform, scl_crs,
                mosaic_array.shape[-2:], mosaic_transform, crs,
            )
            mosaic_array = mask_clouds(mosaic_array, scl_resampled, cloud_mask_classes, nodata)

    if atmos_correction == "dos" and band_code != SCL_BAND_CODE:
        mosaic_array = dark_object_subtraction(mosaic_array, nodata)

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
    parser.add_argument("--cloud-mask", dest="cloud_mask", action="store_true", default=True, help="Mask clouds/shadows using the SCL band (default: on)")
    parser.add_argument("--no-cloud-mask", dest="cloud_mask", action="store_false", help="Disable SCL-based cloud masking")
    parser.add_argument("--cloud-mask-classes", nargs="+", type=int, default=DEFAULT_CLOUD_MASK_CLASSES, help=f"SCL class values to mask out (default: {DEFAULT_CLOUD_MASK_CLASSES})")
    parser.add_argument("--atmos-correction", choices=["dos", "none"], default="dos", help="Atmospheric correction method: 'dos' (Dark Object Subtraction) or 'none' if input is already surface reflectance (default: dos)")
    return parser.parse_args()


def main():
    args = parse_args()
    for band_code in args.bands:
        output_path = process_band(
            args.input_dir, band_code, args.aoi, args.output_dir,
            apply_cloud_mask=args.cloud_mask,
            cloud_mask_classes=args.cloud_mask_classes,
            atmos_correction=args.atmos_correction,
        )
        print(f"{band_code}: wrote {output_path}")


if __name__ == "__main__":
    main()
