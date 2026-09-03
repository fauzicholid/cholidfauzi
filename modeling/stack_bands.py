#!/usr/bin/env python3
"""Stack the per-band GeoTIFFs produced by preprocessing/sentinel2_mosaic_clip.py
into a single multi-band raster, ready for CNN patch extraction.

sentinel2_mosaic_clip.py writes one file per band, e.g.:
    B02_mosaic_clipped.tif
    B03_mosaic_clipped.tif
    B04_mosaic_clipped.tif
    B08_mosaic_clipped.tif
All of them share the same grid (same transform/CRS/shape) because they were
produced from the same AOI clip, so they can be stacked band-for-band.

Usage:
    python stack_bands.py \\
        --input-dir /path/to/preprocessing/output \\
        --bands B02 B03 B04 B08 \\
        --output /path/to/stack.tif
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio

DEFAULT_BANDS = ["B02", "B03", "B04", "B08"]


def find_band_file(input_dir: Path, band_code: str) -> Path:
    matches = sorted(input_dir.glob(f"{band_code}_mosaic_clipped.tif"))
    if not matches:
        raise FileNotFoundError(
            f"No '{band_code}_mosaic_clipped.tif' found under {input_dir}. "
            "Run preprocessing/sentinel2_mosaic_clip.py first."
        )
    return matches[0]


def stack_bands(input_dir: Path, bands: list[str], output_path: Path) -> Path:
    band_paths = [find_band_file(input_dir, b) for b in bands]

    with rasterio.open(band_paths[0]) as ref:
        profile = ref.profile.copy()
        ref_shape = (ref.height, ref.width)
        ref_transform = ref.transform
        ref_crs = ref.crs

    arrays = []
    for band_code, path in zip(bands, band_paths):
        with rasterio.open(path) as src:
            if (src.height, src.width) != ref_shape:
                raise ValueError(
                    f"Band {band_code} ({path}) has shape {(src.height, src.width)}, "
                    f"expected {ref_shape}. All bands must share the same grid -- "
                    "re-run preprocessing with a consistent AOI/resolution."
                )
            if src.transform != ref_transform or src.crs != ref_crs:
                raise ValueError(
                    f"Band {band_code} ({path}) has a different transform/CRS than "
                    f"{bands[0]}. All bands must be co-registered before stacking."
                )
            arrays.append(src.read(1))

    stacked = np.stack(arrays, axis=0)

    profile.update(count=len(bands), dtype=stacked.dtype)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(stacked)
        dst.descriptions = tuple(bands)

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory with per-band '<BAND>_mosaic_clipped.tif' files")
    parser.add_argument("--bands", nargs="+", default=DEFAULT_BANDS, help=f"Band codes to stack, in order (default: {DEFAULT_BANDS})")
    parser.add_argument("--output", required=True, type=Path, help="Output multi-band GeoTIFF path")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = stack_bands(args.input_dir, args.bands, args.output)
    print(f"Stacked {len(args.bands)} bands -> {output_path}")


if __name__ == "__main__":
    main()
