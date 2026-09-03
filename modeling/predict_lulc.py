#!/usr/bin/env python3
"""Run the trained CNN over a full stacked satellite raster and write out a
classified LULC (land use / land cover) GeoTIFF, one predicted class per pixel.

The image is padded and swept with a sliding window so every pixel gets a
patch-centered prediction; row-blocks are processed at a time so memory use
stays bounded regardless of raster size.

Usage:
    python predict_lulc.py \\
        --image stack.tif --model-dir runs/lulc_v1 --output lulc_prediction.tif
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from numpy.lib.stride_tricks import sliding_window_view
from tensorflow import keras

NODATA_CLASS = 255

# A small default palette (RGB) so the output GeoTIFF renders with sensible
# colors out of the box; extend/replace as needed for your class scheme.
DEFAULT_PALETTE = [
    (0, 100, 0),     # e.g. forest / vegetation
    (34, 139, 34),   # e.g. agriculture / cropland
    (210, 180, 140), # e.g. bare land
    (30, 144, 255),  # e.g. water
    (220, 20, 60),   # e.g. built-up / urban
    (255, 215, 0),
    (128, 0, 128),
    (0, 206, 209),
]


def load_metadata(model_dir: Path) -> dict:
    with open(model_dir / "model_metadata.json") as f:
        return json.load(f)


def predict_block(model, block: np.ndarray, patch_size: int, n_bands: int, batch_size: int) -> np.ndarray:
    """block: (rows_padded, cols_padded, n_bands) -> (rows, cols) predicted class indices."""
    rows_padded, cols_padded, _ = block.shape
    rows = rows_padded - patch_size + 1
    cols = cols_padded - patch_size + 1

    windows = sliding_window_view(block, (patch_size, patch_size, n_bands))
    # windows shape: (rows, cols, 1, patch_size, patch_size, n_bands) -> squeeze the band-window axis
    windows = windows.reshape(rows, cols, patch_size, patch_size, n_bands)
    flat = windows.reshape(rows * cols, patch_size, patch_size, n_bands)

    preds = model.predict(flat, batch_size=batch_size, verbose=0)
    class_idx = np.argmax(preds, axis=1)
    return class_idx.reshape(rows, cols)


def predict_lulc(image_path: Path, model_dir: Path, output_path: Path, row_chunk: int, batch_size: int):
    metadata = load_metadata(model_dir)
    patch_size = metadata["patch_size"]
    n_bands = metadata["n_bands"]
    band_mean = np.array(metadata["band_mean"], dtype=np.float32)
    band_std = np.array(metadata["band_std"], dtype=np.float32)
    classes = metadata["classes"]
    class_labels = metadata["class_labels"]

    model = keras.models.load_model(model_dir / "lulc_cnn.keras")

    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)  # (bands, H, W)
        profile = src.profile.copy()
        nodata = src.nodata

    if image.shape[0] != n_bands:
        raise ValueError(
            f"Image has {image.shape[0]} bands but the model was trained on {n_bands}; "
            "use the same --bands order as during training."
        )

    valid_mask = None
    if nodata is not None:
        valid_mask = ~np.all(image == nodata, axis=0)

    image = (image - band_mean[:, None, None]) / band_std[:, None, None]
    image_hwc = np.transpose(image, (1, 2, 0))  # (H, W, bands)

    half = patch_size // 2
    padded = np.pad(image_hwc, ((half, half), (half, half), (0, 0)), mode="reflect")

    H, W = image_hwc.shape[:2]
    output = np.full((H, W), NODATA_CLASS, dtype=np.uint8)

    for row_start in range(0, H, row_chunk):
        row_end = min(row_start + row_chunk, H)
        block = padded[row_start: row_end + 2 * half, :, :]
        class_idx = predict_block(model, block, patch_size, n_bands, batch_size)
        output[row_start:row_end, :] = class_idx.astype(np.uint8)
        print(f"  predicted rows {row_start}:{row_end} / {H}")

    if valid_mask is not None:
        output[~valid_mask] = NODATA_CLASS

    profile.update(count=1, dtype="uint8", nodata=NODATA_CLASS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output, 1)
        colormap = {
            i: (*DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)], 255) for i in range(len(classes))
        }
        colormap[NODATA_CLASS] = (0, 0, 0, 0)
        dst.write_colormap(1, colormap)
        dst.update_tags(class_labels=json.dumps(dict(zip(range(len(classes)), class_labels))))

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, type=Path, help="Stacked multi-band raster to classify (same band order used for training)")
    parser.add_argument("--model-dir", required=True, type=Path, help="Directory containing lulc_cnn.keras and model_metadata.json (from train_cnn.py)")
    parser.add_argument("--output", required=True, type=Path, help="Output classified LULC GeoTIFF path")
    parser.add_argument("--row-chunk", type=int, default=256, help="Rows processed per batch to bound memory use (default: 256)")
    parser.add_argument("--batch-size", type=int, default=512, help="Model prediction batch size (default: 512)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = predict_lulc(args.image, args.model_dir, args.output, args.row_chunk, args.batch_size)
    print(f"Wrote classified LULC map -> {output_path}")


if __name__ == "__main__":
    main()
