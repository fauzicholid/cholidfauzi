#!/usr/bin/env python3
"""Build a CNN training set of labeled image patches from a stacked satellite
raster (see stack_bands.py) and a land-use/land-cover (LULC) label source.

The label source can be either:
  * a label raster co-registered with the image (integer class codes, one
    value per pixel; 0 is treated as "unlabeled" and ignored), or
  * a vector file (GeoJSON/Shapefile) of labeled polygons/points with an
    attribute column holding the class name or code -- it is rasterized onto
    the image grid first.

For every labeled pixel a square patch of size --patch-size, centered on that
pixel, is cut from the image (the image is edge-padded so patches near the
border are still valid) and paired with the pixel's class. The result is
saved as a single .npz file containing:
    X            float32 array, shape (N, patch_size, patch_size, n_bands)
    y            int64 array, shape (N,) -- class indices into `classes`
    classes      the class codes/names, in the order y indexes into
    band_mean    per-band mean used for normalization, shape (n_bands,)
    band_std     per-band std used for normalization, shape (n_bands,)

Usage (label raster):
    python prepare_training_patches.py \\
        --image stack.tif --label-raster lulc_labels.tif \\
        --patch-size 15 --output training_patches.npz

Usage (label vector):
    python prepare_training_patches.py \\
        --image stack.tif --label-vector lulc_samples.geojson \\
        --label-field class_name --patch-size 15 --output training_patches.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize


def load_image(image_path: Path):
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)  # (bands, H, W)
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)
    return image, transform, crs, shape


def load_label_raster(label_path: Path, ref_shape, ref_transform, ref_crs) -> np.ndarray:
    with rasterio.open(label_path) as src:
        if (src.height, src.width) != ref_shape:
            raise ValueError(
                f"Label raster shape {(src.height, src.width)} does not match "
                f"image shape {ref_shape}; labels must be co-registered with the image "
                "(same grid as the stacked raster)."
            )
        labels = src.read(1)
    return labels


def rasterize_label_vector(vector_path: Path, label_field: str, ref_shape, ref_transform, ref_crs):
    import geopandas as gpd

    gdf = gpd.read_file(vector_path)
    if gdf.crs is None:
        raise ValueError(f"Label vector {vector_path} has no CRS defined")
    if gdf.crs != ref_crs:
        gdf = gdf.to_crs(ref_crs)
    if label_field not in gdf.columns:
        raise ValueError(f"Label field '{label_field}' not found in {vector_path}; columns: {list(gdf.columns)}")

    class_values = sorted(gdf[label_field].dropna().unique().tolist())
    class_to_code = {cls: i + 1 for i, cls in enumerate(class_values)}  # reserve 0 = unlabeled

    shapes = [(geom, class_to_code[val]) for geom, val in zip(gdf.geometry, gdf[label_field]) if val in class_to_code]
    labels = rasterize(
        shapes,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0,
        dtype=np.int32,
    )
    code_to_class = {v: k for k, v in class_to_code.items()}
    return labels, code_to_class


def extract_patches(image: np.ndarray, labels: np.ndarray, patch_size: int, max_per_class: int | None, seed: int):
    if patch_size % 2 == 0:
        raise ValueError("--patch-size must be odd so patches have a well-defined center pixel")

    rng = np.random.default_rng(seed)
    half = patch_size // 2
    n_bands = image.shape[0]

    padded = np.pad(image, ((0, 0), (half, half), (half, half)), mode="reflect")

    rows, cols = np.nonzero(labels)
    class_codes = labels[rows, cols]

    if max_per_class is not None:
        keep_idx = []
        for code in np.unique(class_codes):
            idx = np.nonzero(class_codes == code)[0]
            if len(idx) > max_per_class:
                idx = rng.choice(idx, size=max_per_class, replace=False)
            keep_idx.append(idx)
        keep_idx = np.concatenate(keep_idx)
        rng.shuffle(keep_idx)
        rows, cols, class_codes = rows[keep_idx], cols[keep_idx], class_codes[keep_idx]

    classes = sorted(np.unique(class_codes).tolist())
    code_to_index = {code: i for i, code in enumerate(classes)}

    n = len(rows)
    X = np.empty((n, patch_size, patch_size, n_bands), dtype=np.float32)
    y = np.empty((n,), dtype=np.int64)
    for i, (r, c, code) in enumerate(zip(rows, cols, class_codes)):
        pr, pc = r + half, c + half  # offset into padded array
        patch = padded[:, pr - half: pr + half + 1, pc - half: pc + half + 1]
        X[i] = np.transpose(patch, (1, 2, 0))  # (patch, patch, bands)
        y[i] = code_to_index[code]

    return X, y, classes


def compute_band_stats(image: np.ndarray):
    n_bands = image.shape[0]
    mean = np.zeros(n_bands, dtype=np.float32)
    std = np.ones(n_bands, dtype=np.float32)
    for b in range(n_bands):
        band = image[b]
        valid = band[np.isfinite(band)]
        if valid.size:
            mean[b] = valid.mean()
            std[b] = valid.std() or 1.0
    return mean, std


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, type=Path, help="Stacked multi-band satellite raster (see stack_bands.py)")
    label_group = parser.add_mutually_exclusive_group(required=True)
    label_group.add_argument("--label-raster", type=Path, help="Co-registered label raster (integer class codes, 0 = unlabeled)")
    label_group.add_argument("--label-vector", type=Path, help="Vector file (GeoJSON/Shapefile) of labeled samples")
    parser.add_argument("--label-field", help="Attribute column holding the class name/code (required with --label-vector)")
    parser.add_argument("--patch-size", type=int, default=15, help="Odd patch side length in pixels (default: 15)")
    parser.add_argument("--max-per-class", type=int, default=None, help="Cap samples per class to balance the dataset (default: no cap)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for per-class subsampling")
    parser.add_argument("--output", required=True, type=Path, help="Output .npz path")
    args = parser.parse_args()
    if args.label_vector and not args.label_field:
        parser.error("--label-field is required when using --label-vector")
    return args


def main():
    args = parse_args()
    image, transform, crs, shape = load_image(args.image)

    if args.label_raster:
        labels = load_label_raster(args.label_raster, shape, transform, crs)
        class_names = {int(c): str(int(c)) for c in np.unique(labels) if c != 0}
    else:
        labels, class_names = rasterize_label_vector(args.label_vector, args.label_field, shape, transform, crs)

    n_labeled = int(np.count_nonzero(labels))
    if n_labeled == 0:
        raise ValueError("No labeled pixels found -- check the label source and that it overlaps the image AOI")
    print(f"Found {n_labeled} labeled pixels across {len(np.unique(labels)) - 1} classes")

    X, y, classes = extract_patches(image, labels, args.patch_size, args.max_per_class, args.seed)
    band_mean, band_std = compute_band_stats(image)

    class_labels = [class_names.get(code, str(code)) for code in classes]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=X,
        y=y,
        classes=np.array(classes),
        class_labels=np.array(class_labels),
        band_mean=band_mean,
        band_std=band_std,
        patch_size=args.patch_size,
    )

    counts = {class_labels[i]: int(np.sum(y == i)) for i in range(len(classes))}
    print(f"Wrote {len(y)} patches ({args.patch_size}x{args.patch_size}, {X.shape[-1]} bands) -> {args.output}")
    print("Class distribution:", json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
