# LULC CNN prediction

Predicts land use / land cover (LULC) classes from Sentinel-2 satellite
imagery using a patch-based CNN. Builds on the rasters produced by
[`preprocessing/sentinel2_mosaic_clip.py`](../preprocessing/sentinel2_mosaic_clip.py).

## Pipeline

```
Sentinel-2 tiles
      │  preprocessing/sentinel2_mosaic_clip.py
      ▼
per-band clipped GeoTIFFs (B02_mosaic_clipped.tif, B03_..., ...)
      │  stack_bands.py
      ▼
stack.tif (multi-band raster, one AOI, one grid)
      │  prepare_training_patches.py  (+ labeled samples)
      ▼
training_patches.npz
      │  train_cnn.py
      ▼
runs/<name>/lulc_cnn.keras + model_metadata.json
      │  predict_lulc.py  (+ stack.tif, incl. areas with no labels)
      ▼
lulc_prediction.tif  (one predicted class per pixel)
```

## 1. Stack the preprocessed bands

```bash
python stack_bands.py \
    --input-dir /path/to/preprocessing/output \
    --bands B02 B03 B04 B08 \
    --output stack.tif
```

## 2. Prepare training patches

You need labeled samples: either a co-registered label raster, or a vector
file of polygons/points with a class attribute (e.g. digitized from visual
interpretation, an existing LULC map, or field survey points).

```bash
python prepare_training_patches.py \
    --image stack.tif \
    --label-vector lulc_samples.geojson --label-field class_name \
    --patch-size 15 --max-per-class 5000 \
    --output training_patches.npz
```

`--patch-size` must be odd; 15 is a reasonable default for 10 m Sentinel-2
bands (a ~150 m x 150 m context window per prediction).

## 3. Train the CNN

```bash
python train_cnn.py \
    --patches training_patches.npz \
    --model-dir runs/lulc_v1 \
    --epochs 50
```

Writes `runs/lulc_v1/lulc_cnn.keras`, `model_metadata.json` (class list,
band normalization stats, patch size) and `training_history.json`.

## 4. Predict the full LULC map

```bash
python predict_lulc.py \
    --image stack.tif \
    --model-dir runs/lulc_v1 \
    --output lulc_prediction.tif
```

Sweeps the whole raster in row-blocks (`--row-chunk`) so memory use stays
bounded regardless of AOI size, and writes a single-band classified GeoTIFF
with a colormap and class labels embedded as metadata tags. Pixels where the
source image is entirely nodata are written as nodata (255) in the output.

## Model

`cnn_model.py` defines a small CNN (three conv blocks + global average
pooling) that classifies each patch by its center pixel's LULC class. Global
average pooling (rather than flattening) keeps the architecture valid for
any `--patch-size` used upstream, so patch size can be tuned per dataset
without touching the model code.

## Trying it with RTRW labels

West Java's RTRW (Rencana Tata Ruang Wilayah) pola-ruang layer is a natural
label source for step 2 -- it's an existing province-wide land-use
classification. See [`sample_data/`](sample_data/) for a placeholder version
(real class names and coordinates, made-up geometry/imagery) that proves the
pipeline runs end-to-end, plus links to where to get the real dataset and
how to swap it in.
