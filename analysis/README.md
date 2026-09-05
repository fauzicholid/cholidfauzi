# Land cover change analysis: CA-Markov and ConvLSTM

Scripts used to build and validate two 2025 land cover projections for West
Java (Jawa Barat) from Sentinel-2 imagery preprocessed by
`../preprocessing/sentinel2_mosaic_clip.py`: a classical Cellular
Automata-Markov Chain (CA-Markov) model, and a Convolutional LSTM
(ConvLSTM) deep-learning alternative.

## Pipeline

1. **Classification inputs** (not included here — produced upstream from the
   preprocessed Sentinel-2 mosaics): a 4-class NDVI-based classification
   (0 = water, 1 = non-vegetated, 2 = sparse vegetation, 3 = dense
   vegetation) for each year, radiometrically normalized to a common
   reference year via median-shift on NDVI, on a shared analysis grid.

2. `build_sequence_dataset.py` — loads the 2017-2024 per-year classifications,
   downsamples them to a common tractable grid, and saves the stacked
   sequence (`seq_X.npy`, `seq_cls.npy`, `seq_valid_common.npy`) used by
   `train_convlstm.py`.

3. `ca_markov.py` — applies an annualized transition matrix
   (`P1_adjusted.npy`, estimated from the 2019 and 2024 classifications and
   annualized via matrix fractional power) to the 2024 classification,
   using a 7x7-neighborhood-density, quota-based Cellular Automata rule, to
   project 2025 land cover (`proj2025.npy`).

4. `ca_validate.py` — the same CA allocation procedure run out-of-sample:
   a transition matrix estimated only from the independent 2019-2021 pair
   is projected 3 years forward and applied to the 2021 classification to
   simulate 2024 (`sim2024.npy`), for comparison against the actual 2024
   classification the transition matrix never saw.

5. `train_convlstm.py` — trains a custom ConvLSTM (2-layer CNN encoder,
   one ConvLSTM cell, 2-layer CNN decoder) on 3-year-input to 1-year-target
   sliding windows over the 2017-2024 sequence, holds out the window
   targeting 2024 as the test set, and produces a genuine forward
   projection to 2025 from real 2022-2024 input.

Per-class accuracy metrics reported in the paper (Overall Accuracy, Kappa,
Producer's/User's Accuracy, and Figure of Merit for the CA-Markov model)
were computed from the confusion matrices between each script's output and
the corresponding actual classification, following the standard formulas in
Pontius et al. (2007), *Annals of Regional Science* 42(1):11-37 — the
`train_convlstm.py` script includes this computation for its own test set as
a worked example; the same procedure was applied to `sim2024.npy` vs. the
actual 2024 classification for the CA-Markov validation numbers, but that
step was run ad hoc rather than saved as a standalone script in this
repository.

## Data layout

Each script reads from `analysis/data/` by default; override with the
`LC_DATA_DIR` environment variable. None of the intermediate `.npy`/`.pkl`
arrays are committed here (they are large per-pixel rasters regenerated from
the Sentinel-2 mosaics) — place the following in that directory before
running:

- `geo.pkl`, `valid.npy` — reference grid transform/CRS and the AOI validity
  mask
- `cls24_4.npy`, `cls21_adj.npy` — 2024 and radiometrically-normalized 2021
  classifications
- `P1_adjusted.npy`, `P3_sim2024.npy` — annualized transition matrices (full
  and validation)
- per-year band arrays and classifications for `build_sequence_dataset.py`
  (see that script for the exact filenames expected per year, 2017-2024)

## Requirements

`numpy`, `scipy`, `rasterio`, `Pillow`, `torch` (CPU is sufficient).
