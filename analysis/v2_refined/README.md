# Independent ground-truth validation, NDMI-refined classification, and rebuilt models

Every accuracy figure in `../ca_validate.py`, `../train_convlstm.py`, and
`../experiments/` is self-referential: model output is checked only against
other classifications produced by this study's own NDVI-threshold pipeline,
never against an external reference. These scripts close that gap, diagnose
what it found, and rebuild both model families on a corrected classification.

## Pipeline

1. **Independent reference layer** (not included here — a third-party,
   17-class Indonesian government land-cover vector product; point
   `LC_REF_FILE` at a local copy with a `"Legenda"` attribute using the
   class names in `CROSSWALK`, or adjust the crosswalk for a different
   reference layer's schema). No acquisition-date field is present in the
   layer's own metadata, so its exact vintage is undocumented — treated
   throughout as a check on spatial pattern and classification logic, not
   as a perfectly contemporaneous 2024 reference.

2. `validate_against_reference.py` — rasterizes the reference layer directly
   onto this study's own analysis grid (`rasterio.features.rasterize`, not a
   point sample) and computes a pixel-exact confusion matrix against the
   existing pure-NDVI 2024 classification (`cls24_4.npy`), plus a sensitivity
   check excluding three crosswalk-ambiguous reference classes. *Finding:*
   OA = 43.8%, Kappa = 0.155 — far below any self-referential figure in this
   project — concentrated in reference-Sparse pixels misclassified as Dense
   (PA = 25.2%), consistent with a single greenness index being unable to
   separate closed-canopy forest from healthy, high-NDVI cropland/plantation.
   The sensitivity check moves OA by less than 1 point for the *refined*
   classification (55.5% → 56.4%) but by more for the *initial* one (43.8%
   → 40.5%, same held-out split) — in the direction of an even lower
   initial-classification accuracy, not a higher one, so the ambiguous
   classes are not the reason the initial classification looks poor.
   Per-class Producer's Accuracy shifts by ≤1 point either way; User's
   Accuracy for Dense vegetation drops 5-6 points once the ambiguous
   Perkebunan polygons are excluded, under both classifications.

3. `improve_classification.py` — computes NDMI = (B08−B11)/(B08+B11) (B11
   fetched, cloud-masked, and reprojected onto the analysis grid the same
   way B02/B03/B04/B08 were) and fits a threshold, on a spatial checkerboard
   train split against the reference layer only, that reassigns NDVI≥0.55
   candidates from Dense to Sparse vegetation when NDMI falls below it.
   *Finding:* best threshold = 0.2680; on the held-out test split, OA rises
   43.8% → 55.5% and Sparse PA rises 25.2% → 64.3%, at a real cost to Dense
   PA (91.1% → 53.2%) — a genuine but partial correction, not a solved
   classification problem.

4. `build_sequence_dataset_v2.py` — rebuilds the full 2017-2024 sequence
   using the NDMI-refined classifier. NDMI turned out to need the same kind
   of cross-date median-shift normalization already applied to NDVI (worst
   in 2019 and 2023, the same two years with the largest NDVI offsets) —
   applied here before classification, the same as `../build_sequence_dataset.py`
   does for NDVI. Saves `sequence_raw_v2.pkl`.

5. `rebuild_ca_markov_v2.py` — re-runs `../ca_markov.py` / `../ca_validate.py`
   in full on the refined classifications, with the same cap-ordering
   convention (annualized rate capped for the main 5-year matrix, raw
   2-year rate capped for the validation matrix). *Finding:* refined 2025
   Non-vegetated projection +19.0% (vs. +15.9% on the original
   classification); out-of-sample validation OA falls (68.3% → 61.7%) but
   FoM rises (22.2% → 25.7%) — a single, stable, directionally plausible
   forecast either way.

6. `train_convlstm_v2.py` / `train_convlstm_v3.py` — retrain ConvLSTM on the
   refined labels with a three-channel input (NDVI, brightness, NDMI — added
   so the model has access to the same information its own targets depend
   on), 5 seeds each. v2 uses full inverse-frequency class weighting; v3
   dampens the same weights by a square root. *Finding:* v2 reaches
   Non-vegetated PA = 73.1% but UA = 13% (severe false-positive inflation,
   an implausible 2025 forecast of 8,420.6±706.6 km², a near-quadrupling in
   one year). v3 improves overall OA/Kappa substantially (72.73%/0.4509,
   both exceeding refined CA-Markov) but leaves Non-vegetated recall
   unstable across seeds (12.9-42.4%, one seed collapsing) and its own 2025
   forecast noisy (1,580.0±550.3 km²). **v3 is the configuration used in the
   final paper**; v2 is retained as a documented cautionary finding about
   weighting-scheme sensitivity, not discarded.

7. `finalize_v2_comparison.py` — computes 2025 province-wide area
   projections for ConvLSTM v2 across all 5 seeds and an exploratory
   coarse-grid three-way check of the refined classification against the
   independent reference. The corresponding v3 areas
   (998.8±126.9 / 1,580.0±550.3 / 25,333.5±617.5 / 7,610.1±619.6 km² for
   Air / Non-vegetasi / Veg. jarang / Veg. rapat) were computed with a short
   one-off snippet in the original session rather than a saved script; they
   are reported here for completeness since the paper cites them directly.

8. `rebuild_ca_markov_v2_coarse.py` — re-runs CA-Markov a second time on the
   refined classifications, at ConvLSTM's own 300x233 coarse grid (mirroring
   `../experiments/exp1_ca_markov_coarse_grid.py`, but on `sequence_raw_v2.pkl`
   instead of the original `sequence_raw.pkl`). Added during a third
   peer-review pass: comparing refined CA-Markov's *native*-resolution
   validation figures directly against refined ConvLSTM's necessarily
   *coarse*-grid figures (as the paper first did) repeats the exact
   resolution confound Section III-H/IV-F diagnosed for the original
   classification, without ever re-checking whether it recurs under the
   refined labels. *Finding:* it does — refined CA-Markov's Non-vegetated PA
   collapses from 46.3% (native) to 10.2% (coarse), matching the same
   pattern found for the original classification (44.2% → 8.4%). At that
   shared, resolution-matched grid, refined ConvLSTM v3 (OA 72.73%, Kappa
   0.4509, PA[Non-veg] 34.4%) still exceeds refined CA-Markov (OA 61.82%,
   Kappa 0.2871, PA[Non-veg] 10.2%) on every metric — the Kappa gap is
   larger under the fair comparison (0.4509 vs. 0.2871) than the unmatched
   one had suggested (0.4509 vs. 0.3194), so the fix strengthens rather than
   undermines the paper's ConvLSTM-Kappa claim.

9. `render_2025_maps.py` — renders the two 2025 projection maps used as
   Fig. 3 and Fig. 4 in the paper: refined CA-Markov's projection at native
   resolution, and refined ConvLSTM v3's projection (majority vote across
   the 5 seeds) at its actual native 300x233 grid, upsampled by
   nearest-neighbor rather than smoothed, so the resolution difference
   between the two models stays visually honest. Saves to `../../paper/figures/`
   by default (override with `LC_OUT_DIR`). Uses the same RGB palette as the
   existing 2024 land-cover map (Fig. 2).

## Net result

Rebuilding both models on the refined classification did not produce a
clean verdict. Refined CA-Markov trades OA for FoM and keeps a single,
stable 2025 Non-vegetated forecast; refined ConvLSTM (v3), even measured
fairly at matched resolution (step 8), reaches a higher Kappa but cannot,
under either weighting scheme tested, produce a Non-vegetated forecast that
is simultaneously precise, sensitive, and stable across seeds. See the
paper's Discussion (Section V) for the full read of this result.

## Data layout

Each script reads from `../data` by default; override with the
`LC_DATA_DIR` environment variable (same convention as `../` and
`../experiments`). In addition to everything `../build_sequence_dataset.py`
already expects there, these scripts also expect:

- `ndmi.npy`, `b11.npy` — 2024 NDMI and B11 on the analysis grid
- `{year}_B11_on2024grid.npy` for 2017-2023 — B11 for each historical year,
  fetched and cloud-masked the same way as the existing `{year}_B0X.npy`
  arrays, reprojected directly onto the 2024 grid
- a copy of the independent reference layer, pointed to by `LC_REF_FILE`
  (see step 1 above; not committed to this repository)

## Requirements

Same as `../` (`numpy`, `scipy`, `rasterio`, `Pillow`, `torch`), plus
`geopandas` for reading and rasterizing the independent reference layer.
