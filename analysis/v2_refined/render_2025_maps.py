"""Renders the two 2025 projection maps used as Fig. 3 and Fig. 4 in the
paper: refined CA-Markov's 2025 projection at native resolution, and
refined ConvLSTM v3's 2025 projection (majority vote across 5 seeds) at
its native coarse 300x233 grid, upsampled by nearest-neighbor so the true
prediction resolution stays visible rather than being smoothed away.

Uses the same RGB palette as the existing 2024 land-cover map (Fig. 2):
dark green = dense vegetation/forest, light green = sparse vegetation/
agriculture, red = non-vegetated, blue = water, black = outside AOI/no data.
"""
import os
import numpy as np
from PIL import Image
from scipy import stats as sp_stats

DATA_DIR = os.environ.get("LC_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
OUT_DIR = os.environ.get("LC_OUT_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures"))

PALETTE = {
    0: (58, 110, 165),   # Water - blue
    1: (150, 60, 60),    # Non-vegetated - red/maroon
    2: (150, 189, 101),  # Sparse vegetation - light green
    3: (35, 102, 64),    # Dense vegetation - dark green
    -1: (0, 0, 0),       # nodata - black
}


def colorize(cls):
    h, w = cls.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, rgb in PALETTE.items():
        out[cls == k] = rgb
    return out


os.makedirs(OUT_DIR, exist_ok=True)

# ---- CA-Markov 2025 (refined, native resolution) ----
proj = np.load(f"{DATA_DIR}/proj2025_v2.npy")
valid = np.load(f"{DATA_DIR}/valid.npy")
ca_cls = np.where(valid, proj, -1)
Image.fromarray(colorize(ca_cls)).save(f"{OUT_DIR}/camarkov_2025_map.png")
print("Saved camarkov_2025_map.png:", ca_cls.shape,
      "class dist:", [int((ca_cls == i).sum()) for i in range(4)])

# ---- ConvLSTM v3 2025 (refined, majority vote across 5 seeds, coarse grid) ----
valid_common = np.load(f"{DATA_DIR}/seq_valid_common_v2.npy")  # (233,300)
preds = np.stack(
    [np.load(f"{DATA_DIR}/convlstm_v3_pred2025_seed{s}.npy") for s in range(5)], axis=0
)  # (5,233,300)
maj = sp_stats.mode(preds, axis=0, keepdims=False).mode
conv_cls_coarse = np.where(valid_common, maj, -1).astype(np.int8)
print("ConvLSTM coarse class dist (majority vote):",
      [int((conv_cls_coarse == i).sum()) for i in range(4)])

# Upsample via NEAREST to the native grid's shape for side-by-side comparability.
# This is deliberately blocky -- it is the model's actual prediction resolution,
# not a rendering artifact.
H, W = valid.shape
conv_cls_up_img = Image.fromarray((conv_cls_coarse + 1).astype(np.uint8), mode="L").resize(
    (W, H), resample=Image.NEAREST
)
conv_cls_native = np.array(conv_cls_up_img).astype(np.int16) - 1
conv_cls_native = np.where(valid, conv_cls_native, -1)  # clip to the true province boundary
Image.fromarray(colorize(conv_cls_native)).save(f"{OUT_DIR}/convlstm_2025_map.png")
print("Saved convlstm_2025_map.png:", conv_cls_native.shape)
