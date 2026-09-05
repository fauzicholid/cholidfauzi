"""EXPERIMENT 2 — Retrain ConvLSTM with two candidate fixes, isolated so you
can tell which one (if either) actually matters:

  FIX_LABELS  : replace Image.NEAREST label downsampling with
                reclassify-after-bilinear-downsample-of-NDVI (exp0 option C).
  FIX_WEIGHT  : replace plain cross-entropy with inverse-frequency class-
                weighted cross-entropy.

Runs all 4 combinations (baseline, +labels, +weight, +both) x N_SEEDS seeds
each, and reports OA / Kappa / per-class PA as mean +/- SD across seeds.
This directly answers: "is the collapse a labeling artifact, a class-
imbalance/loss artifact, both, or neither?"

Identical architecture/window/epoch settings to the original train_convlstm.py
so results are comparable line-for-line.
"""
import time, pickle, json, os, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

DATA_DIR = os.environ.get("LC_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
TARGET_SIZE = (300, 233)
N_CLASSES = 4
NAMES = ["Air", "Non-vegetasi", "Veg. jarang", "Veg. rapat"]
N_SEEDS = 5          # bump to 10 if you have time; 5 is enough to see the pattern
EPOCHS = 60          # unchanged from original, for comparability

raw = pickle.load(open(f"{DATA_DIR}/sequence_raw.pkl", "rb"))
W, H = TARGET_SIZE


def classify4(ndvi, valid):
    c = np.full(ndvi.shape, -1, dtype=np.int8)
    c[valid & (ndvi < 0.05)] = 0
    c[valid & (ndvi >= 0.05) & (ndvi < 0.25)] = 1
    c[valid & (ndvi >= 0.25) & (ndvi < 0.55)] = 2
    c[valid & (ndvi >= 0.55)] = 3
    return c


def resize_float(arr, size, resample):
    img = Image.fromarray(np.nan_to_num(arr, nan=0.0).astype(np.float32), mode="F")
    return np.array(img.resize(size, resample), dtype=np.float32)


def build_sequence(fix_labels: bool):
    ndvi_seq = np.zeros((len(YEARS), H, W), dtype=np.float32)
    bright_seq = np.zeros((len(YEARS), H, W), dtype=np.float32)
    valid_seq = np.zeros((len(YEARS), H, W), dtype=np.float32)
    cls_seq = np.zeros((len(YEARS), H, W), dtype=np.int8)

    for t, y in enumerate(YEARS):
        ndvi_seq[t] = resize_float(raw[y]["ndvi_adj"], (W, H), Image.BILINEAR)
        bright_seq[t] = resize_float(raw[y]["bright"], (W, H), Image.BILINEAR)
        valid_frac = resize_float(raw[y]["valid"].astype(np.float32), (W, H), Image.BOX)
        valid_seq[t] = (valid_frac > 0.5).astype(np.float32)

        if fix_labels:
            # OPTION C from exp0: classify the already-downsampled, smoothly
            # averaged NDVI, instead of nearest-subsampling discrete labels.
            cls_seq[t] = classify4(ndvi_seq[t], valid_seq[t] > 0.5)
        else:
            # ORIGINAL behaviour: nearest-neighbor subsample of native-res labels.
            cls_img = Image.fromarray(raw[y]["cls"].astype(np.uint8), mode="L")
            cls_seq[t] = np.array(cls_img.resize((W, H), Image.NEAREST), dtype=np.int8)

    bright_mean = bright_seq[valid_seq > 0].mean()
    bright_std = bright_seq[valid_seq > 0].std()
    bright_norm = (bright_seq - bright_mean) / (bright_std + 1e-6)
    X = np.stack([ndvi_seq, bright_norm], axis=1)
    valid_common = valid_seq.min(axis=0) > 0
    return X, cls_seq, valid_common


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=k // 2)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class CNNLSTM(nn.Module):
    def __init__(self, in_ch=2, cnn_ch=16, lstm_hid=16, n_classes=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, cnn_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(cnn_ch, cnn_ch, 3, padding=1), nn.ReLU(),
        )
        self.convlstm = ConvLSTMCell(cnn_ch, lstm_hid, k=3)
        self.decoder = nn.Sequential(
            nn.Conv2d(lstm_hid, cnn_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(cnn_ch, n_classes, 1),
        )
        self.lstm_hid = lstm_hid

    def forward(self, x_seq):
        Tt, C, H, W = x_seq.shape
        h = torch.zeros(1, self.lstm_hid, H, W)
        c = torch.zeros(1, self.lstm_hid, H, W)
        for t in range(Tt):
            feat = self.encoder(x_seq[t:t+1])
            h, c = self.convlstm(feat, h, c)
        out = self.decoder(h)
        return out


def run_one(seed, fix_labels, fix_weight, X, cls_seq, valid_common):
    torch.manual_seed(seed)
    np.random.seed(seed)

    window = 3
    windows = []
    for start in range(0, len(YEARS) - window):
        in_idx = list(range(start, start + window))
        tgt_idx = start + window
        windows.append((in_idx, tgt_idx, YEARS[tgt_idx]))
    train_windows = [w for w in windows if w[2] != 2024]
    test_window = [w for w in windows if w[2] == 2024][0]

    X_t = torch.from_numpy(X)
    valid_mask_t = torch.from_numpy(valid_common).float()

    class_weight = None
    if fix_weight:
        # inverse-frequency weights computed from TRAIN targets only (not test),
        # to avoid any leakage of test-year class balance into the loss.
        train_target_years = [w[2] for w in train_windows]
        counts = np.zeros(N_CLASSES)
        for _, tgt_idx, _ in train_windows:
            tgt = cls_seq[tgt_idx]
            for c in range(N_CLASSES):
                counts[c] += ((tgt == c) & valid_common).sum()
        counts = np.maximum(counts, 1)
        w = counts.sum() / (N_CLASSES * counts)
        class_weight = torch.tensor(w, dtype=torch.float32)

    model = CNNLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        for in_idx, tgt_idx, tgt_year in train_windows:
            model.train()
            opt.zero_grad()
            x_in = X_t[in_idx]
            tgt_arr = cls_seq[tgt_idx].astype(np.int64)
            tgt_arr = np.where(tgt_arr < 0, 0, tgt_arr)
            target = torch.from_numpy(tgt_arr).unsqueeze(0)
            logits = model(x_in)
            loss = F.cross_entropy(logits, target, weight=class_weight, reduction="none")[0]
            loss = (loss * valid_mask_t).sum() / valid_mask_t.sum()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        in_idx, tgt_idx, tgt_year = test_window
        logits = model(X_t[in_idx])
        pred = torch.argmax(logits, dim=1)[0].numpy()

    actual = cls_seq[tgt_idx]
    mask = valid_common & (actual >= 0) & (pred >= 0)

    CM = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for a in range(N_CLASSES):
        for p in range(N_CLASSES):
            CM[a, p] = int(((actual == a) & (pred == p) & mask).sum())
    total = CM.sum()
    OA = np.trace(CM) / total
    row_sums, col_sums = CM.sum(axis=1), CM.sum(axis=0)
    pe = (row_sums * col_sums).sum() / (total ** 2)
    kappa = (OA - pe) / (1 - pe) if pe < 1 else float("nan")
    PA = np.divide(np.diag(CM), row_sums, out=np.zeros(N_CLASSES), where=row_sums > 0)
    UA = np.divide(np.diag(CM), col_sums, out=np.zeros(N_CLASSES), where=col_sums > 0)
    return dict(OA=float(OA), kappa=float(kappa), PA=PA.tolist(), UA=UA.tolist(),
                CM=CM.tolist(), n=int(total))


configs = [
    ("baseline (original)",        False, False),
    ("+ fix_labels only",          True,  False),
    ("+ fix_weight only",          False, True),
    ("+ fix_labels + fix_weight",  True,  True),
]

all_results = {}
for name, fix_labels, fix_weight in configs:
    log(f"=== Config: {name} ===")
    X, cls_seq, valid_common = build_sequence(fix_labels)
    runs = []
    for seed in range(N_SEEDS):
        r = run_one(seed, fix_labels, fix_weight, X, cls_seq, valid_common)
        runs.append(r)
        log(f"  seed {seed}: OA={r['OA']*100:.2f}%  Kappa={r['kappa']:.3f}  "
            f"PA[Non-veg]={r['PA'][1]*100:.2f}%")
    all_results[name] = runs

    OAs = np.array([r["OA"] for r in runs])
    kappas = np.array([r["kappa"] for r in runs])
    PA1s = np.array([r["PA"][1] for r in runs])  # Non-vegetated PA
    log(f"  >>> OA = {OAs.mean()*100:.2f}% +/- {OAs.std()*100:.2f}%   "
        f"Kappa = {kappas.mean():.3f} +/- {kappas.std():.3f}   "
        f"PA[Non-veg] = {PA1s.mean()*100:.2f}% +/- {PA1s.std()*100:.2f}%")

json.dump(all_results, open(f"{DATA_DIR}/exp2_ablation_results.json", "w"), indent=2)
log("Saved exp2_ablation_results.json")

print("\n" + "=" * 90)
print(f"{'Config':32s} {'OA (mean±SD)':>18s} {'Kappa (mean±SD)':>18s} {'PA[Non-veg] (mean±SD)':>22s}")
print("=" * 90)
for name, _, _ in configs:
    runs = all_results[name]
    OAs = np.array([r["OA"] for r in runs]) * 100
    kappas = np.array([r["kappa"] for r in runs])
    PA1s = np.array([r["PA"][1] for r in runs]) * 100
    print(f"{name:32s} {OAs.mean():6.2f}±{OAs.std():4.2f}%     "
          f"{kappas.mean():5.3f}±{kappas.std():5.3f}      "
          f"{PA1s.mean():6.2f}±{PA1s.std():5.2f}%")
print("=" * 90)
print("\nRead this top to bottom: whichever fix produces the biggest jump in")
print("PA[Non-veg] mean (and doesn't blow up its SD) is the dominant real cause")
print("of the original collapse -- report that finding explicitly in Discussion,")
print("citing the mean+/-SD here as your evidence instead of a single run.")
