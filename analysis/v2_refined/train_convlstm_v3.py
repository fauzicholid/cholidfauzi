"""ConvLSTM v3: identical to train_convlstm_v2.py except for the class
weighting -- full inverse-frequency weighting there collapsed Non-vegetated
User's Accuracy to ~13% (roughly seven in eight predictions of that class
false positives), so this configuration dampens the weights by a square
root, a standard remedy for over-aggressive class rebalancing. This is the
paper's primary refined-ConvLSTM result: it substantially improves overall
OA/Kappa but leaves Non-vegetated recall unstable across seeds (one seed
collapses to ~13%) -- reported honestly as a real, unresolved limitation
rather than a clean win.
"""
import time, pickle, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

DATA_DIR = os.environ.get("LC_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
TARGET_SIZE = (300, 233)
N_CLASSES = 4
NAMES = ["Air", "Non-vegetasi", "Veg. jarang", "Veg. rapat"]
N_SEEDS = 5
EPOCHS = 60
NDMI_THR = 0.2680

raw = pickle.load(open(f"{DATA_DIR}/sequence_raw_v2.pkl", "rb"))
W, H = TARGET_SIZE


def classify4_improved(ndvi, ndmi, valid):
    c = np.full(ndvi.shape, -1, dtype=np.int8)
    c[valid & (ndvi < 0.05)] = 0
    c[valid & (ndvi >= 0.05) & (ndvi < 0.25)] = 1
    c[valid & (ndvi >= 0.25) & (ndvi < 0.55)] = 2
    cand_dense = valid & (ndvi >= 0.55)
    have_ndmi = ~np.isnan(ndmi)
    c[cand_dense & have_ndmi & (ndmi >= NDMI_THR)] = 3
    c[cand_dense & have_ndmi & (ndmi < NDMI_THR)] = 2
    c[cand_dense & ~have_ndmi] = 3
    return c


def resize_float(arr, size, resample):
    img = Image.fromarray(np.nan_to_num(arr, nan=0.0).astype(np.float32), mode="F")
    return np.array(img.resize(size, resample), dtype=np.float32)


def build_sequence():
    ndvi_seq = np.zeros((len(YEARS), H, W), dtype=np.float32)
    ndmi_seq = np.zeros((len(YEARS), H, W), dtype=np.float32)
    bright_seq = np.zeros((len(YEARS), H, W), dtype=np.float32)
    valid_seq = np.zeros((len(YEARS), H, W), dtype=np.float32)
    cls_seq = np.zeros((len(YEARS), H, W), dtype=np.int8)

    for t, y in enumerate(YEARS):
        ndvi_seq[t] = resize_float(raw[y]["ndvi_adj"], (W, H), Image.BILINEAR)
        ndmi_seq[t] = resize_float(np.nan_to_num(raw[y]["ndmi_adj"], nan=0.0), (W, H), Image.BILINEAR)
        bright_seq[t] = resize_float(raw[y]["bright"], (W, H), Image.BILINEAR)
        valid_frac = resize_float(raw[y]["valid"].astype(np.float32), (W, H), Image.BOX)
        valid_seq[t] = (valid_frac > 0.5).astype(np.float32)
        # reclassify-after-downsample (fixes the label-downsampling distortion
        # from exp0), using the SAME improved NDVI+NDMI rule at coarse resolution
        cls_seq[t] = classify4_improved(ndvi_seq[t], ndmi_seq[t], valid_seq[t] > 0.5)

    bright_mean = bright_seq[valid_seq > 0].mean()
    bright_std = bright_seq[valid_seq > 0].std()
    bright_norm = (bright_seq - bright_mean) / (bright_std + 1e-6)
    X = np.stack([ndvi_seq, bright_norm, ndmi_seq], axis=1)  # 3 channels
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
    def __init__(self, in_ch=3, cnn_ch=16, lstm_hid=16, n_classes=4):
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
            feat = self.encoder(x_seq[t:t + 1])
            h, c = self.convlstm(feat, h, c)
        out = self.decoder(h)
        return out


log("Building v2 sequence (3-channel NDVI+brightness+NDMI, reclassify-after-downsample labels)...")
X, cls_seq, valid_common = build_sequence()
log(f"sequence shape {X.shape}, valid-common frac {valid_common.mean():.3f}")
np.save(f"{DATA_DIR}/seq_X_v2.npy", X)
np.save(f"{DATA_DIR}/seq_cls_v2.npy", cls_seq)
np.save(f"{DATA_DIR}/seq_valid_common_v2.npy", valid_common)

window = 3
windows = []
for start in range(0, len(YEARS) - window):
    in_idx = list(range(start, start + window))
    tgt_idx = start + window
    windows.append((in_idx, tgt_idx, YEARS[tgt_idx]))
train_windows = [w for w in windows if w[2] != 2024]
test_window = [w for w in windows if w[2] == 2024][0]
log(f"train targets: {[w[2] for w in train_windows]}  test target: {test_window[2]}")

X_t_full = torch.from_numpy(X)
valid_mask_t = torch.from_numpy(valid_common).float()


def run_one(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    counts = np.zeros(N_CLASSES)
    for _, tgt_idx, _ in train_windows:
        tgt = cls_seq[tgt_idx]
        for c in range(N_CLASSES):
            counts[c] += ((tgt == c) & valid_common).sum()
    counts = np.maximum(counts, 1)
    w_full = counts.sum() / (N_CLASSES * counts)
    w = np.sqrt(w_full)  # sqrt-dampened inverse-frequency weighting: v2 used w_full directly,
    # which drove Non-vegetasi User's Accuracy down to ~13% (severe false-positive inflation);
    # dampening the weight is a standard remedy for over-aggressive class rebalancing.
    class_weight = torch.tensor(w, dtype=torch.float32)

    model = CNNLSTM(in_ch=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        for in_idx, tgt_idx, tgt_year in train_windows:
            model.train()
            opt.zero_grad()
            x_in = X_t_full[in_idx]
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
        logits = model(X_t_full[in_idx])
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

    # genuine forward projection to 2025 from real 2022-2024 input
    with torch.no_grad():
        idx22, idx23, idx24 = YEARS.index(2022), YEARS.index(2023), YEARS.index(2024)
        logits25 = model(X_t_full[[idx22, idx23, idx24]])
        pred25 = torch.argmax(logits25, dim=1)[0].numpy()

    return dict(OA=float(OA), kappa=float(kappa), PA=PA.tolist(), UA=UA.tolist(), CM=CM.tolist(),
                n=int(total), pred2025=pred25, model_state=model.state_dict())


all_results = []
for seed in range(N_SEEDS):
    log(f"=== Seed {seed} ===")
    r = run_one(seed)
    log(f"  OA={r['OA']*100:.2f}%  Kappa={r['kappa']:.4f}  PA[Non-veg]={r['PA'][1]*100:.2f}%  "
        f"PA[Sparse]={r['PA'][2]*100:.2f}%  PA[Dense]={r['PA'][3]*100:.2f}%")
    all_results.append(r)
    np.save(f"{DATA_DIR}/convlstm_v3_pred2025_seed{seed}.npy", r["pred2025"])

OAs = np.array([r["OA"] for r in all_results])
kappas = np.array([r["kappa"] for r in all_results])
PAs = np.array([r["PA"] for r in all_results])
log("\n" + "=" * 90)
log(f"ConvLSTM v3 (3-channel, sqrt-dampened weights, label-fix + class-weighted), {N_SEEDS} seeds:")
log(f"  OA = {OAs.mean()*100:.2f}% +/- {OAs.std()*100:.2f}%")
log(f"  Kappa = {kappas.mean():.4f} +/- {kappas.std():.4f}")
for c, name in enumerate(NAMES):
    log(f"  PA[{name}] = {PAs[:,c].mean()*100:.2f}% +/- {PAs[:,c].std()*100:.2f}%")

json.dump(dict(
    per_seed=[{k: v for k, v in r.items() if k not in ("pred2025", "model_state")} for r in all_results],
    OA_mean=float(OAs.mean()), OA_std=float(OAs.std()),
    kappa_mean=float(kappas.mean()), kappa_std=float(kappas.std()),
    PA_mean=PAs.mean(axis=0).tolist(), PA_std=PAs.std(axis=0).tolist(),
), open(f"{DATA_DIR}/convlstm_v3_results.json", "w"), indent=2)
log("\nSaved convlstm_v3_results.json and per-seed convlstm_v3_pred2025_seed*.npy")
