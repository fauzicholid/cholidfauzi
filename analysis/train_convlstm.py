"""Train and evaluate a custom ConvLSTM (CNN encoder + ConvLSTM cell + CNN
decoder) on the 8-year NDVI-classification sequence, with a held-out 2024
test window and a genuine forward projection to 2025.
"""
import time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

DATA_DIR = os.environ.get("LC_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
TARGET_SIZE = (300, 233)  # (W, H) downsample target for tractable CPU ConvLSTM training

raw = pickle.load(open(f"{DATA_DIR}/sequence_raw.pkl", "rb"))

def resize_float(arr, size, resample):
    img = Image.fromarray(np.nan_to_num(arr, nan=0.0).astype(np.float32), mode="F")
    return np.array(img.resize(size, resample), dtype=np.float32)

W, H = TARGET_SIZE
ndvi_seq = np.zeros((len(YEARS), H, W), dtype=np.float32)
bright_seq = np.zeros((len(YEARS), H, W), dtype=np.float32)
valid_seq = np.zeros((len(YEARS), H, W), dtype=np.float32)
cls_seq = np.zeros((len(YEARS), H, W), dtype=np.int8)

for t, y in enumerate(YEARS):
    ndvi_seq[t] = resize_float(raw[y]["ndvi_adj"], (W, H), Image.BILINEAR)
    bright_seq[t] = resize_float(raw[y]["bright"], (W, H), Image.BILINEAR)
    valid_frac = resize_float(raw[y]["valid"].astype(np.float32), (W, H), Image.BOX)
    valid_seq[t] = (valid_frac > 0.5).astype(np.float32)
    cls_img = Image.fromarray(raw[y]["cls"].astype(np.uint8), mode="L")
    cls_seq[t] = np.array(cls_img.resize((W, H), Image.NEAREST), dtype=np.int8)

log(f"downsampled sequence shape: {ndvi_seq.shape}")

# normalize bright to roughly unit scale
bright_mean, bright_std = bright_seq[valid_seq > 0].mean(), bright_seq[valid_seq > 0].std()
bright_norm = (bright_seq - bright_mean) / (bright_std + 1e-6)

X = np.stack([ndvi_seq, bright_norm], axis=1)  # (T, C=2, H, W)
valid_common = valid_seq.min(axis=0) > 0  # pixels valid across ALL years (for clean windows)
log(f"pixels valid across all {len(YEARS)} years: {valid_common.sum()} / {valid_common.size} ({100*valid_common.mean():.1f}%)")

np.save(f"{DATA_DIR}/seq_X.npy", X)
np.save(f"{DATA_DIR}/seq_cls.npy", cls_seq)
np.save(f"{DATA_DIR}/seq_valid_common.npy", valid_common)

# ---- ConvLSTM ----
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
        # x_seq: (T, C, H, W)
        Tt, C, H, W = x_seq.shape
        h = torch.zeros(1, self.lstm_hid, H, W)
        c = torch.zeros(1, self.lstm_hid, H, W)
        for t in range(Tt):
            feat = self.encoder(x_seq[t:t+1])
            h, c = self.convlstm(feat, h, c)
        out = self.decoder(h)
        return out  # (1, n_classes, H, W)

model = CNNLSTM()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# sliding windows: input 3 years -> predict next year
window = 3
windows = []
for start in range(0, len(YEARS) - window):
    in_idx = list(range(start, start + window))
    tgt_idx = start + window
    windows.append((in_idx, tgt_idx, YEARS[tgt_idx]))
log(f"windows: {[(  [YEARS[i] for i in w[0]], w[2]) for w in windows]}")

train_windows = [w for w in windows if w[2] != 2024]
test_window = [w for w in windows if w[2] == 2024][0]
log(f"train targets: {[w[2] for w in train_windows]}  test target: {test_window[2]}")

X_t = torch.from_numpy(X)
valid_mask_t = torch.from_numpy(valid_common)

EPOCHS = 60
for epoch in range(EPOCHS):
    total_loss = 0.0
    for in_idx, tgt_idx, tgt_year in train_windows:
        model.train()
        opt.zero_grad()
        x_in = X_t[in_idx]
        tgt_arr = cls_seq[tgt_idx].astype(np.int64)
        tgt_arr = np.where(tgt_arr < 0, 0, tgt_arr)  # invalid pixels masked out of loss below
        target = torch.from_numpy(tgt_arr).unsqueeze(0)
        logits = model(x_in)
        loss = F.cross_entropy(logits, target, reduction="none")[0]
        loss = (loss * valid_mask_t.float()).sum() / valid_mask_t.float().sum()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        log(f"epoch {epoch}: train_loss={total_loss/len(train_windows):.4f}")

# held-out test: predict 2024 from (2021,2022,2023)
model.eval()
with torch.no_grad():
    in_idx, tgt_idx, tgt_year = test_window
    x_in = X_t[in_idx]
    logits = model(x_in)
    pred = torch.argmax(logits, dim=1)[0].numpy()

actual = cls_seq[tgt_idx]
mask = valid_common & (actual >= 0) & (pred >= 0)
log(f"test pixels: {mask.sum()}")

np.save(f"{DATA_DIR}/convlstm_pred2024.npy", pred)
np.save(f"{DATA_DIR}/convlstm_actual2024_ds.npy", actual)
np.save(f"{DATA_DIR}/convlstm_test_mask.npy", mask)

n = 4
CM = np.zeros((n, n), dtype=np.int64)
for a in range(n):
    for p in range(n):
        CM[a, p] = int(((actual == a) & (pred == p) & mask).sum())
log(f"Confusion matrix:\n{CM}")
total = CM.sum()
OA = np.trace(CM) / total
row_sums = CM.sum(axis=1); col_sums = CM.sum(axis=0)
pe = (row_sums * col_sums).sum() / (total ** 2)
kappa = (OA - pe) / (1 - pe) if pe < 1 else float("nan")
log(f"Overall Accuracy: {OA*100:.2f}%  Kappa: {kappa:.4f}")
PA = np.divide(np.diag(CM), row_sums, out=np.zeros(n), where=row_sums>0)
UA = np.divide(np.diag(CM), col_sums, out=np.zeros(n), where=col_sums>0)
log(f"PA: {PA}  UA: {UA}")

import json
json.dump(dict(OA=OA, kappa=kappa, PA=PA.tolist(), UA=UA.tolist(), n=int(total),
               CM=CM.tolist(), row_sums=row_sums.tolist(), col_sums=col_sums.tolist(),
               train_targets=[w[2] for w in train_windows], test_target=tgt_year,
               window=window, epochs=EPOCHS),
          open(f"{DATA_DIR}/convlstm_metrics.json", "w"))

torch.save(model.state_dict(), f"{DATA_DIR}/convlstm_weights.pt")

# genuine forward projection: predict 2025 from actual (2022,2023,2024)
with torch.no_grad():
    idx_2022, idx_2023, idx_2024 = YEARS.index(2022), YEARS.index(2023), YEARS.index(2024)
    x_future = X_t[[idx_2022, idx_2023, idx_2024]]
    logits_2025 = model(x_future)
    pred_2025 = torch.argmax(logits_2025, dim=1)[0].numpy()
np.save(f"{DATA_DIR}/convlstm_pred2025.npy", pred_2025)
log(f"2025 projection saved, class dist: {[int((pred_2025[valid_common]==i).sum()) for i in range(4)]}")

log("DONE")
