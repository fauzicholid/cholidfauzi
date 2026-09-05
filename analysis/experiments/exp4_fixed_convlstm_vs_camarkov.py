"""EXPERIMENT 4 -- Significance test: CA-Markov vs. the FIXED ConvLSTM.

Closes the gap left by exp3, which only tested CA-Markov against the
original, unweighted ConvLSTM. Retrains ConvLSTM with the winning combined
fix from exp2 (corrected label downsampling [Option C: reclassify-after-
bilinear-downsampled-NDVI] + inverse-frequency class-weighted loss) across
the same 5 seeds as exp2, saves each seed's per-pixel 2024 prediction, and
runs the same bootstrap-CI + McNemar test as exp3 against CA-Markov's
coarse-grid prediction (from exp1) -- for EACH seed, so the significance
result is checked for consistency rather than resting on one run.

Requires: exp1 (ca_markov_sim2024_coarse.npy, ca_markov_coarse_actual2024.npy,
ca_markov_coarse_mask.npy) and sequence_raw.pkl (from build_sequence_dataset.py).
"""
import time, pickle, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import chi2

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

DATA_DIR = os.environ.get("LC_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
TARGET_SIZE = (300, 233)
N_CLASSES = 4
NAMES = ["Air", "Non-vegetasi", "Veg. jarang", "Veg. rapat"]
N_SEEDS = 5
EPOCHS = 60
N_BOOT = 2000
RNG_BOOT = np.random.default_rng(0)

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


def build_sequence_fixed():
    """fix_labels=True (Option C) + built for fix_weight (inverse-frequency
    class weighting computed inside run_one from train-window targets only)."""
    ndvi_seq = np.zeros((len(YEARS), H, W), dtype=np.float32)
    bright_seq = np.zeros((len(YEARS), H, W), dtype=np.float32)
    valid_seq = np.zeros((len(YEARS), H, W), dtype=np.float32)
    cls_seq = np.zeros((len(YEARS), H, W), dtype=np.int8)
    for t, y in enumerate(YEARS):
        ndvi_seq[t] = resize_float(raw[y]["ndvi_adj"], (W, H), Image.BILINEAR)
        bright_seq[t] = resize_float(raw[y]["bright"], (W, H), Image.BILINEAR)
        valid_frac = resize_float(raw[y]["valid"].astype(np.float32), (W, H), Image.BOX)
        valid_seq[t] = (valid_frac > 0.5).astype(np.float32)
        cls_seq[t] = classify4(ndvi_seq[t], valid_seq[t] > 0.5)
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
            feat = self.encoder(x_seq[t:t + 1])
            h, c = self.convlstm(feat, h, c)
        out = self.decoder(h)
        return out


def run_one(seed, X, cls_seq, valid_common):
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

    # inverse-frequency class weights from TRAIN targets only (no test leakage)
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
    return pred, actual, mask


def oa(correct):
    return correct.mean()


def pa_for_class(actual, pred, cls):
    is_cls = actual == cls
    if is_cls.sum() == 0:
        return float("nan")
    return (pred[is_cls] == cls).mean()


def bootstrap_ci(stat_fn, n, n_boot=N_BOOT):
    idx_all = np.arange(n)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG_BOOT.choice(idx_all, size=n, replace=True)
        stats[b] = stat_fn(idx)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return stats.mean(), lo, hi


# ---- Load CA-Markov coarse-grid prediction (from exp1) ----
ca_pred_full = np.load(f"{DATA_DIR}/ca_markov_sim2024_coarse.npy")
ca_actual_full = np.load(f"{DATA_DIR}/ca_markov_coarse_actual2024.npy")
ca_mask_full = np.load(f"{DATA_DIR}/ca_markov_coarse_mask.npy")

log("Building fixed (label-corrected) sequence for ConvLSTM training...")
X, cls_seq, valid_common = build_sequence_fixed()

all_seed_results = []
for seed in range(N_SEEDS):
    log(f"=== Seed {seed}: training fixed ConvLSTM (fix_labels + fix_weight) ===")
    cl_pred, cl_actual, cl_mask = run_one(seed, X, cls_seq, valid_common)

    common_mask = ca_mask_full & cl_mask & (ca_actual_full == cl_actual)
    n_disagree = int((ca_mask_full & cl_mask & (ca_actual_full != cl_actual)).sum())
    ys, xs = np.where(common_mask)
    n = len(ys)
    actual = ca_actual_full[ys, xs]
    pred_ca = ca_pred_full[ys, xs]
    pred_cl = cl_pred[ys, xs]
    correct_ca = (pred_ca == actual)
    correct_cl = (pred_cl == actual)

    oa_ca_mean, oa_ca_lo, oa_ca_hi = bootstrap_ci(lambda idx: oa(correct_ca[idx]), n)
    oa_cl_mean, oa_cl_lo, oa_cl_hi = bootstrap_ci(lambda idx: oa(correct_cl[idx]), n)

    pa_ca_mean, pa_ca_lo, pa_ca_hi = bootstrap_ci(
        lambda idx: pa_for_class(actual[idx], pred_ca[idx], 1), n)
    pa_cl_mean, pa_cl_lo, pa_cl_hi = bootstrap_ci(
        lambda idx: pa_for_class(actual[idx], pred_cl[idx], 1), n)

    b = int(((correct_ca) & (~correct_cl)).sum())
    c = int(((~correct_ca) & (correct_cl)).sum())
    if b + c == 0:
        stat, p_value = float("nan"), float("nan")
    else:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(stat, df=1)

    is_nonveg = actual == 1
    b_nv = int(((correct_ca) & (~correct_cl) & is_nonveg).sum())
    c_nv = int(((~correct_ca) & (correct_cl) & is_nonveg).sum())
    if b_nv + c_nv == 0:
        stat_nv, p_nv = float("nan"), float("nan")
    else:
        stat_nv = (abs(b_nv - c_nv) - 1) ** 2 / (b_nv + c_nv)
        p_nv = 1 - chi2.cdf(stat_nv, df=1)

    result = dict(
        seed=seed, n_common=n, n_disagree_actual=n_disagree,
        oa_ca=[oa_ca_mean, oa_ca_lo, oa_ca_hi],
        oa_cl_fixed=[oa_cl_mean, oa_cl_lo, oa_cl_hi],
        pa_nonveg_ca=[pa_ca_mean, pa_ca_lo, pa_ca_hi],
        pa_nonveg_cl_fixed=[pa_cl_mean, pa_cl_lo, pa_cl_hi],
        mcnemar_overall=dict(b=b, c=c, chi2=stat, p=p_value),
        mcnemar_nonveg=dict(n=int(is_nonveg.sum()), b=b_nv, c=c_nv, chi2=stat_nv, p=p_nv),
    )
    all_seed_results.append(result)
    log(f"  n_common={n} (excluded {n_disagree} ground-truth-disagreement pixels)")
    log(f"  OA:  CA-Markov={oa_ca_mean*100:.2f}%  Fixed-ConvLSTM={oa_cl_mean*100:.2f}%")
    log(f"  PA[Non-veg]:  CA-Markov={pa_ca_mean*100:.2f}%  Fixed-ConvLSTM={pa_cl_mean*100:.2f}%")
    log(f"  McNemar overall: chi2={stat:.2f} p={p_value:.4g} | "
        f"Non-veg-only (n={int(is_nonveg.sum())}): chi2={stat_nv:.2f} p={p_nv:.4g}")

json.dump(all_seed_results, open(f"{DATA_DIR}/exp4_fixed_vs_camarkov_results.json", "w"), indent=2)

print("\n" + "=" * 100)
print("SUMMARY ACROSS 5 SEEDS: CA-Markov (coarse grid, exp1) vs. FIXED ConvLSTM (fix_labels+fix_weight)")
print("=" * 100)
print(f"{'Seed':>4s} {'OA CA':>8s} {'OA Fixed':>9s} {'PA[NV] CA':>10s} {'PA[NV] Fixed':>13s} "
      f"{'McNemar p (overall)':>20s} {'McNemar p (Non-veg)':>20s}")
for r in all_seed_results:
    print(f"{r['seed']:>4d} {r['oa_ca'][0]*100:7.2f}% {r['oa_cl_fixed'][0]*100:8.2f}% "
          f"{r['pa_nonveg_ca'][0]*100:9.2f}% {r['pa_nonveg_cl_fixed'][0]*100:12.2f}% "
          f"{r['mcnemar_overall']['p']:20.4g} {r['mcnemar_nonveg']['p']:20.4g}")

n_sig_overall = sum(1 for r in all_seed_results if r['mcnemar_overall']['p'] < 0.05)
n_sig_nonveg = sum(1 for r in all_seed_results if r['mcnemar_nonveg']['p'] < 0.05)
n_fixed_wins_nonveg = sum(1 for r in all_seed_results if r['pa_nonveg_cl_fixed'][0] > r['pa_nonveg_ca'][0])
print(f"\nSignificant overall difference in {n_sig_overall}/{N_SEEDS} seeds (p<0.05).")
print(f"Significant Non-vegetated-restricted difference in {n_sig_nonveg}/{N_SEEDS} seeds (p<0.05).")
print(f"Fixed ConvLSTM PA[Non-veg] > CA-Markov PA[Non-veg] in {n_fixed_wins_nonveg}/{N_SEEDS} seeds.")
print("\nSaved exp4_fixed_vs_camarkov_results.json")
