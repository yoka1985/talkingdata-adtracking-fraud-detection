#!/usr/bin/env python
"""
TalkingData AdTracking Fraud Detection - FINAL pipeline (memory-safe, disk-safe, with built-in ablation / tests).

Models   : lgbm_full (CPU) | ftrl (Numba FM-FTRL, CPU) | nn (PyTorch embedding MLP, GPU optional, multi-seed)
           lgbm_r (R-kernel features) is OFF by default: it stopped at iteration 1 in both earlier runs.
Safety   : every large array lives in a memory-mapped .npy file written atomically (tmp file -> rename);
           features are written column by column; models predict in row chunks; predictions are cached;
           the cache goes to the roomiest scratch disk and a disk preflight runs BEFORE the long builds.
Tests    : (1) single-feature drift scan, train-in-test-hours vs test, for supplement AND test.csv-only features
           (2) leave-one-group-out + parameter ablation with seed-noise-aware significance
           (3) NN multi-seed spread (seed noise)
           (4) time-blocked cross-fit blending, compared on ONE metric (pooled out-of-fold AUC)
           (5) paired block bootstrap: blend vs best single model
           (6) AUC inside vs outside the test hours
           (7) floor/ceiling truncation, applied only if confirmed on a second block
Test rows: the real test rows inside test_supplement.csv are located by KEY (ip, app, device, os, channel, time),
           not by the supplement's click_id column (which is filled on every row in the real file).
GPU      : only the NN uses it (automatic, falls back to CPU). LightGBM / FTRL / feature building are CPU work.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import numba as nb
import lightgbm as lgb
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
TESTING = os.environ.get("TD_TESTING") == "1"           # tiny smoke-test mode (never set on Kaggle)
TEST_ROWS = int(os.environ.get("TD_TEST_ROWS", 300_000))
INPUT_DIR = os.environ.get("TD_INPUT", "/kaggle/input")
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUT_DIR = WORK / "outputs"                              # small deliverables only (counts toward the 20 GB quota)
CACHE_DIR = OUT_DIR / "cache"                           # overwritten in main() by pick_cache_dir()

TOTAL_TRAIN_ROWS = 184_903_890
N_TEST_EXPECTED = 18_790_469
TRAIN_ROWS = TEST_ROWS if TESTING else 40_000_000
VAL_ROWS = TEST_ROWS // 5 if TESTING else 2_500_000
SEED = 42
TEST_HOURS = (4, 5, 9, 10, 13, 14)                      # hours of the test day (assumption: UTC clock of the CSVs)

USE_SUPPLEMENT = True                                   # full test-day features; falls back to test.csv if RAM is short
SUPP_MIN_FREE_GB = 14.0
DRIFT_N = 20_000 if TESTING else 300_000
DRIFT_DROP_AUC = 0.70                                   # drop a feature whose single-feature drift AUC exceeds this
DRIFT_MAX_DROP = 6

ENABLE = dict(lgbm_full=True, lgbm_r=False, ftrl=True, nn=True)
RUN_ABLATION = False                                     # runs AFTER the submission is written
FORCE_RECOMPUTE = False                                 # True = ignore cached predictions

PARAMS_FULL = dict(objective="binary", metric="auc", learning_rate=0.08, num_leaves=31, max_depth=-1,
                   min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                   scale_pos_weight=10.0, reg_lambda=5.0, max_bin=63, verbose=-1, seed=SEED)
PARAMS_R = dict(objective="binary", metric="auc", learning_rate=0.08, num_leaves=7, max_depth=4,
                min_child_samples=100, subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
                scale_pos_weight=99.7, reg_lambda=0.0, max_bin=63, verbose=-1, seed=SEED)
MAX_ROUNDS, EARLY_STOP = (60, 10) if TESTING else (800, 50)

FTRL = dict(alpha=0.05, beta=1.0, l1=1.0, l2=0.1, alpha_fm=0.01, k=4, D=2 ** 18 if TESTING else 2 ** 24,
            chunk=2_000_000, epochs=2)

CAT_CAPS = dict(app=1024, device=5000, os=1024, channel=512, hour=24)
NN = dict(enabled=True, pool_neg=100_000 if TESTING else 6_000_000, neg_ratio=4,
          epochs_cpu=2 if TESTING else 12, epochs_gpu=2 if TESTING else 30,
          seeds_cpu=2, seeds_gpu=3, batch=4096, lr=2e-3, val_neg_frac=0.25)

BLEND_K = 5                                             # contiguous time blocks for cross-fitting
BLEND_NEG_FRAC = 0.2                                    # negatives kept when fitting blend weights (AUC stays unbiased)
BLEND_MIN_GAIN = 2e-5                                   # blend must beat best single (pooled OOF AUC) by this much
BOOT_N = 20 if TESTING else 100
TAIL_LO, TAIL_HI, MIN_GAIN = [0, 1, 2, 5, 10], [0, 0.25, 0.5, 1], 1e-5

ABL = dict(train_rows=60_000 if TESTING else 6_000_000, val_neg_frac=0.3, rounds=40 if TESTING else 300,
           early=10 if TESTING else 30, full_seeds=(0, 1, 2) if not TESTING else (0, 1), z=2.0)
PARAMS_ABL = dict(objective="binary", metric="auc", learning_rate=0.15, num_leaves=31, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, scale_pos_weight=10.0,
                  reg_lambda=5.0, max_bin=63, verbose=-1)


# =============================================================================
# UTILITIES
# =============================================================================
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def auc(y, s) -> float:
    return float(roc_auc_score(y, s))


def mem_limit_gb() -> float:
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
            if v.isdigit() and int(v) < (1 << 50):
                return int(v) / 1e9
        except Exception:
            pass
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        return float("nan")


def rss_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return float("nan")


def log_mem(tag: str = "") -> None:
    log(f"mem {tag}: process {rss_gb():.1f} GB (limit ~{mem_limit_gb():.1f} GB)")


def save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def save_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def find_file(name: str) -> Path:
    for root in (Path(INPUT_DIR), Path("."), Path("data")):
        if root.exists():
            hits = sorted(root.rglob(name), key=lambda h: ("talkingdata" not in str(h).lower(), len(h.parts)))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"{name} not found under {INPUT_DIR}")


def to_seconds(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S").values.astype("datetime64[s]").astype(np.int64)


# ---- disk handling ------------------------------------------------------------------------------------------
def pick_cache_dir() -> Path:
    """Scratch disk with the most free space; the 20 GB /kaggle/working area is only a last resort."""
    if TESTING:
        return OUT_DIR / "cache"
    best, best_free = None, -1.0
    for c in (Path("/kaggle/temp"), Path("/tmp")):
        try:
            c.mkdir(parents=True, exist_ok=True)
            if not os.access(c, os.W_OK):
                continue
            free = shutil.disk_usage(c).free / 1e9
            log(f"disk: {c} free {free:.1f} GB")
            if free > best_free:
                best, best_free = c / "td_cache", free
        except Exception:
            pass
    if best is None:
        best = OUT_DIR / "cache"
        log(f"disk: falling back to {best} (inside the 20 GB output area)")
    return best


def clean_tmp(d: Path) -> None:
    """Remove half-written temp files left by a killed/failed run (they can fill the disk)."""
    for f in list(d.glob("*.tmp*")):
        try:
            f.unlink()
            log(f"removed stale temp file {f.name}")
        except Exception:
            pass


def disk_preflight(n_cols_est: int = 42) -> None:
    need = (TRAIN_ROWS * (n_cols_est * 4 + 26) + N_TEST_EXPECTED * (n_cols_est * 4 + 33)) / 1e9 * 1.15
    free = shutil.disk_usage(CACHE_DIR).free / 1e9
    log(f"disk preflight: need ~{need:.1f} GB, free {free:.1f} GB at {CACHE_DIR}")
    if free < need:
        raise RuntimeError(f"Not enough disk for the feature cache: need ~{need:.1f} GB, free {free:.1f} GB. "
                           f"Reduce TRAIN_ROWS or free space.")


# =============================================================================
# DATA LOADING & CHRONOLOGICAL SPLIT
# =============================================================================
def load_train() -> pd.DataFrame:
    f = find_file("train.csv")
    skip = 0 if TESTING else TOTAL_TRAIN_ROWS - TRAIN_ROWS
    log(f"reading {TRAIN_ROWS:,} train rows (skipping first {skip:,})")
    dt = dict(ip="uint32", app="uint16", device="uint16", os="uint16", channel="uint16", is_attributed="uint8")
    df = pd.read_csv(f, skiprows=range(1, skip + 1) if skip else None, nrows=TRAIN_ROWS,
                     usecols=["ip", "app", "device", "os", "channel", "click_time", "is_attributed"], dtype=dt)
    df["t"] = to_seconds(df.pop("click_time"))
    if not np.all(df["t"].to_numpy()[1:] >= df["t"].to_numpy()[:-1]):
        df = df.iloc[np.argsort(df["t"].to_numpy(), kind="stable")].reset_index(drop=True)
    return df


def load_test(source: str) -> tuple[pd.DataFrame, np.ndarray]:
    """source='supplement': full test day (test rows must be located by key). source='test.csv': only the test sample."""
    dt = dict(ip="uint32", app="uint16", device="uint16", os="uint16", channel="uint16")
    name = "test_supplement.csv" if source == "supplement" else "test.csv"
    log(f"reading {name}")
    df = pd.read_csv(find_file(name), nrows=TEST_ROWS if TESTING else None,
                     usecols=["click_id", "ip", "app", "device", "os", "channel", "click_time"], dtype=dt)
    df["t"] = to_seconds(df.pop("click_time"))
    if not np.all(df["t"].to_numpy()[1:] >= df["t"].to_numpy()[:-1]):
        df = df.sort_values("t", kind="stable").reset_index(drop=True)
    cid = df["click_id"].to_numpy(np.float64)
    mask = np.isfinite(cid) & (cid >= 0)
    return df, mask


KEYS = ["ip", "app", "device", "os", "channel", "t"]


def _occurrence(k: np.ndarray) -> np.ndarray:
    """Occurrence number of each row among rows with an identical key (0 for the first)."""
    o = np.argsort(k, kind="stable")
    ks = k[o]
    start = np.r_[True, ks[1:] != ks[:-1]]
    idx = np.arange(len(k))
    first = np.maximum.accumulate(np.where(start, idx, 0))
    occ = np.empty(len(k), np.int32)
    occ[o] = idx - first
    return occ


def locate_test_rows(sup: pd.DataFrame, test: pd.DataFrame):
    """Boolean mask over supplement rows that are real test rows + the test click_id of each of them."""
    ks = pd.util.hash_pandas_object(sup[KEYS], index=False).to_numpy()
    kt = pd.util.hash_pandas_object(test[KEYS], index=False).to_numpy()
    S = pd.DataFrame({"k": ks, "o": _occurrence(ks), "i": np.arange(len(ks), dtype=np.int32)})
    T = pd.DataFrame({"k": kt, "o": _occurrence(kt), "cid": test["click_id"].to_numpy(np.int64)})
    M = S.merge(T, on=["k", "o"], how="inner")
    log(f"test rows located in supplement: {len(M):,} of {len(test):,}")
    if len(M) < 0.999 * len(test):
        raise RuntimeError("could not match >= 99.9% of test.csv rows inside test_supplement.csv")
    mask = np.zeros(len(sup), bool)
    mask[M["i"].to_numpy()] = True
    cid_full = np.full(len(sup), -1, np.int64)
    cid_full[M["i"].to_numpy()] = M["cid"].to_numpy()
    return mask, cid_full


def decide_test_source() -> str:
    if not USE_SUPPLEMENT:
        return "test.csv"
    try:
        find_file("test_supplement.csv")
    except FileNotFoundError:
        log("test_supplement.csv not found -> using test.csv only")
        return "test.csv"
    free = mem_limit_gb() - rss_gb()
    if not TESTING and free < SUPP_MIN_FREE_GB:
        log(f"only ~{free:.1f} GB RAM free (< {SUPP_MIN_FREE_GB}) -> falling back to test.csv features")
        return "test.csv"
    return "supplement"


def chronological_test_hour_split(h_all: np.ndarray, y_all: np.ndarray, max_val_rows: int) -> tuple[int, int]:
    """
    Hold out the final contiguous TEST_HOURS block (up to max_val_rows) so validation mirrors the
    competition test-hour distribution while keeping train strictly prior to validation (no future rows).
    """
    in_th = np.isin(h_all, TEST_HOURS)
    th_idx = np.flatnonzero(in_th)
    if len(th_idx) > 0:
        va_end = int(th_idx[-1]) + 1
        va_start = va_end - 1
        while va_start > 0 and in_th[va_start - 1] and (va_end - va_start) < max_val_rows:
            va_start -= 1
        yv = y_all[va_start:va_end]
        if va_start > 0 and len(yv) >= min(1_000, max_val_rows // 2) and 0 < yv.sum() < len(yv):
            return va_start, va_end
    return len(y_all) - max_val_rows, len(y_all)


# =============================================================================
# FEATURE ENGINEERING  (pruned per ablation_v2: newness & ratios dropped for +0.0003 AUC)
# =============================================================================
def _codes(cols):
    key = cols[0].astype(np.int64)
    for v in cols[1:]:
        key = key * (int(v.max()) + 1) + v.astype(np.int64)
    return pd.factorize(key)[0].astype(np.int64)


def _deltas(c, t, steps=(1, 2, 3)):
    n = len(c)
    order = np.lexsort((t, c))
    cs, ts = c[order], t[order]
    out = {}
    for k in steps:
        nxt = np.full(n, np.nan, np.float32)
        prv = np.full(n, np.nan, np.float32)
        same = cs[k:] == cs[:-k]
        d = (ts[k:] - ts[:-k]).astype(np.float32, copy=False)
        d[~same] = np.nan
        nxt[: n - k] = d
        prv[k:] = d
        a, b = np.empty(n, np.float32), np.empty(n, np.float32)
        a[order], b[order] = nxt, prv
        out[f"next{k}"], out[f"prev{k}"] = a, b
    return out


def _nunique(a, b):
    bm = int(b.max()) + 1
    u = np.unique(a * bm + b)
    cnt = np.bincount(u // bm, minlength=int(a.max()) + 1)
    return cnt[a].astype(np.float32, copy=False)


GROUPS = {
    "ip":            (["ip"],                         dict(cnt=1, nu=["channel", "app", "device", "os"])),
    "app":           (["app"],                        dict(cnt=1, nu=["channel"])),
    "ip_app":        (["ip", "app"],                  dict(cnt=1)),
    "ip_app_os":     (["ip", "app", "os"],            dict(cnt=1)),
    "ip_dev_os":     (["ip", "device", "os"],         dict(cnt=1)),
    "ip_app_dev_os": (["ip", "app", "device", "os"],  dict(cnt=1, delta=1)),
    "ip_chan":       (["ip", "channel"],              dict(cnt=1, delta=1)),
    "ip_app_chan":   (["ip", "app", "channel"],       dict(cnt=1, delta=1)),
    "app_chan":      (["app", "channel"],             dict(cnt=1)),
    "app_os":        (["app", "os"],                  dict(cnt=1)),
    "ip_dh":         (["ip", "dh"],                   dict(cnt=1, nu=["channel", "app"])),
    "ip_app_dh":     (["ip", "app", "dh"],            dict(cnt=1)),
}
RAW_FEATS = ["app", "device", "os", "channel", "hour"]       # 'dow' removed: it only encodes train-day vs test-day
RAW6 = ["ip", "app", "device", "os", "channel", "hour"]


def build_features(df: pd.DataFrame, keep: np.ndarray | None = None) -> dict:
    """Features over the WHOLE frame; only rows in `keep` (boolean mask) are materialised."""
    t = df["t"].to_numpy()
    base = {c: df[c].to_numpy().astype(np.int32) for c in ["ip", "app", "device", "os", "channel"]}
    base["dh"] = (t // 3600).astype(np.int32)
    base["hour"] = ((t % 86400) // 3600).astype(np.int32)
    out: dict = {}

    def put(name, arr):
        out[name] = (arr if keep is None else arr[keep]).astype(np.float32, copy=False)

    for c in RAW_FEATS:
        put(c, base[c])
    for g, (cols, fl) in GROUPS.items():
        code = _codes([base[c] for c in cols])
        if fl.get("cnt"):
            put(f"cnt_{g}", np.bincount(code)[code])
        if fl.get("delta"):
            for k, v in _deltas(code, t).items():
                put(f"dt_{g}_{k}", v)
        for tgt in fl.get("nu", []):
            put(f"nu_{g}_{tgt}", _nunique(code, base[tgt]))
        del code
        gc.collect()
    return out


def group_of(c: str) -> str:
    if c in RAW_FEATS:
        return "raw"
    for p, g in (("cnt_", "counts"), ("new_", "newness"), ("dt_", "deltas"), ("nu_", "uniques"), ("r_", "ratios")):
        if c.startswith(p):
            return g
    return "other"


def core_cols(cols):
    return [c for c in cols if not c.startswith(("dt_", "nu_"))]


# =============================================================================
# ON-DISK FEATURE CACHE  (atomic writes: a killed run can never leave a half-written file that looks complete)
# =============================================================================
def _writemm(path: Path, cols, out: dict, order=None):
    tmp = path.with_name(path.stem + ".tmp.npy")
    n = len(out[cols[0]])
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, len(cols)))
    for j, c in enumerate(cols):
        col = out.pop(c)
        mm[:, j] = col if order is None else col[order]
    mm.flush()
    del mm
    os.replace(tmp, path)


def ensure_train_cache() -> list[str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    P = {k: CACHE_DIR / v for k, v in dict(x="X_all.npy", r="raw_all.npy", y="y_all.npy",
                                           h="hours_all.npy", c="cols.json").items()}
    if all(p.exists() for p in P.values()):
        log("train cache: found")
        return json.load(open(P["c"]))
    tr = load_train()
    y = tr.pop("is_attributed").to_numpy()
    log("building train features")
    out = build_features(tr)
    cols = list(out)
    raw = np.empty((len(tr), 6), np.int32)
    raw[:, :5] = tr[["ip", "app", "device", "os", "channel"]].to_numpy()
    raw[:, 5] = out["hour"].astype(np.int32)
    hours = out["hour"].astype(np.uint8)
    del tr
    gc.collect()
    _writemm(P["x"], cols, out)
    save_npy_atomic(P["r"], raw)
    save_npy_atomic(P["h"], hours)
    save_npy_atomic(P["y"], y)
    save_json_atomic(P["c"], cols)                                  # completion marker (written last)
    del out, raw
    gc.collect()
    log_mem("train cache built")
    return cols


def ensure_test_cache(cols_all: list[str], source: str) -> str:
    P = {k: CACHE_DIR / v for k, v in dict(x="X_te.npy", r="raw_te.npy", c="cid.npy",
                                           h="hours_te.npy", m="test_meta.json").items()}
    if all(p.exists() for p in P.values()) and json.load(open(P["m"]))["source"] == source:
        log(f"test cache: found (source={source})")
        return source
    sup, _ = load_test(source)
    if source == "supplement":
        test, _ = load_test("test.csv")
        mask, cid_full = locate_test_rows(sup, test)
        del test
        cid = cid_full[mask]
        del cid_full
    else:
        mask = np.ones(len(sup), bool)
        cid = sup["click_id"].to_numpy(np.int64)
    sup = sup.drop(columns="click_id")
    n = int(mask.sum())
    if not TESTING and n != N_TEST_EXPECTED:
        raise RuntimeError(f"expected {N_TEST_EXPECTED:,} test rows, got {n:,}")
    order = np.argsort(cid, kind="stable")
    log(f"building test features | source={source} | rows in frame {len(sup):,} | test rows {n:,}")
    out = build_features(sup, keep=mask)
    assert list(out) == cols_all, "feature columns differ between train and test builds"
    hour = out["hour"].astype(np.int32)[order]
    raw = np.empty((n, 6), np.int32)
    raw[:, :5] = sup.loc[mask, ["ip", "app", "device", "os", "channel"]].to_numpy()[order]
    raw[:, 5] = hour
    del sup
    gc.collect()
    _writemm(P["x"], cols_all, out, order)
    save_npy_atomic(P["r"], raw)
    save_npy_atomic(P["c"], cid[order])
    save_npy_atomic(P["h"], hour.astype(np.uint8))
    save_json_atomic(P["m"], dict(source=source, rows=n))
    log_mem("test cache built")
    return source


# =============================================================================
# DRIFT SCAN (label-free, single-feature)
# =============================================================================
def drift_scan(A_tr: np.ndarray, A_te: np.ndarray, cols: list[str], scan_cols: list[str]) -> pd.DataFrame:
    y = np.r_[np.zeros(len(A_tr)), np.ones(len(A_te))]
    rows = []
    for c in scan_cols:
        j = cols.index(c)
        v = np.nan_to_num(np.r_[A_tr[:, j], A_te[:, j]], nan=-1.0)
        a = auc(y, v)
        rows.append((c, max(a, 1 - a)))
    return pd.DataFrame(rows, columns=["feature", "drift_auc"]).sort_values("drift_auc", ascending=False)


def sample_rows(mm, pool_idx, n, rng):
    idx = np.sort(rng.choice(pool_idx, min(n, len(pool_idx)), replace=False))
    return np.asarray(mm[idx])


def testcsv_feature_sample(cols: list[str], n: int, rng) -> np.ndarray:
    df, _ = load_test("test.csv")
    keep = np.zeros(len(df), bool)
    keep[rng.choice(len(df), min(n, len(df)), replace=False)] = True
    out = build_features(df, keep=keep)
    return np.column_stack([out[c] for c in cols])


# =============================================================================
# LIGHTGBM
# =============================================================================
def train_lgb(params, rounds, early, Xtr, ytr, Xva, yva, names, log_every=100):
    d_tr = lgb.Dataset(Xtr, ytr, feature_name=names, free_raw_data=True).construct()
    d_va = lgb.Dataset(Xva, yva, reference=d_tr).construct()
    m = lgb.train(params, d_tr, num_boost_round=rounds, valid_sets=[d_va],
                  callbacks=[lgb.early_stopping(early, verbose=False), lgb.log_evaluation(log_every)])
    return m, (m.best_iteration or rounds)


def predict_chunks(m, A, ix, bi, chunk=1_000_000):
    out = np.empty(len(A), np.float64)
    for i in range(0, len(A), chunk):                                # slice ROWS first, then columns
        out[i:i + chunk] = m.predict(np.asarray(A[i:i + chunk])[:, ix].astype(np.float32, copy=False), num_iteration=bi)
    return out


def run_lgb(name, params, A_tr, y_tr, A_va, y_va, A_te, cols, feats):
    ix = [cols.index(c) for c in feats]
    log(f"{name}: {len(feats)} features | train {len(y_tr):,} | val {len(y_va):,}")
    Xtr = np.asarray(A_tr)[:, ix].astype(np.float32, copy=False)
    Xva = np.asarray(A_va)[:, ix].astype(np.float32, copy=False)
    m, bi = train_lgb(params, MAX_ROUNDS, EARLY_STOP, Xtr, y_tr, Xva, y_va, feats)
    del Xtr
    gc.collect()
    log(f"{name}: best_iter={bi} val_auc={m.best_score['valid_0']['auc']:.5f}")
    log_mem(f"{name} before predict")
    pv = predict_chunks(m, A_va, ix, bi)
    pt = predict_chunks(m, A_te, ix, bi)
    log(f"{name}: predictions done")
    return pv, pt


# =============================================================================
# FM-FTRL (Numba)
# =============================================================================
@nb.njit(cache=True, fastmath=True)
def _ftrl_fm(X, y, z, n, V, G, alpha, beta, l1, l2, alpha_fm, update):
    N, F = X.shape
    K = V.shape[1]
    out = np.empty(N, np.float64)
    s = np.empty(K)
    wv = np.empty(F)
    for i in range(N):
        lin = 0.0
        sq = 0.0
        for k in range(K):
            s[k] = 0.0
        for j in range(F):
            idx = X[i, j]
            zi = z[idx]
            if abs(zi) <= l1:
                w = 0.0
            else:
                w = -(zi - np.sign(zi) * l1) / ((beta + np.sqrt(n[idx])) / alpha + l2)
            wv[j] = w
            lin += w
            for k in range(K):
                v = V[idx, k]
                s[k] += v
                sq += v * v
        fm = 0.0
        for k in range(K):
            fm += s[k] * s[k]
        t = lin + 0.5 * (fm - sq)
        if t > 35.0:
            t = 35.0
        elif t < -35.0:
            t = -35.0
        p = 1.0 / (1.0 + np.exp(-t))
        out[i] = p
        if update:
            g = p - y[i]
            for j in range(F):
                idx = X[i, j]
                sigma = (np.sqrt(n[idx] + g * g) - np.sqrt(n[idx])) / alpha
                z[idx] += g - sigma * wv[j]
                n[idx] += g * g
                for k in range(K):
                    gv = g * (s[k] - V[idx, k])
                    G[idx, k] += gv * gv
                    V[idx, k] -= alpha_fm * gv / (np.sqrt(G[idx, k]) + 1.0)
    return out


_M32 = np.uint64(0xFFFFFFFF)


def _fmix32(x):
    x = x.astype(np.uint64) & _M32
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x85EBCA6B)) & _M32
    x ^= x >> np.uint64(13)
    x = (x * np.uint64(0xC2B2AE35)) & _M32
    x ^= x >> np.uint64(16)
    return x


_CROSSES = [(a, b) for i, a in enumerate(RAW6) for b in RAW6[i + 1:]] + [("app", "os", "channel")]


def hash_matrix(raw: np.ndarray, D: int) -> np.ndarray:
    cols = {c: raw[:, i].astype(np.uint64) for i, c in enumerate(RAW6)}
    fields = [(c,) for c in RAW6] + _CROSSES
    X = np.empty((len(raw), len(fields) + 1), np.int32)
    for j, f in enumerate(fields):
        key = np.zeros(len(raw), np.uint64)
        for c in f:
            key = (key << np.uint64(20)) | cols[c]
        key ^= np.uint64((j + 1) * 0x9E3779B1)
        X[:, j] = (_fmix32(key) % np.uint64(D)).astype(np.int32)
    X[:, -1] = 0
    return X


def run_ftrl(raw_tr, y_tr, raw_va, y_va, raw_te):
    P = FTRL
    z, n = np.zeros(P["D"]), np.zeros(P["D"])
    V = np.random.RandomState(SEED).normal(0, 0.01, (P["D"], P["k"]))
    G = np.zeros_like(V)
    args = (P["alpha"], P["beta"], P["l1"], P["l2"], P["alpha_fm"])
    N = len(y_tr)
    for ep in range(P["epochs"]):
        for i in range(0, N, P["chunk"]):
            j = min(i + P["chunk"], N)
            _ftrl_fm(hash_matrix(np.asarray(raw_tr[i:j]), P["D"]), y_tr[i:j].astype(np.float64), z, n, V, G, *args, True)
        log(f"FM-FTRL epoch {ep + 1}/{P['epochs']} done")

    def pred(raw):
        return np.concatenate([_ftrl_fm(hash_matrix(np.asarray(raw[i:i + P["chunk"]]), P["D"]),
                                        np.zeros(min(P["chunk"], len(raw) - i)), z, n, V, G, *args, False)
                               for i in range(0, len(raw), P["chunk"])])
    pv = pred(raw_va)
    log(f"FM-FTRL val_auc={auc(y_va, pv):.5f}")
    return pv, pred(raw_te)


# =============================================================================
# EMBEDDING NN (PyTorch). GPU used automatically if usable; multi-seed; fresh negatives every epoch.
# =============================================================================
def _prep_num(a):
    a = np.nan_to_num(a, nan=-1.0)
    return np.sign(a) * np.log1p(np.abs(a))


def run_nn(A_tr, y_tr, A_va, y_va, A_te, cols, feats):
    if torch is None or not NN["enabled"]:
        log("NN skipped (torch missing or disabled)")
        return None
    use_cuda = False
    try:
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.zeros(1, device="cuda")
    except Exception as e:
        log(f"CUDA unusable ({e!r}) -> CPU")
        use_cuda = False
    dev = torch.device("cuda" if use_cuda else "cpu")
    if not use_cuda:
        torch.set_num_threads(os.cpu_count() or 4)
    epochs = NN["epochs_gpu"] if use_cuda else NN["epochs_cpu"]
    seeds = NN["seeds_gpu"] if use_cuda else NN["seeds_cpu"]
    log(f"NN device={dev} | epochs={epochs} | seeds={seeds}")

    cat_names = list(CAT_CAPS)
    num_names = [c for c in feats if c not in CAT_CAPS]
    ix = [cols.index(c) for c in cat_names + num_names]
    nc = len(cat_names)
    caps = np.array(list(CAT_CAPS.values())) - 1
    rng = np.random.RandomState(SEED)

    pos = np.flatnonzero(y_tr == 1)
    negs = np.flatnonzero(y_tr == 0)
    pool = rng.choice(negs, min(NN["pool_neg"], len(negs)), replace=False)
    rows = np.sort(np.r_[pos, pool])                                # sorted -> sequential memmap reads
    blk = np.asarray(A_tr[rows])[:, ix].astype(np.float32, copy=False)
    is_pos = y_tr[rows] == 1
    pos_in, neg_in = np.flatnonzero(is_pos), np.flatnonzero(~is_pos)
    num0 = _prep_num(blk[:, nc:])
    mu, sd = num0.mean(0), num0.std(0) + 1e-6

    def featurize(b):
        c = torch.from_numpy(np.clip(b[:, :nc].astype(np.int64), 0, caps)).to(dev)
        n_ = torch.from_numpy(((_prep_num(b[:, nc:]) - mu) / sd).astype(np.float32, copy=False)).to(dev)
        return c, n_
    cat_t, num_t = featurize(blk)
    y_t = torch.from_numpy(y_tr[rows].astype(np.float32, copy=False)).to(dev)
    del blk, num0
    gc.collect()
    log_mem("NN after sampling")

    vpos, vneg = np.flatnonzero(y_va == 1), np.flatnonzero(y_va == 0)
    vsub = np.sort(np.r_[vpos, rng.choice(vneg, int(NN["val_neg_frac"] * len(vneg)), replace=False)])
    vc, vn = featurize(np.asarray(A_va[vsub])[:, ix].astype(np.float32, copy=False))
    y_vsub = y_va[vsub]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.ModuleList([nn.Embedding(cap, 8) for cap in CAT_CAPS.values()])
            self.mlp = nn.Sequential(nn.Linear(8 * nc + len(num_names), 256), nn.ReLU(), nn.Dropout(0.2),
                                     nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

        def forward(self, cat, num):
            return self.mlp(torch.cat([m(cat[:, i]) for i, m in enumerate(self.emb)] + [num], 1)).squeeze(1)

    @torch.no_grad()
    def predict(net, A, chunk=500_000):
        net.eval()
        out = []
        for i in range(0, len(A), chunk):
            c, n_ = featurize(np.asarray(A[i:i + chunk])[:, ix].astype(np.float32, copy=False))
            out.append(torch.sigmoid(net(c, n_)).cpu().numpy())
        return np.concatenate(out)

    pv_sum, pt_sum, seed_aucs = 0.0, 0.0, []
    nneg = min(NN["neg_ratio"] * len(pos_in), len(neg_in))
    steps_ep = int(np.ceil((len(pos_in) + nneg) / NN["batch"]))
    for sd_i in range(seeds):
        torch.manual_seed(SEED + sd_i)
        r = np.random.RandomState(SEED + sd_i)
        net = Net().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=NN["lr"], weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_ep, eta_min=1e-5)
        lossf = nn.BCEWithLogitsLoss()
        best, best_state = -1.0, None
        for ep in range(epochs):
            net.train()
            idx = np.r_[pos_in, r.choice(neg_in, nneg, replace=False)]
            idx = torch.from_numpy(idx[r.permutation(len(idx))]).to(dev)
            for i in range(0, len(idx), NN["batch"]):
                b = idx[i:i + NN["batch"]]
                opt.zero_grad()
                lossf(net(cat_t[b], num_t[b]), y_t[b]).backward()
                opt.step()
                sch.step()
            net.eval()
            with torch.no_grad():
                a = auc(y_vsub, torch.sigmoid(net(vc, vn)).cpu().numpy())
            log(f"NN seed {sd_i} epoch {ep + 1}/{epochs} val_auc(sub)={a:.5f}")
            if a > best:
                best, best_state = a, {k: v.clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        pv = predict(net, A_va)
        seed_aucs.append(auc(y_va, pv))
        log(f"NN seed {sd_i}: full val AUC {seed_aucs[-1]:.5f} -> predicting test")
        pv_sum = pv_sum + pv
        pt_sum = pt_sum + predict(net, A_te)
        del net
        gc.collect()
    save_json_atomic(OUT_DIR / "nn_seed_aucs.json", dict(seed_aucs=seed_aucs, mean=float(np.mean(seed_aucs)),
                                                         std=float(np.std(seed_aucs)), device=str(dev)))
    return pv_sum / seeds, pt_sum / seeds


# =============================================================================
# BLENDING: time-blocked cross-fit, one metric (pooled OOF AUC), non-negative weights
# =============================================================================
def rank_norm(p):
    return rankdata(p, method="average") / len(p)


def fit_weights(method, R, y, strong, rng):
    k = R.shape[1]
    if k == 1:
        return np.ones(1)
    if method == "equal_strong":
        return strong / strong.sum()
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    keep = np.sort(np.r_[pos, rng.choice(neg, int(len(neg) * BLEND_NEG_FRAC), replace=False)])
    Rs, ys = R[keep], y[keep]
    if method == "logreg":
        w = np.clip(LogisticRegression(C=1e6, max_iter=300).fit(Rs, ys).coef_[0], 0, None)
        return w / w.sum() if w.sum() > 0 else np.ones(k) / k
    if method == "nm_auc":
        soft = lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()
        res = minimize(lambda x: -auc(ys, Rs @ soft(x)), np.zeros(k), method="Nelder-Mead",
                       options=dict(maxiter=60 if TESTING else 150, xatol=1e-3, fatol=1e-7))
        return soft(res.x)
    raise ValueError(method)


def crossfit(method, R, y, strong, rng):
    n = len(y)
    edges = np.linspace(0, n, BLEND_K + 1).astype(int)
    oof = np.zeros(n)
    for b in range(BLEND_K):
        tr = np.ones(n, bool)
        tr[edges[b]:edges[b + 1]] = False
        w = fit_weights(method, R[tr], y[tr], strong, rng)
        oof[edges[b]:edges[b + 1]] = R[edges[b]:edges[b + 1]] @ w
    return oof


def run_blend(names, pv, pt, y_va):
    rng = np.random.RandomState(SEED)
    Rv = np.column_stack([rank_norm(pv[k]) for k in names])
    Rt = np.column_stack([rank_norm(pt[k]) for k in names])
    single = np.array([auc(y_va, Rv[:, j]) for j in range(len(names))])
    jb = int(single.argmax())
    strong = (single >= single.max() - 0.01).astype(float)
    rows = [dict(method=f"single:{names[jb]}", pooled_oof_auc=float(single[jb]))]
    cand = {}
    if len(names) > 1:
        for m in ("equal_strong", "logreg", "nm_auc"):
            oof = crossfit(m, Rv, y_va, strong, rng)
            a = auc(y_va, oof)
            rows.append(dict(method=m, pooled_oof_auc=a))
            cand[m] = a
    tbl = pd.DataFrame(rows)
    best_m = max(cand, key=cand.get) if cand else None
    if best_m and cand[best_m] > single[jb] + BLEND_MIN_GAIN:
        w = fit_weights(best_m, Rv, y_va, strong, rng)
        chosen = best_m
        oof_score = crossfit(best_m, Rv, y_va, strong, np.random.RandomState(SEED))
    else:
        w, chosen = np.eye(len(names))[jb], f"single:{names[jb]}"
        oof_score = Rv[:, jb]
    log(f"blend: chosen={chosen} | weights={dict(zip(names, np.round(w, 4)))}")
    return dict(table=tbl, chosen=chosen, weights=dict(zip(names, w.tolist())), s_val=Rv @ w, s_test=Rt @ w,
                oof=oof_score, best_single=Rv[:, jb], best_single_name=names[jb], single=dict(zip(names, single)))


def paired_block_bootstrap(y, a, b, block, n_boot, seed=SEED):
    rng = np.random.RandomState(seed)
    nb_ = max(1, len(y) // block)
    ids = [np.arange(k * block, min((k + 1) * block, len(y))) for k in range(nb_)]
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([ids[k] for k in rng.randint(0, nb_, nb_)])
        if y[idx].min() == y[idx].max():
            continue
        d.append(auc(y[idx], a[idx]) - auc(y[idx], b[idx]))
    d = np.array(d)
    return dict(mean_diff=float(d.mean()), ci_lo=float(np.percentile(d, 2.5)), ci_hi=float(np.percentile(d, 97.5)),
                share_le_zero=float((d <= 0).mean()), n=int(len(d)))


def hour_auc_table(y, hours, scores: dict):
    th = np.isin(hours, TEST_HOURS)
    rows = []
    for name, s in scores.items():
        r = dict(model=name, overall=auc(y, s))
        r["in_test_hours"] = auc(y[th], s[th]) if th.sum() and 0 < y[th].sum() < th.sum() else np.nan
        r["out_test_hours"] = auc(y[~th], s[~th]) if (~th).sum() and 0 < y[~th].sum() < (~th).sum() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# =============================================================================
# TAILS (floor -> 0, ceiling -> 1): applied only if confirmed on a second block
# =============================================================================
def make_thresholds(s, lo_pct, hi_pct):
    lo = float(np.quantile(s, lo_pct / 100)) if lo_pct > 0 else None
    hi = float(np.quantile(s, 1 - hi_pct / 100)) if hi_pct > 0 else None
    return lo, hi


def apply_thresholds(s, lo, hi):
    r = s.copy()
    if lo is not None:
        r[r <= lo] = 0.0
    if hi is not None:
        r[r >= hi] = 1.0
    return r


def tail_decision(y, s):
    h = len(y) // 2
    yt, st, yc, sc = y[:h], s[:h], y[h:], s[h:]
    base_t, base_c = auc(yt, st), auc(yc, sc)
    best = (0.0, 0.0, base_t)
    for lo in TAIL_LO:
        for hi in TAIL_HI:
            if lo == 0 and hi == 0:
                continue
            a = auc(yt, apply_thresholds(st, *make_thresholds(st, lo, hi)))
            if a > best[2]:
                best = (lo, hi, a)
    lo, hi, a_t = best
    if lo == 0 and hi == 0:
        return dict(applied=False, reason="no candidate beat baseline on block 1", lo_pct=0, hi_pct=0)
    lo_v, hi_v = make_thresholds(st, lo, hi)
    gain_c = auc(yc, apply_thresholds(sc, lo_v, hi_v)) - base_c
    ok = (a_t - base_t > MIN_GAIN) and gain_c > MIN_GAIN
    return dict(applied=bool(ok), lo_pct=lo, hi_pct=hi, lo_thr=lo_v, hi_thr=hi_v, gain_block1=float(a_t - base_t),
                gain_block2=float(gain_c))


# =============================================================================
# ABLATION (LightGBM on the most recent ABL['train_rows'] train rows; validation subsample keeps all positives)
# =============================================================================
def run_ablation(mm_all, y_tr, ntr, A_va, y_va, h_va, cols, feats, dropped):
    log("ABLATION: start")
    rng = np.random.RandomState(SEED)
    a0 = max(0, ntr - ABL["train_rows"])
    Xtr = np.asarray(mm_all[a0:ntr])
    ytr = y_tr[a0:ntr]
    vpos, vneg = np.flatnonzero(y_va == 1), np.flatnonzero(y_va == 0)
    vsub = np.sort(np.r_[vpos, rng.choice(vneg, int(ABL["val_neg_frac"] * len(vneg)), replace=False)])
    Xv, yv, hv = np.asarray(A_va[vsub]), y_va[vsub], h_va[vsub]
    th = np.isin(hv, TEST_HOURS)

    def evaluate(fcols, params, seed):
        ix = [cols.index(c) for c in fcols]
        p = {**PARAMS_ABL, **params, "seed": SEED + seed}
        m, bi = train_lgb(p, ABL["rounds"], ABL["early"], Xtr[:, ix], ytr, Xv[:, ix], yv, fcols, log_every=0)
        pr = m.predict(Xv[:, ix], num_iteration=bi)
        a_th = auc(yv[th], pr[th]) if th.sum() and 0 < yv[th].sum() < th.sum() else np.nan
        return auc(yv, pr), a_th, bi

    variants = [("full (final feature set)", feats, {})]
    if len(cols) != len(feats):
        variants.append(("all features before drift drop", cols, {}))
    variants.append(("R-kernel set (no deltas/uniques)", core_cols(feats), {}))
    for g in sorted({group_of(c) for c in feats}):
        variants.append((f"minus group: {g}", [c for c in feats if group_of(c) != g], {}))
    variants.append(("shallow trees (R-parity params)", feats, dict(num_leaves=7, max_depth=4)))
    variants.append(("no scale_pos_weight", feats, dict(scale_pos_weight=1.0)))

    rows, full_aucs = [], []
    for name, fcols, par in variants:
        seeds = ABL["full_seeds"] if name.startswith("full") else (0,)
        res = [evaluate(fcols, par, s) for s in seeds]
        r = np.array(res, dtype=float)
        if name.startswith("full"):
            full_aucs = r[:, 0]
        rows.append(dict(variant=name, n_features=len(fcols), val_auc=r[:, 0].mean(), auc_in_test_hours=np.nanmean(r[:, 1]),
                         best_iter=int(r[:, 2].mean())))
        log(f"ablation | {name}: AUC {rows[-1]['val_auc']:.5f}")
    df = pd.DataFrame(rows)
    full = df.loc[0, "val_auc"]
    noise = float(np.std(full_aucs, ddof=1)) if len(full_aucs) > 1 else float("nan")
    df["delta_vs_full"] = df["val_auc"] - full
    df["seed_noise_std"] = noise
    df["significant"] = df["delta_vs_full"].abs() > ABL["z"] * noise if noise == noise else False
    df.to_csv(OUT_DIR / "ablation_results.csv", index=False)
    log("ABLATION results:\n" + df.to_string(index=False))
    return df


# =============================================================================
# MAIN
# =============================================================================
def sig_of(*objs) -> str:
    return hashlib.md5(json.dumps(objs, sort_keys=True, default=str).encode()).hexdigest()[:8]


def cached(name, fn, sig=""):
    """Predictions are cached per (model, config signature): changed features/params never reuse stale files."""
    pdir = OUT_DIR / "preds"
    tag = f"{name}_{sig}" if sig else name
    fv, ft = pdir / f"{tag}_val.npy", pdir / f"{tag}_test.npy"
    if fv.exists() and ft.exists() and not FORCE_RECOMPUTE:
        log(f"{name}: loaded cached predictions ({tag})")
        return np.load(fv), np.load(ft)
    res = fn()
    if res is None:
        return None
    save_npy_atomic(fv, res[0])
    save_npy_atomic(ft, res[1])
    return res


def main():
    global CACHE_DIR
    t0 = time.time()
    (OUT_DIR / "preds").mkdir(parents=True, exist_ok=True)
    info: dict = dict(testing=TESTING, train_rows=TRAIN_ROWS, val_rows=VAL_ROWS)
    log_mem("start")

    CACHE_DIR = pick_cache_dir()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clean_tmp(CACHE_DIR)
    if not TESTING:
        disk_preflight(n_cols_est=42)                                 # fail in seconds, not after 30+ minutes

    cols = ensure_train_cache()
    source = ensure_test_cache(cols, decide_test_source())
    info["test_feature_source"] = source

    mm_all = np.load(CACHE_DIR / "X_all.npy", mmap_mode="r")
    raw_mm = np.load(CACHE_DIR / "raw_all.npy", mmap_mode="r")
    mm_te = np.load(CACHE_DIR / "X_te.npy", mmap_mode="r")
    raw_te = np.load(CACHE_DIR / "raw_te.npy", mmap_mode="r")
    y_all = np.load(CACHE_DIR / "y_all.npy")
    h_all = np.load(CACHE_DIR / "hours_all.npy")
    cid = np.load(CACHE_DIR / "cid.npy")

    ntr, va_end = chronological_test_hour_split(h_all, y_all, VAL_ROWS)
    nv = va_end - ntr
    A_tr, A_va = mm_all[:ntr], np.asarray(mm_all[ntr:va_end])
    y_tr, y_va, h_tr, h_va = y_all[:ntr], y_all[ntr:va_end], h_all[:ntr], h_all[ntr:va_end]
    info.update(val_slice=[ntr, va_end], val_hours=sorted(np.unique(h_va).astype(int).tolist()))
    log(f"train [0..{ntr:,}) (pos {y_tr.mean():.4%}) | chronological TEST_HOURS val [{ntr:,}..{va_end:,}) "
        f"(hours {info['val_hours']}, pos {y_va.mean():.4%}) | test {len(cid):,}")
    if not TESTING:
        assert np.array_equal(cid, np.arange(len(cid))), "click_ids are not 0..N-1 after sorting"

    # ---- (1) drift scan --------------------------------------------------------------------------------------
    rng = np.random.RandomState(SEED)
    in_hours = np.flatnonzero(np.isin(h_tr, TEST_HOURS))
    pool = in_hours if len(in_hours) >= DRIFT_N else np.arange(ntr)
    Atr_s = sample_rows(mm_all, pool, DRIFT_N, rng)
    Ate_s = sample_rows(mm_te, np.arange(len(mm_te)), DRIFT_N, rng)
    scan_cols = [c for c in cols if c not in RAW_FEATS]
    drift = drift_scan(Atr_s, Ate_s, cols, scan_cols)
    drift.to_csv(OUT_DIR / f"drift_{source}.csv", index=False)
    log("drift scan (top 8):\n" + drift.head(8).to_string(index=False))
    dropped = drift[drift.drift_auc > DRIFT_DROP_AUC].feature.tolist()[:DRIFT_MAX_DROP]
    feats = [c for c in cols if c not in dropped]
    log(f"dropped drifting features: {dropped}")
    info.update(dropped=dropped, kept_features=feats)
    if RUN_ABLATION and source == "supplement":                      # test: do supplement features reduce drift?
        try:
            Acsv = testcsv_feature_sample(cols, DRIFT_N, rng)
            d2 = drift_scan(Atr_s, Acsv, cols, scan_cols)
            d2.to_csv(OUT_DIR / "drift_testcsv_only.csv", index=False)
            info["drift_supplement"] = dict(max=float(drift.drift_auc.max()), mean=float(drift.drift_auc.mean()),
                                            n_over=int((drift.drift_auc > DRIFT_DROP_AUC).sum()))
            info["drift_testcsv_only"] = dict(max=float(d2.drift_auc.max()), mean=float(d2.drift_auc.mean()),
                                              n_over=int((d2.drift_auc > DRIFT_DROP_AUC).sum()))
            log(f"drift supplement vs test.csv-only: {info['drift_supplement']} vs {info['drift_testcsv_only']}")
            del Acsv
        except Exception as e:
            log(f"source drift comparison skipped: {e!r}")
    del Atr_s, Ate_s
    gc.collect()

    # ---- models (each cached; a Python error in one never loses the others) ------------------------------------
    preds = {}

    def attempt(name, fn, sig):
        try:
            r = cached(name, fn, sig)
            if r is not None:
                preds[name] = r
        except Exception as e:
            log(f"{name} FAILED, continuing without it: {e!r}")
        gc.collect()
        log_mem(f"after {name}")

    if ENABLE["lgbm_full"]:
        attempt("lgbm_full", lambda: run_lgb("lgbm_full", PARAMS_FULL, A_tr, y_tr, A_va, y_va, mm_te, cols, feats),
                sig_of(PARAMS_FULL, MAX_ROUNDS, EARLY_STOP, feats, source, ntr, va_end))
    if ENABLE["lgbm_r"]:
        attempt("lgbm_r", lambda: run_lgb("lgbm_r", PARAMS_R, A_tr, y_tr, A_va, y_va, mm_te, cols, core_cols(feats)),
                sig_of(PARAMS_R, MAX_ROUNDS, EARLY_STOP, core_cols(feats), source, ntr, va_end))
    if ENABLE["ftrl"]:
        attempt("ftrl", lambda: run_ftrl(raw_mm[:ntr], y_tr, np.asarray(raw_mm[ntr:va_end]), y_va, raw_te),
                sig_of(FTRL, source, ntr, va_end))
    if ENABLE["nn"]:
        attempt("nn", lambda: run_nn(A_tr, y_tr, A_va, y_va, mm_te, cols, feats),
                sig_of(NN, CAT_CAPS, feats, source, ntr, va_end))
    assert preds, "no model produced predictions"
    names = list(preds)
    info["val_auc_per_model"] = {k: auc(y_va, v[0]) for k, v in preds.items()}
    log(f"per-model val AUC: {info['val_auc_per_model']}")
    if (OUT_DIR / "nn_seed_aucs.json").exists():
        info["nn_seed_spread"] = json.load(open(OUT_DIR / "nn_seed_aucs.json"))

    # ---- (4) blend, (5) bootstrap, (6) hour AUC, (7) tails ---------------------------------------------------------
    bl = run_blend(names, {k: preds[k][0] for k in names}, {k: preds[k][1] for k in names}, y_va)
    bl["table"].to_csv(OUT_DIR / "blend_methods.csv", index=False)
    log("blend methods (pooled OOF AUC):\n" + bl["table"].to_string(index=False))
    info.update(blend_chosen=bl["chosen"], blend_weights=bl["weights"],
                blend_pooled_oof_auc=auc(y_va, bl["oof"]), best_single=bl["best_single_name"])

    vpos, vneg = np.flatnonzero(y_va == 1), np.flatnonzero(y_va == 0)
    bsub = np.sort(np.r_[vpos, rng.choice(vneg, int(0.25 * len(vneg)), replace=False)])
    info["bootstrap_blend_vs_best_single"] = paired_block_bootstrap(
        y_va[bsub], bl["oof"][bsub], bl["best_single"][bsub], block=max(2_000, len(bsub) // 40), n_boot=BOOT_N)
    log(f"paired block bootstrap (blend - best single): {info['bootstrap_blend_vs_best_single']}")

    scores = {k: preds[k][0] for k in names}
    scores["BLEND(oof)"] = bl["oof"]
    ht = hour_auc_table(y_va, h_va.astype(int), scores)
    ht.to_csv(OUT_DIR / "auc_by_test_hours.csv", index=False)
    log("AUC inside / outside test hours:\n" + ht.to_string(index=False))

    tail = tail_decision(y_va, bl["s_val"])
    info["tails"] = tail
    log(f"tails: {tail}")
    final = apply_thresholds(bl["s_test"], tail.get("lo_thr"), tail.get("hi_thr")) if tail["applied"] else bl["s_test"]

    pd.DataFrame({"click_id": cid, "is_attributed": final}).to_csv(WORK / "submission.csv", index=False)
    save_json_atomic(OUT_DIR / "final_report.json", info)
    log(f"submission.csv written ({len(final):,} rows) after {(time.time() - t0) / 60:.1f} min")

    # ---- (2)(3) ablation LAST, so a crash here can never lose the submission -------------------------------------
    if RUN_ABLATION:
        try:
            run_ablation(mm_all, y_tr, ntr, A_va, y_va, h_va.astype(int), cols, feats, dropped)
        except Exception as e:
            log(f"ablation failed: {e!r}")
    log("done")


if __name__ == "__main__":
    main()
