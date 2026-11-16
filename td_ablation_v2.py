#!/usr/bin/env python
"""
TalkingData AdTracking Fraud Detection - ABLATION v2 (standalone, ablation only).

WHY v2: in the first ablation 10 of 11 variants stopped at best_iter=1 (a single tree), so every delta compared
one-tree models; the validation window (last ~45 min of the day) contained none of the test hours (NaN column);
significance came from 3 seeds on one split; and early-stopping + scoring used the same rows.

WHAT v2 DOES
  1. Forward-chained folds on TEST-HOUR blocks (eval hours 9-10 and 13-14 of the last day): train strictly before,
     early-stop on the hour just before, score on the test-hour block. No row is used twice.
  2. Calibration step: picks the base LightGBM config (scale_pos_weight x with/without 'hour') on fold-1 early-stop
     AUC only, and flags every model that stops < HEALTHY iterations as UNSTABLE (so one-tree comparisons can't hide).
  3. Ablation families: leave-one-group-out, only-group, build-up waterfall, R-kernel set, drift-dropped feature,
     'hour' handling, training-setup (spw, depth, max_bin, main-run params, negative-subsample size),
     permutation + gain importance (feature and group level), drop-one for the top-K features.
  4. Statistics: paired per-block (time-block) delta-AUC with bootstrap CI, seed-noise floor from 3 seeds,
     per-fold consistency, verdict = CI excludes 0 AND |delta| > noise floor.
  5. Outputs (outputs/ablation_v2/): ablation_v2_report.md, ablation_v2_results.csv, ablation_v2_importance.csv,
     ablation_v2_calibration.csv, ablation_v2_learning_curve.csv, 4 PNG plots. Fully resumable (jobs.jsonl).

RUN (Kaggle):   !python td_ablation_v2.py                    # standard
                !TD_ABL_QUICK=1 python td_ablation_v2.py     # 1 fold, 2 seeds, no build-up/top-K/scaling  (~1/4 time)
The script reuses the main run's train cache (/kaggle/temp/td_cache) if it exists; otherwise it builds a
40M-row train-only cache (same window as the main run, ~35 min).
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
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
TESTING = os.environ.get("TD_TESTING") == "1"           # tiny smoke-test mode (never set on Kaggle)
QUICK = os.environ.get("TD_ABL_QUICK") == "1"
TEST_ROWS = int(os.environ.get("TD_TEST_ROWS", 300_000))
INPUT_DIR = os.environ.get("TD_INPUT", "/kaggle/input")
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUT_DIR = WORK / "outputs"
ADIR = OUT_DIR / "ablation_v2"
PRED_DIR, EXTRA_DIR = ADIR / "preds", ADIR / "extras"
JOBS_FILE = ADIR / "jobs.jsonl"
CACHE_DIR = OUT_DIR / "cache"

TOTAL_TRAIN_ROWS = 184_903_890
TRAIN_ROWS = TEST_ROWS if TESTING else int(os.environ.get("TD_ABL_ROWS", 40_000_000))   # = main-run window
SEED = 42
TEST_HOURS = (4, 5, 9, 10, 13, 14)
EVAL_BLOCKS = ((9, 10), (13, 14))                       # test-hour blocks used as forward-chained evaluation folds
DROPPED = ["r_iph_ip"]                                  # what the drift scan dropped in the main run
RAW_FEATS = ["app", "device", "os", "channel", "hour"]

ABL = dict(
    neg_frac_train=0.5 if TESTING else 0.10,            # negatives kept for TRAIN (all positives kept)
    neg_frac_eval=1.0 if TESTING else 0.30,             # negatives kept for ES + EVAL sets (AUC stays unbiased)
    es_frac=0.12,                                       # last share of the pre-eval period = early-stopping set
    min_pre_rows=40_000 if TESTING else 3_000_000,
    min_eval_pos=30 if TESTING else 500,
    blocks=4 if TESTING else 12,                        # time blocks per fold for paired delta-AUC
    rounds=25 if TESTING else 500, early=5 if TESTING else 40,
    seeds=(0, 1) if TESTING else (0, 1, 2),
    healthy_iter=3 if TESTING else 25,                  # best_iter below this => flagged UNSTABLE
    topk_drop=2 if TESTING else 5,
    n_boot=500 if TESTING else 5000,
    scale_negfracs=(0.2, 0.8) if TESTING else (0.03, 0.30),
    perm_rows=30_000 if TESTING else 300_000,           # negatives used for permutation importance
    budget_min=float(os.environ.get("TD_ABL_BUDGET_MIN", 0)),   # 0 = unlimited; else stop launching new models
)
if QUICK:
    ABL.update(topk_drop=0, seeds=ABL["seeds"][:2], scale_negfracs=())

PARAMS_BASE = dict(objective="binary", metric="auc", learning_rate=0.1, num_leaves=31, max_depth=-1,
                   min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                   reg_lambda=5.0, max_bin=63, verbose=-1)
# scale_pos_weight is always expressed as on the FULL data and converted for the negative-subsampled train set
# (spw_sub = spw_full * neg_frac_train), so "spw=50" means exactly what it meant in the main run.
MAIN_RUN_OVER = dict(learning_rate=0.08, min_child_samples=200, max_bin=255, spw_full=50.0)

BASE: dict = {}                                         # filled by calibration: feats, spw_full, use_hour
COLS: list = []                                         # cached feature columns (canonical order)


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


def sig_of(*objs) -> str:
    return hashlib.md5(json.dumps(objs, sort_keys=True, default=str).encode()).hexdigest()[:10]


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
    for f in list(d.glob("*.tmp*")):
        try:
            f.unlink()
            log(f"removed stale temp file {f.name}")
        except Exception:
            pass


def disk_preflight(n_cols_est: int = 50) -> None:
    need = TRAIN_ROWS * (n_cols_est * 4 + 26) / 1e9 * 1.15
    free = shutil.disk_usage(CACHE_DIR).free / 1e9
    log(f"disk preflight: need ~{need:.1f} GB, free {free:.1f} GB at {CACHE_DIR}")
    if free < need:
        raise RuntimeError(f"Not enough disk for the feature cache: need ~{need:.1f} GB, free {free:.1f} GB.")


# =============================================================================
# DATA LOADING (train only)
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


# =============================================================================
# FEATURE ENGINEERING  (identical to the main pipeline)
# =============================================================================
def _codes(cols):
    key = cols[0].astype(np.int64)
    for v in cols[1:]:
        key = key * (int(v.max()) + 1) + v.astype(np.int64)
    return pd.factorize(key)[0].astype(np.int64)


def _cumcount(c):
    order = np.argsort(c, kind="stable")
    cs = c[order]
    start = np.r_[True, cs[1:] != cs[:-1]]
    idx = np.arange(len(c))
    first = np.maximum.accumulate(np.where(start, idx, 0))
    out = np.empty(len(c), np.int32)
    out[order] = idx - first
    return out


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
    "ip_app":        (["ip", "app"],                  dict(cnt=1, new=1)),
    "ip_app_os":     (["ip", "app", "os"],            dict(cnt=1, new=1)),
    "ip_dev_os":     (["ip", "device", "os"],         dict(cnt=1)),
    "ip_app_dev_os": (["ip", "app", "device", "os"],  dict(cnt=1, new=1, delta=1)),
    "ip_chan":       (["ip", "channel"],              dict(cnt=1, new=1, delta=1)),
    "ip_app_chan":   (["ip", "app", "channel"],       dict(cnt=1, delta=1)),
    "app_chan":      (["app", "channel"],             dict(cnt=1)),
    "app_os":        (["app", "os"],                  dict(cnt=1)),
    "ip_dh":         (["ip", "dh"],                   dict(cnt=1, nu=["channel", "app"])),
    "ip_app_dh":     (["ip", "app", "dh"],            dict(cnt=1)),
}


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
        if fl.get("new"):
            put(f"new_{g}", _cumcount(code))
        if fl.get("delta"):
            for k, v in _deltas(code, t).items():
                put(f"dt_{g}_{k}", v)
        for tgt in fl.get("nu", []):
            put(f"nu_{g}_{tgt}", _nunique(code, base[tgt]))
        del code
        gc.collect()
    eps = 1.0
    out["r_ipapp_ip"] = out["cnt_ip_app"] / (out["cnt_ip"] + eps)
    out["r_newness_usr"] = out["new_ip_app_dev_os"] / (out["cnt_ip_app_dev_os"] + eps)
    out["r_ipchan_ip"] = out["cnt_ip_chan"] / (out["cnt_ip"] + eps)
    out["r_iph_ip"] = out["cnt_ip_dh"] / (out["cnt_ip"] + eps)
    return out


def expected_cols() -> list[str]:
    out = list(RAW_FEATS)
    for g, (_, fl) in GROUPS.items():
        if fl.get("cnt"):
            out.append(f"cnt_{g}")
        if fl.get("new"):
            out.append(f"new_{g}")
        if fl.get("delta"):
            for k in (1, 2, 3):
                out += [f"dt_{g}_next{k}", f"dt_{g}_prev{k}"]
        for tgt in fl.get("nu", []):
            out.append(f"nu_{g}_{tgt}")
    return out + ["r_ipapp_ip", "r_newness_usr", "r_ipchan_ip", "r_iph_ip"]


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
# ON-DISK FEATURE CACHE (same files/format as the main pipeline -> the main run's cache is reusable)
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


def find_cache() -> Path | None:
    """Reuse a complete train cache (e.g. the main run's) when it is at least 90% as large as the wanted window."""
    cands = [os.environ.get("TD_CACHE"), "/kaggle/temp/td_cache", "/tmp/td_cache", "/kaggle/temp/td_cache_abl"]
    if TESTING:
        cands = [str(OUT_DIR / "cache")]
    for c in cands:
        if not c:
            continue
        d = Path(c)
        if not all((d / f).exists() for f in ("X_all.npy", "raw_all.npy", "y_all.npy", "hours_all.npy", "cols.json")):
            continue
        try:
            n = len(np.load(d / "y_all.npy", mmap_mode="r"))
            same = set(json.load(open(d / "cols.json"))) == set(expected_cols())
        except Exception:
            continue
        if same and n >= 0.9 * TRAIN_ROWS:
            return d
        log(f"cache at {d} not reusable (rows {n:,}, same feature set: {same})")
    return None


# =============================================================================
# FOLDS: forward-chained on test-hour blocks
# =============================================================================
def plan_folds(h: np.ndarray, y: np.ndarray) -> list[dict]:
    """Rows are time-sorted. Fold = [train | early-stop (last es_frac of the pre-eval period) | eval = test-hour block]."""
    n = len(h)
    dec = np.flatnonzero(np.diff(h.astype(np.int16)) < 0)          # hour wraps at midnight -> keep only the last day
    s0 = int(dec[-1] + 1) if len(dec) else 0
    hs = h[s0:]
    folds = []
    for ev in EVAL_BLOCKS:
        pos = np.flatnonzero(np.isin(hs, ev))
        if len(pos) == 0:
            log(f"fold hours {ev}: no rows in the cache window -> skipped")
            continue
        a, b = s0 + int(pos[0]), s0 + int(pos[-1]) + 1
        n_pre, n_pos = a - s0, int(y[a:b].sum())
        if len(pos) != b - a or n_pre < ABL["min_pre_rows"] or n_pos < ABL["min_eval_pos"]:
            log(f"fold hours {ev}: pre-eval rows {n_pre:,}, eval positives {n_pos:,} -> skipped")
            continue
        folds.append(dict(name=f"h{ev[0]}-{ev[-1]}", s0=s0, es0=a - int(ABL["es_frac"] * n_pre), a=a, b=b))
    if not folds:
        a = s0 + int(0.8 * (n - s0))
        folds.append(dict(name="tail20", s0=s0, es0=a - int(ABL["es_frac"] * (a - s0)), a=a, b=n))
        log("WARNING: window has no usable test-hour block -> single chronological 80/20 fold (auc_test_hours not meaningful)")
    return folds


class FoldData:
    """Row indices are cheap (labels only); feature matrices are loaded lazily, only when a model must be trained."""

    def __init__(self, mm, y, f):
        self.mm, self.y, self.f, self.name = mm, y, f, f["name"]
        rng = np.random.RandomState(SEED + 7)
        self.idx_es = self._sub(f["es0"], f["a"], ABL["neg_frac_eval"], rng)
        self.idx_ev = self._sub(f["a"], f["b"], ABL["neg_frac_eval"], rng)
        self.yes, self.yev = y[self.idx_es], y[self.idx_ev]
        self.sig = sig_of(f, len(y), int(y.sum()), ABL["neg_frac_eval"], ABL["es_frac"])
        self.Xtr = self.ytr = self.Xes = self.Xev = None
        self.tr_nf = None
        self._psel = None

    def _sub(self, lo, hi, frac, rng):
        yy = self.y[lo:hi]
        pos, neg = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
        if frac < 1.0:
            neg = rng.choice(neg, int(round(frac * len(neg))), replace=False)
        return np.sort(np.r_[pos, neg]) + lo

    def train_idx(self, nf):
        return self._sub(self.f["s0"], self.f["es0"], nf, np.random.RandomState(SEED + 13 + int(nf * 1000)))

    def load(self, nf):
        if self.Xes is None:
            self.Xes = np.asarray(self.mm[self.idx_es])
            self.Xev = np.asarray(self.mm[self.idx_ev])
        if self.tr_nf != nf:
            self.Xtr = None
            gc.collect()
            idx = self.train_idx(nf)
            self.Xtr, self.ytr, self.tr_nf = np.asarray(self.mm[idx]), self.y[idx], nf

    def perm_sel(self):
        if self._psel is None:
            rng = np.random.RandomState(SEED + 99)
            pos, neg = np.flatnonzero(self.yev == 1), np.flatnonzero(self.yev == 0)
            neg = rng.choice(neg, min(ABL["perm_rows"], len(neg)), replace=False)
            self._psel = np.sort(np.r_[pos, neg])
        return self._psel

    def release(self):
        self.Xtr = self.ytr = self.Xes = self.Xev = None
        self.tr_nf = None
        gc.collect()


# =============================================================================
# SPECS / JOBS
# =============================================================================
def canon(fl):
    s = set(fl)
    return [c for c in COLS if c in s]


def mk(family, label, feats, over=None, folds="all", nf=None) -> dict:
    spec = dict(family=family, label=label, feats=canon(feats), folds=folds,
                over={"spw_full": BASE.get("spw_full", 10.0), **(over or {})})
    if nf is not None:
        spec["neg_frac"] = nf
    return spec


def spec_params(spec, nf) -> dict:
    over = dict(spec["over"])
    spw_full = over.pop("spw_full")
    p = {**PARAMS_BASE, **over}
    p["scale_pos_weight"] = float(spw_full) * float(nf)
    return p


def job_key(fd, fcols, params, seed, nf) -> str:
    return sig_of(fd.sig, list(fcols), params, int(seed), float(nf), ABL["rounds"], ABL["early"])


def load_jobs() -> dict:
    jobs = {}
    if JOBS_FILE.exists():
        for line in open(JOBS_FILE):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    jobs[r["key"]] = r
                except Exception:
                    pass
    return jobs


class Runner:
    def __init__(self):
        self.jobs = load_jobs()
        self.cix = {c: i for i, c in enumerate(COLS)}
        self.ds, self.dkey = None, None
        self.n_run = 0
        self.t0 = time.time()
        self.deadline = self.t0 + 60 * ABL["budget_min"] if ABL["budget_min"] > 0 else None
        self.over_budget = False

    def key_of(self, spec, fd, seed=0):
        nf = spec.get("neg_frac", ABL["neg_frac_train"])
        return job_key(fd, spec["feats"], spec_params(spec, nf), seed, nf)

    def extras(self, spec, fd):
        p = EXTRA_DIR / f"{self.key_of(spec, fd)}.json"
        return json.load(open(p)) if p.exists() else None

    def reset(self):
        self.ds, self.dkey = None, None
        gc.collect()

    def run(self, spec, fd, seed=0, keep_extras=False):
        nf = spec.get("neg_frac", ABL["neg_frac_train"])
        fcols = spec["feats"]
        params = spec_params(spec, nf)
        key = job_key(fd, fcols, params, seed, nf)
        rec = self.jobs.get(key)
        if rec and (PRED_DIR / f"{key}.npy").exists() and (not keep_extras or (EXTRA_DIR / f"{key}.json").exists()):
            return rec
        if self.deadline and time.time() > self.deadline:
            if not self.over_budget:
                log("TIME BUDGET reached: no new models will be started (analysis runs on what is finished)")
                self.over_budget = True
            return None
        t = time.time()
        fd.load(nf)
        ix = [self.cix[c] for c in fcols]
        mb = params["max_bin"]
        dkey = (fd.name, tuple(fcols), mb, nf)
        if dkey != self.dkey:                                        # re-use binned data across param-only variants
            self.ds = None
            gc.collect()
            dp = dict(max_bin=mb, verbose=-1, feature_pre_filter=False)
            d_tr = lgb.Dataset(fd.Xtr[:, ix], fd.ytr, feature_name=list(fcols), params=dp, free_raw_data=True).construct()
            d_es = lgb.Dataset(fd.Xes[:, ix], fd.yes, reference=d_tr, params=dp, free_raw_data=True).construct()
            self.ds, self.dkey = (d_tr, d_es), dkey
        d_tr, d_es = self.ds
        store: dict = {}
        m = lgb.train({**params, "seed": SEED + seed}, d_tr, num_boost_round=ABL["rounds"], valid_sets=[d_es],
                      valid_names=["es"], callbacks=[lgb.early_stopping(ABL["early"], verbose=False),
                                                     lgb.record_evaluation(store)])
        bi = int(m.best_iteration or ABL["rounds"])
        pe = m.predict(fd.Xev[:, ix], num_iteration=bi).astype(np.float32)
        rec = dict(key=key, label=spec["label"], family=spec["family"], fold=fd.name, seed=int(seed), n_features=len(fcols),
                   neg_frac=nf, best_iter=bi, es_auc=float(m.best_score["es"]["auc"]), eval_auc=auc(fd.yev, pe),
                   unstable=bool(bi < ABL["healthy_iter"]), secs=round(time.time() - t, 1))
        save_npy_atomic(PRED_DIR / f"{key}.npy", pe)
        if keep_extras:
            self._extras(m, bi, store, fd, fcols, ix, key)
        with open(JOBS_FILE, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        self.jobs[key] = rec
        self.n_run += 1
        flag = "  <-- UNSTABLE (stopped very early)" if rec["unstable"] else ""
        log(f"#{self.n_run} {spec['label']} | {fd.name} s{seed} | {len(fcols)}f | iters {bi} | "
            f"ES {rec['es_auc']:.5f} | EVAL {rec['eval_auc']:.5f} | {rec['secs']:.0f}s{flag}")
        del m
        return rec

    def _extras(self, m, bi, store, fd, fcols, ix, key):
        """Learning curve, gain + permutation importance (feature and group level) for the base model."""
        sel = fd.perm_sel()
        Xp = fd.Xev[sel][:, ix]
        yp = fd.yev[sel]
        cur = m.current_iteration()
        ks = sorted({k for k in (1, 2, 5, 10, 20, 50, 100, 200, 400, bi) if 1 <= k <= cur})
        ck = {int(k): auc(yp, m.predict(Xp, num_iteration=k)) for k in ks}
        base_a = auc(yp, m.predict(Xp, num_iteration=bi))
        rng = np.random.RandomState(SEED + 5)
        perm_f = []
        for j, c in enumerate(fcols):
            col, drops = Xp[:, j].copy(), []
            for _ in range(2):
                Xp[:, j] = col[rng.permutation(len(col))]
                drops.append(base_a - auc(yp, m.predict(Xp, num_iteration=bi)))
            Xp[:, j] = col
            perm_f.append(dict(feature=c, group=group_of(c), drop_mean=float(np.mean(drops)), drop_sd=float(np.std(drops))))
        gidx: dict = {}
        for j, c in enumerate(fcols):
            gidx.setdefault(group_of(c), []).append(j)
        perm_g = []
        for g, js in gidx.items():
            src, drops = Xp[:, js].copy(), []
            for _ in range(2):
                Xp[:, js] = src[rng.permutation(len(src))]
                drops.append(base_a - auc(yp, m.predict(Xp, num_iteration=bi)))
            Xp[:, js] = src
            perm_g.append(dict(group=g, n=len(js), drop_mean=float(np.mean(drops)), drop_sd=float(np.std(drops))))
        gain = m.feature_importance("gain", iteration=bi).astype(float)
        gain = gain / max(gain.sum(), 1e-12)
        save_json_atomic(EXTRA_DIR / f"{key}.json", dict(
            fold=fd.name, best_iter=bi, es_curve=[float(v) for v in store["es"]["auc"]], eval_ck=ck, base_sub_auc=base_a,
            gain={c: float(g) for c, g in zip(fcols, gain)}, perm_features=perm_f, perm_groups=perm_g))


# =============================================================================
# CALIBRATION: choose the base config on EARLY-STOP AUC only (never on eval rows)
# =============================================================================
def calibrate(runner, fd, feats_all):
    nohour = [c for c in feats_all if c != "hour"]
    rows = []
    for use_hour in (True, False):
        for s in (1.0, 10.0, 50.0):
            spec = mk("calibration", f"calib hour={use_hour} spw={s:g}", feats_all if use_hour else nohour, over=dict(spw_full=s))
            rec = runner.run(spec, fd, 0)
            if rec:
                rows.append(dict(use_hour=use_hour, spw_full=s, es_auc=rec["es_auc"], best_iter=rec["best_iter"],
                                 healthy=not rec["unstable"]))
    if not rows:
        return dict(use_hour=True, spw_full=10.0), pd.DataFrame()
    df = pd.DataFrame(rows)
    d = df.sort_values(["healthy", "es_auc"], ascending=[False, False]).reset_index(drop=True)
    best = d.iloc[0]
    with_h = d[d.use_hour].iloc[0]
    # keep the final feature set (with 'hour') unless dropping it clearly wins
    if (not best.use_hour) and with_h.healthy == best.healthy and best.es_auc - with_h.es_auc <= 2e-4:
        best = with_h
    if not best.healthy:
        log("WARNING: no calibration config trained past the healthy-iteration floor; the window itself is shifted. "
            "Results below are flagged UNSTABLE.")
    log(f"calibration -> use_hour={bool(best.use_hour)}  spw_full={best.spw_full:g}  (ES AUC {best.es_auc:.5f}, iters {int(best.best_iter)})")
    return dict(use_hour=bool(best.use_hour), spw_full=float(best.spw_full)), df


def pick_topk(ex, k):
    if not ex or k <= 0:
        return []
    pf = sorted(ex["perm_features"], key=lambda r: -r["drop_mean"])
    return [r["feature"] for r in pf if r["feature"] in BASE["feats"]][:k]


def build_plan(topk) -> list[dict]:
    fb = BASE["feats"]
    G: dict = {}
    for c in fb:
        G.setdefault(group_of(c), []).append(c)
    plan = [mk("base", "BASE", fb)]
    add = lambda *a, **k: plan.append(mk(*a, **k))                       # noqa: E731
    for g in G:
        add("leave-one-group-out", f"minus group: {g}", [c for c in fb if group_of(c) != g])
    add("only-group", "raw only", G.get("raw", []))
    for g in G:
        if g != "raw":
            add("only-group", f"raw + {g}", G.get("raw", []) + G[g])
    if not QUICK:
        acc = list(G.get("raw", []))
        order = [g for g in ("counts", "newness", "uniques", "deltas", "ratios") if g in G]
        BASE["last_group"] = order[-1] if order else ""
        for i, g in enumerate(order[:-1]):                              # the last step is BASE itself
            acc = acc + G[g]
            add("build-up", f"build-up {i + 1}: +{g}", list(acc))
    add("feature-set", "R-kernel set (no deltas/uniques)", core_cols(fb))
    extra = [c for c in DROPPED if c in COLS and c not in fb]
    if extra:
        add("feature-set", "+ " + ",".join(extra) + " (drift-dropped)", fb + extra)
    if "hour" in fb:
        add("feature-set", "minus hour", [c for c in fb if c != "hour"])
    else:
        add("feature-set", "plus hour (final set)", fb + ["hour"])
    for s in (1.0, 10.0, 50.0):
        if abs(s - BASE["spw_full"]) > 1e-9:
            add("training-setup", f"spw={s:g} (full-data equiv.)", fb, over=dict(spw_full=s))
    add("training-setup", "shallow trees (7 leaves, depth 4)", fb, over=dict(num_leaves=7, max_depth=4))
    add("training-setup", "max_bin=255", fb, over=dict(max_bin=255))
    add("training-setup", "main-run params (lr .08, mcs 200, bin 255, spw 50)", fb, over=dict(MAIN_RUN_OVER))
    for f in topk:
        add("drop-one (top permutation features)", f"minus feature: {f}", [c for c in fb if c != f])
    for nf in ABL["scale_negfracs"]:
        add("training-setup", f"train negatives kept = {nf:g}", fb, nf=nf, folds="last")
    return plan


# =============================================================================
# STATISTICS
# =============================================================================
def block_aucs(y, p, nb=None):
    nb = nb or ABL["blocks"]
    edges = np.linspace(0, len(y), nb + 1).astype(int)
    out = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        yy = y[e0:e1]
        out.append(auc(yy, p[e0:e1]) if 0 < yy.sum() < len(yy) else np.nan)
    return np.array(out)


def boot_ci(d, n_boot=None, seed=SEED):
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), (n_boot or ABL["n_boot"], len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def verdict(d, lo, hi, thr):
    if lo > 0 and d > thr:
        return "BETTER than base"
    if hi < 0 and -d > thr:
        return "WORSE than base"
    if lo > 0 or hi < 0:
        return "tiny (below noise floor)"
    return "no measurable difference"


def analyse(runner, fds, plan):
    names = [fd.name for fd in fds]

    def load(spec, fd, seed=0):
        key = runner.key_of(spec, fd, seed)
        rec, p = runner.jobs.get(key), PRED_DIR / f"{key}.npy"
        return (rec, np.load(p)) if rec is not None and p.exists() else (None, None)

    base_spec = plan[0]
    base_blk, base_auc = {}, {}
    for fd in fds:
        rec, p = load(base_spec, fd)
        if rec is not None:
            base_blk[fd.name], base_auc[fd.name] = block_aucs(fd.yev, p), rec["eval_auc"]
    sds = []
    for fd in fds:
        a = [r["eval_auc"] for s in ABL["seeds"] for r, _ in [load(base_spec, fd, s)] if r is not None]
        if len(a) > 1:
            sds.append(np.std(a, ddof=1))
    seed_sd = float(np.mean(sds)) if sds else float("nan")
    thr = max(2 * seed_sd, 2e-4) if seed_sd == seed_sd else 5e-4
    log(f"seed-noise sd(AUC) = {seed_sd:.5f} -> materiality threshold {thr:.5f}")

    rows = []
    for spec in plan:
        parts, aucs, iters, unst, per = [], [], [], 0, {}
        for fd in fds:
            if spec["folds"] == "last" and fd.name != names[-1]:
                continue
            rec, p = load(spec, fd)
            if rec is None or fd.name not in base_blk:
                continue
            parts.append(block_aucs(fd.yev, p) - base_blk[fd.name])
            aucs.append(rec["eval_auc"])
            iters.append(rec["best_iter"])
            unst += int(rec["unstable"])
            per[fd.name] = (rec["eval_auc"], rec["eval_auc"] - base_auc[fd.name])
        if not aucs:
            continue
        d = np.concatenate(parts)
        d = d[np.isfinite(d)]
        mean, lo, hi = boot_ci(d)
        is_base = spec["family"] == "base"
        row = dict(family=spec["family"], variant=spec["label"], n_features=len(spec["feats"]), folds=len(aucs),
                   auc=float(np.mean(aucs)), delta=0.0 if is_base else mean, ci_lo=0.0 if is_base else lo,
                   ci_hi=0.0 if is_base else hi,
                   blocks_worse="-" if is_base else f"{int((d < 0).sum())}/{len(d)}",
                   best_iter=int(np.mean(iters)), unstable_folds=unst,
                   verdict="BASE" if is_base else verdict(mean, lo, hi, thr))
        for n in names:
            row[f"auc_{n}"], row[f"d_{n}"] = per.get(n, (np.nan, np.nan))
        rows.append(row)
    return pd.DataFrame(rows), seed_sd, thr


# =============================================================================
# REPORTING
# =============================================================================
def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)

    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if v != v else f"{v:.5f}"
        return str(v)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f(r[c]) for c in cols) + " |")
    return "\n".join(out)


def importance_tables(runner, fds, base_spec):
    feat, grp, curve = [], [], []
    for fd in fds:
        ex = runner.extras(base_spec, fd)
        if not ex:
            continue
        for r in ex["perm_features"]:
            feat.append(dict(fold=fd.name, **r, gain_share=ex["gain"].get(r["feature"], np.nan)))
        for r in ex["perm_groups"]:
            grp.append(dict(fold=fd.name, **r))
        ck = {int(k): v for k, v in ex["eval_ck"].items()}
        for i, a in enumerate(ex["es_curve"], 1):
            curve.append(dict(fold=fd.name, iteration=i, es_auc=a, eval_auc_sub=ck.get(i, np.nan)))
    fi = pd.DataFrame(feat)
    if len(fi):
        fi = fi.groupby(["feature", "group"], as_index=False).agg(
            perm_drop_auc=("drop_mean", "mean"), perm_drop_sd=("drop_sd", "mean"), gain_share=("gain_share", "mean")
        ).sort_values("perm_drop_auc", ascending=False)
        fi.insert(0, "level", "feature")
    gi = pd.DataFrame(grp)
    if len(gi):
        gi = gi.groupby("group", as_index=False).agg(
            perm_drop_auc=("drop_mean", "mean"), perm_drop_sd=("drop_sd", "mean"), n=("n", "first")
        ).sort_values("perm_drop_auc", ascending=False)
        gi.insert(0, "level", "group")
    return fi, gi, pd.DataFrame(curve)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_forest(res, thr, path):
    plt = _plt()
    d = res[res.family != "base"].sort_values("delta")
    fams = list(dict.fromkeys(d.family))
    colors = {f: plt.cm.tab10(i % 10) for i, f in enumerate(fams)}
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(d) + 1.5)))
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.errorbar(r.delta, yi, xerr=[[r.delta - r.ci_lo], [r.ci_hi - r.delta]], fmt="o", color=colors[r.family], capsize=2)
    ax.axvline(0, color="k", lw=1)
    ax.axvspan(-thr, thr, color="grey", alpha=0.15, label=f"seed-noise band (+/-{thr:.4f})")
    ax.set_yticks(y)
    ax.set_yticklabels(d.variant, fontsize=8)
    ax.set_xlabel("delta AUC vs BASE (paired per-block mean, 95% bootstrap CI)  | left = worse")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=colors[f], label=f) for f in fams] + [ax.patches[0]],
              fontsize=7, loc="upper left")
    ax.set_title("TalkingData ablation v2: effect of each change on test-hour AUC")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_buildup(res, path):
    plt = _plt()
    steps = res[res.variant == "raw only"].to_dict("records") + res[res.family == "build-up"].to_dict("records") \
        + res[res.family == "base"].to_dict("records")
    if len(steps) < 3:
        return
    fold_cols = [c for c in res.columns if c.startswith("auc_")]
    labels = ["raw"] + ["+" + s["variant"].split("+")[-1] for s in steps[1:-1]] + [f"+{BASE.get('last_group', '')} (=BASE)"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(steps))
    for c in fold_cols:
        ax.plot(x, [s[c] for s in steps], "o--", alpha=0.5, label=c.replace("auc_", "fold "))
    ax.plot(x, [s["auc"] for s in steps], "ko-", lw=2, label="mean")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("eval AUC (test hours)")
    ax.set_title("Build-up: AUC as feature groups are added (order = how they were engineered)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_importance(fi, gi, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw=dict(width_ratios=[2, 1]))
    t = fi.head(20).iloc[::-1]
    axes[0].barh(t.feature, t.perm_drop_auc, xerr=t.perm_drop_sd)
    axes[0].set_title("Permutation importance (drop in AUC), top 20 features")
    g = gi.iloc[::-1]
    axes[1].barh(g.group, g.perm_drop_auc, xerr=g.perm_drop_sd, color="tab:orange")
    axes[1].set_title("Group-level permutation")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_curve(curve, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (fold, g) in enumerate(curve.groupby("fold")):
        ax.plot(g.iteration, g.es_auc, color=plt.cm.tab10(i), label=f"{fold}: early-stop set")
        e = g.dropna(subset=["eval_auc_sub"])
        ax.plot(e.iteration, e.eval_auc_sub, "s--", color=plt.cm.tab10(i), label=f"{fold}: eval (test hours)")
    ax.set_xlabel("boosting iteration")
    ax.set_ylabel("AUC")
    ax.set_title("Base model learning curve - a curve that peaks at iteration ~1 means train/eval shift")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(res, fi, gi, calib, seed_sd, thr, fds, runner, n_rows):
    L = ["# TalkingData AdTracking - Ablation v2", ""]
    L.append(f"**Window**: last {n_rows:,} train rows (same window/features as the main run). **Protocol**: forward-chained folds; "
             f"train = rows strictly before the early-stop slice, early-stop = last {ABL['es_frac']:.0%} of the pre-eval period, "
             f"eval = a test-hour block. Train keeps all positives + {ABL['neg_frac_train']:.0%} of negatives; ES/eval keep "
             f"{ABL['neg_frac_eval']:.0%} of negatives (AUC is unbiased under uniform negative subsampling). "
             f"Metric: AUC on eval; delta = mean paired per-block delta ({ABL['blocks']} time blocks/fold), 95% bootstrap CI; "
             f"verdict requires CI to exclude 0 AND |delta| > noise floor ({thr:.5f}; seed sd {seed_sd:.5f}, {len(ABL['seeds'])} seeds).")
    L += ["", "## Folds", ""]
    fr = []
    for fd in fds:
        fr.append(dict(fold=fd.name, train_rows=len(fd.train_idx(ABL["neg_frac_train"])), es_rows=len(fd.idx_es),
                       eval_rows=len(fd.idx_ev), eval_pos=int(fd.yev.sum())))
    L.append(md_table(pd.DataFrame(fr)))
    if len(calib):
        L += ["", "## Calibration (chosen on early-stop AUC only)", "", md_table(calib), "",
              f"Chosen: use_hour={BASE['use_hour']}, spw_full={BASE['spw_full']:g}. "
              + ("`hour` was excluded from the base: with forward-chained folds the eval hours are unseen by construction, so "
                 "`hour` cannot be assessed fairly in a single-day window (it IS seen at test time)." if not BASE["use_hour"] else "")]
    nun = int(res.unstable_folds.sum())
    L += ["", "## Health check", "",
          f"{nun} model(s) stopped before iteration {ABL['healthy_iter']} (flagged UNSTABLE in results.csv). "
          + ("**Comparisons involving them are not reliable.**" if nun else "All models trained past the floor, so deltas compare real models.")]
    cols = ["variant", "n_features", "auc", "delta", "ci_lo", "ci_hi", "blocks_worse", "best_iter", "verdict"] + \
           [c for c in res.columns if c.startswith("d_")]
    for fam, title in (("base", "Base"), ("leave-one-group-out", "Leave-one-group-out (delta < 0 => the group helps)"),
                       ("only-group", "Only-group (raw + one group)"), ("build-up", "Build-up"),
                       ("feature-set", "Feature-set variants"), ("training-setup", "Training-setup ablations"),
                       ("drop-one (top permutation features)", "Drop-one for the top permutation features")):
        t = res[res.family == fam]
        if len(t):
            L += ["", f"## {title}", "", md_table(t[cols])]
    if len(gi):
        L += ["", "## Importance", "", "Group-level permutation (drop in AUC):", "", md_table(gi.drop(columns="level")), "",
              "Top 15 features (permutation drop in AUC, gain share):", "",
              md_table(fi.head(15).drop(columns="level"))]
    logo = res[res.family == "leave-one-group-out"]
    hurts = logo[logo.verdict == "WORSE than base"].sort_values("delta")
    helps = logo[logo.verdict == "BETTER than base"]
    L += ["", "## Takeaways (auto-generated; read together with the tables)", ""]
    L.append("- Groups whose removal measurably hurts: " + (", ".join(f"{r.variant.split(': ')[1]} ({r.delta:+.4f})" for r in hurts.itertuples()) or "none beyond noise") + ".")
    L.append("- Groups whose removal measurably helps (prune candidates): " + (", ".join(f"{r.variant.split(': ')[1]} ({r.delta:+.4f})" for r in helps.itertuples()) or "none") + ".")
    ts = res[res.family == "training-setup"].sort_values("delta")
    if len(ts):
        L.append("- Largest training-setup effects: " + ", ".join(f"{r.variant} ({r.delta:+.4f})" for r in ts.head(3).itertuples()) + ".")
    L.append("- Limit: one-day window => `hour` and day-level drift cannot be validated; features are computed over the whole window "
             "(as in the main pipeline), the absolute AUC is on negative-subsampled eval rows (unbiased).")
    (ADIR / "ablation_v2_report.md").write_text("\n".join(L), encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================
def main_ablation():
    global CACHE_DIR, COLS
    t0 = time.time()
    for d in (ADIR, PRED_DIR, EXTRA_DIR):
        d.mkdir(parents=True, exist_ok=True)
    log_mem("start")
    log(f"lightgbm {lgb.__version__} | QUICK={QUICK} TESTING={TESTING}")

    found = find_cache()
    if found:
        CACHE_DIR = found
        log(f"re-using train cache at {CACHE_DIR}")
    else:
        CACHE_DIR = pick_cache_dir()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        clean_tmp(CACHE_DIR)
        if not TESTING:
            disk_preflight()
    COLS = ensure_train_cache()
    mm = np.load(CACHE_DIR / "X_all.npy", mmap_mode="r")
    y = np.load(CACHE_DIR / "y_all.npy")
    h = np.load(CACHE_DIR / "hours_all.npy")
    assert len(mm) == len(y) == len(h)
    log(f"cache: {len(y):,} rows, {len(COLS)} features, positives {y.mean():.4%}, hours {int(h.min())}..{int(h.max())}")

    feats_all = [c for c in COLS if c not in DROPPED]
    folds = plan_folds(h, y)
    if QUICK:
        folds = folds[-1:]
    fds = [FoldData(mm, y, f) for f in folds]
    for fd in fds:
        log(f"fold {fd.name}: train rows [{fd.f['s0']:,}..{fd.f['es0']:,}) | ES [{fd.f['es0']:,}..{fd.f['a']:,}) | "
            f"eval [{fd.f['a']:,}..{fd.f['b']:,}) -> {len(fd.idx_ev):,} rows, {int(fd.yev.sum()):,} positives")

    runner = Runner()

    def safe(spec, fd, seed=0, **kw):
        try:
            return runner.run(spec, fd, seed, **kw)
        except Exception as e:                                        # one failed model never loses the others
            log(f"FAILED {spec['label']} on {fd.name}: {e!r}")
            runner.reset()
            return None

    # ---- stage 1: calibration on the first fold -------------------------------------------------------------------
    cfg, calib = calibrate(runner, fds[0], feats_all)
    BASE.update(spw_full=cfg["spw_full"], use_hour=cfg["use_hour"],
                feats=canon(feats_all if cfg["use_hour"] else [c for c in feats_all if c != "hour"]))
    save_json_atomic(ADIR / "champion.json", dict(use_hour=BASE["use_hour"], spw_full=BASE["spw_full"], n_features=len(BASE["feats"])))
    calib.to_csv(ADIR / "ablation_v2_calibration.csv", index=False)

    # ---- stage 2: base + every variant, fold by fold --------------------------------------------------------------
    plan = None
    for i, fd in enumerate(fds):
        log(f"=== fold {fd.name} ({i + 1}/{len(fds)}) | elapsed {(time.time() - t0) / 60:.1f} min ===")
        base_spec = mk("base", "BASE", BASE["feats"])
        safe(base_spec, fd, 0, keep_extras=True)
        if plan is None:
            plan = build_plan(pick_topk(runner.extras(base_spec, fd), ABL["topk_drop"]))
            log(f"plan: {len(plan)} variants x {len(fds)} fold(s); top-K drop-one = {[s['label'] for s in plan if s['family'].startswith('drop-one')]}")
        for s in ABL["seeds"][1:]:
            safe(base_spec, fd, s)
        last = i == len(fds) - 1
        for spec in plan[1:]:
            if spec.get("neg_frac") is None:
                safe(spec, fd, 0)
        if last:
            for spec in plan[1:]:
                if spec.get("neg_frac") is not None:
                    safe(spec, fd, 0)
        runner.reset()
        fd.release()

    # ---- stage 3: statistics + report ---------------------------------------------------------------------------
    res, seed_sd, thr = analyse(runner, fds, plan)
    res.to_csv(ADIR / "ablation_v2_results.csv", index=False)
    fi, gi, curve = importance_tables(runner, fds, mk("base", "BASE", BASE["feats"]))
    if len(fi):
        pd.concat([fi, gi], ignore_index=True).to_csv(ADIR / "ablation_v2_importance.csv", index=False)
    if len(curve):
        curve.to_csv(ADIR / "ablation_v2_learning_curve.csv", index=False)
    write_report(res, fi, gi, calib, seed_sd, thr, fds, runner, len(y))
    for name, fn in (("ablation_v2_forest.png", lambda p: plot_forest(res, thr, p)),
                     ("ablation_v2_buildup.png", lambda p: plot_buildup(res, p)),
                     ("ablation_v2_importance.png", lambda p: plot_importance(fi, gi, p)),
                     ("ablation_v2_learning_curve.png", lambda p: plot_curve(curve, p))):
        try:
            fn(ADIR / name)
        except Exception as e:
            log(f"plot {name} skipped: {e!r}")

    show = res[["family", "variant", "auc", "delta", "ci_lo", "ci_hi", "best_iter", "unstable_folds", "verdict"]]
    log("RESULTS:\n" + show.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {ADIR}")


if __name__ == "__main__":
    main_ablation()
