# Generic cohort feature-discovery script — leak-safe univariate + discriminative ranking.
#
# March 2026 enhancements:
#   - Composite score redesigned: 2×SHAP + LGBM_gain + MI
#     (F-test dropped: linear/redundant for binary data; per-feature AUC reported but not scored)
#   - LightGBM + SHAP as primary importance signals (multivariate, imbalance-aware)
#   - CV bug fixed: StratifiedGroupKFold(groups=arange) → StratifiedKFold
#   - Directional screen is BIDIRECTIONAL: protective features are kept, direction reported
#   - Module-level cache (get_preprocessed_data) lets dashboard recompute CV in real time
#   - run_cv_from_cache(): fast CV recomputation when user changes Top-N slider
#   - detect_leakers(): post-hoc audit flags features that may be leaky
#
# Requires: Python 3.7+, scikit-learn, statsmodels
# Optional but strongly recommended: lightgbm, shap
# Last updated: March 2026, Maxim Seferovic
#!/usr/bin/env python3

import os, re, pickle, warnings, hashlib, gzip, time
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, accuracy_score

from statsmodels.stats.multitest import multipletests

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    print("WARNING: lightgbm not installed. Falling back to L1 LogReg for feature importance.")

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    print("WARNING: shap not installed. SHAP component will be skipped.")

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42

_RUN_T0: float = 0.0   # set at entry of run_analysis(); used by _tlog()


def _tlog(msg: str) -> None:
    """Print a progress line with elapsed-time prefix, always flushed immediately."""
    elapsed = time.time() - _RUN_T0
    print(f"[+{elapsed:5.0f}s] {msg}", flush=True)


PICKLE_IN           = "PBDBfinal_ready_forML_IHCP_paper3.pkl"
MIN_BINARY_COUNT    = 10
DIRECTIONAL_MARGIN  = 0.002
TOP_N               = 50

# ---------------------------------------------------------------------------
# Module-level in-memory cache — populated by run_analysis(), read by get_preprocessed_data()
# Allows the dashboard to recompute CV in real time without re-running the full pipeline.
# ---------------------------------------------------------------------------
_cached_Xn     = None
_cached_y      = None
_cached_n_pos  = None
_cached_n_neg  = None
_cached_Xf_raw = None

# ---------------------------------------------------------------------------
# Two-tier cache system:
#   Tier 1 — RAM (module-level vars): survives Streamlit reruns within the same process.
#             Repeat "Run Analysis" on the same cohort returns in < 1 s with zero disk I/O.
#   Tier 2 — Disk (gzip pickle): survives Streamlit restarts; ~5 s load on cold start.
#
#   prep cache: Xn/y/means/delta after hygiene+screen+imputation+corr-filter (20–60 s saved)
#   rank cache: complete rank_df after MI+LGB+SHAP (60–120 s additional saved)
#
# Cache keys encode every parameter that affects the output — any change auto-invalidates.
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pm_cache")

# RAM prep cache
_mem_prep_key  = None   # str  — hash of last preprocessed cohort
_mem_prep_data = None   # dict — Xn, y, means, delta, …

# RAM rank cache
_mem_rank_key  = None   # str          — hash of last ranked result
_mem_rank_df   = None   # pd.DataFrame — cached rank_df


def _prep_cache_key(cohort_ids, search_terms, min_binary_count,
                    directional_margin, pickle_file, control_ids=None):
    """Return a short hash string that uniquely identifies this preprocessing configuration."""
    mtime = os.path.getmtime(pickle_file)
    blob  = str((
        sorted(cohort_ids),
        sorted(search_terms or []),
        min_binary_count,
        round(directional_margin, 6),
        mtime,
        sorted(control_ids) if control_ids else None,
    )).encode()
    return hashlib.md5(blob).hexdigest()[:10]


def _load_prep_cache(key):
    """Return cached preprocessing dict if it exists and is readable, else None."""
    path = os.path.join(_CACHE_DIR, f"prep_{key}.pkl.gz")
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"  Preprocessing cache found — skipping preprocessing "
              f"({obj['Xn'].shape[1]} features, {len(obj['y'])} subjects)")
        return obj
    except Exception as e:
        print(f"  Cache load failed ({e}) — recomputing.")
        return None


def _save_prep_cache(key, obj):
    """Save preprocessing results to disk and remove any stale cache files."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"prep_{key}.pkl.gz")
    with gzip.open(path, "wb", compresslevel=1) as f:   # level 1: ~5–10× faster write
        pickle.dump(obj, f, protocol=4)
    # Remove stale cache files (keep only the current one)
    for fname in os.listdir(_CACHE_DIR):
        if fname.startswith("prep_") and fname.endswith(".pkl.gz") and fname != f"prep_{key}.pkl.gz":
            try:
                os.remove(os.path.join(_CACHE_DIR, fname))
            except OSError:
                pass
    print(f"  Saved preprocessing cache ({path})")


# ── Rank cache helpers ───────────────────────────────────────────────────────

def _rank_cache_key(prep_key: str, cv_shap: bool, compute_extras: bool = False,
                    exclude_cols=None) -> str:
    """Hash of (prep_key, cv_shap, compute_extras, exclude_cols) — identifies a unique ranking."""
    return hashlib.md5(
        str((prep_key, bool(cv_shap), bool(compute_extras), sorted(exclude_cols or []))).encode()
    ).hexdigest()[:10]


def _load_rank_cache(key: str):
    """Return cached rank_df if it exists and is readable, else None."""
    path = os.path.join(_CACHE_DIR, f"rank_{key}.pkl.gz")
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rb") as f:
            df = pickle.load(f)
        print(f"  Rank cache: disk hit — {len(df)} features loaded, skipping MI/LGB/SHAP")
        return df
    except Exception as e:
        print(f"  Rank cache load failed ({e}) — recomputing.")
        return None


def _save_rank_cache(key: str, rank_df):
    """Save rank_df to disk and evict stale rank cache files."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"rank_{key}.pkl.gz")
    with gzip.open(path, "wb", compresslevel=1) as f:   # level 1: ~5–10× faster write
        pickle.dump(rank_df, f, protocol=4)
    for fname in os.listdir(_CACHE_DIR):
        if (fname.startswith("rank_") and fname.endswith(".pkl.gz")
                and fname != f"rank_{key}.pkl.gz"):
            try:
                os.remove(os.path.join(_CACHE_DIR, fname))
            except OSError:
                pass
    print(f"  Saved rank cache ({path})")


def get_preprocessed_data():
    """Return (Xn, y, n_pos, n_neg) from the most recent run_analysis() call.
    Use with run_cv_from_cache() to recompute CV metrics when Top-N changes."""
    return _cached_Xn, _cached_y, _cached_n_pos, _cached_n_neg


def get_raw_data():
    """Return NaN-preserved preprocessed data (same columns as Xn, before imputation)."""
    return _cached_Xf_raw


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
ERA_RE          = re.compile(r"(year_|_year\b|clinic|provider|doctor)", re.I) #hospital formerly there.
TEXT_JUNK_RE    = re.compile(r"(other\s*\(details\)|_x_)", re.I)
JUNK_SHORT_MED_RE = re.compile(r"(COMBINED LIST|Medications).*_[a-z]{1,2}$", re.I)
NEG_TOKENS = ("_No", "_Negative", "_False", "_None", "_Never")
POS_TOKENS = ("_Yes", "_Positive", "_True", "_Present", "_Ever", "_Sometimes", "_Often", "_History", "_Hx")


def _feature_category(col_name: str) -> str:
    m = re.match(r"^([A-Za-z][A-Za-z0-9]*)__", str(col_name))
    return m.group(1) if m else "Other"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_binary_col(s: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(s):
        return False
    vals = pd.unique(s.dropna())
    return len(vals) <= 2 and set(vals).issubset({0, 1})


def _safe_auc(y, x):
    """Per-feature AUC, always >= 0.5 (direction-agnostic for ranking)."""
    try:
        x = pd.to_numeric(x, errors="coerce").fillna(0).to_numpy()
        if np.unique(x).size <= 1:
            return 0.5
        score = float(roc_auc_score(y, x))
        return max(score, 1.0 - score)
    except Exception:
        return 0.5


def _rank_norm(a):
    """Rank-based normalisation → uniform [0, 1]. Robust to outliers."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    if n <= 1:
        return np.zeros_like(a)
    ranks = pd.Series(a).rank(method="average", ascending=True, na_option="bottom")
    return ((ranks - 1) / (n - 1)).to_numpy()


def _unpack_shap(sv):
    """Normalise SHAP output to 2D (samples × features) for binary classification.
    Handles API differences across SHAP library versions:
      - older SHAP (<0.40): returns list [neg_class_array, pos_class_array]
      - some mid versions:  returns 3D array (samples, features, classes)
      - current SHAP:       returns 2D array (samples, features) for binary clf
    """
    if isinstance(sv, list):
        return sv[1]                                      # older: pick pos-class
    if hasattr(sv, 'ndim') and sv.ndim == 3:
        return sv[:, :, 1]                                # 3D: pick pos-class slice
    return sv                                             # 2D: already correct


def _sens_spec_at_threshold(y_true, y_score, target=0.90):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1.0 - fpr
    mask_spec = spec >= target
    sens_at_spec = float(tpr[mask_spec].max()) if mask_spec.any() else 0.0
    mask_sens = tpr >= target
    spec_at_sens = float(spec[mask_sens].max()) if mask_sens.any() else 0.0
    return sens_at_spec, spec_at_sens


def _build_lgbm(n_neg, n_pos, n_jobs: int = -1):
    """Build a LightGBM classifier scaled to cohort size.

    n_jobs controls internal LGB threading.  Pass n_jobs=1 when running folds
    in parallel via joblib (avoids CPU oversubscription).
    """
    spw = max(1.0, n_neg / max(n_pos, 1))
    # Scale model complexity with cohort size to avoid overfitting small cohorts.
    if n_pos < 300:
        num_leaves, min_child = 15, max(20, n_pos // 10)
    elif n_pos < 1000:
        num_leaves, min_child = 31, max(20, n_pos // 15)
    else:
        num_leaves, min_child = 63, 20
    return lgb.LGBMClassifier(
        n_estimators=150,          # 300→150: ranking stable at 150; ~40% faster training
        learning_rate=0.05,
        num_leaves=num_leaves,
        min_child_samples=min_child,
        lambda_l1=0.05,            # mild L1 regularisation reduces noise feature importance
        feature_fraction=0.7,      # random 70% features per tree — faster + regularises
        subsample=0.8,             # random 80% rows per tree (bagging)
        subsample_freq=1,          # required to enable subsample
        scale_pos_weight=spw,
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# Hygiene filter
# ---------------------------------------------------------------------------
class HygieneFilter(BaseEstimator, TransformerMixin):
    def __init__(self, min_count=10):
        self.min_count = min_count

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        # 1. Name-based drops — regex on column names (O(ncols), no row scans)
        name_drop = np.array([
            bool(ERA_RE.search(str(c)))
            for c in X.columns
        ], dtype=bool)

        # 2. Operate on numeric sub-frame only — no full-DataFrame copy.
        #    Non-numeric cols are rare after booleanisation; handled separately.
        num_mask = X.dtypes.map(pd.api.types.is_numeric_dtype).values  # (ncols,)
        Xnum     = X.iloc[:, num_mask]           # view, no copy

        # 3. Constant-column drop — one vectorised nunique() call
        n_unique = np.empty(len(X.columns), dtype=np.intp)
        n_unique[num_mask] = Xnum.nunique(dropna=True).values
        for i in np.where(~num_mask)[0]:         # non-numeric fallback (rare)
            n_unique[i] = X.iloc[:, i].nunique(dropna=True)
        const_drop = n_unique <= 1

        # 4. Binary min-count check — vectorised min / max / sum on numeric cols
        col_min_n  = Xnum.min().values                        # NaN-safe (skips NaN)
        col_max_n  = Xnum.max().values
        is_bin_n   = ((n_unique[num_mask] <= 2) &
                      (col_min_n >= 0) & (col_max_n <= 1))   # NaN comparison → False ✓

        col_sums_n = Xnum.sum().values                        # count of 1s (NaN skipped)
        n_rows     = len(X)
        minority_n = np.minimum(col_sums_n.clip(0), n_rows - col_sums_n.clip(0))
        low_cnt_n  = minority_n < self.min_count

        # Expand binary flags back to full column width (non-numeric → False)
        is_bin  = np.zeros(len(X.columns), dtype=bool)
        low_cnt = np.zeros(len(X.columns), dtype=bool)
        is_bin[num_mask]  = is_bin_n
        low_cnt[num_mask] = low_cnt_n

        drop_mask = name_drop | const_drop | (is_bin & low_cnt)

        # Diagnostic logging — surface which columns are being dropped and why
        dropped = X.columns[drop_mask].tolist()
        if dropped:
            print(f"HygieneFilter: dropping {len(dropped)} columns (min_count={self.min_count})")
            for c in dropped[:20]:
                reason = []
                if c in X.columns[name_drop]: reason.append("name_pattern")
                if c in X.columns[const_drop]: reason.append("constant")
                if c in X.columns[is_bin & low_cnt]: reason.append("binary_min_count")
                print(f"  {c}  ({', '.join(reason)})")
            if len(dropped) > 20:
                print(f"  ... and {len(dropped) - 20} more")

        self.keep_cols_ = X.columns[~drop_mask].tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X)[self.keep_cols_].copy()
        non_num = ~X.dtypes.map(pd.api.types.is_numeric_dtype)
        if non_num.any():
            for c in X.columns[non_num]:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        return X


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _collapse_synonyms(cols):
    new = pd.Index(cols)
    # Original rules
    new = new.str.replace(r"hep(at(?:ic)?|a)", "hepatic", regex=True)
    new = new.str.replace(r"\btransaminases?\b", "transaminase", regex=True)
    new = new.str.replace(r"\bmult(\.|iple)\b", "multiple", regex=True)
    # Perinatal abbreviation → full-term rules (prevents importance splitting)
    new = new.str.replace(r"\bpec\b", "preeclampsia", regex=True)
    new = new.str.replace(r"\bc[\-/]?s(?:ection)?\b", "cesarean", regex=True)
    new = new.str.replace(r"\bgestational\s*diabet\w*", "gdm", regex=True)
    new = new.str.replace(r"\bhtn\b", "hypertension", regex=True)
    new = new.str.replace(r"\bptb\b", "preterm", regex=True)
    new = new.str.replace(r"\bptl\b", "preterm_labor", regex=True)
    new = new.str.replace(r"\b(iugr|fgr)\b", "growth_restriction", regex=True)
    new = new.str.replace(r"\bpprom\b", "preterm_premature_rupture", regex=True)
    new = new.str.replace(r"\bhellp\b", "hellp_syndrome", regex=True)
    new = new.str.replace(r"\b(sga|small for gestational age)\b", "small_for_ga", regex=True)
    new = new.str.replace(r"\b(lga|large for gestational age)\b", "large_for_ga", regex=True)
    new = new.str.replace(r"\b(ama|advanced maternal age)\b", "advanced_maternal_age", regex=True)
    new = new.str.replace(r"\b(pih|pregnancy induced hypertension)\b", "pih", regex=True)
    new = new.str.replace(r"\bihcp\b", "intrahepatic_cholestasis", regex=True)
    new = new.str.replace(r"\b(ppd|postpartum depression)\b", "postpartum_depression", regex=True)
    new = new.str.replace(r"\b(uti|urinary tract infection)\b", "urinary_tract_infection", regex=True)
    new = new.str.replace(r"\b(gbs|group b strep)\b", "group_b_strep", regex=True)
    return new


def _drop_trivial_no_columns(X, max_no=0.95):
    to_drop = []
    for c in X.columns:
        if str(c).endswith("_No"):
            root = str(c)[:-3]
            if (root + "_Yes") not in X.columns:
                col = pd.to_numeric(X[c], errors="coerce").fillna(0)
                if col.mean() > max_no:
                    to_drop.append(c)
    if to_drop:
        X = X.drop(columns=to_drop, errors="ignore")
        print(f"Dropped {len(to_drop)} trivial '_No' columns.")
    return X


def _drop_negative_reference_dummies(X):
    cols   = list(X.columns)
    colset = set(cols)
    drops  = []
    for c in cols:
        cs = str(c)
        if cs.endswith(NEG_TOKENS):
            for neg in NEG_TOKENS:
                if cs.endswith(neg):
                    root = cs[:-len(neg)]; break
            if any((root + pos) in colset for pos in POS_TOKENS):
                drops.append(c)
    if drops:
        X = X.drop(columns=drops, errors="ignore")
        print(f"Dropped {len(drops)} negative reference dummies.")
    return X


# ---------------------------------------------------------------------------
# Plausibility caps for continuous features
# ---------------------------------------------------------------------------
# Data-entry errors (glucose=137M, WBC=13087, APGAR=729) destroy the signal
# for the entire continuous feature by distorting imputation medians and
# scaling.  Caps set implausible values to NaN before imputation so the
# clean median fills them.
_PLAUSIBILITY_CAPS = {
    # ── Derived scalars (DB_2) ───────────────────────────────────────────────
    'GA at admit':              (18.0,  46.0),
    'GA at 1st prenatal visit': (2.0,   42.0),
    'GA at delivery':           (18.0,  46.0),
    'PP height':                (3.5,   7.5),
    'Height':                   (3.5,   7.5),
    'Mat age':                  (12.0,  65.0),
    # ── Demographics / prenatal ──────────────────────────────────────────────
    'MatInfo__Total years living in US':      (0, 80),
    'MatInfo__Number of people in household': (1, 25),
    'Prenatal__PP weight':                    (70, 500),
    'Prenatal__PP BMI':                       (12, 70),
    'Prenatal__PP SBP':                       (70, 200),
    'Prenatal__PP DBP':                       (30, 140),
    'Prenatal__1st trimester weight':         (70, 500),
    'Prenatal__Total number prenatal visits':  (0, 60),
    'Prenatal__Gravida':                       (1, 25),
    # ── Intrapartum ─────────────────────────────────────────────────────────
    'Intrapartum__Length of stay (days)':      (0, 90),
    'Intrapartum__Number of babies':           (1, 5),
    'Intrapartum__Cervical dilation':          (0, 10),
    'Intrapartum__WBC (lowest)':               (1, 50),
    'Intrapartum__WBC (highest)':              (1, 50),
    'Intrapartum__HgB (lowest)':               (3, 20),
    'Intrapartum__HgB (highest)':              (3, 20),
    'Intrapartum__HCT (lowest)':               (10, 60),
    'Intrapartum__HCT (highest)':              (10, 60),
    'Intrapartum__PLT (lowest)':               (5, 800),
    'Intrapartum__PLT (highest)':              (5, 800),
    'Intrapartum__ALT (lowest)':               (1, 5000),
    'Intrapartum__ALT (highest)':              (1, 5000),
    'Intrapartum__AST (lowest)':               (1, 5000),
    'Intrapartum__AST (highest)':              (1, 5000),
    'Intrapartum__LDH (lowest)':               (50, 5000),
    'Intrapartum__LDH (highest)':              (50, 5000),
    'Intrapartum__Glucose (lowest)':           (10, 500),
    'Intrapartum__Glucose (highest)':          (10, 500),
    'Intrapartum__Total Bilirubin (lowest)':   (0.1, 30),
    'Intrapartum__Total Bilirubin (highest)':  (0.1, 30),
    'Intrapartum__Weight (pounds)':            (70, 500),
    'Intrapartum__BMI':                        (12, 70),
    'Intrapartum__Intrapartum HCT (lowest)':   (10, 60),
    'Intrapartum__SBP on admission':           (70, 250),
    'Intrapartum__DBP on admission':           (30, 160),
    # ── Antepartum (mirrors intrapartum ranges) ─────────────────────────────
    'Antepartum__Cervical dilation':            (0, 10),
    'Antepartum__WBC (lowest)':                 (1, 50),
    'Antepartum__WBC (highest)':                (1, 50),
    'Antepartum__HgB (lowest)':                 (3, 20),
    'Antepartum__HgB (highest)':                (3, 20),
    'Antepartum__HCT (lowest)':                 (10, 60),
    'Antepartum__HCT (highest)':                (10, 60),
    'Antepartum__PLT (lowest)':                 (5, 800),
    'Antepartum__PLT (highest)':                (5, 800),
    'Antepartum__ALT (lowest)':                 (1, 5000),
    'Antepartum__ALT (highest)':                (1, 5000),
    'Antepartum__AST (lowest)':                 (1, 5000),
    'Antepartum__AST (highest)':                (1, 5000),
    'Antepartum__LDH (lowest)':                 (50, 5000),
    'Antepartum__LDH (highest)':                (50, 5000),
    'Antepartum__Glucose (lowest)':             (10, 500),
    'Antepartum__Glucose (highest)':            (10, 500),
    'Antepartum__Total Bilirubin (lowest)':     (0.1, 30),
    'Antepartum__Total Bilirubin (highest)':    (0.1, 30),
    'Antepartum__SBP on admission':             (70, 250),
    'Antepartum__DBP on admission':             (30, 160),
    # ── Delivery / Newborn ──────────────────────────────────────────────────
    'Delivery__Length':                          (20, 65),
    'Delivery__Birth weight':                   (200, 6500),
    'Delivery__Head circumference':             (20, 45),
    'Newborn__APGAR at 1 minute':               (0, 10),
    'Newborn__APGAR at 5 minute':               (0, 10),
    'Newborn__Highest total bilirubin':          (0, 40),
    'Newborn__Head circumference at discharge':  (20, 45),
}


def _cap_outliers(X):
    """Apply plausibility caps to continuous features — data-entry errors -> NaN.

    Runs BEFORE imputation so outliers are replaced with NaN and subsequently
    filled with the clean median rather than distorting it.
    """
    n_capped = 0
    for col, (lo, hi) in _PLAUSIBILITY_CAPS.items():
        if col not in X.columns:
            continue
        s = pd.to_numeric(X[col], errors='coerce')
        mask = s.notna() & ((s < lo) | (s > hi))
        if mask.any():
            n = int(mask.sum())
            X.loc[mask, col] = np.nan
            n_capped += n
    if n_capped:
        print(f"  Plausibility caps: {n_capped} outlier values -> NaN")
    return X


# ---------------------------------------------------------------------------
# Parallel CV utilities
# ---------------------------------------------------------------------------

def _parallel_budget(n_folds: int = 5):
    """Return (n_outer, n_lgb_jobs) that use all CPUs without oversubscription.

    LightGBM uses n_lgb_jobs threads internally; we run n_outer folds in parallel
    via threading (LGB releases the GIL).  Product ≤ physical core count.
    """
    n_cpu   = os.cpu_count() or 1
    n_outer = min(n_folds, max(1, n_cpu // 2))
    n_lgb   = max(1, n_cpu // n_outer)
    return n_outer, n_lgb


def _run_ranking_fold(tr, te, X, y, n_pos, n_neg, n_lgb_jobs, has_shap):
    """Fit one ranking fold (cv_shap path) — called via joblib.Parallel threads.

    X can contain NaN — LightGBM handles it natively.
    """
    m = _build_lgbm(n_neg, n_pos, n_jobs=n_lgb_jobs)
    m.fit(X[tr], y[tr])
    gain = m.booster_.feature_importance(importance_type="gain").astype(np.float64)
    sh   = np.zeros(X.shape[1], dtype=np.float64)
    if has_shap:
        import shap as _shap
        sv = _unpack_shap(_shap.TreeExplainer(m).shap_values(X[te]))
        sh = np.abs(sv).mean(axis=0).astype(np.float64)
    return gain, sh


def _run_cv_fold(tr, te, X_vals, y, n_pos, n_neg, n_lgb_jobs, base_fpr):
    """Fit one performance-CV fold — called via joblib.Parallel threads."""
    from sklearn.preprocessing import MinMaxScaler as _MMS
    from sklearn.metrics import roc_auc_score as _auc, roc_curve as _rc
    sc  = _MMS()
    Xtr = sc.fit_transform(X_vals[tr])
    Xte = sc.transform(X_vals[te])
    clf = _build_lgbm(n_neg, n_pos, n_jobs=n_lgb_jobs)
    clf.fit(Xtr, y[tr])
    probs = clf.predict_proba(Xte)[:, 1]
    f_fpr, f_tpr, _ = _rc(y[te], probs)
    tpr_interp = np.interp(base_fpr, f_fpr, f_tpr)
    return y[te], probs, float(_auc(y[te], probs)), tpr_interp


# ---------------------------------------------------------------------------
# Public: CV recomputation (for real-time Top-N slider in dashboard)
# ---------------------------------------------------------------------------
def run_cv_from_cache(Xn, y, rank_df, top_n, n_pos, n_neg):
    """Re-run 5-fold CV on a different top_n slice of an already-ranked feature set.

    Call get_preprocessed_data() after run_analysis() to retrieve Xn and y,
    then call this whenever the user changes the Top-N slider.

    Returns a cv_results dict with the same keys as run_analysis().
    """
    if Xn is None or y is None:
        return None

    top_feats = [f for f in rank_df.head(top_n)["feature"].tolist() if f in Xn.columns]
    if not top_feats:
        return None

    if not _HAS_LGB:
        # LogReg fallback — sequential (fast enough)
        cv_clf  = LogisticRegression(max_iter=5000, solver="lbfgs")
        cv_pipe = Pipeline([("scaler", MinMaxScaler()), ("clf", cv_clf)])
        base_fpr = np.linspace(0, 1, 101)
        all_y_true, all_y_score, fold_aucs, fold_tprs = [], [], [], []
        for tr, te in StratifiedKFold(5, shuffle=True,
                                      random_state=RANDOM_STATE).split(Xn[top_feats], y):
            cv_pipe.fit(Xn[top_feats].iloc[tr], y[tr])
            probs = cv_pipe.predict_proba(Xn[top_feats].iloc[te])[:, 1]
            fold_aucs.append(roc_auc_score(y[te], probs))
            all_y_true.extend(y[te].tolist())
            all_y_score.extend(probs.tolist())
            f_fpr, f_tpr, _ = roc_curve(y[te], probs)
            fold_tprs.append(np.interp(base_fpr, f_fpr, f_tpr))
        all_y_true  = np.array(all_y_true)
        all_y_score = np.array(all_y_score)
        fold_tprs   = np.array(fold_tprs)
    else:
        # LightGBM — run folds in parallel threads (LGB releases the GIL)
        from joblib import Parallel, delayed as _delayed
        base_fpr        = np.linspace(0, 1, 101)
        X_vals          = Xn[top_feats].values
        n_outer, n_lgb_jobs = _parallel_budget(5)
        results = Parallel(n_jobs=n_outer, prefer="threads")(
            _delayed(_run_cv_fold)(tr, te, X_vals, y, n_pos, n_neg, n_lgb_jobs, base_fpr)
            for tr, te in StratifiedKFold(5, shuffle=True,
                                          random_state=RANDOM_STATE).split(X_vals, y)
        )
        all_y_true  = np.concatenate([r[0] for r in results])
        all_y_score = np.concatenate([r[1] for r in results])
        fold_aucs   = [r[2] for r in results]
        fold_tprs   = np.array([r[3] for r in results])

    fpr, tpr, _ = roc_curve(all_y_true, all_y_score)
    s90sp, sp90s = _sens_spec_at_threshold(all_y_true, all_y_score)

    return {
        "roc_auc_mean":      float(np.mean(fold_aucs)),
        "roc_auc_std":       float(np.std(fold_aucs)),
        "pr_auc_mean":       float(average_precision_score(all_y_true, all_y_score)),
        "accuracy_mean":     float(accuracy_score(all_y_true, (all_y_score > 0.5).astype(int))),
        "sens_at_90spec":    s90sp,
        "spec_at_90sens":    sp90s,
        "top_n":             len(top_feats),
        "n_positive":        n_pos,
        "n_total":           n_pos + n_neg,
        "roc_fpr":           fpr.tolist(),
        "roc_tpr":           tpr.tolist(),
        "roc_fpr_band":      base_fpr.tolist(),
        "roc_tpr_band_mean": fold_tprs.mean(axis=0).tolist(),
        "roc_tpr_band_std":  fold_tprs.std(axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Public: Single 80/20 split evaluation (fast alternative to 5-fold CV)
# ---------------------------------------------------------------------------
def run_single_split_from_cache(Xn, y, rank_df, top_n, n_pos, n_neg):
    """Run a single 80/20 stratified split for fast performance evaluation.

    Faster alternative to run_cv_from_cache() — one model instead of five.
    Returns the same dict shape so the dashboard can use either interchangeably.
    """
    if Xn is None or y is None:
        return None

    top_feats = [f for f in rank_df.head(top_n)["feature"].tolist() if f in Xn.columns]
    if not top_feats:
        return None

    from sklearn.model_selection import StratifiedShuffleSplit

    X_vals = Xn[top_feats].values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr, te = next(sss.split(X_vals, y))

    sc = MinMaxScaler()
    Xtr = sc.fit_transform(X_vals[tr])
    Xte = sc.transform(X_vals[te])

    if _HAS_LGB:
        clf = _build_lgbm(n_neg, n_pos)
    else:
        clf = LogisticRegression(max_iter=5000, solver="lbfgs")
    clf.fit(Xtr, y[tr])
    probs = clf.predict_proba(Xte)[:, 1]

    fpr, tpr, _ = roc_curve(y[te], probs)
    s90sp, sp90s = _sens_spec_at_threshold(y[te], probs)

    return {
        "roc_auc_mean":      float(roc_auc_score(y[te], probs)),
        "roc_auc_std":       0.0,
        "pr_auc_mean":       float(average_precision_score(y[te], probs)),
        "accuracy_mean":     float(accuracy_score(y[te], (probs > 0.5).astype(int))),
        "sens_at_90spec":    s90sp,
        "spec_at_90sens":    sp90s,
        "top_n":             len(top_feats),
        "n_positive":        n_pos,
        "n_total":           n_pos + n_neg,
        "roc_fpr":           fpr.tolist(),
        "roc_tpr":           tpr.tolist(),
        "roc_fpr_band":      [],
        "roc_tpr_band_mean": [],
        "roc_tpr_band_std":  [],
        "_single_split":     True,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis(
    cohort_ids,
    pickle_file=None,
    search_terms=None,
    label_name="TARGET",
    min_binary_count=MIN_BINARY_COUNT,
    directional_margin=DIRECTIONAL_MARGIN,
    top_n=TOP_N,
    save_csv=False,
    out_prefix=None,
    cv_shap=False,
    compute_extras=False,
    run_cv=True,
    exclude_cols=None,
    control_ids=None,
):
    """Run the generic feature-ranking analysis.

    Composite score (when lightgbm + shap available):
        2 × rank_norm(SHAP_mean_abs)   — primary: model-consistent, multivariate
      + 1 × rank_norm(MI)              — secondary: non-linear univariate
      [LGBM_gain: reported in table but excluded from composite — it double-
       counts the LightGBM model already represented by SHAP]
      [F-test dropped: linear/redundant for binary sparse data]
      [Per-feature AUC: optional — only when compute_extras=True]

    Args:
        cv_shap       : bool — if True, aggregate SHAP across 5 CV folds (stable
                        but ~5× slower). Default False.
        compute_extras: bool — if True, also compute per-feature F-statistic,
                        FDR q-value, and AUC (adds ~1–3 min). Not used in
                        composite. Default False.
        exclude_cols  : set[str] — column names to drop before ranking.
                        Used to remove columns whose name matched the search term
                        (definitional leakage — the cohort was defined by that column).
        run_cv        : bool — if True, run 5-fold CV at end of analysis (adds
                        3–10 min). If False, return a stub cv_results with NaN
                        metrics; use the dashboard 'Recompute CV' button later.
                        Default True.
        control_ids   : set[str] | None — if provided, the analysis population is
                        restricted to cohort_ids ∪ control_ids (dual-cohort mode).
                        Cohort A = cohort_ids (y=1), Cohort B = control_ids (y=0).
                        When None, cohort is compared against all other subjects in
                        the pickle (original behaviour).

    Preprocessed matrices (Xn, y) are cached in module globals after each call.
    Retrieve with get_preprocessed_data() to enable real-time CV recomputation.

    Returns:
        rank_df    : pd.DataFrame  — features ranked by composite score
        cv_results : dict          — CV summary metrics + ROC curve points
    """
    global _cached_Xn, _cached_y, _cached_n_pos, _cached_n_neg, _cached_Xf_raw
    global _mem_prep_key, _mem_prep_data, _mem_rank_key, _mem_rank_df
    global _RUN_T0
    _RUN_T0 = time.time()

    if pickle_file is None: pickle_file = PICKLE_IN
    if search_terms is None: search_terms = []
    if out_prefix is None: out_prefix = label_name
    if exclude_cols is None: exclude_cols = set()

    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Missing pickle: {pickle_file}")

    # ── Two-tier preprocessing cache ────────────────────────────────────────
    # Tier 1: RAM (module-level vars) — survives Streamlit reruns, zero disk I/O.
    # Tier 2: Disk (gzip pickle) — survives Streamlit restarts.
    # A cache hit skips ~20–60 s of hygiene filter + screen + imputation + corr filter.
    _ckey      = _prep_cache_key(cohort_ids, search_terms, min_binary_count,
                                 directional_margin, pickle_file, control_ids)
    _cache_hit = False
    _cached    = None

    # Tier 1: RAM
    if _ckey == _mem_prep_key and _mem_prep_data is not None:
        _cached    = _mem_prep_data
        _cache_hit = True
        print("  Prep cache: RAM hit")
    else:
        # Tier 2: disk
        _cached = _load_prep_cache(_ckey)
        if _cached is not None:
            _mem_prep_key  = _ckey
            _mem_prep_data = _cached
            _cache_hit     = True

    if _cache_hit:
        Xn          = _cached["Xn"]
        y           = _cached["y"]
        n_pos       = _cached["n_pos"]
        n_neg       = _cached["n_neg"]
        n_total     = _cached.get("n_total", len(_cached["y"]))
        means_pos   = _cached["means_pos"]
        means_neg   = _cached["means_neg"]
        delta       = _cached["delta"]
        Xf_raw      = _cached.get("Xf_raw")       # None for old caches
        pct_imputed = _cached.get("pct_imputed")   # None for old caches
        # Old caches lack Xf_raw (NaN-preserved data needed for feature detail plots).
        # Force a full recompute so the cache is rebuilt with Xf_raw included.
        if Xf_raw is None:
            _cache_hit = False
            _cached    = None

    if not _cache_hit:
        # ── Full preprocessing ───────────────────────────────────────────────
        with open(pickle_file, "rb") as f:
            df = pickle.load(f)
        _tlog(f"Loaded data: {df.shape}")

        if "Pregnancy ID" not in df.columns:
            df = df.reset_index().rename(columns={"index": "Pregnancy ID"})
        if control_ids is not None:
            # Dual-cohort mode: restrict population to cohort_ids ∪ control_ids
            _all_ids = set(str(x) for x in cohort_ids) | set(str(x) for x in control_ids)
            df = df[df["Pregnancy ID"].astype(str).isin(_all_ids)].copy()
            _tlog(f"Dual-cohort: restricted to {len(df)} subjects "
                  f"({len(cohort_ids)} A + {len(control_ids)} B)")
        df[label_name] = df["Pregnancy ID"].astype(str).isin(cohort_ids)
        n_pos   = int(df[label_name].sum())
        n_total = len(df)
        n_neg   = n_total - n_pos
        _tlog(f"{label_name}: {n_pos} positives ({df[label_name].mean():.2%})  |  "
              f"imbalance {n_neg/max(n_pos,1):.1f}:1")

        drop_admin = [c for c in df.columns if re.search(
            r"pregnancy id|study id|record id|subject id|chart|visit|mrn", str(c), re.I)]
        if drop_admin:
            df = df.drop(columns=drop_admin)

        y = df[label_name].astype(np.uint8).to_numpy()
        X = df.drop(columns=[label_name])
        X.columns = _collapse_synonyms(pd.Index(X.columns))

        junk_cols = [c for c in X.columns
                     if TEXT_JUNK_RE.search(str(c)) or JUNK_SHORT_MED_RE.search(str(c))]
        if junk_cols:
            X = X.drop(columns=junk_cols, errors="ignore")

        X = _drop_trivial_no_columns(X)
        X = _drop_negative_reference_dummies(X)

        # Plausibility caps: data-entry errors (glucose=137M, APGAR=729, WBC=13k)
        # destroy continuous-feature signal by distorting imputation medians and
        # MinMaxScaler ranges.  Capping before hygiene so outlier-NaN'd columns
        # aren't mistakenly dropped as constant.
        X = _cap_outliers(X)

        filt = HygieneFilter(min_count=min_binary_count)
        Xf   = filt.fit_transform(X, y)
        _tlog(f"After hygiene filter: {Xf.shape[1]} features")

        missing_cols = [c for c in Xf.columns if str(c).endswith("_missing")]
        if missing_cols:
            Xf = Xf.drop(columns=missing_cols)

        # Bidirectional directional screen — absolute delta OR relative enrichment.
        # Relative criterion (≥2× enrichment/depletion) preserves rare features with
        # low absolute prevalence but strong relative association (e.g. 0.1% → 0.3%).
        y_bool    = (y == 1)
        means_pos = Xf[y_bool].mean(numeric_only=True)
        means_neg = Xf[~y_bool].mean(numeric_only=True)
        delta     = means_pos - means_neg
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = pd.Series(
                np.where(means_neg > 1e-6,
                         means_pos / means_neg,
                         np.where(means_pos > 0, np.inf, 1.0)),
                index=delta.index
            )
        keep_dir = (delta.abs() > directional_margin) | (rel > 2.0) | (rel < 0.5)
        Xf        = Xf.loc[:, keep_dir.values]
        delta     = delta[keep_dir]
        n_higher  = int((delta > 0).sum())
        n_lower   = int((delta < 0).sum())
        _tlog(f"After bidirectional screen: {Xf.shape[1]} features "
              f"({n_higher} higher, {n_lower} lower in cohort)")

        # Densify any sparse columns once — prevents SimpleImputer from
        # densifying each column independently (causes the double-warning and
        # allocates the full matrix twice).  Result (Xn) is stored in the prep
        # cache so this conversion never runs again after the first pass.
        _sparse_cols = [c for c in Xf.columns if hasattr(Xf[c], "sparse")]
        if _sparse_cols:
            Xf = Xf.copy()
            for _sc in _sparse_cols:
                Xf[_sc] = Xf[_sc].sparse.to_dense()

        # Impute
        _tlog(f"Imputing missing values ({Xf.shape[1]} features)…")
        imp = SimpleImputer(strategy="median")
        Xn  = pd.DataFrame(imp.fit_transform(Xf), columns=Xf.columns)

        # Correlation filtering — remove one of each pair with |r| > 0.95.
        # Highly correlated features (OHE siblings, medication synonyms) split model
        # importance between them, burying individually important features.
        # Run after imputation so correlation is computed on clean numeric data.
        #
        # Adaptive strategy:
        #   ≤ _CHUNK_THRESH features → single BLAS matrix multiply (fast, bounded memory)
        #   > _CHUNK_THRESH features → block-by-block: never materialises the full n×n
        #                              matrix, prints progress every ~10% of block-pairs
        #
        # Row subsample reduced 5 000 → 2 000:
        #   |r| > 0.95 is deterministic at n ≈ 500; 2 000 is very conservative
        #   and gives a 2.5× speed-up on the matrix multiply + halves peak RAM.
        _MAX_CORR_ROWS = 2_000
        _CHUNK_THRESH  = 4_000

        n_feats = Xn.shape[1]
        est_mb  = n_feats * n_feats * 4 / 1e6
        _tlog(f"Correlation filter: {n_feats} features  "
              f"(full matrix would be ~{est_mb:.0f} MB)")

        X_arr = Xn.values.astype(np.float32)
        if X_arr.shape[0] > _MAX_CORR_ROWS:
            _rng  = np.random.RandomState(RANDOM_STATE)
            X_arr = X_arr[_rng.choice(X_arr.shape[0], _MAX_CORR_ROWS, replace=False)]
            print(f"  Subsampled to {_MAX_CORR_ROWS} rows "
                  f"(|r|>0.95 stable at this n)", flush=True)

        # Mean-centre and L2-normalise each feature column once
        X_arr -= X_arr.mean(axis=0)
        norms  = np.linalg.norm(X_arr, axis=0); norms[norms == 0] = 1.0
        X_arr /= norms
        X_norm = X_arr.T.astype(np.float32)   # (n_feats, n_rows_sub) — features as rows
        n_sub  = X_arr.shape[0]
        del X_arr                              # free (n_rows_sub, n_feats) copy

        cols_arr = np.array(Xn.columns)

        # Pre-compute absolute point-biserial correlation with target (y) for
        # each feature — used to break ties in the correlation filter.  When two
        # features are |r| > 0.95, keep the one with more target signal.
        _y_f = y.astype(np.float64)
        _y_c = _y_f - _y_f.mean()
        _y_n = np.sqrt((_y_c ** 2).sum())
        _Xv  = Xn.values.astype(np.float64)
        _Xc  = _Xv - _Xv.mean(axis=0)
        _Xn2 = np.sqrt((_Xc ** 2).sum(axis=0))
        _Xn2[_Xn2 == 0] = 1.0
        _target_corr = np.abs((_Xc.T @ _y_c) / (_Xn2 * _y_n))
        _target_corr = np.nan_to_num(_target_corr, nan=0.0)
        del _Xv, _Xc, _Xn2, _y_c

        if n_feats <= _CHUNK_THRESH:
            # ── Fast path: single BLAS multiply ───────────────────────────────
            print("  Computing correlation matrix (BLAS dot product)…", flush=True)
            corr_mat  = (X_norm @ X_norm.T) / n_sub        # (n_feats, n_feats)
            print(f"  Matrix done — scanning {n_feats*(n_feats-1)//2:,} pairs…",
                  flush=True)
            iu        = np.triu_indices(n_feats, k=1)
            high_corr = np.abs(corr_mat[iu]) > 0.95
            del corr_mat
            # Signal-aware dropping: for each correlated pair, drop the one with
            # lower correlation to the target — keeps the more informative member.
            drop_set = set()
            for idx in np.where(high_corr)[0]:
                i, j = int(iu[0][idx]), int(iu[1][idx])
                if i in drop_set or j in drop_set:
                    continue
                if _target_corr[i] >= _target_corr[j]:
                    drop_set.add(j)
                else:
                    drop_set.add(i)
            to_drop_corr = list(set(cols_arr[list(drop_set)]))
        else:
            # ── Memory-safe path: block-by-block ──────────────────────────────
            # Peak RAM ≈ 2×BLOCK×n_sub×4 + BLOCK²×4 bytes  (never the full n²)
            BLOCK      = 2_000
            n_blocks   = (n_feats + BLOCK - 1) // BLOCK
            n_pairs    = n_blocks * (n_blocks + 1) // 2
            peak_mb    = (2 * BLOCK * n_sub + BLOCK * BLOCK) * 4 / 1e6
            print(f"  Chunked path ({n_blocks} blocks ×{BLOCK}): "
                  f"peak RAM ~{peak_mb:.0f} MB vs {est_mb:.0f} MB full", flush=True)
            drop_idx_set: set = set()
            pair_idx          = 0
            _log_every        = max(1, n_pairs // 10)
            for i_start in range(0, n_feats, BLOCK):
                i_end  = min(i_start + BLOCK, n_feats)
                blk_i  = X_norm[i_start:i_end]              # (Bi, n_sub)
                for j_start in range(i_start, n_feats, BLOCK):
                    j_end  = min(j_start + BLOCK, n_feats)
                    blk_j  = X_norm[j_start:j_end]          # (Bj, n_sub)
                    cblock = (blk_i @ blk_j.T) / n_sub      # (Bi, Bj)
                    rr, cc = np.where(np.abs(cblock) > 0.95)
                    for r, c in zip(rr, cc):
                        gi = i_start + r; gj = j_start + c
                        if gj > gi and gi not in drop_idx_set and gj not in drop_idx_set:
                            if _target_corr[gi] >= _target_corr[gj]:
                                drop_idx_set.add(gj)
                            else:
                                drop_idx_set.add(gi)
                    del cblock
                    pair_idx += 1
                    if pair_idx % _log_every == 0:
                        print(f"  … {pair_idx/n_pairs*100:.0f}%  "
                              f"({pair_idx}/{n_pairs} block-pairs)", flush=True)
            to_drop_corr = list(set(cols_arr[list(drop_idx_set)]))

        del X_norm

        if to_drop_corr:
            Xn    = Xn.drop(columns=to_drop_corr)
            delta = delta.reindex(Xn.columns)
            _tlog(f"  Removed {len(to_drop_corr)} correlated features — "
                  f"{Xn.shape[1]} remain")
        else:
            _tlog("  No highly correlated features found.")

        # NaN-preserving view for LightGBM (trees handle NaN natively)
        Xf_raw      = Xf[Xn.columns]       # same columns, NaN preserved
        pct_imputed = Xf_raw.isna().mean()  # per-column imputation rate

        # ── Save preprocessing to disk + RAM ────────────────────────────────
        _prep_payload = {
            "Xn":          Xn,
            "y":           y,
            "n_pos":       n_pos,
            "n_neg":       n_neg,
            "n_total":     n_total,
            "means_pos":   means_pos,
            "means_neg":   means_neg,
            "delta":       delta,
            "Xf_raw":      Xf_raw,
            "pct_imputed": pct_imputed,
        }
        _save_prep_cache(_ckey, _prep_payload)
        _mem_prep_key  = _ckey
        _mem_prep_data = _prep_payload

    # Drop columns whose name matched the search term — the cohort was defined by
    # having a value in that column, so its rank would be circular/definitional.
    if exclude_cols:
        _excl = [c for c in Xn.columns if c in exclude_cols]
        if _excl:
            Xn = Xn.drop(columns=_excl)
            if Xf_raw is not None:
                Xf_raw = Xf_raw.drop(columns=[c for c in _excl if c in Xf_raw.columns],
                                      errors="ignore")
            print(f"  Excluded {len(_excl)} definitional column(s) from ranking: {_excl}")

    # Populate module-level cache for real-time CV recomputation
    _cached_Xn     = Xn
    _cached_y      = y
    _cached_n_pos  = n_pos
    _cached_n_neg  = n_neg
    _cached_Xf_raw = Xf_raw

    # ── Two-tier rank cache ──────────────────────────────────────────────────
    # Tier 1: RAM — survives Streamlit reruns in the same process, zero disk I/O.
    # Tier 2: Disk — survives Streamlit restarts (~5 s load).
    # A rank cache hit skips MI + LGB training + SHAP (60–120 s per run).
    _rkey     = _rank_cache_key(_ckey, cv_shap, compute_extras, exclude_cols)
    _rank_hit = False
    # Tier 1: RAM
    if _rkey == _mem_rank_key and _mem_rank_df is not None:
        rank_df   = _mem_rank_df
        _rank_hit = True
        print("  Rank cache: RAM hit — skipping MI/LGB/SHAP")
    else:
        # Tier 2: disk
        _disk_rank = _load_rank_cache(_rkey)
        if _disk_rank is not None:
            rank_df        = _disk_rank
            _mem_rank_key  = _rkey
            _mem_rank_df   = rank_df
            _rank_hit      = True

    if not _rank_hit:
        # ── MI ──────────────────────────────────────────────────────────────
        # Detect binary 0/1 features: sklearn uses the exact formula for discrete
        # features (O(n) per feature) instead of k-NN estimation (O(n log n)).
        # Binary-heavy datasets (like this one) complete MI in 1–3 s vs 10–30 s.
        _is_discrete = np.array([
            set(Xn.iloc[:, j].dropna().unique()).issubset({0, 1, 0.0, 1.0})
            for j in range(Xn.shape[1])
        ])
        _tlog(f"Computing MI ({_is_discrete.sum()} binary exact, "
              f"{(~_is_discrete).sum()} continuous k-NN)…")
        mi = mutual_info_classif(Xn, y, discrete_features=_is_discrete,
                                  random_state=RANDOM_STATE)

        if compute_extras:
            print("Computing F-tests and per-feature AUC…")
            sel      = SelectKBest(score_func=f_classif, k="all").fit(Xn, y)
            f_scores = sel.scores_
            f_pvals  = sel.pvalues_
            f_qvals  = multipletests(f_pvals, method="fdr_bh")[1]
            aucs     = np.array([_safe_auc(y, Xn[c]) for c in Xn.columns])
        else:
            print("  (F-test and per-feature AUC skipped — enable 'Compute extra metrics' to include)")
            f_scores = np.full(Xn.shape[1], np.nan)
            f_pvals  = np.full(Xn.shape[1], np.nan)
            f_qvals  = np.full(Xn.shape[1], np.nan)
            aucs     = np.full(Xn.shape[1], np.nan)

        # ── LightGBM + SHAP ─────────────────────────────────────────────────
        lgbm_gain     = np.zeros(Xn.shape[1])
        shap_mean_abs = np.zeros(Xn.shape[1])
        l1_w          = np.full(Xn.shape[1], np.nan)

        # LightGBM uses Xf_raw (NaN preserved — trees handle it natively).
        # LogReg fallback still needs MinMaxScaler (L1 is scale-sensitive).
        _use_raw = _HAS_LGB and Xf_raw is not None

        if _HAS_LGB:
            if _use_raw:
                X_lgb = Xf_raw.values
            else:
                # Old cache without Xf_raw — fall back to scaled Xn
                scaler = MinMaxScaler()
                X_lgb  = scaler.fit_transform(Xn)

            if cv_shap:
                # CV-aggregated SHAP: parallel 5-fold, LGB releases GIL → threading safe.
                # More stable rankings than single model — useful for novel discovery.
                from joblib import Parallel, delayed as _delayed
                n_outer, n_lgb_jobs = _parallel_budget(5)
                _tlog(f"CV-aggregated SHAP (scale_pos_weight={n_neg/max(n_pos,1):.1f}, "
                      f"{n_outer} parallel folds)…")
                results = Parallel(n_jobs=n_outer, prefer="threads")(
                    _delayed(_run_ranking_fold)(tr, te, X_lgb, y, n_pos, n_neg,
                                               n_lgb_jobs, _HAS_SHAP)
                    for tr, te in StratifiedKFold(n_splits=5, shuffle=True,
                                                  random_state=RANDOM_STATE).split(X_lgb, y)
                )
                lgbm_gain     = sum(r[0] for r in results) / 5
                shap_mean_abs = (sum(r[1] for r in results) / 5
                                 if _HAS_SHAP else np.zeros(Xn.shape[1]))
            else:
                # Fast single-model path (default)
                _tlog(f"Training LightGBM (scale_pos_weight={n_neg/max(n_pos,1):.1f})…")
                lgbm_model = _build_lgbm(n_neg, n_pos)
                lgbm_model.fit(X_lgb, y)
                lgbm_gain  = lgbm_model.booster_.feature_importance(importance_type="gain")
                if _HAS_SHAP:
                    _tlog("Computing SHAP values…")
                    # 2 000-row subsample for ranking estimation (stable at this n,
                    # ~2.5× faster than the previous 5 000-row subsample).
                    _SHAP_MAX_ROWS = 2_000
                    if X_lgb.shape[0] > _SHAP_MAX_ROWS:
                        _shap_rng = np.random.RandomState(RANDOM_STATE)
                        _shap_idx = _shap_rng.choice(X_lgb.shape[0], _SHAP_MAX_ROWS,
                                                     replace=False)
                        X_lgb_shap = X_lgb[_shap_idx]
                        print(f"  (SHAP ranking estimated on {_SHAP_MAX_ROWS}-row "
                              f"subsample; model trained on full {X_lgb.shape[0]} rows)")
                    else:
                        X_lgb_shap = X_lgb
                    sv = _unpack_shap(shap.TreeExplainer(lgbm_model).shap_values(X_lgb_shap))
                    shap_mean_abs = np.abs(sv).mean(axis=0)
                    _tlog("SHAP done.")
        else:
            _tlog("Computing L1 LogReg weights (lightgbm unavailable)…")
            scaler = MinMaxScaler()
            Xs     = scaler.fit_transform(Xn)
            l1   = LogisticRegression(penalty="l1", solver="saga", max_iter=5000,
                                      random_state=RANDOM_STATE, n_jobs=-1)
            l1.fit(Xs, y)
            l1_w = np.abs(l1.coef_.ravel())

        # ── Composite score ──────────────────────────────────────────────────
        # Primary signal: SHAP (2×) + MI
        # LGBM_gain excluded: double-counts the LightGBM model already in SHAP
        # F-test excluded: linear/redundant for binary sparse data
        # Per-feature AUC: reported in table (useful for Leak Audit) but not scored
        if _HAS_LGB and _HAS_SHAP:
            comp = (
                2.0 * _rank_norm(shap_mean_abs) +
                1.0 * _rank_norm(mi)
            )
        elif _HAS_LGB:
            comp = (
                2.0 * _rank_norm(lgbm_gain) +
                1.0 * _rank_norm(mi)        +
                1.0 * _rank_norm(aucs)
            )
        else:
            # Fallback: MI primary (non-linear) + AUC (interpretable)
            comp = _rank_norm(mi) + _rank_norm(aucs)

        categories = [_feature_category(c)  for c in Xn.columns]
        directions = ["Higher in cohort" if d > 0 else "Lower in cohort"
                      for d in delta.reindex(Xn.columns).values]

        _pct_imp_vals = (pct_imputed.reindex(Xn.columns).values
                        if pct_imputed is not None
                        else np.full(Xn.shape[1], np.nan))

        rank_df = pd.DataFrame({
            "feature":              Xn.columns,
            "category":             categories,
            "direction":            directions,
            "Composite":            comp,
            "SHAP_mean_abs":        shap_mean_abs,
            "LGBM_gain":            lgbm_gain,
            "MI":                   mi,
            "AUC":                  aucs,
            "F":                    f_scores,
            "p":                    f_pvals,
            "q":                    f_qvals,
            "L1_abs_weight":        l1_w,
            "pct_imputed":          _pct_imp_vals,
            f"Mean_{label_name}":   means_pos.reindex(Xn.columns).values,
            f"Mean_No{label_name}": means_neg.reindex(Xn.columns).values,
            "Delta":                delta.reindex(Xn.columns).values,
        }).sort_values("Composite", ascending=False)

        # Save rank result to both cache tiers
        _save_rank_cache(_rkey, rank_df)
        _mem_rank_key = _rkey
        _mem_rank_df  = rank_df

    if save_csv:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv = f"{out_prefix}_feature_rank_{ts}.csv"
        rank_df.to_csv(csv, index=False)
        print(f"Saved: {csv}")

    # 5-fold CV (optional — skip for fast exploratory runs; use dashboard Recompute button later)
    if run_cv:
        cv_results = run_cv_from_cache(Xn, y, rank_df, top_n, n_pos, n_neg)
        if cv_results is None:
            cv_results = {
                "roc_auc_mean": 0.5, "roc_auc_std": 0.0,
                "pr_auc_mean": 0.0,  "accuracy_mean": 0.0,
                "sens_at_90spec": 0.0, "spec_at_90sens": 0.0,
                "top_n": top_n, "n_positive": n_pos, "n_total": n_total,
                "roc_fpr": [0, 1], "roc_tpr": [0, 1],
                "roc_fpr_band": [], "roc_tpr_band_mean": [], "roc_tpr_band_std": [],
            }
        else:
            cv_results["n_total"] = n_total
        model_label = "LightGBM+SHAP" if (_HAS_LGB and _HAS_SHAP) else \
                      "LightGBM"       if _HAS_LGB else "LogReg"
        _tlog(
            f"5-fold CV [{model_label}, top {top_n}]: "
            f"ROC-AUC {cv_results['roc_auc_mean']:.3f} ± {cv_results['roc_auc_std']:.3f}  |  "
            f"PR-AUC {cv_results['pr_auc_mean']:.3f}  |  "
            f"Sens@90%Spec {cv_results['sens_at_90spec']:.3f}  |  "
            f"Spec@90%Sens {cv_results['spec_at_90sens']:.3f}"
        )
    else:
        print("\nCV skipped (run_cv=False) — use dashboard 'Recompute CV' to compute performance metrics.")
        cv_results = {
            "roc_auc_mean": float("nan"), "roc_auc_std": float("nan"),
            "pr_auc_mean":  float("nan"), "accuracy_mean": float("nan"),
            "sens_at_90spec": float("nan"), "spec_at_90sens": float("nan"),
            "top_n": top_n, "n_positive": n_pos, "n_total": n_total,
            "roc_fpr": [], "roc_tpr": [],
            "roc_fpr_band": [], "roc_tpr_band_mean": [], "roc_tpr_band_std": [],
            "_cv_skipped": True,
        }

    return rank_df, cv_results


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
def main():
    from ML_1_Subject_search import search_cohort

    terms_input  = input("Enter search terms (comma-separated): ").strip()
    search_terms = [t.strip() for t in terms_input.split(",") if t.strip()]
    if not search_terms:
        print("No search terms provided. Exiting."); return

    db_file     = input("Database file (Enter = 'PBDBfinal.txt'): ").strip() or "PBDBfinal.txt"
    label       = input("Cohort label (Enter = 'TARGET'): ").strip() or "TARGET"
    pickle_file = input(f"ML pickle (Enter = '{PICKLE_IN}'): ").strip() or PICKLE_IN

    print(f"\nSearching {db_file} for: {search_terms}")
    cohort_ids, _, _, _, _ = search_cohort(search_terms, db_file)
    print(f"Found {len(cohort_ids)} matching subjects.\n")

    run_analysis(cohort_ids=cohort_ids, pickle_file=pickle_file,
                 search_terms=search_terms, label_name=label, save_csv=True)


if __name__ == "__main__":
    main()
