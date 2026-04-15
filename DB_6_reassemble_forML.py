#This script combines the three processed PBDB data frames then
#processes them into a single database that is saved as a binary 
#and suitable for ML mining. The script has an option to include
#scalers that have mising values to a certain percent or remove scalers, 
#keeping just Boolean and categorical data.
#
# Requires: Python 3.7+
# Last updated: 7 November 2025, Maxim Seferovic
#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pickle, json, datetime, os, re, warnings
warnings.filterwarnings("ignore")  # keep output clean

EXPECTED_ROWS = 47291

TO_FLOAT = [
    # --- Computed scalers (no prefix — created by DB_2) ---
    'GA at admit', 'GA at 1st prenatal visit', 'GA at delivery',
    'PP height', 'Height', 'Mat age',

    # --- Maternal demographics (MatInfo / Prenatal) ---
    'MatInfo__Total years living in US',
    'MatInfo__Number of people in household',
    'Prenatal__PP weight', 'Prenatal__PP BMI',
    'Prenatal__PP SBP', 'Prenatal__PP DBP',
    'Prenatal__1st trimester weight',
    'Prenatal__Total number prenatal visits',
    'Prenatal__Gravida',

    # --- Intrapartum (DB_7) scalers (previously _y-suffixed) ---
    'Intrapartum__Length of stay (days)',
    'Intrapartum__Number of babies',
    'Intrapartum__Cervical dilation',
    'Intrapartum__WBC (lowest)',  'Intrapartum__WBC (highest)',
    'Intrapartum__HgB (lowest)',  'Intrapartum__HgB (highest)',
    'Intrapartum__HCT (lowest)',  'Intrapartum__HCT (highest)',
    'Intrapartum__PLT (lowest)',  'Intrapartum__PLT (highest)',
    'Intrapartum__Proteinuria (lowest)', 'Intrapartum__Proteinuria (highest)',
    'Intrapartum__ALT (lowest)',  'Intrapartum__ALT (highest)',
    'Intrapartum__AST (lowest)',  'Intrapartum__AST (highest)',
    'Intrapartum__LDH (lowest)',  'Intrapartum__LDH (highest)',
    'Intrapartum__Glucose (lowest)', 'Intrapartum__Glucose (highest)',
    'Intrapartum__Total Bilirubin (lowest)', 'Intrapartum__Total Bilirubin (highest)',
    'Intrapartum__Weight (pounds)', 'Intrapartum__BMI',
    'Intrapartum__Intrapartum HCT (lowest)',
    'Intrapartum__SBP on admission', 'Intrapartum__DBP on admission',

    # --- Antepartum (DB_6) scalers ---
    'Antepartum__Cervical dilation',
    'Antepartum__WBC (lowest)',  'Antepartum__WBC (highest)',
    'Antepartum__HgB (lowest)',  'Antepartum__HgB (highest)',
    'Antepartum__HCT (lowest)',  'Antepartum__HCT (highest)',
    'Antepartum__PLT (lowest)',  'Antepartum__PLT (highest)',
    'Antepartum__ALT (lowest)',  'Antepartum__ALT (highest)',
    'Antepartum__AST (lowest)',  'Antepartum__AST (highest)',
    'Antepartum__LDH (lowest)',  'Antepartum__LDH (highest)',
    'Antepartum__Glucose (lowest)', 'Antepartum__Glucose (highest)',
    'Antepartum__Total Bilirubin (lowest)', 'Antepartum__Total Bilirubin (highest)',
    'Antepartum__SBP on admission', 'Antepartum__DBP on admission',
    'Antepartum__Proteinuria (lowest)', 'Antepartum__Proteinuria (highest)',

    # --- Delivery / Newborn scalers ---
    'Delivery__Length', 'Delivery__Birth weight', 'Delivery__Head circumference',
    'Newborn__APGAR at 1 minute', 'Newborn__APGAR at 5 minute',
    'Newborn__Highest total bilirubin', 'Newborn__Head circumference at discharge',

    # --- Auto-detected numeric columns (Bug fix: were falling through to fillna(0)) ---
    'MatID__How old when moved to US',
    'PriorPreg__Delivery classification at GA (wks)',
    'Prenatal__Delay to 1st prenatal visit',
    'Prenatal__Maternal smoking cig/day during 1 st trimester',
    'Prenatal__Maternal smoking cig/day during 2 nd /3 rd trimester',
    'Prenatal__Maternal smoking cig/day currently',
    'Prenatal__Maternal alcohol drink/week during 1 st trimester',
    'Prenatal__Maternal alcohol drink/week during 2 nd /3 rd trimester',
    'Prenatal__Maternal alcohol drink/week currently',
    'Prenatal__Maternal marijuana joints/week during 1 st trimester',
    'Prenatal__Maternal marijuana joints/week during 2 nd /3 rd trimester',
    'Prenatal__Maternal marijuana joints/week currently',
    'FamHxMatGrandma__Recurrent pregnancy loss number',
    'FamHxMatGrandma__Preterm birth at (wks)',
    'Intrapartum__Mag sulfate dose before delivery',
    'Newborn__Cord arterial PH',
    'Newborn__Arterial base deficit',
    'Newborn__Venous base deficit',
    'Prenatal__T', 'Prenatal__P', 'Prenatal__A', 'Prenatal__L',
]

def _mem_mb(df):
    try:
        return round(df.memory_usage(deep=True).sum()/1024**2, 2)
    except Exception:
        return None

def _is_binary_col(s):
    dt = s.dtype
    return str(dt) in ("uint8", "bool") or (
        hasattr(dt, "subtype") and str(getattr(dt, "subtype", "")) in ("uint8", "bool"))

# For normalization of multi-value cells (take first token)
_SPLIT_RE = re.compile(r'[;,|]+')
def _normalize_cell(x):
    if isinstance(x, str):
        s = x.strip()
        if (',' in s) or (';' in s) or ('|' in s):
            for part in _SPLIT_RE.split(s):
                p = part.strip()
                if p:
                    return p
            return ''
        return s
    return x

def _normalize_series(col: pd.Series) -> pd.Series:
    """Vectorized column-level equivalent of applymap(_normalize_cell).

    For columns with no delimiters (the common case) this is a single
    str.strip() call.  For the rare cells that contain ; , | it finds
    the first non-empty token using a Series-level apply — still far
    faster than a full DataFrame applymap because it only touches the
    small subset of delimited cells.
    """
    if col.dtype != object:
        return col
    s = col.str.strip()
    mask = s.str.contains(r'[;,|]', na=False, regex=True)
    if mask.any():
        s = s.copy()
        s[mask] = (s[mask]
                   .str.split(r'[;,|]')
                   .apply(lambda parts: next((p.strip() for p in parts if p.strip()), '')))
    return s

def one_hot_encode(df, to_float_cols):
    print("One-hot encoding categorical columns")

    present = [c for c in to_float_cols if c in df.columns]
    float_columns = df[present].copy() if present else pd.DataFrame(index=df.index)
    categorical_columns = df.drop(columns=present, axis=1)

    # Fast cardinality check — drop columns with too many unique values to OHE usefully
    nunique = categorical_columns.nunique(dropna=False)
    high_card = list(nunique[nunique > 100].index)
    if high_card:
        print(f"Dropping {len(high_card)} high-cardinality cols (>100 levels)")
        categorical_columns.drop(columns=high_card, inplace=True, errors='ignore')

    # Low-variance filter removed: cohort-blind global threshold silently drops
    # condition-specific features before ML_2's cohort-aware min_binary_count can
    # evaluate them. ML_2 handles zero-variance and near-zero features at analysis time.

    # Auto-detect numeric-looking categorical columns:
    # If >80% of non-empty values parse as float, treat as scalar rather than OHE.
    # This catches columns like "Newborn__Arterial base deficit" that are stored as
    # strings in the CSVs but are not in the hardcoded TO_FLOAT list — without this,
    # they get OHE'd into useless dummies like "_24.0", "_25.0", etc.
    _NUMERIC_BLOCKLIST = ('vaccine year',)  # calendar years are confounders, not clinical values
    _numeric_detect = []
    for _col in list(categorical_columns.columns):
        _vals = categorical_columns[_col].replace('', pd.NA).dropna()
        if len(_vals) < 10:
            continue
        if pd.to_numeric(_vals, errors='coerce').notna().mean() > 0.80:
            if any(bl in _col.lower() for bl in _NUMERIC_BLOCKLIST):
                print(f"  Blocked from auto-numeric: '{_col}' (confounder)")
                continue
            _numeric_detect.append(_col)
    if _numeric_detect:
        print(f"Auto-detected {len(_numeric_detect)} numeric-looking categorical cols "
              f"→ scalar treatment")
        print(f"  Examples: {_numeric_detect[:5]}")
        _num_extra = pd.DataFrame(index=categorical_columns.index)
        for _col in _numeric_detect:
            _num_extra[_col] = pd.to_numeric(categorical_columns[_col], errors='coerce')
            if _num_extra[_col].isna().mean() <= 0.95:
                _num_extra[_col] = _num_extra[_col].fillna(_num_extra[_col].median())
            else:
                _num_extra.drop(columns=_col, inplace=True)
        float_columns = pd.concat([float_columns, _num_extra], axis=1)
        categorical_columns.drop(columns=_numeric_detect, inplace=True)

    # --- Split binary (Yes/No) vs true categorical columns ---
    # Binary columns: _normalize_series (first token wins) then get_dummies
    # Categorical columns: str.get_dummies to preserve multi-value cells from
    #   DB_1 duplicate aggregation (e.g. "Hispanic/Latino,Non-Hispanic")
    _BINARY_VALS = {"yes", "no", "true", "false", ""}
    def _is_yesno_col(col):
        uniq = set(col.dropna().astype(str).str.strip().str.lower().unique())
        return uniq <= _BINARY_VALS

    binary_cols = [c for c in categorical_columns.columns
                   if _is_yesno_col(categorical_columns[c])]
    cat_cols    = [c for c in categorical_columns.columns if c not in binary_cols]
    print(f"Categorical split: {len(binary_cols)} binary Yes/No cols, "
          f"{len(cat_cols)} true categorical cols")

    # Binary Yes/No columns: normalize (first token wins), then OHE
    if binary_cols:
        bin_df  = categorical_columns[binary_cols].apply(_normalize_series)
        bin_ohe = pd.get_dummies(bin_df, sparse=True, dtype='uint8')
    else:
        bin_ohe = pd.DataFrame(index=categorical_columns.index)

    # Categorical columns: str.get_dummies to handle multi-value cells
    cat_ohe_parts = []
    for c in cat_cols:
        s = categorical_columns[c].fillna('').astype(str).str.strip().str.lower()
        s = s.str.replace(r'\s*[;\t|]\s*', ',', regex=True)  # normalize delimiters (incl tabs from PeriBank multi-select)
        dummies = s.str.get_dummies(sep=',')
        dummies = dummies.loc[:, dummies.columns != '']      # drop blank-value column
        dummies.columns = [f"{c}_{v}" for v in dummies.columns]
        cat_ohe_parts.append(dummies.astype('uint8'))
    cat_ohe = pd.concat(cat_ohe_parts, axis=1) if cat_ohe_parts else pd.DataFrame(index=categorical_columns.index)

    one_hot = pd.concat([bin_ohe, cat_ohe], axis=1)
    print(f"OHE result width: {one_hot.shape[1]} columns "
          f"(binary={bin_ohe.shape[1]}, categorical={cat_ohe.shape[1]})")

    # Drop blank-category OHE columns — a blank cell means the user left the field
    # empty, which is administrative (not informative) in a registry database.
    # Previously these were renamed to "_missing"; now they are dropped entirely.
    missing_cols = [c for c in one_hot.columns if re.search(r'_$', c)]
    if missing_cols:
        print(f"Dropping {len(missing_cols)} blank-category OHE columns "
              f"(blank = not filled, not informative)")
        one_hot.drop(columns=missing_cols, inplace=True)

    # Drop _missing indicator columns — in a registry database, a column encoding
    # whether a field was unfilled is administrative metadata, not clinical signal.
    # These can also correlate with era/site patterns (some form versions lack
    # certain fields) and dilute model attention away from real features.
    missing_indicator = [c for c in one_hot.columns if c.endswith('_missing')]
    if missing_indicator:
        print(f"Dropping {len(missing_indicator)} '_missing' indicator OHE columns")
        one_hot.drop(columns=missing_indicator, inplace=True)

    # Collapse _Yes/_No pairs without dense casts: keep single binary "<root>"
    yes_cols = [c for c in one_hot.columns if c.endswith('_Yes')]
    for y in yes_cols:
        root = y[:-4]
        n = root + '_No'
        if n in one_hot.columns:
            one_hot[root] = one_hot[y]
            one_hot.drop([y, n], axis=1, inplace=True, errors='ignore')

    # Collapse _True/_False pairs — same redundancy as _Yes/_No
    true_cols = [c for c in one_hot.columns if c.endswith('_True')]
    for t in true_cols:
        root = t[:-5]
        f = root + '_False'
        if f in one_hot.columns:
            one_hot[root] = one_hot[t]
            one_hot.drop([t, f], axis=1, inplace=True, errors='ignore')

    encoded_df = pd.concat([float_columns, one_hot], axis=1)
    print(f"Encoded df shape: {encoded_df.shape} | mem≈{_mem_mb(encoded_df)} MB")
    return encoded_df

def normalize_colnames(df):
    # Keep original names; only trim outer whitespace so we don't collapse distinct columns
    new_cols = df.columns.astype(str).str.strip()
    # Detect (but do not change) duplicates
    if new_cols.duplicated().any():
        dups = new_cols[new_cols.duplicated()].unique()
        print(f"⚠️ Detected {len(dups)} duplicate column names (leaving as-is). Example: {list(dups[:10])}")
    df.columns = new_cols
    return df

# Plausibility caps for continuous features — applied before median imputation
# so that data-entry errors are NaN'd and replaced by clean medians.
_SCALAR_CAPS = {
    'GA at admit':              (18.0,  46.0),
    'GA at 1st prenatal visit': (2.0,   42.0),
    'GA at delivery':           (18.0,  46.0),
    'PP height':                (3.5,   7.5),
    'Height':                   (3.5,   7.5),
    'Mat age':                  (12.0,  65.0),
    'Prenatal__PP weight':                    (70, 500),
    'Prenatal__PP BMI':                       (12, 70),
    'Prenatal__PP SBP':                       (70, 200),
    'Prenatal__PP DBP':                       (30, 140),
    'Prenatal__1st trimester weight':         (70, 500),
    'Prenatal__Total number prenatal visits':  (0, 60),
    'Prenatal__Gravida':                       (1, 25),
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
    'Delivery__Length':                          (20, 65),
    'Delivery__Birth weight':                   (200, 6500),
    'Delivery__Head circumference':             (20, 45),
    'Newborn__APGAR at 1 minute':               (0, 10),
    'Newborn__APGAR at 5 minute':               (0, 10),
    'Newborn__Highest total bilirubin':          (0, 40),
    'Newborn__Head circumference at discharge':  (20, 45),
    # --- Bug 2: existing TO_FLOAT columns that were missing caps ---
    'MatInfo__Total years living in US':           (0, 100),
    'MatInfo__Number of people in household':      (1, 30),
    'Prenatal__1st trimester weight':              (70, 500),
    'Intrapartum__Proteinuria (lowest)':           (0, 10000),
    'Intrapartum__Proteinuria (highest)':          (0, 10000),
    # --- Improvement 1: antepartum proteinuria ---
    'Antepartum__Proteinuria (lowest)':            (0, 10000),
    'Antepartum__Proteinuria (highest)':           (0, 10000),
    # --- Bug 1: auto-detected numeric columns ---
    'MatID__How old when moved to US':                          (0, 80),
    'PriorPreg__Delivery classification at GA (wks)':           (18, 46),
    'Prenatal__Delay to 1st prenatal visit':                    (0, 42),
    'Prenatal__Maternal smoking cig/day during 1 st trimester': (0, 60),
    'Prenatal__Maternal smoking cig/day during 2 nd /3 rd trimester': (0, 60),
    'Prenatal__Maternal smoking cig/day currently':              (0, 60),
    'Prenatal__Maternal alcohol drink/week during 1 st trimester': (0, 50),
    'Prenatal__Maternal alcohol drink/week during 2 nd /3 rd trimester': (0, 50),
    'Prenatal__Maternal alcohol drink/week currently':           (0, 50),
    'Prenatal__Maternal marijuana joints/week during 1 st trimester': (0, 50),
    'Prenatal__Maternal marijuana joints/week during 2 nd /3 rd trimester': (0, 50),
    'Prenatal__Maternal marijuana joints/week currently':        (0, 50),
    'FamHxMatGrandma__Recurrent pregnancy loss number':          (0, 20),
    'FamHxMatGrandma__Preterm birth at (wks)':                   (18, 42),
    'Intrapartum__Mag sulfate dose before delivery':             (0, 100),
    'Newborn__Cord arterial PH':                                 (6.5, 7.6),
    'Newborn__Arterial base deficit':                            (-30, 30),
    'Newborn__Venous base deficit':                              (-30, 30),
    'Prenatal__T':  (0, 20),
    'Prenatal__P':  (0, 15),
    'Prenatal__A':  (0, 20),
    'Prenatal__L':  (0, 20),
}


def impute_scalers(df, scalers):
    """Impute scalar columns with median; drop columns >95% missing.

    Applies plausibility caps before imputation so that data-entry errors
    (e.g. glucose=137M, APGAR=729) are NaN'd and replaced by clean medians
    rather than distorting them.

    No _missing flags are created: in PeriBank, missing data reflects unfilled
    form fields (administrative), not a clinical state.
    """
    existing = [c for c in scalers if c in df.columns]
    print(f"Imputing {len(existing)} scalar columns")
    dropped = []
    n_capped = 0
    for c in existing:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        # Plausibility caps — data-entry errors -> NaN before median computation
        if c in _SCALAR_CAPS:
            lo, hi = _SCALAR_CAPS[c]
            mask = df[c].notna() & ((df[c] < lo) | (df[c] > hi))
            if mask.any():
                n = int(mask.sum())
                n_capped += n
                df.loc[mask, c] = np.nan
        miss_rate = float(df[c].isna().mean())
        if miss_rate > 0.95:
            df.drop(columns=c, inplace=True)
            dropped.append(c)
        else:
            df[c] = df[c].fillna(df[c].median())
    if n_capped:
        print(f"  Plausibility caps: {n_capped} outlier values -> NaN (then imputed with median)")
    if dropped:
        print(f"Dropped {len(dropped)} scalar columns (>95% missing): "
              f"{', '.join(dropped[:5])}{'...' if len(dropped) > 5 else ''}")
    return df

# ---- robust summary that works with sparse/uint8 ----
def quick_summary(df, n=10):
    from pandas.api.types import is_float_dtype
    # Continuous (float-like) columns
    cont_cols = [c for c in df.columns if is_float_dtype(df[c].dtype)]
    if cont_cols:
        print("\nContinuous scalers (sample):")
        try:
            print(df[cont_cols].describe().transpose().head(n))
        except Exception:
            desc = df[cont_cols].agg(['count','mean','min','max']).transpose()
            print(desc.head(n))
    else:
        print("\nNo continuous scaler columns detected.")

    # Binary-like columns (bool/uint8/sparse uint8)
    def _is_binary(s):
        dt = s.dtype
        if str(dt) in ("bool", "uint8"):
            return True
        # detect pandas sparse dtype without importing SparseDtype explicitly
        return hasattr(dt, "subtype") and str(getattr(dt, "subtype", "")) in ("uint8", "bool")

    bin_cols = [c for c in df.columns if _is_binary(df[c])]
    if bin_cols:
        prev = df[bin_cols].mean().sort_values(ascending=False)
        print("\nTop binary feature prevalence:")
        print(prev.head(n).to_frame("prevalence"))
        print("\nBottom binary feature prevalence:")
        print(prev.tail(n).to_frame("prevalence"))
    else:
        print("\nNo binary/sparse columns detected.")

# ---- robust boolean normalization for df2/df3 ----
TRUTHY  = {"1", "TRUE", "T", "YES", "Y"}
FALSY   = {"0", "FALSE", "F", "NO", "N", ""}
# Explicitly coded ambiguous values — log their count but still encode as 0.
# In PeriMiner, missing data reflects unfilled fields (non-informative), so
# "unknown"/"refused"/etc. are treated the same as blank rather than as NaN.
UNKNOWN = {"UNKNOWN", "N/A", "NA", "PENDING", "NOT RECORDED",
           "REFUSED", "?", "UNKNOWN/NOT REPORTED"}

def to_bool_frame(df):
    # Vectorized: column-wise str.strip().str.upper(), then isin() for TRUTHY check.
    # Replaces 3 × applymap(lambda) — ~20–50× faster on large frames.
    s = df.astype(str).apply(lambda col: col.str.strip().str.upper())
    n_unknown = int(s.isin(UNKNOWN).sum().sum())
    if n_unknown:
        print(f"  Note: {n_unknown:,} cells contained ambiguous values "
              f"(unknown/N/A/refused) — encoded as 0")
    return s.isin(TRUTHY).astype("uint8")

def main():
    print("Reading CSVs...")
    df1 = pd.read_csv('PBDBfinal_cleaned.csv', index_col='Pregnancy ID', sep='|', dtype=str)
    df2 = pd.read_csv('PBDBfinal_meds_dictcorrect_bool.csv', index_col='Pregnancy ID', sep='|', dtype=str)
    df3 = pd.read_csv('PBDBfinal_details_tok.csv', index_col='Pregnancy ID', sep='|', dtype=str)

    for name, d in (('df1', df1), ('df2', df2), ('df3', df3)):
        print(f"{name}: rows={len(d):,}, cols={d.shape[1]:,}, mem≈{_mem_mb(d)} MB")

    # All features (including post-birth outcomes like APGAR, birth weight, etc.)
    # are kept in the pickle.  PeriMiner is an association tool as much as a
    # prediction tool — leak filtering is handled at analysis time by the
    # dashboard and ML_2's leak_re.

    df1.fillna('', inplace=True)
    df2.fillna('', inplace=True)
    df3.fillna('', inplace=True)

    # OHE only for df1
    df1 = one_hot_encode(df1, TO_FLOAT)

    # Robust boolean conversion for df2/df3
    df2 = to_bool_frame(df2)
    df3 = to_bool_frame(df3)

    print("\nConcatenating…")
    before_cols = (df1.shape[1], df2.shape[1], df3.shape[1])
    df = pd.concat([df1, df2, df3], axis=1)
    print(f"Concatenated shape (pre-normalize): {df.shape} | mem≈{_mem_mb(df)} MB")

    # Normalize names once (no collapsing)
    df = normalize_colnames(df)
    print(f"After normalize_colnames: {df.shape} | mem≈{_mem_mb(df)} MB")

    if df.columns.duplicated().any():
        dup_names = df.columns[df.columns.duplicated(keep=False)].unique()
        print(f"Merging {len(dup_names)} duplicate column name(s) "
              f"(binary→max, float→mean):")
        deduped = []
        seen = set()
        for col in df.columns:
            if col in seen:
                continue
            seen.add(col)
            group = df.loc[:, df.columns == col]
            if group.shape[1] == 1:
                deduped.append(group.iloc[:, 0])
            else:
                print(f"  {col!r}  ({group.shape[1]} copies)")
                if _is_binary_col(group.iloc[:, 0]):
                    deduped.append(group.max(axis=1).astype("uint8").rename(col))
                else:
                    deduped.append(group.mean(axis=1).rename(col))
        df = pd.concat(deduped, axis=1)
        print(f"After dedup: {df.shape} | mem≈{_mem_mb(df)} MB")

    # Impute scalers
    df = impute_scalers(df, TO_FLOAT)

    # Drop exact zero-variance binary columns (0% or 100% prevalence).
    # Caused by parent checkboxes stored as blank-when-unchecked (not False),
    # leaving a _True column with all 1s after OHE. Never informative for ML.
    # Distinct from near-zero filter (intentionally removed): this only targets exact 0/1.
    zero_var = [c for c in df.columns if _is_binary_col(df[c]) and (df[c].mean() < 1e-9 or df[c].mean() > 1 - 1e-9)]
    if zero_var:
        print(f"Dropping {len(zero_var)} zero-variance binary columns (0% or 100%):")
        for c in zero_var:
            print(f"  {df[c].mean():.0%}  {c}")
        df.drop(columns=zero_var, inplace=True)

    # Handle lingering NaNs — previously ALL columns with any NaN were dropped,
    # which silently killed important features with minor parse residuals.
    # Now: heavy NaN (>50%) → drop; light NaN → fill with 0 (safe for binary/OHE).
    df.replace('', pd.NA, inplace=True)
    cols_with_nan = df.columns[df.isna().any()]
    if len(cols_with_nan) > 0:
        heavy_nan = [c for c in cols_with_nan if df[c].isna().mean() > 0.50]
        light_nan = [c for c in cols_with_nan if c not in heavy_nan]
        if heavy_nan:
            print(f"Dropping {len(heavy_nan)} columns with >50% residual NaN")
            df.drop(columns=heavy_nan, inplace=True)
        if light_nan:
            print(f"Imputing {len(light_nan)} columns with minor residual NaN (fillna 0)")
            df[light_nan] = df[light_nan].fillna(0)

    print(f"Final df shape: {df.shape} | mem≈{_mem_mb(df)} MB")
    print(f"Index unique: {df.index.is_unique} | duplicated: {df.index.duplicated().sum()}")

    out_name = "PBDBfinal_ready_forML_IHCP_paper3.pkl"
    df.to_pickle(out_name)
    meta = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "rows": len(df),
        "columns": df.shape[1],
        "memory_MB": _mem_mb(df),
        "inputs": {
            "cleaned": "PBDBfinal_cleaned.csv",
            "meds": "PBDBfinal_meds_dictcorrect_bool.csv",
            "details": "PBDBfinal_details_tok.csv"
        },
        "scalers_retained": len([c for c in TO_FLOAT if c in df.columns]),
        "feature_sources": {
            "df1_cols": before_cols[0],
            "df2_cols": before_cols[1],
            "df3_cols": before_cols[2]
        }
    }
    meta_path = os.path.splitext(out_name)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved pickle and metadata: {out_name}, {meta_path}")

    # Quiet, robust summary
    quick_summary(df, n=10)

if __name__ == '__main__':
    main()

