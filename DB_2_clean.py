# Cleans reassembled peribank database. Curation, data reduction & feature selection, set to Boolean.
# Updated March 2026 for PREFIX__column naming scheme from DB_1_recreate.py rewrite.
# Last originally written 31 March 2023, Maxim Seferovic, seferovi@bcm.edu
#!/usr/bin/env python3

import re, pickle
import pandas as pd
import numpy as np
from datetime import timedelta

#Set filter thresholds for data processing
##########################
preg_stringency = 0.3       #Maximum fraction of missing data per pregnancy before removal (keeps rows with ≥30% filled)
feat_stringency = 0.0002    #Minimum fraction of filled data per feature before removal (~12 rows at 58k, aligned with ML_2 min_binary_count=10)
# feat_variance removed: DB_4 applies a stricter 99.5% uniformity filter during OHE,
# and ML_2 catches zero-variance features. A weaker pass here is redundant.
##########################

# Column name substrings that indicate the column should remain numeric (float)
# rather than being booleanised into one-hot dummies.
_NUMERIC_FORCE_PATTERNS = (
    # ── general numeric indicators ─────────────────────────────────────────
    "how old",        # age at immigration / age at event
    " age",           # any age column (leading space avoids "dosage", "package")
    "years living",
    "total years",
    "number of",
    "how many",
    "bmi",
    "weight",
    "height",
    "gravida",
    # ── gestational timing ─────────────────────────────────────────────────
    "week",           # gestational timing at treatment ("week tocolytic agents are given")
    # NOTE: "gestational" removed — it matches Yes/No checkbox columns like
    #   "Gestational diabetes" and "Preeclampsia/gestational hypertension",
    #   destroying ~14 clinical columns by forcing Yes/No → NaN.
    #   Numeric GA columns are already caught by "wks" and "week".
    "wks",            # "(wks)" notation: "betamethasone at GA (wks)", "preterm birth at (wks)"
    "preterm birth at",  # belt-and-suspenders alongside "wks"
    "ptb at ga",         # prior preterm birth GA (weeks) — abbreviated form
    # ── lab / vital sign measurements ──────────────────────────────────────
    " ph",            # pH scale: venous pH, arterial pH
                      #   leading space avoids "phosphate", "phosphorus", "graph", etc.
    "base deficit",   # venous/arterial base deficit (mEq/L)
    "(lowest)",       # lab minimum values: "WBC (lowest)", "PLT (lowest)", …
                      #   parenthetical format avoids matching non-lab columns
    "(highest)",      # lab maximum values: "WBC (highest)", "PLT (highest)", …
                      #   avoids matching "Newborn__Highest level of resuscitation" (categorical)
    "inr",            # INR coagulation metric
    "albumin",        # serum albumin (g/dL)
    "proteinur",      # proteinuria — truncated to cover spelling variants
    "temper",         # chorioamnionitis max temperature (°F/°C)
    "cord",           # cord arterial/venous pH, base deficit
    "apgar",          # APGAR scores (already in TO_FLOAT but belt-and-suspenders)
    "pt (",           # PT (lowest), PT (highest)
    "fibrinogen",     # fibrinogen level
    "creatinine",     # serum creatinine
    "tsh",            # thyroid-stimulating hormone
    # ── duration / length-of-stay ─────────────────────────────────────────
    "length of stay",  # hospital length-of-stay (days)
    "length of time",  # e.g. "length of time it took to become pregnant"
)


def convert_date(string):
    """Convert PeriBank date string to ISO 'YYYY-MM-DD', handling three formats:
      1. 'MM/DD/YYYY'                  — original PeriBank export
      2. 'MM/DD/YYYY HH:MM:SS a. m.'  — Excel locale re-save artifact
      3. 'YYYY-MM-DD'                  — ISO format (already correct)
    Multiple dates (twins around midnight) → keep last entry.
    Returns 'YYYY-MM-DD' string or np.nan.
    """
    if not isinstance(string, str):
        return np.nan
    # Multiple dates (twins ~midnight) — use last
    if ',' in string:
        string = string.split(',')[-1].strip()
    string = string.strip()
    parts = string.split('/')
    if len(parts) == 3:
        # Strip any trailing time Excel appended (e.g. "1979 12:00:00 a. m.")
        year_part = parts[2].strip().split()[0]   # take only the numeric year
        return f"{year_part}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
    # ISO format 'YYYY-MM-DD' (no slashes) — take first 10 chars, ignore trailing time
    if re.match(r'^\d{4}-\d{2}-\d{2}', string):
        return string[:10]
    return np.nan   # unparseable — will become NaT in pd.to_datetime


def keep_a2z(string):
    pattern = re.compile('[^a-zA-Z ,]')
    return ' '.join(pattern.sub(' ', string.lower()).split())


def strp_lst(x):
    if x is None: return []
    else: return [i.strip() for i in x]


def strip_terms(lst, things_to_strip):
    stripped_list = []
    for medication in lst:
        for ending in things_to_strip:
            if medication.endswith(ending):
                medication = medication[: -len(ending)].strip()
        stripped_list.append(medication)
    return stripped_list


_NULL_MEDS = {'', 'none', 'no', 'nil', 'n a', 'na', 'nka', 'nkda', 'o', 'unknown', 'deny', 'denies'}

def separate_meds(series):
    series.fillna('', inplace=True)
    series = series.str.replace('\t', ',', regex=False)
    #***Pull out what to keep — only a-z, commas & whitespace***
    series = series.apply(keep_a2z)
    series = series.str.split(',').apply(lambda x: strp_lst(x))
    series = series.apply(lambda meds: [m for m in meds if m.strip() not in _NULL_MEDS])
    things_to_strip = [' ', '/t', ' crm', ' cream', ' vaginal', ' vag', ' top', ' sq', ' topical',
        ' po', ' odt', ' tab', ' pm', ' extra',  ' inhaler', ' jelly', ' xl', ' plus',
        ' mg', ' soln', ' ml', ' flush', ' tab', ' mcg', ' pf', ' id', ' in', ' premix', ' pot',
        ' continous', ' infusion', ' cap',' bolus', ' gel', ' patch', ' sodium', ' injec',
        ' inh', ' neb', ' or', ' ophthalmic', ' ointment', ' hbr', ' tabs', ' lr', ' multi',
        ' ivpb', ' drip', ' ac', ' acet', ' in', ' g', ' id', ' sol', ' sal', ' push', ' vac',
        ' buffered', ' nacl', ' ivpb', ' cq', ' gram', ' epidural infursion', ' infus', ' suppository',
        ' oral', ' dm', ' hfa', ' x', ' iv', ' tablet', ' hci', ' topical', ' hcl', ' w',
        ' lotion', ' pack', ' nasal',' drops',' strength',' pf',' inj',' inhaler', ' inf',
        ' bid', 'regular', ' nasal', ' spray', ' micronized', ' injections', ]

    while True: #strip iteratively in case multiple/stacked instances.
        prev_series = series.copy()
        series = series.apply(strip_terms, args=(things_to_strip,))
        if series.equals(prev_series): break
    series = series.apply(lambda meds: ','.join(meds))
    while True: #get rid of excessive whitespace
        prev_series = series.copy()
        series = series.str.replace('  ', ' ')
        if series.equals(prev_series): break
    return series


def separate_infections(series, name):
    series.fillna('', inplace=True)
    # PeriBank exports multi-select checkboxes as TAB-delimited; split on both tab and comma.
    infec_lists = series.str.split(r'[\t,]+').apply(lambda x: [item.strip() for item in x])
    all_infec = set().union(*infec_lists)
    # Match against split tokens (set membership), not substring of the original cell.
    # str.contains caused false positives: "UTI" matched cells containing "ESBL-UTI", etc.
    return pd.DataFrame({
        name + '_' + infection: infec_lists.apply(lambda lst: infection in lst)
        for infection in all_infec
    })


def separate_csvdata(series, name):
    # Skip columns that are already numeric — they should stay as continuous values,
    # not be exploded into one binary column per unique number.
    numeric_ratio = pd.to_numeric(series.dropna(), errors='coerce').notna().mean()
    if numeric_ratio > 0.7:
        return pd.DataFrame()   # caller checks for empty and skips concat
    series.fillna('', inplace=True)
    # PeriBank multi-select checkboxes use tabs as delimiters; normalize to commas
    series = series.str.replace('\t', ',', regex=False)
    series = series.apply(keep_a2z)
    items_lists = series.str.split(',').apply(lambda x: [item.strip().lower() for item in x])
    all_items = set().union(*items_lists)
    # Match against split tokens (set membership), not substring of the original cell.
    # str.contains caused false positives when one item was a substring of another.
    return pd.DataFrame({
        name + '_' + item: items_lists.apply(lambda lst: item in lst)
        for item in all_items
    })


#***Main code***
##################################################################
df = pd.read_csv("PBDBfinal.txt", sep="|", index_col='Pregnancy ID', dtype=str)
start_size = len(df.columns)
#remove any with missing pregnancy ID
df = df.drop(index=df.index[df.index.isna()])
#First pass reduction of columns/features and pregnancies by missing or repetitive data
threshold = int(len(df.index)*feat_stringency) #require at least this fraction of feature data filled for column to survive
df.dropna(axis='columns', thresh=threshold, inplace=True)
#Filter pregnancies with a lot of missing data (rows by % empty)
# Keep pregnancies where at least preg_stringency fraction of columns are filled.
# rows_to_remove = those with MORE THAN (1 - preg_stringency) fraction missing.
fraction_missing = df.isna().sum(axis=1)/df.shape[1]
rows_to_remove = fraction_missing[fraction_missing > (1 - preg_stringency)].index
df.drop(rows_to_remove, inplace=True)
print(f"Dropped {start_size - len(df.columns)} features for too much missing data, {len(df.columns)} remain")
print(f"Dropped {len(rows_to_remove)} pregnancies for too much missing data, {len(df.index)} remain")


# Fix tab corruption in income brackets (PeriBank exports $-commas as tabs)
_income_col = 'MatInfo__Household income'
if _income_col in df.columns:
    df[_income_col] = df[_income_col].str.replace('\t', '', regex=False)
    print(f"Fixed tab corruption in '{_income_col}'")

#Consolidate all instances of wk/day, inches/feet, make it scalar data
print("Consolidating and filtering data...")
print("Replace wk/d with scalars")
df.fillna('', inplace=True)
#deal with few instances where GA/length/other scalar data has commas (multiple entries = ambiguous)
checkforcommas = [
    'Delivery__Birth weight', 'Delivery__Length',
    'Delivery__GA at delivery (wks)', 'Delivery__GA at delivery (d)',
    'Delivery__Head circumference',
    'Newborn__APGAR at 1 minute', 'Newborn__APGAR at 5 minute',
    'Intrapartum__GA at admit (wks)', 'Intrapartum__GA at admit (d)',
    'Prenatal__GA at 1st prenatal visit (wks)', 'Prenatal__GA at 1st prenatal visit (d)',
    'Prenatal__PP height (ft)', 'Prenatal__PP height (in)',
    'Intrapartum__Height (ft)', 'Intrapartum__Height (in)']
for col in checkforcommas:
    if col in df.columns:
        df.loc[df[col].str.contains(',', na=False), col] = np.nan

df.replace('', np.nan, inplace=True)

# GA at admit: use Intrapartum (DB_7) as the delivery admission GA
# Batch all derived scalar columns into one concat to avoid DataFrame fragmentation
_derived = {
    'GA at admit': (
        pd.to_numeric(df['Intrapartum__GA at admit (wks)'], errors='coerce') +
        pd.to_numeric(df['Intrapartum__GA at admit (d)'], errors='coerce') / 7
    ),
    'GA at 1st prenatal visit': (
        pd.to_numeric(df['Prenatal__GA at 1st prenatal visit (wks)'], errors='coerce') +
        pd.to_numeric(df['Prenatal__GA at 1st prenatal visit (d)'], errors='coerce') / 7
    ),
    'GA at delivery': (
        pd.to_numeric(df['Delivery__GA at delivery (wks)'], errors='coerce') +
        pd.to_numeric(df['Delivery__GA at delivery (d)'], errors='coerce') / 7
    ),
    'PP height': (
        pd.to_numeric(df['Prenatal__PP height (ft)'], errors='coerce') +
        pd.to_numeric(df['Prenatal__PP height (in)'], errors='coerce') / 12
    ),
    'Height': (
        pd.to_numeric(df['Intrapartum__Height (ft)'], errors='coerce') +
        pd.to_numeric(df['Intrapartum__Height (in)'], errors='coerce') / 12
    ),
}
df = pd.concat([df, pd.DataFrame(_derived, index=df.index)], axis=1)

df.drop(columns=[c for c in [
    'Prenatal__PP height (ft)', 'Prenatal__PP height (in)',
    'Prenatal__GA at 1st prenatal visit (wks)', 'Prenatal__GA at 1st prenatal visit (d)',
    'Delivery__GA at delivery (wks)', 'Delivery__GA at delivery (d)',
    'Intrapartum__GA at admit (wks)', 'Intrapartum__GA at admit (d)',
    'Antepartum__GA at admit (wks)', 'Antepartum__GA at admit (d)',
    'Intrapartum__Height (ft)', 'Intrapartum__Height (in)',
] if c in df.columns], inplace=True)

# Drop columns that are unhelpful, logistically redundant, or handled elsewhere
colums_to_drop = (
    'ContraceptiveHx__Prepregnancy contraception use',
    'ContraceptiveHx__Planned',
    'ContraceptiveHx__Ovulation prediction methods',
    'ContraceptiveHx__Fertility calendar',
    'Prenatal__Prenatal care location',
    'FamHxMat__Substance frequency of use',
    'Intrapartum__Indication for induction other',
    'Antepartum__Significant findings - Other (details)',
    'Postpartum__Postpartum course other conditions details',
)
for col in colums_to_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Drop by substring patterns (still work with PREFIX__col_name format)
patterns_togo = (
    'ip code',          # zip codes — privacy + low ML value
    'Ultrasound',       # timepoint counts — low ML value
    'Time',             # time of day columns
    r'(?<!of) time',  # time-of-day columns, but not "length of time"
    'Dose',             # raw dose sub-form columns (processed separately)
    'Frz No', 'Transport', 'barcode', 'Status', 'Shelf ', 'Rack ', 'Box ', 'Pos ',
    'Collected/storage', 'Hemolyzed', 'Volume', 'Intermediate temp',
    'Specimen ', 'RBC contam', 'Comments', 'processed difference'
)
for pattern in patterns_togo:
    df.drop(df.filter(regex=pattern).columns, axis=1, inplace=True)


#Get maternal age from dates and remove all date data
print("Calculate maternal age from dates - scrub all dates")
df['MatID__Date of birth'] = df['MatID__Date of birth'].apply(convert_date)
df['Delivery__Date of delivery'] = df['Delivery__Date of delivery'].apply(convert_date)
df['MatID__Date of birth'] = pd.to_datetime(df['MatID__Date of birth'], errors='coerce')
df['Delivery__Date of delivery'] = pd.to_datetime(df['Delivery__Date of delivery'], errors='coerce')
_mat_age = (df['Delivery__Date of delivery'] - df['MatID__Date of birth']) / timedelta(days=365)
df = pd.concat([df, pd.DataFrame({'Mat age': _mat_age}, index=df.index)], axis=1)

# Plausibility caps: data-entry errors (e.g. Mat age=131, Height=8.75ft,
# GA=0.3 wks) are nulled out rather than silently skewing downstream models.
print("Applying plausibility caps to derived scalar columns...")
_caps = {
    'Mat age':                  (12.0, 65.0),
    'Height':                   (3.5,   7.5),
    'PP height':                (3.5,   7.5),
    'GA at admit':              (18.0,  46.0),
    'GA at 1st prenatal visit': (4.0,   42.0),
    'GA at delivery':           (18.0,  46.0),
}
_n_capped = 0
for _col, (_lo, _hi) in _caps.items():
    if _col in df.columns:
        _mask = df[_col].notna() & ((df[_col] < _lo) | (df[_col] > _hi))
        if _mask.any():
            print(f"  Outlier cap: {int(_mask.sum())} in '{_col}' [{_lo}–{_hi}] → NaN")
            df.loc[_mask, _col] = np.nan
            _n_capped += int(_mask.sum())
if _n_capped:
    print(f"  Total outlier-capped: {_n_capped} values")

df.drop(df.filter(regex='Date ').columns, axis=1, inplace=True)
df.drop(df.filter(regex=' date').columns, axis=1, inplace=True)
df.drop(columns=[c for c in ['Prenatal__LMP', 'Prenatal__1st prenatal visit'] if c in df.columns], inplace=True)


#Deal with parity — separate TPAL into individual columns
# Note: G (Gravida) is already kept as a continuous column via _NUMERIC_FORCE_PATTERNS.
# Prenatal__Parity contains the TPAL string, so only T, P, A, L need parsing here.
# Data format is space-separated: "2 0 0 2" → split by whitespace, not char-index.
print("Separating TPAL (G already captured as continuous Prenatal__Gravida)")
def _parse_tpal(x, idx):
    if not isinstance(x, str):
        return np.nan
    parts = x.split()
    if len(parts) > idx:
        return parts[idx]
    # Fallback for legacy non-spaced format ("2002")
    if len(x) > idx:
        return x[idx]
    return np.nan

_tpal = {v: df['Prenatal__Parity'].apply(_parse_tpal, idx=i)
         for i, v in enumerate(('Prenatal__T', 'Prenatal__P', 'Prenatal__A', 'Prenatal__L'))}
df = pd.concat([df, pd.DataFrame(_tpal, index=df.index)], axis=1)
df.drop(columns=['Prenatal__Parity'], inplace=True)


#Simplify smoking/marijuana/nicotine detail columns to single boolean (already captured by parent column)
print("Dropping details of drug/smoking/nicotine exposures")
df.drop(df.filter(regex='Other forms of nicotine exposure -').columns, axis=1, inplace=True)
df.drop(df.filter(regex='Secondhand smoke -').columns, axis=1, inplace=True)
df.drop(df.filter(regex='Secondhand marijuana -').columns, axis=1, inplace=True)


#Normalize karyotype free-text into ~8 categories before OHE
def _normalize_karyotype(val):
    if not isinstance(val, str):
        return val
    v = re.sub(r'[^a-z0-9 ]', ' ', val.lower())
    v = ' '.join(v.split())
    if not v or v in ('none', 'na', 'n a', 'unknown', 'not done', 'pending', 'declined'):
        return np.nan
    if re.search(r'trisomy\s*21|down', v):
        return 'trisomy 21'
    if re.search(r'trisomy\s*18|edward', v):
        return 'trisomy 18'
    if re.search(r'trisomy\s*13|patau', v):
        return 'trisomy 13'
    if re.search(r'turner|45\s*x(?!x|y)', v):
        return 'turner syndrome'
    if re.search(r'klinefelter|47\s*xxy', v):
        return 'klinefelter syndrome'
    if re.search(r'mosaic', v):
        return 'mosaic'
    if re.search(r'46\s*x{2}', v) or re.search(r'normal\s*(female|girl)', v):
        return 'normal female'
    if re.search(r'46\s*xy', v) or re.search(r'normal\s*(male|boy)', v):
        return 'normal male'
    return 'other abnormal'

_karyo_cols = [c for c in df.columns if 'karyotype' in c.lower()]
if _karyo_cols:
    print(f"Normalizing {len(_karyo_cols)} karyotype column(s) to categories...")
    for c in _karyo_cols:
        before_uniq = df[c].nunique()
        df[c] = df[c].apply(_normalize_karyotype)
        print(f"  '{c}': {before_uniq} unique → {df[c].nunique()} categories")


#Save free-filled columns separately for NLP script (DB_5b)
print("Saving free filled columns separately for later NLP")
# Match columns with known free-text indicators:
#   (details), (diagnosis), (notes) in parens
#   "Infection details", "Infection Abcess location" (Postpartum)
#   "Other infections" (PostpartumReadmit)
#   "- Text" suffix (Harvey free-text narratives)
#   "alcohol notes", "Other text", trailing "description"
_detail_re = (r'\(details\)|\(diagnosis\)|\(notes\)|Infection details'
              r'|Infection Abcess location|Other infections|- Text$'
              r'|alcohol notes|Other text|description$')
details = df.filter(regex=_detail_re)
df.drop(columns=details.columns, inplace=True)
print(f"  Matched {len(details.columns)} columns by known free-text patterns")
# NOTE: details.to_csv deferred to after general free-text detection below


#Deal with medication list columns — first-pass clean, save for DB_5a_meds processing
print('Separating/cleaning medication data and saving separately for further processing')
# Detect all medication combined-list columns regardless of prefix
med_col_names = df.filter(
    regex=r'COMBINED LIST'
).columns.tolist()
processed_meds = []
for series_name in med_col_names:
    processed_meds.append(separate_meds(df[series_name]))
    print(f"  Processed '{series_name}'...")
    df.drop(columns=[series_name], inplace=True)
# ── Individual medication name columns (not COMBINED LIST) ─────────────────
# These small sections (2 cols each: Name/Medication + Dose) are NOT matched by
# the regex above.  Without explicit handling they stay in cleaned.csv and get
# silently discarded by DB_4's cardinality filter — losing data and creating a
# fragile implicit dependency on the 100-unique-value threshold.
#
# Fix: route the NAME/MEDICATION columns through the same separate_meds() path
# so DB_4a fuzzy-matches and boolean-expands them alongside the COMBINED LISTs.
# DOSE columns are explicitly dropped — dose strings ("200 mg", "1 tab tid")
# are not useful for boolean feature extraction.
#
# Includes paternal meds (PatMedsConception, PatMeds6Mo) — user decision March 2026.
# ContraceptiveHxMeds is the 2026 equivalent of ConceptionMedsDetail for the
# ContraceptiveHx individual-entry subsection (PeribankDB_2026_3 tail).
_indiv_med_name_pats = (r'MedsDetail__Name', r'MedsConception__Medication',
                         r'Meds6Mo__Medication', r'HxMeds__Name',
                         r'ContraceptiveHxMeds__Name',
                         r'__Medication(?:[._]\w+)?$',  # individual med cols (incl merge/dedup suffixes)
                        )
_indiv_med_cols = [c for c in df.columns
                   if any(re.search(p, c) for p in _indiv_med_name_pats)]
_INDIV_MED_DOSE_COLS = (
    'ConceptionMedsDetail__Dose',
    'PatMedsConception__Dose',
    'PatMeds6Mo__Dose',
    'ContraceptiveHxMeds__Dose',      # 2026: dose counterpart (also caught by patterns_togo 'Dose')
)
for col in _indiv_med_cols:
    if col in df.columns:
        processed_meds.append(separate_meds(df[col]))
        print(f"  Processed individual medication column '{col}'...")
        df.drop(columns=[col], inplace=True)

dose_cols_found = [c for c in _INDIV_MED_DOSE_COLS if c in df.columns]
if dose_cols_found:
    print(f"  Dropping {len(dose_cols_found)} medication dose column(s) "
          f"(not useful for boolean feature extraction).")
    df.drop(columns=dose_cols_found, inplace=True, errors='ignore')

if processed_meds:
    meds_df = pd.concat(processed_meds, axis=1)
    meds_df.to_csv("PBDBfinal_meds.csv", sep='|', index=True)


# --- General free-text detection: catch remaining high-cardinality string columns ---
# After known patterns (details/diagnosis/infections/meds) have been extracted,
# any remaining string column with high diversity is likely free-text and should
# go to NLP rather than being silently OHE'd into garbage or dropped by DB_6.
print("Detecting remaining free-text columns for NLP routing...")
_freetext_extra = []
for _col in list(df.columns):
    if df[_col].dtype != object:
        continue
    _vals = df[_col].replace('', pd.NA).dropna()
    if len(_vals) < 20:
        continue
    _nunique = _vals.nunique()
    _median_len = _vals.str.len().median()
    # Heuristic: >50 unique values AND median cell length > 10 chars
    # Catches free-text narratives while excluding checkbox/categorical columns
    if _nunique > 50 and _median_len > 10:
        _freetext_extra.append(_col)
        print(f"  Free-text detected: '{_col}' ({_nunique} unique, median {_median_len:.0f} chars)")

if _freetext_extra:
    freetext_df = df[_freetext_extra]
    df.drop(columns=_freetext_extra, inplace=True)
    details = pd.concat([details, freetext_df], axis=1)
    print(f"  Routed {len(_freetext_extra)} additional free-text columns to NLP pipeline")

# Save all free-text columns (from _detail_re + general detection) for DB_5b
details.to_csv("PBDBfinal_details.csv", sep='|', index=True)
print(f"  Saved {len(details.columns)} total free-text columns to PBDBfinal_details.csv")


# Pre-pass: coerce columns whose names imply a continuous numeric value to float.
# This prevents them from being swept up by separate_csvdata / separate_infections
# and exploded into one binary column per unique value (e.g. age "12" → col_12, col_23…).
print("Coercing numeric-named columns to float before booleanisation…")
for _col in list(df.columns):
    _col_lower = _col.lower()
    if any(pat in _col_lower for pat in _NUMERIC_FORCE_PATTERNS):
        df[_col] = pd.to_numeric(df[_col], errors='coerce')
        print(f"  Kept as numeric: '{_col}'")

#Deal with infections — make Boolean
# Match main infections columns (not "other/details" variants).
# Regex catches both plural "Infections" and singular "Infection" (Postpartum).
inf_col_names = [
    c for c in df.columns
    if re.search(r'__Infections?(\s+intrapartum)?$', c, re.I)
]
for series_name in inf_col_names:
    table = separate_infections(df[series_name], series_name)
    df = pd.concat([df, table], axis=1)
    df.drop(columns=[series_name], inplace=True)
    print(f"  Processed and incorporated as Boolean data from '{series_name}'")


#Deal with allergies, vaccines and similar CSV-separated data — make Boolean
csv_columns = [
    'Allergies__Allergy',
    'OtherImmunizations__Vaccine',
    'FamHxMat__Substance',              # family history substance type
    'PatComorbid__White classification', # diabetes severity classifier
]
for col in csv_columns:
    if col in df.columns:
        table = separate_csvdata(df[col], col)
        if table.empty:
            print(f"  Skipped '{col}' — already numeric, kept as continuous")
        else:
            df = pd.concat([df, table], axis=1)
            df.drop(columns=[col], inplace=True)
            print(f"  Processed and incorporated as Boolean '{col}' data")


# Low-variance filter removed: DB_4 applies a stricter 99.5% uniformity filter during OHE,
# and ML_2's HygieneFilter catches zero-variance features at analysis time.
# A weaker pass here was nearly a no-op and redundant with downstream logic.
print(f'Started with {start_size} features, expanded to {len(df.columns)} as Boolean')
print('(Low-variance filter removed — handled by DB_4 and ML_2 downstream)')


# Duplicate-column check removed: in a sparse clinical database, unrelated columns
# can appear value-identical because the same small patient subset filled out both
# form sections. Shared sparsity patterns are coincidence, not redundancy.
# ML_2's high-correlation filter (r > 0.95) handles genuine redundancy downstream.


#Save final dataframe
print("\nSaving ...")
df.to_csv("PBDBfinal_cleaned.csv", sep='|', index=True)
print("\nSaved remaining data as dataframe: PBDBfinal_cleaned.csv")
