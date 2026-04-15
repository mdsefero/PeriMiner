#!/usr/bin/env python3
"""DB_1_recreate.py — Assemble and prefix-label the PeriBank database.

Consolidates PeribankDB_2026_1.txt through PeribankDB_2026_10.txt into PBDBfinal.txt.
Also accepts the legacy naming scheme (PeribankDB_1.txt … PeribankDB_9.txt).

All non-ID columns are prefixed with their clinical category, e.g.:
    MatComorbid__Hypertension
    Prenatal__PP BMI
    Intrapartum__Gestational diabetes
This eliminates pandas _x/_y suffix collisions on merge.

Run:
    python DB_1_recreate.py
Requires Python 3.7+, pandas.
Last updated 2026-03, Maxim Seferovic, seferovi@bcm.edu
"""

import csv
import os
import glob
import re
import multiprocessing

import pandas as pd

# Python's csv module caps field size at 131 072 chars by default; some medical
# free-text fields exceed this.  Raise the limit to 10 MB.
csv.field_size_limit(10_000_000)


# ---------------------------------------------------------------------------
# Section map
#   Key   : canonical file stem, e.g. "PeribankDB_3" (legacy) or
#           "PeribankDB_2026_3" (2026 export).  Keys are SEPARATE because
#           the 2026 files reorganise sections across different file numbers.
#   Value : list of (prefix, column_count)
#             column_count=None  →  all remaining non-ID columns in this file
#
# Counts are derived from Headings.txt (Desktop/PeriMiner).
# A runtime warning is printed if the actual file has more columns than mapped.
# ---------------------------------------------------------------------------
SECTION_MAP: dict = {
    # ── Legacy naming scheme (PeribankDB_1.txt … PeribankDB_9.txt) ──────────
    "PeribankDB_1": [
        ("MatID",       10),   # Date of birth … Race - Not reported/Unknown
        ("MatInfo",     22),   # Hospital … Hospital (other)
        ("MatComorbid", 38),   # No maternal comorbidities … Cystic fibrosis
        ("MatBirthHx",   5),   # Were you born prematurely … Do you have or have you had
        ("Harvey",      None), # Affected by Hurricane Harvey … Received mental health
    ],
    "PeribankDB_2": [
        ("PriorPreg",              9),   # Pregnancy order … Cesarean indication (other)
        ("PriorPregComplication",  23),  # No prior obstetrical complications … Other (details)
        ("ContraceptiveHx",        None),# Prepregnancy contraception use … COMBINED LIST
    ],
    "PeribankDB_3": [
        ("Prenatal",             116),   # Gravida … Conception med COMBINED LIST
        ("ConceptionMedsDetail",   2),   # Name, Dose
        ("Allergies",              1),   # Allergy
        ("OtherImmunizations",    None), # Vaccine, Vaccine date
    ],
    "PeribankDB_4": [
        ("FamHxMat",        35),   # None … Other (details)   [maternal family Hx]
        ("FamHxMatSisters", 25),   # None … Hypothyroid       [sisters OB history]
        ("FamHxMatGrandma", None), # None … Hypothyroid       [grandmother/aunt OB history]
    ],
    "PeribankDB_5": [
        ("PatInfo",           36),  # Father info unknown … Medications 6mo COMBINED LIST
        ("PatMedsConception",  2),  # Medication, Dose  (at conception)
        ("PatMeds6Mo",         2),  # Medication, Dose  (6 months prior)
        ("PatComorbid",       38),  # None … Muscular dystrophy
        ("PatFamOB",          25),  # None … Fetal birth anomaly (details)
        ("PatFamMedHx",       28),  # None … Other (details)
        ("PatBirthHx",        None),# Were you born prematurely … Do you have or have you had
    ],
    "PeribankDB_6": [("Antepartum",  None)],
    "PeribankDB_7": [("Intrapartum", None)],
    "PeribankDB_8": [
        ("Delivery", 27),   # FHR at admission … Major congenital malformations details
        ("Newborn",  None), # APGAR at 1 minute … Cause of death
    ],
    "PeribankDB_9": [
        ("Postpartum",        37),   # Infections … Date ICU discharge
        ("PostpartumReadmit", None), # Hospital … Medication
    ],

    # ── 2026 naming scheme (PeribankDB_2026_1.txt … PeribankDB_2026_10.txt) ─
    # Section organisation differs from legacy — DO NOT share keys between schemes.
    # Column counts derived from Headings.txt (6 March 2026 download).
    #
    # Key differences vs legacy:
    #   • MatComorbid/MatBirthHx/Harvey moved from file 1 → file 2
    #   • Prenatal gains 7 COVID-19 columns (123 total, was 116)
    #   • FamHxMat is 34 cols (was 35 in legacy)
    #   • Antepartum individual-medication rows split to start of file 8
    #   • DB_10 now contains Postpartum data (was excluded in legacy as biobank)
    "PeribankDB_2026_1": [
        ("MatID",    10),   # Date of birth … Race - Not reported/Unknown
        ("MatInfo", None),  # Hospital … Hospital (other)  [22 cols]
    ],
    "PeribankDB_2026_2": [
        ("MatComorbid", 38),   # No maternal comorbidities … Cystic fibrosis
        ("MatBirthHx",   5),   # Were you born prematurely … Do you have or have you had
        ("Harvey",      None), # Affected by Hurricane Harvey … Received mental health
    ],
    "PeribankDB_2026_3": [
        ("PriorPreg",              9),   # Pregnancy order … Cesarean indication (other)
        ("PriorPregComplication",  23),  # No prior obstetrical complications … Other (details)
        ("ContraceptiveHx",        18),  # Prepregnancy contraception use … COMBINED LIST
        ("ContraceptiveHxMeds",  None),  # Medications and dose at conception: Name, Dose
    ],
    "PeribankDB_2026_4": [
        ("Prenatal",             122),   # Gravida … Wanted to become pregnant (incl COVID-19)
        ("ConceptionMedsDetail",   2),   # Name, Dose  (Prenatal - Medications at conception)
        ("Allergies",              1),   # Allergy
        ("OtherImmunizations",    None), # Vaccine, Vaccine date
    ],
    "PeribankDB_2026_5": [
        ("FamHxMat",        34),   # None … Other (details) [maternal family Hx; 34 cols in 2026]
        ("FamHxMatSisters", 25),   # None … Hypothyroid [sisters OB history]
        ("FamHxMatGrandma", None), # None … Hypothyroid [grandmother/aunt OB history]
    ],
    "PeribankDB_2026_6": [
        ("PatInfo",           36),  # Father info unknown … Medications 6mo COMBINED LIST
        ("PatMedsConception",  2),  # Medication, Dose  (at conception)
        ("PatMeds6Mo",         2),  # Medication, Dose  (6 months prior)
        ("PatComorbid",       38),  # None … Muscular dystrophy
        ("PatFamOB",          25),  # None … Fetal birth anomaly (details)
        ("PatFamMedHx",       28),  # None … Other (details)
        ("PatBirthHx",        None),# Were you born prematurely … Do you have or have you had
    ],
    "PeribankDB_2026_7": [("Antepartum",  None)],  # Full antepartum section
    "PeribankDB_2026_8": [
        # Antepartum course - Medications (individual entry rows) split here in 2026
        ("Antepartum",    1),    # Medication  [1 individual-entry col]
        ("Intrapartum", None),   # Hospital … Medications - COMBINED LIST (134 cols)
                                 #   + Intrapartum course - Medications at end (1 col)
    ],
    "PeribankDB_2026_9": [
        ("Delivery", 27),   # FHR at admission … Major congenital malformations details
        ("Newborn",  None), # APGAR at 1 minute … Cause of death
    ],
    "PeribankDB_2026_10": [
        ("Postpartum",        37),   # Infections … Date ICU discharge
        ("PostpartumReadmit", None), # Hospital … Medications - COMBINED LIST (66 cols)
                                     #   + Postpartum re-admission - Medications at end (1 col)
    ],
}

_ID_COLS = frozenset({"Subject ID", "Pregnancy ID"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_key(path: str) -> str:
    """Return the SECTION_MAP lookup key for a file path.

    2026 naming:  PeribankDB_2026_3.txt  → "PeribankDB_2026_3"
    Legacy naming: PeribankDB_3.txt      → "PeribankDB_3"

    The year component is intentionally preserved for 2026 files because the
    2026 export reorganises clinical sections across different file numbers —
    the two schemes require separate SECTION_MAP entries.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    # 2026 format: PeribankDB_YYYY_N — keep year, normalise case
    m = re.match(r"PeribankDB_(\d{4})_(\d+)$", stem, re.IGNORECASE)
    if m:
        return f"PeribankDB_{m.group(1)}_{m.group(2)}"   # → "PeribankDB_2026_3"
    # Legacy format: PeribankDB_N — normalise case
    m = re.match(r"PeribankDB_(\d+)$", stem, re.IGNORECASE)
    if m:
        return f"PeribankDB_{m.group(1)}"                # → "PeribankDB_3"
    return stem


def _apply_prefix(df: pd.DataFrame, sections: list, path: str = "") -> pd.DataFrame:
    """Rename every non-ID column as PREFIX__original_name (positional assignment).

    Uses position-based renaming (not a name→name dict) so that files with two
    columns sharing the same name (e.g. two 'Medication' subsections in DB_8)
    each get their own independent prefix without dict-key collisions.
    """
    # Positions (in df.columns) of non-ID columns
    non_id_pos = [i for i, c in enumerate(df.columns) if c not in _ID_COLS]
    new_cols = list(df.columns)
    pos = 0
    for prefix, count in sections:
        end = len(non_id_pos) if count is None else pos + count
        for j in range(pos, min(end, len(non_id_pos))):
            col_i = non_id_pos[j]
            new_cols[col_i] = f"{prefix}__{df.columns[col_i]}"
        pos = end
        if pos >= len(non_id_pos):
            break

    # Warn and absorb any extra columns into the last named section
    if pos < len(non_id_pos):
        last_prefix = sections[-1][0]
        extra_count = len(non_id_pos) - pos
        print(
            f"  [WARNING] {os.path.basename(path)}: {extra_count} extra column(s) beyond "
            f"section map — prefixed as {last_prefix}__."
        )
        for j in range(pos, len(non_id_pos)):
            col_i = non_id_pos[j]
            new_cols[col_i] = f"{last_prefix}__{df.columns[col_i]}"

    df.columns = pd.Index(new_cols)

    # Strip pandas dedup suffixes (.1, .2, …) that are now redundant because
    # the clinical prefix disambiguates (e.g. "FamHxMatGrandma__Hypothyroid.1"
    # → "FamHxMatGrandma__Hypothyroid").  Re-deduplicate if stripping creates
    # new collisions (rare, but defensive).
    cleaned = [re.sub(r'\.\d+$', '', c) if '__' in c else c for c in df.columns]
    if len(set(cleaned)) < len(cleaned):
        seen: dict = {}
        final = []
        for c in cleaned:
            if c in seen:
                seen[c] += 1
                final.append(f"{c}.{seen[c]}")
            else:
                seen[c] = 0
                final.append(c)
        cleaned = final
    df.columns = pd.Index(cleaned)
    return df


# ---------------------------------------------------------------------------
# Main processing functions
# ---------------------------------------------------------------------------

def data_clean(all_filenames: list):
    """
    Load, clean, and prefix-rename each input file.

    Returns
    -------
    ready : dict {path: DataFrame}
        Files with no duplicate Pregnancy IDs.
    to_process : list of [path, dup_ids, dup_subdf, full_df]
        Files with duplicate Pregnancy IDs needing aggregation.
    """
    ready: dict = {}
    to_process: list = []

    for path in sorted(all_filenames):
        # Auto-detect separator: PeriBank native files use "|"; Excel "Text (Macintosh)"
        # re-saves use tab.  Open with default universal-newlines mode (newline=None) so
        # that readline() correctly reads only the first line even with \r-only Mac endings.
        # (newline="" would disable \r translation, causing readline() to consume the
        #  entire file and falsely detect a tab in some data cell.)
        with open(path, encoding="utf-8", errors="replace") as _f:
            _hdr = _f.readline().rstrip()  # strip \r, \n, trailing tabs (Excel artifact)
        # Compare counts: pipe files with trailing Excel tabs have many more pipes
        # than tabs; genuinely tab-delimited files have many more tabs than pipes.
        sep = "\t" if _hdr.count("\t") > _hdr.count("|") else "|"
        df = pd.read_csv(path, sep=sep, dtype="object", engine="python",
                         encoding="utf-8", encoding_errors="replace",
                         quoting=csv.QUOTE_NONE)
        df.columns = df.columns.str.strip()   # Excel sometimes pads header names
        # Stripping can turn e.g. "Medication " and "Medication" into the same string,
        # bypassing mangle_dupe_cols which ran earlier on the unstripped names.
        # Re-de-duplicate explicitly so no column appears twice downstream.
        if df.columns.duplicated().any():
            seen: dict = {}
            new_names = []
            for c in df.columns:
                if c in seen:
                    seen[c] += 1
                    new_names.append(f"{c}.{seen[c]}")
                else:
                    seen[c] = 0
                    new_names.append(c)
            df.columns = new_names

        # Drop Subject ID — Pregnancy ID is the sole merge key
        df.drop(columns=[c for c in df.columns if c == "Subject ID"], inplace=True)

        # Move Pregnancy ID to the first column
        df = df[["Pregnancy ID"] + [c for c in df.columns if c != "Pregnancy ID"]]

        df.drop_duplicates(inplace=True)

        df["Pregnancy ID"] = pd.to_numeric(df["Pregnancy ID"], errors="coerce")
        df.dropna(subset=["Pregnancy ID"], inplace=True)

        df.dropna(axis="columns", how="all", inplace=True)
        df.fillna("", inplace=True)

        # Apply clinical category prefixes
        sections = SECTION_MAP.get(_file_key(path))
        if sections:
            df = _apply_prefix(df, sections, path)
        else:
            print(f"  [WARNING] No section map for {os.path.basename(path)} — columns unprefixed.")

        # Identify rows with duplicate Pregnancy IDs
        dups_mask = df.duplicated(subset=["Pregnancy ID"], keep=False)
        dup_df    = df[dups_mask]
        dup_ids   = list(dict.fromkeys(dup_df["Pregnancy ID"].tolist()))

        bname = os.path.basename(path)
        if not dup_ids:
            ready[path] = df
            print(f"  Loaded (no duplicates):  {bname}  ({len(df):,} rows, {len(df.columns):,} cols)")
        else:
            to_process.append([path, dup_ids, dup_df, df])
            print(
                f"  Loaded (has duplicates): {bname}  "
                f"({len(df):,} rows, {len(dup_ids):,} duplicate Pregnancy IDs)"
            )

    return ready, to_process


def aggregate_duplicates(data: list):
    """
    Collapse duplicate Pregnancy ID rows by CSV-concatenating differing values.
    Returns (path, cleaned_df).  Designed to run inside a multiprocessing Pool.
    """
    path, dup_ids, dup_df, df = data
    for n, pid in enumerate(dup_ids, 1):
        idx         = df.index[df["Pregnancy ID"] == pid].tolist()
        preg        = dup_df.loc[dup_df["Pregnancy ID"] == pid]
        varying     = preg.columns[(preg.nunique() != 1).values]
        for col in varying:
            col_data = preg[col]
            # preg[col] returns a DataFrame if col is a duplicate column name;
            # take the first occurrence as a Series in that case.
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            df.loc[idx, col] = col_data.str.cat(sep=",")
        if n % 10_000 == 0:
            print(f"  {os.path.basename(path)}: {n:,} pregnancies aggregated")
    df.drop_duplicates(inplace=True)
    return path, df


def consolidate(outdf: dict):
    """Merge all cleaned DataFrames on Pregnancy ID and write PBDBfinal.txt."""
    def _sort_key(p):
        # Use the LAST number in the filename as the sort key so that the
        # year component of new-style names (e.g. PeribankDB_2026_3) is
        # ignored and files are merged in clinical order (1, 2, … 10).
        nums = re.findall(r"\d+", os.path.basename(p))
        return int(nums[-1]) if nums else 0

    merged = pd.DataFrame(columns=["Pregnancy ID"])
    for path in sorted(outdf, key=_sort_key):
        print(f"  Merging {os.path.basename(path)} …")
        merged = merged.merge(outdf[path], on="Pregnancy ID", how="outer")

    merged["Pregnancy ID"] = merged["Pregnancy ID"].astype(int)
    merged.sort_values("Pregnancy ID", inplace=True)
    merged.to_csv("PBDBfinal.txt", sep="|", index=False)
    print(f"\nSaved PBDBfinal.txt  ({len(merged):,} rows, {len(merged.columns):,} columns)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Accept both the new 2026 naming scheme (PeribankDB_2026_1.txt … PeribankDB_2026_10.txt)
    # and the legacy scheme (PeribankDB_1.txt … PeribankDB_9.txt).
    # New-style files 1–10 are all included; legacy DB_10–13 (biobank / pathology) remain excluded.
    all_files = glob.glob("*PeribankDB_*.txt")
    keep = [
        f for f in all_files
        if re.search(r"PeribankDB_[1-9]\.txt$", f)                # legacy: 1–9
        or re.search(r"PeribankDB_\d{4}_\d+\.txt$", f, re.I)     # new: YYYY_N (any number)
    ]

    if not keep:
        raise FileNotFoundError(
            "No PeriBank input files found in the current directory.\n"
            "Expected: PeribankDB_2026_1.txt … PeribankDB_2026_10.txt  "
            "(or legacy PeribankDB_1.txt … PeribankDB_9.txt)"
        )

    print(f"Found {len(keep)} input file(s):\n  " +
          "\n  ".join(sorted(os.path.basename(f) for f in keep)) + "\n")

    outdf: dict = {}
    ready, to_process = data_clean(keep)
    outdf.update(ready)

    if to_process:
        workers = max(1, os.cpu_count() - 2)
        print(f"\nAggregating duplicates in {len(to_process)} file(s) "
              f"using {workers} worker(s) …")
        with multiprocessing.Pool(workers) as pool:
            for path, df in pool.imap_unordered(aggregate_duplicates, to_process):
                outdf[path] = df
                print(f"  Done: {os.path.basename(path)}")

    print("\nMerging all files …")
    consolidate(outdf)
