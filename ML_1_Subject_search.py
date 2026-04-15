#!/usr/bin/env python3
# ML_1_Subject_search.py  —  Cohort search module for PeriMiner
#
# March 2026 enhancements:
#   - Morphological stem expansion: strips medical suffixes for higher recall
#   - Optional fuzzy matching via rapidfuzz (catches misspellings in free-text)
#   - Column-level hit tracking: records which DB column each term was found in
#   - match_scores: per-subject confidence (fraction of distinct terms matched)
#
# Returns 5-tuple: (matched_ids, outlist, firstline, match_scores, column_hits)
#
# Importable:
#   from ML_1_Subject_search import search_cohort
#
# Requires: Python 3.7+
# Optional: rapidfuzz  (pip install rapidfuzz)

import re
from typing import Dict, List, Optional, Set, Tuple

try:
    from rapidfuzz import fuzz as _fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

# ---------------------------------------------------------------------------
# Morphological suffix stripping
# Strips common medical suffixes to get a search stem.
# e.g. "cholestasis" → also matches "cholestatic", "cholestatic"
# ---------------------------------------------------------------------------
_MEDICAL_SUFFIXES = re.compile(
    r"(sis|tic|itis|emia|osis|oma|ic|al|ary|ous|ing|ion|tion|ance|ence|"
    r"ment|ive|ated|ate|ase|ent|ant|ary|ory|ory|trophy|pathy|plasty|"
    r"scopy|gram|graph|logy|ectomy|otomy|ostomy|plasty)$",
    re.I,
)


def _stem_expand(term: str) -> Set[str]:
    """Return the original term plus any stem variants created by suffix stripping."""
    variants = {term.lower().strip()}
    t = term.lower().strip()
    m = _MEDICAL_SUFFIXES.search(t)
    if m and m.start() >= 3:          # only strip if stem >= 3 chars
        stem = t[: m.start()]
        if len(stem) >= 3:
            variants.add(stem)
    return variants


def _expand_all_terms(search_terms: List[str]) -> List[str]:
    """Return a flat, deduplicated list of all terms + their stem variants."""
    seen: Set[str] = set()
    out: List[str] = []
    for t in search_terms:
        for v in _stem_expand(t):
            if v not in seen:
                seen.add(v)
                out.append(v)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def search_cohort(
    search_terms: List[str],
    db_file: str = "PBDBfinal.txt",
    fuzzy: bool = False,
    fuzzy_threshold: int = 85,
) -> Tuple[Set[str], Dict, str, Dict[str, float], Dict[str, Dict]]:
    """Search the PeriBank flat-file database for subjects matching search_terms.

    Args:
        search_terms    : list of search strings (case-insensitive).
        db_file         : path to the pipe-delimited PBDBfinal.txt export.
        fuzzy           : if True, also applies rapidfuzz partial_ratio matching
                          (useful for misspellings in free-text fields).
                          Stem expansion is always applied regardless of this flag.
        fuzzy_threshold : rapidfuzz partial_ratio cutoff (0-100). Ignored when fuzzy=False.

    Returns:
        matched_ids  : set[str]  — Pregnancy IDs of matched subjects
        outlist      : dict      — {pid: [list of matching lines/fields]}
        firstline    : str       — the raw header line (for downstream parsing)
        match_scores : dict[str, float]
                         pid → fraction of distinct search_terms matched (0.0–1.0).
                         e.g. if 3 of 5 terms matched, score = 0.60.
        column_hits  : dict[str, dict]
                         {col_name: {"terms": set(matched_terms),
                                     "subjects": set(pids)}}
                         Records which database column each term was found in.
    """
    if not search_terms:
        return set(), {}, "", {}, {}

    expanded = _expand_all_terms(search_terms)
    n_original = len(search_terms)   # denominator for match_scores

    matched_ids:  Set[str]         = set()
    outlist:      Dict             = {}
    firstline:    str              = ""
    match_scores: Dict[str, float] = {}
    column_hits:  Dict[str, Dict]  = {}

    # Per-subject tracking: which original terms matched
    subject_terms: Dict[str, Set[str]] = {}

    # Column indices whose *names* match a search term (populated after header)
    col_name_match_indices: Dict[int, str] = {}   # col_idx -> original term

    def _term_matches_field(term: str, field_val: str) -> bool:
        """Return True if term matches field_val (exact substring or fuzzy)."""
        if term in field_val:
            return True
        if fuzzy and _HAS_RAPIDFUZZ and len(field_val) >= len(term):
            return _fuzz.partial_ratio(term, field_val) >= fuzzy_threshold
        return False

    with open(db_file, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw_line in enumerate(fh):
            line = raw_line.rstrip("\n")

            # ---- Header line ----
            if line_no == 0:
                firstline = line
                header_cols = line.split("|")
                # Find the Pregnancy ID column index (first column by convention)
                pid_col_idx = 0
                for i, h in enumerate(header_cols):
                    if re.search(r"pregnancy.?id", h, re.I):
                        pid_col_idx = i
                        break

                # Find columns whose names match any search term
                for col_idx, col_name in enumerate(header_cols):
                    col_lower = col_name.strip().lower()
                    for term in expanded:
                        if term in col_lower:
                            col_name_match_indices[col_idx] = _original_term(term, search_terms)
                            break
                    if col_idx not in col_name_match_indices and fuzzy and _HAS_RAPIDFUZZ:
                        for term in expanded:
                            if _fuzz.partial_ratio(term, col_lower) >= fuzzy_threshold:
                                col_name_match_indices[col_idx] = _original_term(term, search_terms)
                                break
                continue

            # ---- Fast whole-line filter ----
            line_lower = line.lower()
            if not any(t in line_lower for t in expanded):
                # Only do fuzzy pass if exact stem check fails and fuzzy is on
                if not (fuzzy and _HAS_RAPIDFUZZ and
                        any(_fuzz.partial_ratio(t, line_lower) >= fuzzy_threshold
                            for t in expanded)):
                    # Last chance: does this row have a value in a column whose
                    # *name* matched a search term?
                    if not col_name_match_indices:
                        continue
                    fields_temp = line.split("|")
                    if not any(i < len(fields_temp) and fields_temp[i].strip()
                               for i in col_name_match_indices):
                        continue

            # ---- Split into fields ----
            fields = line.split("|")
            pid = fields[pid_col_idx].strip() if pid_col_idx < len(fields) else ""
            if not pid:
                continue

            # ---- Per-field check (only on lines that passed the fast filter) ----
            for col_idx, field_val in enumerate(fields):
                if col_idx == pid_col_idx:
                    continue
                fv_lower = field_val.lower().strip()
                if not fv_lower:
                    continue

                col_name = (header_cols[col_idx].strip()
                            if col_idx < len(header_cols) else f"col_{col_idx}")

                # Column-name match: non-empty value in a column whose name
                # matched a search term (e.g. searching "length of stay" finds
                # the 'Intrapartum__Length of stay (days)' column)
                if col_idx in col_name_match_indices:
                    orig_term = col_name_match_indices[col_idx]
                    matched_ids.add(pid)
                    outlist.setdefault(pid, []).append(
                        f"[{col_name}] {field_val.strip()}"
                    )
                    subject_terms.setdefault(pid, set()).add(orig_term)
                    ch = column_hits.setdefault(col_name, {"terms": set(), "subjects": set()})
                    ch["name_match"] = True   # column NAME matched the search term
                    ch["terms"].add(orig_term)
                    ch["subjects"].add(pid)
                    continue   # no need to also check term-in-value for this field

                for term in expanded:
                    # Map expanded term back to whichever original term it came from
                    # (for match_score counting we use original terms)
                    if _term_matches_field(term, fv_lower):
                        # Find the original term this variant belongs to
                        orig_term = _original_term(term, search_terms)

                        matched_ids.add(pid)
                        outlist.setdefault(pid, []).append(
                            f"[{col_name}] {field_val.strip()}"
                        )
                        subject_terms.setdefault(pid, set()).add(orig_term)

                        ch = column_hits.setdefault(col_name, {"terms": set(), "subjects": set()})
                        ch.setdefault("name_match", False)  # value matched; don't override True
                        ch["terms"].add(orig_term)
                        ch["subjects"].add(pid)

    # Compute match_scores: fraction of original search_terms matched per subject
    for pid, terms_found in subject_terms.items():
        match_scores[pid] = len(terms_found) / max(n_original, 1)

    return matched_ids, outlist, firstline, match_scores, column_hits


def search_cohort_df(
    df,
    search_terms: List[str],
    fuzzy: bool = False,
    fuzzy_threshold: int = 85,
    _pids_arr=None,
    _str_cols: Optional[Dict[str, any]] = None,
) -> Tuple[Set[str], Dict, str, Dict[str, float], Dict[str, Dict]]:
    """Search an in-memory DataFrame (e.g. the ML-ready pickle) for subjects
    matching search_terms.  Identical return signature to search_cohort().

    Phase 1 — column-name scan:
        Columns whose names contain a search term are flagged as name-match hits.
        Subjects with a truthy (non-zero / non-null) value in those columns are
        included and recorded as cohort members.

    Phase 2 — value scan:
        Object / string columns not already captured in Phase 1 are scanned for
        term substrings in cell values.

    Args:
        df              : pandas DataFrame (the loaded pickle).
        search_terms    : list of search strings (case-insensitive).
        fuzzy           : if True, also applies rapidfuzz partial_ratio matching
                          on column names.  Stem expansion is always applied.
        fuzzy_threshold : rapidfuzz partial_ratio cutoff (0-100).

    Returns:
        Same 5-tuple as search_cohort():
        (matched_ids, outlist, firstline, match_scores, column_hits)
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError as exc:
        raise ImportError("pandas and numpy are required for search_cohort_df()") from exc

    if not search_terms:
        return set(), {}, "", {}, {}

    expanded = _expand_all_terms(search_terms)
    n_original = len(search_terms)

    matched_ids:   Set[str]             = set()
    outlist:       Dict[str, List]      = {}
    match_scores:  Dict[str, float]     = {}
    column_hits:   Dict[str, Dict]      = {}
    subject_terms: Dict[str, Set[str]]  = {}

    # Identify Pregnancy ID column (or fall back to index)
    pid_col = next(
        (c for c in df.columns if re.search(r"pregnancy.?id", c, re.I)), None
    )

    firstline = "|".join(df.columns.tolist())

    # Use pre-computed PID array if provided, otherwise build it once now
    if _pids_arr is None:
        _pids_arr = (df[pid_col] if pid_col else df.index).astype(str).values

    # ------------------------------------------------------------------
    # Phase 1: column-name scan
    # ------------------------------------------------------------------
    col_name_hits: Dict[str, str] = {}   # col -> orig_term
    for col in df.columns:
        if col == pid_col:
            continue
        col_lower = col.strip().lower()
        for term in expanded:
            hit = term in col_lower
            if not hit and fuzzy and _HAS_RAPIDFUZZ:
                hit = _fuzz.partial_ratio(term, col_lower) >= fuzzy_threshold
            if hit:
                col_name_hits[col] = _original_term(term, search_terms)
                break

    for col, orig_term in col_name_hits.items():
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            mask = (series.notna() & (series != 0)).values
        else:
            s = series.astype(str).str.strip()
            mask = (series.notna() & ~s.isin(["", "nan", "NaN", "None", "0"])).values

        ch = column_hits.setdefault(
            col, {"terms": set(), "subjects": set(), "name_match": True}
        )
        ch["terms"].add(orig_term)

        matching_pids = _pids_arr[mask]
        matching_vals = df[col].astype(str).values[mask]
        for pid, val in zip(matching_pids, matching_vals):
            if not pid or pid in ("nan", ""):
                continue
            matched_ids.add(pid)
            outlist.setdefault(pid, []).append(f"[{col}] {val}")
            subject_terms.setdefault(pid, set()).add(orig_term)
            ch["subjects"].add(pid)

    # ------------------------------------------------------------------
    # Phase 2: value scan on object/string columns not in col_name_hits
    # ------------------------------------------------------------------
    for col in df.columns:
        if col == pid_col or col in col_name_hits:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Use pre-computed lowercase array if available, otherwise compute now
        if _str_cols is not None and col in _str_cols:
            col_arr = _str_cols[col]
            col_ser = pd.Series(col_arr)
        else:
            col_ser = df[col].astype(str).str.lower().str.strip()
            col_arr = col_ser.values

        for term in expanded:
            mask = col_ser.str.contains(term, na=False, regex=False).values
            if not mask.any():
                continue

            orig_term = _original_term(term, search_terms)

            ch = column_hits.setdefault(col, {"terms": set(), "subjects": set()})
            ch.setdefault("name_match", False)
            ch["terms"].add(orig_term)

            matching_pids = _pids_arr[mask]
            matching_vals = col_arr[mask]
            for pid, val in zip(matching_pids, matching_vals):
                if not pid or pid in ("nan", ""):
                    continue
                matched_ids.add(pid)
                outlist.setdefault(pid, []).append(f"[{col}] {val}")
                subject_terms.setdefault(pid, set()).add(orig_term)
                ch["subjects"].add(pid)

    for pid, terms_found in subject_terms.items():
        match_scores[pid] = len(terms_found) / max(n_original, 1)

    return matched_ids, outlist, firstline, match_scores, column_hits


def find_matching_columns(
    df,
    search_terms: List[str],
    fuzzy: bool = False,
    fuzzy_threshold: int = 85,
) -> Dict[str, List[Dict]]:
    """Scan column names only (no row iteration). Returns per-term candidate columns.

    For each search term, finds every column whose name contains the term (or a
    stem variant), and counts how many subjects have a truthy value in that column.

    Args:
        df              : pandas DataFrame (the loaded pickle).
        search_terms    : list of search strings (case-insensitive).
        fuzzy           : if True, also applies rapidfuzz partial_ratio on column names.
        fuzzy_threshold : rapidfuzz partial_ratio cutoff (0-100).

    Returns:
        dict: {term: [{"column":      str,   # full column name (PREFIX__label)
                        "section":     str,   # PREFIX part (before __)
                        "col_label":   str,   # readable label (after __)
                        "n_positive":  int,   # subjects with truthy value
                        "pct_positive": float}]}
        Sorted by n_positive descending within each term.
        Each column appears under at most one term (first match wins).
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for find_matching_columns()") from exc

    if not search_terms:
        return {t: [] for t in search_terms}

    n_rows = len(df)
    pid_col = next(
        (c for c in df.columns if re.search(r"pregnancy.?id", c, re.I)), None
    )

    # Use only the original terms (no stem expansion) for column-name matching.
    # Stems are too aggressive on controlled-vocabulary column names
    # (e.g. "genetic" → "gene" would match "general").
    term_lower_map: Dict[str, str] = {t: t.lower().strip() for t in search_terms}

    results: Dict[str, List] = {t: [] for t in search_terms}
    claimed: Set[str] = set()   # columns already assigned to a term

    for col in df.columns:
        if col == pid_col or col in claimed:
            continue
        col_lower = col.strip().lower()
        section   = col.split("__")[0] if "__" in col else "Other"
        col_label = col.split("__", 1)[1] if "__" in col else col

        for orig_term, term_l in term_lower_map.items():
            matched = term_l in col_lower
            if not matched and fuzzy and _HAS_RAPIDFUZZ:
                matched = _fuzz.partial_ratio(term_l, col_lower) >= fuzzy_threshold
            if not matched:
                continue

            # Count truthy subjects
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                n_pos = int((series.notna() & (series != 0)).sum())
            else:
                s = series.astype(str).str.strip()
                n_pos = int((series.notna() & ~s.isin(["", "nan", "NaN", "None", "0"])).sum())

            results[orig_term].append({
                "column":       col,
                "section":      section,
                "col_label":    col_label,
                "n_positive":   n_pos,
                "pct_positive": round(100.0 * n_pos / max(n_rows, 1), 1),
            })
            claimed.add(col)
            break   # first matching term wins

    for t in results:
        results[t].sort(key=lambda x: x["n_positive"], reverse=True)

    return results


def _original_term(expanded_variant: str, original_terms: List[str]) -> str:
    """Map an expanded/stemmed variant back to its source original term."""
    for orig in original_terms:
        if expanded_variant in _stem_expand(orig):
            return orig
    return expanded_variant   # fallback: return as-is


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
def main():
    terms_input  = input("Enter search terms (comma-separated): ").strip()
    search_terms = [t.strip() for t in terms_input.split(",") if t.strip()]
    if not search_terms:
        print("No search terms provided. Exiting.")
        return

    db_file = input("Database file (Enter = 'PBDBfinal.txt'): ").strip() or "PBDBfinal.txt"
    use_fuzzy = input("Use fuzzy matching? (y/N): ").strip().lower() == "y"

    print(f"\nSearching '{db_file}' for: {search_terms}")
    print(f"Fuzzy matching: {'on' if use_fuzzy else 'off (stem expansion always active)'}\n")

    matched_ids, outlist, firstline, match_scores, column_hits = search_cohort(
        search_terms, db_file, fuzzy=use_fuzzy
    )

    print(f"Found {len(matched_ids)} matching subjects.\n")

    if column_hits:
        print("Columns with matches:")
        for col, info in sorted(column_hits.items(),
                                key=lambda kv: len(kv[1]["subjects"]), reverse=True):
            print(f"  {col}: {len(info['subjects'])} subjects, "
                  f"terms: {', '.join(sorted(info['terms']))}")

    if matched_ids:
        top5 = sorted(matched_ids)[:5]
        print(f"\nTop 5 matched IDs: {top5}")
        print(f"Match scores (sample): "
              f"{ {pid: round(match_scores[pid], 2) for pid in top5} }")


if __name__ == "__main__":
    main()
