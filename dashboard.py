# PeriMiner Dashboard  —  streamlit run dashboard.py
#
# Flow:
#   Step 1.   Enter cohort label + search terms → Search Database  (fast)
#   Step 1.5  Column filter — accept/reject columns where terms were found
#             Boolean term filter — Must Have / Must Not Have per term
#   Step 2.   Run Analysis  (LightGBM + SHAP, ~1 min; ~2-3 min with CV)
#   Step 3.   Explore results in Feature Management; adjust Top-N in real time
#
# Modes:
#   Single cohort vs. rest  — original behaviour
#   Compare two cohorts     — define Cohort A and Cohort B; ML compares A vs B

import csv
import hashlib
import os
import pickle as _pkl
import re
import threading
import traceback
from datetime import datetime, timezone

import streamlit as st
import pandas as pd


# PeriBank sections excluded from ML ranking AND hidden by default in Feature Management.
# Users can re-enable any category via the Category filter in the Feature Management tab.
_DEFAULT_EXCLUDED_CATS = {
    "Harvey",
    # Paternal information
    "PatInfo", "PatMedsConception", "PatMeds6Mo", "PatComorbid",
    "PatFamOB", "PatFamMedHx", "PatBirthHx",
    # Extended family history (not direct maternal history)
    "FamHxMat", "FamHxMatSisters", "FamHxMatGrandma",
    # Contraceptive history and medications
    "ContraceptiveHx", "ContraHx", "ContraceptiveMeds", "ContraMeds",
    # Prior pregnancy complications
    "PriorPregComplications", "PriorPregComp",
    # Conception medication details
    "ConceptionMedsDetail",
    # Allergies and other immunizations
    "Allergies", "OtherImmunizations",
    # Postpartum readmission
    "PostpartumReadmit",
}


def _xf_raw_indexed(pickle_path: str, xf_raw):
    """Reindex Xf_raw by Pregnancy ID string so feature detail can look up by PID.

    Returns xf_raw with its index replaced by PID strings, or None if xf_raw is None.
    Uses the already-cached pickle — no extra disk I/O.
    """
    if xf_raw is None:
        return None
    pkl_df = _load_pkl(pickle_path)
    if "Pregnancy ID" not in pkl_df.columns:
        pkl_df = pkl_df.reset_index().rename(columns={"index": "Pregnancy ID"})
    pids = pkl_df["Pregnancy ID"].astype(str).reindex(xf_raw.index)
    result = xf_raw.copy()
    result.index = pids.values
    return result


def _section_excluded_cols(pickle_path: str) -> set:
    """Return the set of column names whose PeriBank section is in _DEFAULT_EXCLUDED_CATS.

    Called before every run_analysis() so excluded sections are never trained on,
    regardless of whether this is a first run, re-run, or re-rank.
    Uses the already-cached pickle — no extra disk I/O.
    """
    df = _load_pkl(pickle_path)
    return {
        c for c in df.columns
        if (c.split("__")[0] if "__" in c else c) in _DEFAULT_EXCLUDED_CATS
    }

_LOG_FILE = "usage_log.csv"
_LOG_LOCK = threading.Lock()
_LOG_COLS = ["timestamp", "event", "cohort_label", "search_terms", "n_terms", "user_hash"]

def _get_user_token() -> str:
    try:
        hdrs = st.context.headers
        ip = hdrs.get("x-forwarded-for") or hdrs.get("host") or "unknown"
        return hashlib.sha256(ip.encode()).hexdigest()[:12]
    except Exception:
        return "unknown"

def _log_event(event: str, cohort_label: str, search_terms: list) -> None:
    try:
        row = {
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "event":        event,
            "cohort_label": cohort_label,
            "search_terms": ";".join(search_terms),
            "n_terms":      len(search_terms),
            "user_hash":    _get_user_token(),
        }
        write_header = not os.path.exists(_LOG_FILE)
        with _LOG_LOCK:
            with open(_LOG_FILE, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=_LOG_COLS)
                if write_header:
                    w.writeheader()
                w.writerow(row)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Boolean cohort filter helper
# ---------------------------------------------------------------------------
def _apply_boolean_filter(matched_ids, subject_term_map, must_have, must_not_have, logic):
    """Filter matched_ids by per-term must-have / must-not-have logic.

    Args:
        matched_ids      : iterable of patient IDs to filter
        subject_term_map : {pid: set(original_search_terms_matched)}
        must_have        : list/set of terms the patient MUST have matched
        must_not_have    : list/set of terms the patient must NOT have matched
        logic            : "ANY" or "ALL" — how must_have terms are combined

    Returns:
        set of patient IDs passing the filter
    """
    must_have     = set(must_have or [])
    must_not_have = set(must_not_have or [])
    if not must_have and not must_not_have:
        return set(matched_ids)
    result = set()
    for pid in matched_ids:
        terms = subject_term_map.get(pid, set())
        if must_not_have and (terms & must_not_have):
            continue
        if must_have:
            if logic == "ANY" and not (terms & must_have):
                continue
            if logic == "ALL" and not must_have.issubset(terms):
                continue
        result.add(pid)
    return result


def _build_subject_term_map(column_hits):
    """Invert column_hits to get {pid: set(original_search_terms_matched)}."""
    subject_term_map: dict = {}
    for _cn, _info in column_hits.items():
        for _pid in _info["subjects"]:
            subject_term_map.setdefault(_pid, set()).update(_info["terms"])
    return subject_term_map


@st.cache_resource(show_spinner="Loading data…")
def _load_pkl(path: str):
    """Load the ML pickle once at startup and keep it in memory across reruns."""
    with open(path, "rb") as f:
        return _pkl.load(f)


@st.cache_resource(show_spinner="Indexing data…")
def _build_search_index(path: str):
    """Pre-compute per-column lowercase string arrays and a PID array.

    Called once per pickle path; result lives in RAM for the session.
    Eliminates repeated astype/lower/strip conversions during every search.
    """
    df = _load_pkl(path)
    pid_col = next(
        (c for c in df.columns if re.search(r"pregnancy.?id", c, re.I)), None
    )
    pids_arr = (df[pid_col] if pid_col else df.index).astype(str).values
    str_cols = {}
    for col in df.columns:
        if col == pid_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            str_cols[col] = df[col].astype(str).str.lower().str.strip().values
    return pids_arr, str_cols


@st.cache_data(show_spinner=False)
def _cached_search(
    pickle_path: str,
    terms_key: tuple,
    fuzzy: bool,
    fuzzy_threshold: int,
):
    """Run search_cohort_df with pre-built index; cache result by (path, terms).

    Identical search terms → instant return on subsequent runs.
    """
    from ML_1_Subject_search import search_cohort_df
    df = _load_pkl(pickle_path)
    pids_arr, str_cols = _build_search_index(pickle_path)
    return search_cohort_df(
        df,
        list(terms_key),
        fuzzy=fuzzy,
        fuzzy_threshold=fuzzy_threshold,
        _pids_arr=pids_arr,
        _str_cols=str_cols,
    )


@st.cache_data(show_spinner=False)
def _get_pkl_ids(path: str) -> set:
    """Return the set of Pregnancy IDs in the pickle — cached by file path."""
    df = _load_pkl(path)
    if "Pregnancy ID" in df.columns:
        return set(df["Pregnancy ID"].astype(str))
    return set(df.index.astype(str))

# ---------------------------------------------------------------------------
# Importance flag computation (cached in session state; updated on analysis events)
# ---------------------------------------------------------------------------
def _compute_importance_flags(rank_df, top_n: int) -> dict:
    """Return {feature: flag_string} based on top-N overlap between SHAP and MI."""
    top_n = int(top_n)
    shap_col = "SHAP_mean_abs" if "SHAP_mean_abs" in rank_df.columns else None
    mi_col   = "MI"            if "MI"            in rank_df.columns else None
    top_shap = set(rank_df.nlargest(top_n, shap_col)["feature"]) if shap_col else set()
    top_mi   = set(rank_df.nlargest(top_n, mi_col)["feature"])   if mi_col   else set()
    flags: dict = {}
    for feat in rank_df["feature"]:
        in_shap, in_mi = feat in top_shap, feat in top_mi
        if in_shap and in_mi:
            flags[feat] = "🟢 High confidence"
        elif in_shap:
            flags[feat] = "🟡 Model-driven"
        elif in_mi:
            flags[feat] = "🔵 Associated signal"
        else:
            flags[feat] = ""
    return flags


# ---------------------------------------------------------------------------
# Hardcoded file paths (edit here if filenames change; not exposed in main UI)
# ---------------------------------------------------------------------------
_PICKLE_FILE = "PBDBfinal_ready_forML_IHCP_paper3.pkl"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PeriMiner", page_icon="🔬", layout="wide")

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
for k, v in {
    # Single-cohort (original keys)
    "search_results":    None,   # 5-tuple from search_cohort()
    "last_terms":        None,
    "ml_results":        None,   # (rank_df, cv_results)
    "cv_override":       None,   # cv_results from real-time Top-N recompute
    "cohort_ids_used":   None,
    "Xn": None, "y": None,       # preprocessed data for Cross Validation recompute
    "n_pos": None, "n_neg": None,
    "removed_feats":       set(),  # features the expert has chosen to remove (session)
    "pending_removed":     set(),  # features ticked in Remove column, not yet Recomputed
    "pinned_exclusions":   set(),  # features permanently excluded across searches
    "committed_exclusions": set(), # exclusions accumulated across re-ranks; survive until Reset/new search
    "name_matched_cols":   set(),  # cols excluded at analysis time due to definitional name-match
    "importance_flags":    {},     # {feature: flag_string}; updated only on analysis events
    "subject_cols":      None,   # precomputed inversion: subject → set of matched columns
    # Boolean term filter (single cohort)
    "subject_term_map":    None,   # {pid: set(original_terms_matched)}
    "must_have_terms":     [],
    "must_not_have_terms": [],
    "match_logic":         "ANY",
    # Dual-cohort mode
    "dual_mode_active":    False,  # True when dual cohort analysis was run
    "cohort_b_ids_used":   None,
    "search_results_b":    None,
    "last_terms_b":        None,
    "subject_cols_b":      None,
    "subject_term_map_b":  None,
    "must_have_terms_b":   [],
    "must_not_have_terms_b": [],
    "match_logic_b":       "ANY",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Sidebar  — analysis hyperparameters only
# ---------------------------------------------------------------------------
with st.sidebar:
    with st.expander("⚙ Advanced Settings", expanded=False):
        # ── Search ───────────────────────────────────────────────────────────
        st.caption("**Search**")
        use_fuzzy = st.checkbox(
            "Fuzzy + stem search", value=True,
            help="Click to catch misspellings and stem expansions in term search.",
        )
        fuzzy_threshold = st.slider(
            "Fuzzy threshold", 70, 99, 85, 1,
            disabled=not use_fuzzy,
            help="Strictness of spelling correction, lower more permissive, **suggest 85.",
        )

        # ── Analysis thresholds ─────────────────────────────────────────────
        st.caption("**Analysis thresholds**")
        min_binary_count = st.number_input(
            "Min binary count", value=30, min_value=1,
            help=(
                "Drop feature with fewer than than this many"
                "subjects. **Suggested: 10** (stable minimum). "
                "Higher if noisey features on the top (20–30)."
                "Lower for very rare conditions (<200 positives)."
            ),
        )
        directional_margin = st.number_input(
            "Directional margin", value=0.01, min_value=0.0, format="%.4f",
            help=(
                "Minimum differnce (by either mean or prevalance) to keep in analysis"
                "**0.002 very permissive; only removes true zero-signal"
                "0.01 is a reasonable tighter option."
            ),
        )

        # ── Ranking quality ──────────────────────────────────────────────────
        st.caption("**Ranking quality**")
        use_cv_shap = st.checkbox(
            "Five-fold cross-validated ranking",
            value=False,
            help=(
                "Averages 5 models for more stable rankings. "
                "Best for refined analyses or rarer variables. ~5× slower."
            ),
        )
        top_n_init = st.slider(
            "Top N features for Cross Validation",
            min_value=5, max_value=500, value=50, step=5,
            key="top_n_init",
            help=(
                "Number of top-ranked features passed to the performance evaluation. "
                "**Suggested: 50** for most analyses. Use fewer for small cohorts (<500 subjects)."
            ),
        )

        # ── Performance evaluation ───────────────────────────────────────────
        st.caption("**Model performance**")
        eval_mode = st.radio(
            "Evaluation mode",
            ["80/20 split", "5-fold CV", "Skip"],
            index=0,
            horizontal=True,
            help=(
                "**80/20 split**: Fast single train/test split — good for exploration.\n\n"
                "**5-fold CV**: More thorough, averages 5 folds. Slower.\n\n"
                "**Skip**: No performance evaluation — fastest initial run."
            ),
        )

        # ── File paths ──────────────────────────────────────────────────────
        st.caption("**File paths**")
        pickle_file = st.text_input("ML pickle file", value=_PICKLE_FILE, key="si_pkl")


# ---------------------------------------------------------------------------
# Main panel  — Mode toggle
# ---------------------------------------------------------------------------
st.title("PeriMiner — Cohort Discovery & Feature Ranking")

app_mode = st.radio(
    "Mode",
    ["Single cohort vs. rest", "Compare two cohorts"],
    horizontal=True,
    key="app_mode",
    help=(
        "**Single cohort vs. rest**: Define one cohort; ML compares it against all "
        "other subjects in the database.\n\n"
        "**Compare two cohorts**: Define Cohort A and Cohort B separately; ML "
        "compares A directly against B."
    ),
)
single_mode = (app_mode == "Single cohort vs. rest")

# ============================================================================
# SINGLE COHORT MODE
# ============================================================================
if single_mode:

    # ── Step 1 ──────────────────────────────────────────────────────────────
    label_name = st.text_input(
        "Cohort label",
        value="TARGET",
        placeholder="e.g. GDM, Hypertension, PTB",
        help="Name your cohort to label outputs (optional).",
    )

    st.header("Step 1 — Search Peribank Patient Records for Term(s) That Define Your Cohort of Interest")
    terms_input = st.text_area(
        "Search terms (one per line):",
        placeholder="e.g.\npreeclampsia\nlength of Stay (days)\nBen Taub",
        height=120,
        help="Case-insensitive. Will search entire record to find these terms.",
    )
    st.caption("Ctrl + Enter to submit")
    search_terms = [t.strip() for t in terms_input.splitlines() if t.strip()]

    if search_terms:
        st.info(f"**{len(search_terms)} term(s):** {', '.join(search_terms)}")
    else:
        st.warning("Enter at least one search term.")

    # Detect stale results
    if search_terms != st.session_state.last_terms and st.session_state.search_results is not None:
        st.info("Search terms changed — click **Search Database** to refresh.")

    # Search button
    search_clicked = st.button("Search Database", disabled=not bool(search_terms))

    if search_clicked and search_terms:
        if not os.path.exists(pickle_file):
            st.error(f"Pickle file not found: {pickle_file}"); st.stop()

        with st.spinner("Searching database…"):
            try:
                results = _cached_search(
                    pickle_file,
                    tuple(sorted(search_terms)),
                    fuzzy=use_fuzzy,
                    fuzzy_threshold=int(fuzzy_threshold),
                )
            except Exception:
                st.error(f"Search failed:\n```\n{traceback.format_exc()}\n```"); st.stop()

        st.session_state.search_results  = results
        st.session_state.last_terms      = search_terms
        # Precompute subject→columns inversion once so it isn't rebuilt on every rerun
        _col_hits = results[4]
        _sc: dict = {}
        for _cn, _info in _col_hits.items():
            for _pid in _info["subjects"]:
                _sc.setdefault(_pid, set()).add(_cn)
        st.session_state.subject_cols = _sc
        # Precompute subject→terms inversion for boolean filter
        st.session_state.subject_term_map = _build_subject_term_map(_col_hits)
        # Reset boolean filter selections when search changes
        st.session_state.must_have_terms     = []
        st.session_state.must_not_have_terms = []
        st.session_state.match_logic         = "ANY"
        _log_event("search", label_name, search_terms)
        st.session_state.ml_results           = None
        st.session_state.cv_override          = None
        st.session_state.cohort_ids_used      = None
        st.session_state.removed_feats        = set()
        st.session_state.pending_removed      = set()
        st.session_state.committed_exclusions = set()
        st.session_state.importance_flags     = {}
        st.session_state.Xn                   = None
        st.session_state.y                    = None
        st.session_state.n_pos                = None
        st.session_state.n_neg                = None

    # ── Step 1.5 ────────────────────────────────────────────────────────────
    if st.session_state.search_results is not None:
        matched_ids, outlist, firstline, match_scores, column_hits = \
            st.session_state.search_results

        st.divider()
        st.header("PeriBank Found Subject Selector")

        if not column_hits:
            st.warning("No matches found. Try different terms."); st.stop()

        with st.expander("How to use this table (explanation)", expanded=False):
            st.markdown(
                "This table shows where your search terms were found in the PeriBank records. "
                "PeriBank is organised into sections. For example, if you searched for "
                "<code>anemia</code>, you might get a result like "
                "<code>MatComorbid__Other(details)anemia</code>. "
                "The section before <code>__</code> is the PeriBank section "
                "(<code>MatComorbid</code> = maternal comorbidity recorded for that pregnancy). "
                "The part after <code>__</code> is the specific field or box within that section "
                "(<code>Other(details)</code> is the free-text details box, and <code>anemia</code> "
                "is the matched value inside it).\n\n"
                "A term like *anemia* will likely appear in many sections — that is expected. "
                "**Where** it is coded defines your research question. For example:\n\n"
                "- Including only the **PriorPreg** section selects pregnancies where anemia was "
                "recorded in a *prior* pregnancy.\n"
                "- Including **Antepartum**, **Intrapartum**, and **Postpartum** sections selects "
                "anemia recorded during the *current* pregnancy.\n\n"
                "Check the rows that are relevant to your design. Unchecked rows are excluded from "
                "the cohort.",
                unsafe_allow_html=True,
            )

        col_rows = [
            {
                "Include":                    False,
                "Section of PeriBank Record": (cn.split("__")[0] if "__" in cn else "Other"),
                "Feature in PeriBank Record": cn,
                "Matched terms":              ", ".join(sorted(info["terms"])),
                "Subjects":                   len(info["subjects"]),
            }
            for cn, info in sorted(column_hits.items(),
                                    key=lambda kv: len(kv[1]["subjects"]), reverse=True)
            if len(info["subjects"]) > 3
        ]

        try:
            edited = st.data_editor(
                pd.DataFrame(col_rows),
                column_config={
                    "Include":                    st.column_config.CheckboxColumn("Include", default=False),
                    "Section of PeriBank Record": st.column_config.TextColumn("Section of PeriBank Record", disabled=True),
                    "Feature in PeriBank Record": st.column_config.TextColumn("Feature in PeriBank Record", disabled=True),
                    "Matched terms":              st.column_config.TextColumn("Matched terms",  disabled=True),
                    "Subjects":                   st.column_config.NumberColumn("Subjects",     disabled=True),
                },
                hide_index=True, use_container_width=True, key="col_filter",
            )
            included_cols = set(edited.loc[edited["Include"], "Feature in PeriBank Record"].tolist())
        except Exception:
            st.warning("Checkbox fallback active.")
            included_cols = set()
            for r in col_rows:
                if st.checkbox(
                    f"{r['Section of PeriBank Record']} — {r['Feature in PeriBank Record']} ({r['Subjects']} subjects)",
                    value=False, key=f"chk_{r['Feature in PeriBank Record']}",
                ):
                    included_cols.add(r["Feature in PeriBank Record"])

        # Precomputed subject → columns inversion
        subject_cols = st.session_state.subject_cols or {}
        col_filtered_ids = (
            {pid for pid in matched_ids if subject_cols.get(pid, set()) & included_cols}
            if included_cols else set()
        )

        # Columns whose NAME matched the search term — excluded from ML ranking
        name_matched_cols = {
            cn for cn, info in column_hits.items()
            if info.get("name_match") and cn in included_cols
        }

        ca, cb = st.columns(2)
        ca.metric("Columns included", f"{len(included_cols)} / {len(column_hits)}")
        cb.metric("Subjects (column filter)", f"{len(col_filtered_ids):,}")

        if not col_filtered_ids:
            st.warning("No subjects remain — include at least one column.")

        # ── Cohort filter (second) ───────────────────────────────────────────
        all_found_terms = sorted({t for info in column_hits.values() for t in info["terms"]})
        subject_term_map = st.session_state.subject_term_map or {}
        boolean_ids = col_filtered_ids  # default: no term filter applied

        with st.expander("Cohort Filter — optional further refinement (Can leave blank - see explanation)", expanded=False):
            if len(all_found_terms) >= 1:
                st.markdown(
                    "Explanation. This is only if you want to further refine your cohort. For example, if "
                    "you wanted 'gestational diabetes' but only wanted to include patients that were not taking "
                    "'metformin'. You would enter both search terms in step 1. Then, in this section, add "
                    "'gestational diabetes' to include and 'metformin' to exlclue."
                )
                bf_col1, bf_col2 = st.columns(2)
                with bf_col1:
                    must_have_sel = st.multiselect(
                        "Must include — patient matched at least one of these:",
                        options=all_found_terms,
                        default=[t for t in st.session_state.must_have_terms if t in all_found_terms],
                        key="must_have_ms",
                        help="Leave empty to include all patients regardless of which terms matched.",
                    )
                with bf_col2:
                    must_not_have_sel = st.multiselect(
                        "Must exclude — remove patients who matched any of these:",
                        options=all_found_terms,
                        default=[t for t in st.session_state.must_not_have_terms if t in all_found_terms],
                        key="must_not_have_ms",
                        help="Patients with ANY of these terms matched will be removed from the cohort.",
                    )

                # Conflict check — same term in both boxes
                conflict = set(must_have_sel) & set(must_not_have_sel)
                if conflict:
                    st.warning(
                        f"The following term(s) appear in both boxes and will be treated as Must Include: "
                        f"{', '.join(sorted(conflict))}"
                    )
                    must_not_have_sel = [t for t in must_not_have_sel if t not in conflict]

                match_logic = "ANY"

                # Persist selections
                st.session_state.must_have_terms     = must_have_sel
                st.session_state.must_not_have_terms = must_not_have_sel
                st.session_state.match_logic         = match_logic

                # Apply boolean filter on column-filtered subjects
                boolean_ids = _apply_boolean_filter(
                    col_filtered_ids, subject_term_map,
                    must_have_sel, must_not_have_sel, match_logic,
                )
            else:
                st.caption("No search terms available to filter on.")

        # Intersect with pkl to ensure counts match what the analysis will actually use
        filtered_ids = boolean_ids & _get_pkl_ids(pickle_file)

        st.metric("Effective cohort (in ML dataset)", f"{len(filtered_ids):,} subjects")

        if not filtered_ids:
            st.warning("No subjects remain after all filters.")

        # ── Step 2 ───────────────────────────────────────────────────────────
        st.divider()
        st.header("Step 2 — Run Analysis")

        run_clicked = st.button(
            "Run Analysis", type="primary", disabled=(not filtered_ids),
            help="Ranks all features",
        )

        if run_clicked and filtered_ids:
            if not os.path.exists(pickle_file):
                st.error(f"Pickle not found: {pickle_file}"); st.stop()
            try:
                from ML_2_most_unique import (run_analysis, get_preprocessed_data,
                                              get_raw_data,
                                              run_cv_from_cache, run_single_split_from_cache)
            except ImportError as e:
                st.error(f"Import error: {e}"); st.stop()

            _spinner_msg = (
                "Analysing your cohort — cross-validated ranking enabled, this may take a few minutes"
                if use_cv_shap else
                "Analysing your cohort — this could take about a minute…"
            )
            with st.spinner(_spinner_msg):
                try:
                    rank_df, _ = run_analysis(
                        cohort_ids=filtered_ids,
                        pickle_file=pickle_file,
                        search_terms=search_terms,
                        label_name=label_name,
                        min_binary_count=int(min_binary_count),
                        directional_margin=float(directional_margin),
                        top_n=int(top_n_init),
                        save_csv=False,
                        cv_shap=bool(use_cv_shap),
                        compute_extras=False,
                        run_cv=False,
                        exclude_cols=name_matched_cols | _section_excluded_cols(pickle_file),
                    )
                    Xn, y, n_pos, n_neg = get_preprocessed_data()
                    # Performance evaluation
                    if eval_mode == "5-fold CV":
                        cv_results = run_cv_from_cache(
                            Xn, y, rank_df, int(top_n_init), n_pos, n_neg)
                    elif eval_mode == "80/20 split":
                        cv_results = run_single_split_from_cache(
                            Xn, y, rank_df, int(top_n_init), n_pos, n_neg)
                    else:
                        cv_results = None
                    if cv_results is None:
                        cv_results = {
                            "roc_auc_mean": float("nan"), "roc_auc_std": float("nan"),
                            "pr_auc_mean": float("nan"), "accuracy_mean": float("nan"),
                            "sens_at_90spec": float("nan"), "spec_at_90sens": float("nan"),
                            "top_n": int(top_n_init), "n_positive": n_pos,
                            "n_total": n_pos + n_neg,
                            "roc_fpr": [], "roc_tpr": [],
                            "roc_fpr_band": [], "roc_tpr_band_mean": [],
                            "roc_tpr_band_std": [],
                            "_cv_skipped": True,
                        }
                    _log_event("analysis", label_name, search_terms)
                except Exception:
                    st.error(f"Analysis failed:\n```\n{traceback.format_exc()}\n```"); st.stop()

            st.session_state.ml_results           = (rank_df, cv_results)
            st.session_state.cv_override          = None
            st.session_state.cohort_ids_used      = filtered_ids
            st.session_state.dual_mode_active     = False
            st.session_state.cohort_b_ids_used    = None
            st.session_state.Xn                   = Xn
            st.session_state.y                    = y
            st.session_state.n_pos                = n_pos
            st.session_state.n_neg                = n_neg
            st.session_state.xf_raw = _xf_raw_indexed(pickle_file, get_raw_data())
            st.session_state.removed_feats        = set()
            st.session_state.pending_removed      = set()
            st.session_state.committed_exclusions = set()
            st.session_state.name_matched_cols    = name_matched_cols
            st.session_state.importance_flags     = _compute_importance_flags(
                rank_df, cv_results.get("top_n", top_n_init))

            st.success("Analysis complete — explore results in the tabs below.")


# ============================================================================
# DUAL COHORT MODE
# ============================================================================
else:
    st.header("Compare Two Cohorts")
    st.caption(
        "Define two cohorts with separate searches. The ML will compare Cohort A "
        "directly against Cohort B (instead of comparing against everyone else)."
    )

    # We need pickle_file — set default if sidebar hasn't initialised it
    if "si_pkl" not in st.session_state:
        pickle_file = _PICKLE_FILE

    # ── Helper: render one search+filter panel ────────────────────────────
    def _render_cohort_panel(suffix: str, default_label: str):
        """Render a single cohort search + boolean filter + column filter panel.
        suffix: "a" or "b" — used to namespace session state and widget keys.
        Returns (cohort_label, filtered_ids, name_matched_cols_set, search_terms_list).
        """
        sr_key      = f"search_results_{suffix}"
        lt_key      = f"last_terms_{suffix}"
        sc_key      = f"subject_cols_{suffix}"
        stm_key     = f"subject_term_map_{suffix}"
        mh_key      = f"must_have_terms_{suffix}"
        mnh_key     = f"must_not_have_terms_{suffix}"
        ml_key      = f"match_logic_{suffix}"

        cohort_label = st.text_input(
            "Cohort label",
            value=default_label,
            key=f"label_{suffix}",
        )
        terms_input = st.text_area(
            "Search terms (one per line):",
            placeholder="e.g.\npreeclampsia\nHELLP",
            height=100,
            key=f"terms_input_{suffix}",
            help="Case-insensitive. Partial string matching — a subject is included if ANY term matches their record.",
        )
        st.caption("Ctrl + Enter to submit")
        search_terms_panel = [t.strip() for t in terms_input.splitlines() if t.strip()]

        if search_terms_panel:
            st.info(f"**{len(search_terms_panel)} term(s):** {', '.join(search_terms_panel)}")

        search_clicked_panel = st.button(
            f"Search Database", key=f"search_btn_{suffix}",
            disabled=not bool(search_terms_panel),
        )

        if search_clicked_panel and search_terms_panel:
            if not os.path.exists(pickle_file):
                st.error(f"Pickle file not found: {pickle_file}")
                return cohort_label, set(), set(), search_terms_panel
            try:
                from ML_1_Subject_search import search_cohort_df
            except ImportError as e:
                st.error(f"Import error: {e}")
                return cohort_label, set(), set(), search_terms_panel

            with st.spinner("Searching…"):
                try:
                    results = search_cohort_df(
                        _load_pkl(pickle_file), search_terms_panel,
                        fuzzy=use_fuzzy,
                        fuzzy_threshold=int(fuzzy_threshold),
                    )
                except Exception:
                    st.error(f"Search failed:\n```\n{traceback.format_exc()}\n```")
                    return cohort_label, set(), set(), search_terms_panel

            st.session_state[sr_key] = results
            st.session_state[lt_key] = search_terms_panel
            _col_hits = results[4]
            _sc: dict = {}
            for _cn, _info in _col_hits.items():
                for _pid in _info["subjects"]:
                    _sc.setdefault(_pid, set()).add(_cn)
            st.session_state[sc_key]  = _sc
            st.session_state[stm_key] = _build_subject_term_map(_col_hits)
            st.session_state[mh_key]  = []
            st.session_state[mnh_key] = []
            st.session_state[ml_key]  = "ANY"

        # ── Show filter UI if results exist ──────────────────────────────────
        if st.session_state.get(sr_key) is None:
            st.caption("Search to define this cohort.")
            return cohort_label, set(), set(), search_terms_panel

        _matched_ids, _, _, _, _column_hits = st.session_state[sr_key]
        _subject_term_map = st.session_state.get(stm_key) or {}

        # ── Column filter (first) ─────────────────────────────────────────────
        _col_rows = [
            {
                "Include":       False,
                "Feature in PeriBank Record": cn,
                "Matched terms": ", ".join(sorted(info["terms"])),
                "Subjects":      len(info["subjects"]),
            }
            for cn, info in sorted(
                _column_hits.items(),
                key=lambda kv: len(kv[1]["subjects"]), reverse=True,
            )
            if len(info["subjects"]) > 3
        ]
        if _col_rows:
            try:
                _edited_cols = st.data_editor(
                    pd.DataFrame(_col_rows),
                    column_config={
                        "Include":                    st.column_config.CheckboxColumn("Include", default=False),
                        "Feature in PeriBank Record": st.column_config.TextColumn("Feature in PeriBank Record", disabled=True),
                        "Matched terms":              st.column_config.TextColumn("Matched terms",  disabled=True),
                        "Subjects":                   st.column_config.NumberColumn("Subjects",     disabled=True),
                    },
                    hide_index=True, use_container_width=True, key=f"col_filter_{suffix}",
                )
                _included_cols = list(_edited_cols.loc[_edited_cols["Include"], "Feature in PeriBank Record"].tolist())
            except Exception:
                st.warning("Checkbox fallback active.")
                _included_cols = []
        else:
            _included_cols = []

        _subject_cols = st.session_state.get(sc_key) or {}
        _col_filtered_ids = (
            {pid for pid in _matched_ids if _subject_cols.get(pid, set()) & set(_included_cols)}
            if _included_cols else _matched_ids
        )
        st.metric("Subjects (column filter)", f"{len(_col_filtered_ids):,}")

        # ── Cohort filter (second) ────────────────────────────────────────────
        _all_terms = sorted({t for info in _column_hits.values() for t in info["terms"]})
        _boolean_ids = _col_filtered_ids  # default: no term filter applied

        with st.expander("Cohort Filter — optional further refinement (leave blank to include all)", expanded=False):
            if _all_terms:
                _mh = st.multiselect(
                    "Must include:",
                    options=_all_terms,
                    default=[t for t in st.session_state.get(mh_key, []) if t in _all_terms],
                    key=f"mh_{suffix}",
                )
                _mnh = st.multiselect(
                    "Must exclude:",
                    options=_all_terms,
                    default=[t for t in st.session_state.get(mnh_key, []) if t in _all_terms],
                    key=f"mnh_{suffix}",
                )
                _conflict = set(_mh) & set(_mnh)
                if _conflict:
                    st.warning(f"In both boxes (treated as Must Include): {', '.join(sorted(_conflict))}")
                    _mnh = [t for t in _mnh if t not in _conflict]

                _logic = "ANY"

                st.session_state[mh_key]  = _mh
                st.session_state[mnh_key] = _mnh
                st.session_state[ml_key]  = _logic

                _boolean_ids = _apply_boolean_filter(
                    _col_filtered_ids, _subject_term_map, _mh, _mnh, _logic,
                )
            else:
                st.caption("No search terms available to filter on.")

        _filtered_ids = _boolean_ids & _get_pkl_ids(pickle_file)
        _name_matched = {
            cn for cn, info in _column_hits.items()
            if info.get("name_match") and cn in set(_included_cols)
        }

        st.metric("Effective cohort", f"{len(_filtered_ids):,}")
        return cohort_label, _filtered_ids, _name_matched, search_terms_panel

    # ── Two panels side by side ──────────────────────────────────────────────
    pan_a, pan_b = st.columns(2)

    with pan_a:
        st.subheader("Cohort A  (positive class)")
        label_a, cohort_a_ids, name_matched_a, terms_a = _render_cohort_panel("a", "COHORT_A")

    with pan_b:
        st.subheader("Cohort B  (comparison class)")
        label_b, cohort_b_ids, name_matched_b, terms_b = _render_cohort_panel("b", "COHORT_B")

    # ── Overlap handling — subjects in both cohorts are included in both ────
    if cohort_a_ids and cohort_b_ids:
        overlap = cohort_a_ids & cohort_b_ids

        ov_a, ov_b, ov_o = st.columns(3)
        ov_a.metric("Cohort A", f"{len(cohort_a_ids):,} subjects")
        ov_b.metric("Cohort B", f"{len(cohort_b_ids):,} subjects")
        if overlap:
            ov_o.metric("Overlap (in both)", f"{len(overlap):,}")

    # ── Run Comparison ───────────────────────────────────────────────────────
    st.divider()
    st.header("Step 2 — Run Comparison")

    both_ready = bool(cohort_a_ids) and bool(cohort_b_ids)
    run_comp = st.button(
        "Run Comparison ML", type="primary", disabled=(not both_ready),
        help=(
            "Run LightGBM + SHAP to find features that distinguish Cohort A from Cohort B."
            if both_ready else
            "Define both cohorts first."
        ),
    )

    if run_comp and both_ready:
        if not os.path.exists(pickle_file):
            st.error(f"Pickle not found: {pickle_file}"); st.stop()
        try:
            from ML_2_most_unique import (run_analysis, get_preprocessed_data,
                                          get_raw_data,
                                          run_cv_from_cache, run_single_split_from_cache)
        except ImportError as e:
            st.error(f"Import error: {e}"); st.stop()

        _exclude_dual = (name_matched_a | name_matched_b) | _section_excluded_cols(pickle_file)
        _spinner_msg = (
            "Comparing cohorts — cross-validated ranking enabled, this may take a few minutes"
            if use_cv_shap else
            "Comparing cohorts — this could take about a minute…"
        )
        with st.spinner(_spinner_msg):
            try:
                rank_df, _ = run_analysis(
                    cohort_ids=cohort_a_ids,
                    pickle_file=pickle_file,
                    search_terms=list(set(terms_a) | set(terms_b)),
                    label_name=label_a,
                    min_binary_count=int(min_binary_count),
                    directional_margin=float(directional_margin),
                    top_n=int(top_n_init),
                    save_csv=False,
                    cv_shap=bool(use_cv_shap),
                    compute_extras=False,
                    run_cv=False,
                    exclude_cols=_exclude_dual,
                    control_ids=cohort_b_ids,
                )
                Xn, y, n_pos, n_neg = get_preprocessed_data()
                if eval_mode == "5-fold CV":
                    cv_results = run_cv_from_cache(
                        Xn, y, rank_df, int(top_n_init), n_pos, n_neg)
                elif eval_mode == "80/20 split":
                    cv_results = run_single_split_from_cache(
                        Xn, y, rank_df, int(top_n_init), n_pos, n_neg)
                else:
                    cv_results = None
                if cv_results is None:
                    cv_results = {
                        "roc_auc_mean": float("nan"), "roc_auc_std": float("nan"),
                        "pr_auc_mean": float("nan"), "accuracy_mean": float("nan"),
                        "sens_at_90spec": float("nan"), "spec_at_90sens": float("nan"),
                        "top_n": int(top_n_init), "n_positive": n_pos,
                        "n_total": n_pos + n_neg,
                        "roc_fpr": [], "roc_tpr": [],
                        "roc_fpr_band": [], "roc_tpr_band_mean": [],
                        "roc_tpr_band_std": [],
                        "_cv_skipped": True,
                    }
                _log_event("comparison", f"{label_a}_vs_{label_b}",
                           list(set(terms_a) | set(terms_b)))
            except Exception:
                st.error(f"Comparison failed:\n```\n{traceback.format_exc()}\n```"); st.stop()

        st.session_state.ml_results        = (rank_df, cv_results)
        st.session_state.cv_override       = None
        st.session_state.cohort_ids_used   = cohort_a_ids
        st.session_state.cohort_b_ids_used = cohort_b_ids
        st.session_state.dual_mode_active  = True
        st.session_state.Xn                = Xn
        st.session_state.y                 = y
        st.session_state.n_pos             = n_pos
        st.session_state.n_neg             = n_neg
        st.session_state.xf_raw = _xf_raw_indexed(pickle_file, get_raw_data())
        st.session_state.removed_feats     = set()
        st.session_state.pending_removed   = set()
        st.session_state.name_matched_cols = _exclude_dual
        # Borrow single-cohort label vars for Step 3 display
        label_name = label_a

        st.success(
            f"Comparison complete — {n_pos} in {label_a}, {n_neg} in {label_b}. "
            "Explore results below."
        )


# ============================================================================
# Step 3 — Results  (shared between both modes)
# ============================================================================
if st.session_state.ml_results is not None:

    # Resolve label_name for dual mode (search_terms used in heading only)
    if not single_mode:
        # Reconstruct label from session state
        _la = st.session_state.get("label_a",
              st.session_state.get("si_label_a", "COHORT_A"))
        _lb = st.session_state.get("label_b",
              st.session_state.get("si_label_b", "COHORT_B"))
        label_name   = st.session_state.get("app_label_a", _la)
        search_terms = []
    else:
        _lb = "Remaining"

    _dual = st.session_state.dual_mode_active

    rank_df, cv_base = st.session_state.ml_results
    cv_results = st.session_state.cv_override or cv_base

    st.divider()
    if _dual:
        st.header("Step 3 — Comparison Results")
    else:
        st.header("Step 3 — Results")

    # ---- Real-time Top-N adjustment ----
    with st.expander("Adjust Top N for Performance Evaluation", expanded=False):
        st.markdown(
            "Change how many top-ranked features are used in performance evaluation. "
            "The feature **ranking itself does not change** — only the performance metrics update."
        )
        max_feats  = min(len(rank_df), 300)
        top_n_live = st.slider("Top N", 5, max_feats,
                               value=int(cv_results.get("top_n", 50)),
                               step=5, key="top_n_live")
        recompute  = st.button("Recompute", key="btn_recompute")

        if recompute:
            from ML_2_most_unique import run_cv_from_cache, run_single_split_from_cache
            Xn    = st.session_state.Xn
            y     = st.session_state.y
            n_pos = st.session_state.n_pos
            n_neg = st.session_state.n_neg
            if Xn is not None:
                excl = st.session_state.removed_feats
                active_rank = rank_df[~rank_df["feature"].isin(excl)] if excl else rank_df
                with st.spinner(f"Evaluating top {top_n_live} features…"):
                    if eval_mode == "5-fold CV":
                        new_cv = run_cv_from_cache(Xn, y, active_rank, top_n_live, n_pos, n_neg)
                    else:
                        new_cv = run_single_split_from_cache(Xn, y, active_rank, top_n_live, n_pos, n_neg)
                if new_cv:
                    st.session_state.cv_override = new_cv
                    cv_results = new_cv
                    st.success(
                        f"Updated — top {top_n_live} features: "
                        f"ROC-AUC {new_cv['roc_auc_mean']:.3f}"
                    )
            else:
                st.warning("Data not available — re-run the analysis.")

    # ---- Metric cards ----
    has_change = st.session_state.cv_override is not None

    def _delta(key):
        if not has_change:
            return None
        try:
            v_new = cv_results.get(key)
            v_old = cv_base.get(key)
            if v_new is None or v_old is None:
                return None
            import math
            diff = float(v_new) - float(v_old)
            return None if math.isnan(diff) else format(diff, "+.3f")
        except (TypeError, ValueError):
            return None

    n_total    = cv_results["n_total"]
    prevalence = cv_results["n_positive"] / max(n_total, 1)
    _cv_skipped = cv_results.get("_cv_skipped", False)

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    if _dual:
        c1.metric("Cohort A size", f"{cv_results['n_positive']:,}")
        c2.metric("Cohort B size", f"{n_total - cv_results['n_positive']:,}")
        c3.metric("Total subjects", f"{n_total:,}")
    else:
        c1.metric("Cohort size",    f"{cv_results['n_positive']:,}")
        c2.metric("Total subjects", f"{n_total:,}")
        c3.metric("Prevalence",     f"{prevalence:.1%}")

    def _fmt_cv(val, fmt=":.3f"):
        return "—" if (val is None or (isinstance(val, float) and val != val)) else format(val, fmt[1:])

    if _cv_skipped:
        c4.metric(f"ROC-AUC (top {cv_results.get('top_n','?')})", "—",
                  help="CV was skipped — click 'Recompute Cross Validation' below.")
        c5.metric("PR-AUC",          "—")
        c6.metric("Sens @ 90% Spec", "—")
        c7.metric("Spec @ 90% Sens", "—")
        st.info(
            "Performance evaluation was skipped. "
            "Click **Recompute** (in the panel above) to compute "
            "ROC-AUC, PR-AUC, and sensitivity metrics."
        )
    else:
        c4.metric(
            f"ROC-AUC (top {cv_results.get('top_n','?')})",
            f"{cv_results['roc_auc_mean']:.3f}",
            delta=_delta("roc_auc_mean"),
            help=(f"5-fold CV std: ± {cv_results['roc_auc_std']:.3f}"
                  if not cv_results.get("_single_split") else "Single 80/20 split"),
        )
        c5.metric("PR-AUC",          f"{cv_results['pr_auc_mean']:.3f}",
                  delta=_delta("pr_auc_mean"))
        c6.metric("Sens @ 90% Spec", f"{cv_results.get('sens_at_90spec', 0):.3f}",
                  delta=_delta("sens_at_90spec"))
        c7.metric("Spec @ 90% Sens", f"{cv_results.get('spec_at_90sens', 0):.3f}",
                  delta=_delta("spec_at_90sens"))
        if has_change:
            st.caption("↑↓ Deltas show change from original analysis.")

    # ---- Tabs ----
    tab_mgmt, tab_roc, tab_dist = st.tabs([
        "Feature Management", "ROC Curve", "Feature Detail",
    ])

    # label_name for column headers in rank_df
    _label_for_df = rank_df.columns[rank_df.columns.str.startswith("Mean_")][0].replace("Mean_", "") \
                    if any(rank_df.columns.str.startswith("Mean_")) else "TARGET"
    mean_pos_col = f"Mean_{_label_for_df}"
    mean_neg_col = f"Mean_No{_label_for_df}"

    # Chronological order matching the MedSciNet form (Headings.txt)
    _CAT_ORDER = [
        "MatID", "MatInfo", "MatComorbid", "MatBirthHx", "Harvey",
        "PriorPreg", "PriorPregComplication", "ContraceptiveHx",
        "Prenatal", "ConceptionMedsDetail", "Allergies", "OtherImmunizations",
        "FamHxMat", "FamHxMatSisters", "FamHxMatGrandma",
        "PatInfo", "PatMedsConception", "PatMeds6Mo", "PatComorbid",
        "PatFamOB", "PatFamMedHx", "PatBirthHx",
        "Antepartum", "Intrapartum",
        "Delivery", "Newborn",
        "Postpartum", "PostpartumReadmit",
        "Other",
    ]
    _present_cats = set(rank_df["category"].unique())
    all_cats = [c for c in _CAT_ORDER if c in _present_cats] + \
               sorted(_present_cats - set(_CAT_ORDER))
    all_dirs = ["Higher in cohort", "Lower in cohort"]


    # =========================================================================
    # Tab: Feature Management
    # =========================================================================
    with tab_mgmt:
        with st.expander("How to read this table", expanded=False):
            st.markdown(
                "Each row is a database field ranked by how strongly it distinguishes your cohort "
                "from the rest of PeriBank.\n\n"
                "**Reading a feature name** — e.g. <code>MatComorbid__Other(details)anemia</code>\n\n"
                "- The text before <code>__</code> is the **section** of the PeriBank record "
                "(<code>MatComorbid</code> = maternal comorbidities noted for that pregnancy).\n"
                "- The text after <code>__</code> is the **field or box** within that section "
                "(<code>Other(details)</code> is the free-text details box, <code>anemia</code> "
                "is the matched value within it).\n\n"
                "**Column guide**\n\n"
                "| Column | Meaning |\n"
                "|--------|---------|\n"
                "| **Composite** | Overall importance score (2×SHAP + MI, rank-normalised). Higher = more distinguishing. |\n"
                "| **SHAP** | Mean absolute SHAP value — how much this feature drives the model's prediction. Best for identifying top predictors, but may ignore correlated features. |\n"
                "| **MI** | Mutual Information — measures association independently of the model. Noisier, but catches features SHAP may suppress. |\n"
                "| **Importance** | 🟢 High confidence (top in both SHAP & MI) · 🟡 Model-driven (SHAP only) · 🔵 Associated signal (MI only). |\n"
                "| **Direction** | Whether the feature is higher or lower in your cohort vs the rest. |",
                unsafe_allow_html=True,
            )

        with st.expander("About the Importance column", expanded=False):
            st.markdown(
                "**Focus on SHAP for prediction, focus on MI for feature discovery.**\n\n"
                "Composite is a weighted rank of both (2×SHAP + MI). In general:\n\n"
                "- **SHAP** finds the best predictors but will ignore some highly correlated "
                "features (can and will miss things).\n"
                "- **MI** (Mutual Information) considers all features independently, but is noisier "
                "(may include artifacts/spurious signals) — however it will also surface "
                "potentially very important features that SHAP missed.\n\n"
                "**Importance classification** (top N features by each metric):\n"
                "- 🟢 **High confidence predictor** — in both SHAP and MI top lists\n"
                "- 🟡 **Model-driven predictor** — in SHAP top list only\n"
                "- 🔵 **Associated signal (possible suppressed feature)** — in MI top list only"
            )

        # Build display DataFrame
        _cols = [
            "feature", "category", "direction", "Composite",
            "SHAP_mean_abs", "MI",
        ]
        _cols = [c for c in _cols if c in rank_df.columns]
        mgmt_df = rank_df[_cols].copy()

        # Importance classification — read from session state cache (updated on analysis events only)
        mgmt_df["Importance"] = (
            mgmt_df["feature"]
            .map(st.session_state.get("importance_flags", {}))
            .fillna("")
        )

        # Filter controls
        _default_cats = [c for c in all_cats if c not in _DEFAULT_EXCLUDED_CATS]
        _fc2, _fc3 = st.columns([1, 2])
        with _fc2:
            dir_filter = st.multiselect("Direction", all_dirs, default=all_dirs, key="fm_dir")
        with _fc3:
            cat_filter = st.multiselect(
                "Category", all_cats, default=_default_cats, key="fm_cat",
                help="Some categories are hidden by default. Add them back here if needed.",
            )
            cat_filter = [c for c in all_cats if c in cat_filter]

        _committed = (st.session_state.removed_feats
                      | st.session_state.pinned_exclusions
                      | st.session_state.committed_exclusions)
        view_df = mgmt_df[
            mgmt_df["direction"].isin(dir_filter) &
            mgmt_df["category"].isin(cat_filter) &
            ~mgmt_df["feature"].isin(_committed)
        ]
        n_committed = len(st.session_state.removed_feats)
        st.caption(
            f"Showing **{len(view_df)}** of {len(mgmt_df)} features — "
            f"**{n_committed}** marked for removal"
        )

        # ---- Feature table with per-row Remove checkboxes ----
        _pending_set = st.session_state.pending_removed
        _disp = view_df.reset_index(drop=True).copy()
        _disp.insert(0, "Remove", [f in _pending_set for f in _disp["feature"]])
        _shap_max = (float(_disp["SHAP_mean_abs"].max())
                     if len(_disp) and _disp["SHAP_mean_abs"].max() > 0 else 1.0)

        import hashlib as _hl
        _view_sig = _hl.md5(
            ",".join(_disp["feature"].tolist()).encode()
        ).hexdigest()[:10]
        _pending_sig = _hl.md5(
            ",".join(sorted(_pending_set)).encode()
        ).hexdigest()[:8]
        _editor_key = f"feat_removal_editor_{_view_sig}_{_pending_sig}"

        try:
            _edited = st.data_editor(
                _disp,
                column_config={
                    "Remove":    st.column_config.CheckboxColumn(
                                     "Remove", default=False,
                                     help="Check to exclude this feature, "
                                          "then click Rerun."),
                    "feature":   st.column_config.TextColumn("Feature",   disabled=True),
                    "category":  st.column_config.TextColumn("Category",  disabled=True),
                    "direction": st.column_config.TextColumn("Direction", disabled=True),
                    "Composite": st.column_config.NumberColumn(
                                     "Composite", format="%.3f", disabled=True,
                                     help="Composite ranking score (2×SHAP + MI, "
                                          "rank-normalised)."),
                    "SHAP_mean_abs": st.column_config.ProgressColumn(
                                     "SHAP", min_value=0, max_value=_shap_max,
                                     format="%.4g",
                                     help="Mean absolute SHAP value — primary driver of "
                                          "composite rank. Bar shows the fall-off across features."),
                    "MI":            st.column_config.NumberColumn("MI",        format="%.4g",
                                                                    disabled=True),
                    "Importance":    st.column_config.TextColumn(
                                         "Importance", disabled=True,
                                         help="Based on top-N overlap between SHAP and MI lists."),
                },
                use_container_width=True,
                hide_index=True,
                height=520,
                key=_editor_key,
            )
        except Exception:
            st.warning(traceback.format_exc())
            _edited = _disp

        _pending_removed_de = set(_edited[_edited["Remove"]]["feature"].tolist())
        _hidden_pending = _pending_set - set(view_df["feature"])
        _new_pending = _pending_removed_de | _hidden_pending
        if _new_pending != st.session_state.pending_removed:
            st.session_state.pending_removed = _new_pending
            st.rerun()
        st.session_state.pending_removed = _new_pending
        _pending_removed = st.session_state.pending_removed | (
            st.session_state.removed_feats - set(view_df["feature"])
        )

        # ---- Button bar: Remove · Rerun CV · Re-rank · Reset ----
        _n_pending    = len(st.session_state.pending_removed)
        _cv_excluded  = (st.session_state.removed_feats
                         | st.session_state.pinned_exclusions
                         | st.session_state.committed_exclusions)
        active_rank_cv = rank_df[
            rank_df["direction"].isin(dir_filter) &
            rank_df["category"].isin(cat_filter) &
            ~rank_df["feature"].isin(_cv_excluded)
        ]
        n_cv_excluded   = len(rank_df) - len(active_rank_cv)
        _dir_cat_excluded = set(mgmt_df[
            ~mgmt_df["direction"].isin(dir_filter) | ~mgmt_df["category"].isin(cat_filter)
        ]["feature"])
        _rerank_excluded = _cv_excluded | _dir_cat_excluded
        _mode_label      = "5-fold CV" if eval_mode == "5-fold CV" else "80/20"

        _rcol1, _rcol2, _rcol3, _rcol4 = st.columns([2, 2, 2, 1])
        _n_dir_cat = len(_dir_cat_excluded - st.session_state.removed_feats
                         - st.session_state.pinned_exclusions
                         - st.session_state.committed_exclusions)
        _n_remove_total = _n_pending + _n_dir_cat
        with _rcol1:
            remove_clicked = st.button(
                f"Remove from table ({_n_remove_total})" if _n_remove_total else "Remove from table",
                disabled=(_n_remove_total == 0), key="btn_remove_feats", type="primary",
                help="Commit checked features and any category/direction-filtered features. "
                     "Does not rerun CV.")
        with _rcol2:
            _cv_label = (f"Rerun CV ({_mode_label}, {n_cv_excluded} excluded)"
                         if n_cv_excluded else f"Rerun CV ({_mode_label})")
            rerun_mgmt = st.button(
                _cv_label, disabled=(n_cv_excluded == 0),
                key="btn_rerun_mgmt", type="secondary",
                help="Re-evaluate performance metrics using the committed exclusions.")
        with _rcol3:
            _rerank_label = (
                f"Re-rank ({len(_rerank_excluded)} excluded)"
                if _rerank_excluded else "Re-rank"
            )
            rerank_clicked = st.button(
                _rerank_label, disabled=(not bool(_rerank_excluded)),
                key="btn_rerank", type="secondary",
                help="Re-run full LightGBM+SHAP without removed features (~1 min). "
                     "Produces a new feature ranking.")
        with _rcol4:
            _reset_mgmt = st.button(
                "↺ Reset", key="btn_reset_mgmt",
                help="Restore original analysis — undo all feature removals.",
                disabled=(
                    st.session_state.cv_override is None
                    and not st.session_state.removed_feats
                    and not st.session_state.pending_removed
                    and not st.session_state.committed_exclusions
                ),
            )

        if _reset_mgmt:
            st.session_state.cv_override          = None
            st.session_state.removed_feats        = set()
            st.session_state.pending_removed      = set()
            st.session_state.committed_exclusions = set()
            st.session_state.pop("fm_cat", None)
            st.session_state.pop("fm_dir", None)
            st.session_state.pop("fm_search", None)
            st.rerun()

        if remove_clicked:
            _to_commit = (
                st.session_state.pending_removed | _dir_cat_excluded
            ) - st.session_state.pinned_exclusions
            st.session_state.removed_feats |= _to_commit
            st.session_state.pending_removed = set()
            # Reset category/direction filters — committed features are now
            # hidden via removed_feats regardless of filter state.
            st.session_state.pop("fm_cat", None)
            st.session_state.pop("fm_dir", None)
            st.rerun()

        if rerun_mgmt and n_cv_excluded > 0:
            from ML_2_most_unique import run_cv_from_cache, run_single_split_from_cache
            Xn    = st.session_state.Xn
            y     = st.session_state.y
            n_pos = st.session_state.n_pos
            n_neg = st.session_state.n_neg
            if Xn is None:
                st.warning("Data not available — re-run the analysis.")
            else:
                top_n_cv = int(cv_results.get("top_n", 50))
                with st.spinner(f"Evaluating {len(active_rank_cv)} features…"):
                    if eval_mode == "5-fold CV":
                        new_cv = run_cv_from_cache(Xn, y, active_rank_cv, top_n_cv, n_pos, n_neg)
                    else:
                        new_cv = run_single_split_from_cache(Xn, y, active_rank_cv, top_n_cv, n_pos, n_neg)
                if new_cv:
                    st.session_state.cv_override      = new_cv
                    st.session_state.importance_flags = _compute_importance_flags(
                        rank_df, new_cv.get("top_n", top_n_cv))
                    st.rerun()

        if rerank_clicked and _rerank_excluded:
            try:
                from ML_2_most_unique import (run_analysis, get_preprocessed_data,
                                              get_raw_data,
                                              run_cv_from_cache, run_single_split_from_cache)
            except ImportError as e:
                st.error(f"Import error: {e}"); st.stop()
            _full_exclude = _rerank_excluded | st.session_state.name_matched_cols | _section_excluded_cols(pickle_file)
            with st.spinner("Re-ranking features — this may take about a minute…"):
                try:
                    _rerank_kwargs = dict(
                        cohort_ids=st.session_state.cohort_ids_used,
                        pickle_file=pickle_file,
                        search_terms=search_terms if single_mode else [],
                        label_name=_label_for_df,
                        min_binary_count=int(min_binary_count),
                        directional_margin=float(directional_margin),
                        top_n=int(cv_results.get("top_n", 50)),
                        save_csv=False,
                        cv_shap=bool(use_cv_shap),
                        compute_extras=False,
                        run_cv=False,
                        exclude_cols=_full_exclude,
                    )
                    if _dual and st.session_state.cohort_b_ids_used:
                        _rerank_kwargs["control_ids"] = st.session_state.cohort_b_ids_used
                    rank_df_new, _ = run_analysis(**_rerank_kwargs)
                    Xn_new, y_new, n_pos_new, n_neg_new = get_preprocessed_data()
                    if eval_mode == "5-fold CV":
                        cv_new = run_cv_from_cache(Xn_new, y_new, rank_df_new,
                                                   int(cv_results.get("top_n", 50)),
                                                   n_pos_new, n_neg_new)
                    elif eval_mode == "80/20 split":
                        cv_new = run_single_split_from_cache(Xn_new, y_new, rank_df_new,
                                                             int(cv_results.get("top_n", 50)),
                                                             n_pos_new, n_neg_new)
                    else:
                        cv_new = None
                    if cv_new is None:
                        cv_new = {
                            "roc_auc_mean": float("nan"), "roc_auc_std": float("nan"),
                            "pr_auc_mean": float("nan"), "accuracy_mean": float("nan"),
                            "sens_at_90spec": float("nan"), "spec_at_90sens": float("nan"),
                            "top_n": int(cv_results.get("top_n", 50)),
                            "n_positive": n_pos_new,
                            "n_total": n_pos_new + n_neg_new,
                            "roc_fpr": [], "roc_tpr": [],
                            "roc_fpr_band": [], "roc_tpr_band_mean": [],
                            "roc_tpr_band_std": [],
                            "_cv_skipped": True,
                        }
                except Exception:
                    st.error(f"Re-rank failed:\n```\n{traceback.format_exc()}\n```"); st.stop()
            st.session_state.ml_results           = (rank_df_new, cv_new)
            st.session_state.cv_override          = None
            st.session_state.committed_exclusions |= _rerank_excluded
            st.session_state.removed_feats        = set()
            st.session_state.pending_removed      = set()
            st.session_state.importance_flags     = _compute_importance_flags(
                rank_df_new, cv_new.get("top_n", 50))
            st.session_state.Xn                   = Xn_new
            st.session_state.y                    = y_new
            st.session_state.n_pos                = n_pos_new
            st.session_state.n_neg                = n_neg_new
            st.session_state.xf_raw = _xf_raw_indexed(pickle_file, get_raw_data())
            st.rerun()

        # Download buttons
        st.divider()
        import io as _io
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button(
                "⬇ Download all rankings (CSV)",
                data=rank_df.to_csv(index=False).encode(),
                file_name=f"{_label_for_df}_feature_rankings.csv",
                mime="text/csv",
            )
        with dl2:
            if st.session_state.removed_feats or st.session_state.pinned_exclusions:
                _excl_all = st.session_state.removed_feats | st.session_state.pinned_exclusions
                clean = rank_df[~rank_df["feature"].isin(_excl_all)]
                st.download_button(
                    f"⬇ Download rankings ({len(_excl_all)} excluded, CSV)",
                    data=clean.to_csv(index=False).encode(),
                    file_name=f"{_label_for_df}_clean_rankings.csv",
                    mime="text/csv",
                )
        with dl3:
            try:
                _buf = _io.BytesIO()
                with pd.ExcelWriter(_buf, engine="openpyxl") as _writer:
                    pd.DataFrame([
                        ("Cohort label",   _label_for_df),
                        ("Cohort A size",  cv_results["n_positive"]),
                        ("Cohort B / Total", cv_results["n_total"]),
                        ("ROC-AUC (mean)", round(cv_results["roc_auc_mean"], 4)),
                        ("ROC-AUC (std)",  round(cv_results["roc_auc_std"],  4)),
                        ("PR-AUC",         round(cv_results["pr_auc_mean"],  4)),
                        ("Sens@90%Spec",   round(cv_results.get("sens_at_90spec", 0), 4)),
                        ("Spec@90%Sens",   round(cv_results.get("spec_at_90sens", 0), 4)),
                    ], columns=["Metric", "Value"]).to_excel(
                        _writer, sheet_name="Summary", index=False)
                    rank_df.to_excel(_writer, sheet_name="Rankings", index=False)
                st.download_button(
                    "⬇ Download as Excel (.xlsx)",
                    data=_buf.getvalue(),
                    file_name=f"{_label_for_df}_PeriMiner.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception:
                st.caption("Excel export unavailable (openpyxl not installed).")

        # Pinned exclusions
        _n_pinned = len(st.session_state.pinned_exclusions)
        with st.expander(f"📌 Pinned Exclusions ({_n_pinned} feature{'s' if _n_pinned != 1 else ''})"):
            st.caption(
                "**Remove** hides a feature from this analysis. It comes back if you click ↺ Reset"
                 "or input new search terms.\n\n"
                "• **Pin** permanently excludes a feature for the rest of this session, "
                "Useful when you are certain a feature should be excluded without having to re-tick it each time."
            )
            pc1, pc2 = st.columns([1, 1])
            with pc1:
                if st.button("📌 Pin current removals", key="btn_pin"):
                    st.session_state.pinned_exclusions |= _pending_removed
                    st.rerun()
            with pc2:
                if _n_pinned and st.button("🗑 Clear all pinned", key="btn_clear_pinned"):
                    st.session_state.pinned_exclusions = set()
                    st.rerun()
            if st.session_state.pinned_exclusions:
                for _pf in sorted(st.session_state.pinned_exclusions):
                    _pc1, _pc2 = st.columns([5, 1])
                    _pc1.text(_pf)
                    if _pc2.button("✕", key=f"unpin_{_pf}"):
                        st.session_state.pinned_exclusions.discard(_pf)
                        st.rerun()
            else:
                st.caption("No features pinned yet.")


    # =========================================================================
    # Tab: ROC Curve
    # =========================================================================
    with tab_roc:
        try:
            import plotly.graph_objects as go
            import numpy as _np
            fpr = cv_results.get("roc_fpr", [0, 1])
            tpr = cv_results.get("roc_tpr", [0, 1])
            fig = go.Figure()
            if "roc_fpr_band" in cv_results:
                _bf    = cv_results["roc_fpr_band"]
                _bm    = _np.array(cv_results["roc_tpr_band_mean"])
                _bs    = _np.array(cv_results["roc_tpr_band_std"])
                _upper = _np.clip(_bm + _bs, 0, 1).tolist()
                _lower = _np.clip(_bm - _bs, 0, 1).tolist()
                fig.add_trace(go.Scatter(
                    x=_bf, y=_upper, mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=_bf, y=_lower, fill="tonexty",
                    fillcolor="rgba(232,92,48,0.15)", line=dict(width=0),
                    name="±1 SD (CV folds)", hoverinfo="skip",
                ))
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"LightGBM  (AUC = {cv_results['roc_auc_mean']:.3f})",
                line=dict(color="#e85c30", width=2.5),
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random (AUC = 0.500)",
                line=dict(color="#aaa", width=1.5, dash="dash"),
            ))
            _roc_title = (
                f"ROC Curve — {_label_for_df} vs {_lb}"
                if _dual else
                f"ROC Curve — {_label_for_df} ({'80/20 split' if cv_results.get('_single_split') else '5-fold pooled'})"
            )
            fig.update_layout(
                title=_roc_title,
                xaxis_title="False Positive Rate (1 − Specificity)",
                yaxis_title="True Positive Rate (Sensitivity)",
                legend=dict(x=0.55, y=0.1), height=480,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Sensitivity at 90% specificity: **{cv_results.get('sens_at_90spec',0):.3f}**  —  "
                f"Specificity at 90% sensitivity: **{cv_results.get('spec_at_90sens',0):.3f}**"
            )
        except Exception:
            st.warning(traceback.format_exc())

    # =========================================================================
    # Tab: Feature Detail
    # =========================================================================
    with tab_dist:
        with st.expander("How to use Feature Detail", expanded=False):
            st.markdown(
                "Select any feature from the dropdown to inspect its statistics and distribution "
                "in your cohort versus the comparison group.\n\n"
                "**Binary features** (yes/no fields): A grouped bar chart shows the prevalence "
                "(proportion of subjects with a value of 1) in your cohort vs the comparison group. "
                "The odds ratio quantifies how much more likely the feature is in the cohort — "
                "an OR > 1 means more common in your cohort, OR < 1 means less common.\n\n"
                "**Continuous features** (numeric fields): Overlaid histograms compare the value "
                "distributions. The rank-biserial correlation (r) is an effect size measure — "
                "|r| > 0.3 is considered medium, |r| > 0.5 is large.\n\n"
                "The stats table on the left shows the Composite score, SHAP, MI, Delta (mean "
                "difference), and the group means for the selected feature."
            )
        try:
            import plotly.graph_objects as go
            sel_feat_d = st.selectbox(
                "Select feature", rank_df["feature"].tolist(), key="dist_feat_sel"
            )
            if sel_feat_d:
                _dr = rank_df[rank_df["feature"] == sel_feat_d].iloc[0]
                _stat_cols_d = [
                    "Composite", "SHAP_mean_abs", "MI",
                    "Delta", mean_pos_col, mean_neg_col,
                ]
                _stat_cols_d = [c for c in _stat_cols_d if c in _dr.index]
                fd_left, fd_right = st.columns([1, 2])
                with fd_left:
                    st.markdown(f"**{sel_feat_d}**")
                    st.markdown(
                        f"Category: `{_dr.get('category', '—')}`  "
                        f"Direction: `{_dr.get('direction', '—')}`"
                    )
                    _sdf = pd.DataFrame(
                        {"Metric": _stat_cols_d,
                         "Value":  [_dr[c] for c in _stat_cols_d]}
                    )
                    st.dataframe(_sdf.set_index("Metric"), use_container_width=True)
                with fd_right:
                    _pos_label = _label_for_df
                    _neg_label = _lb if _dual else "Remaining"
                    # xf_raw is indexed by Pregnancy ID; NaN where missing before
                    # imputation — .dropna() excludes those rows automatically.
                    _xf_raw  = st.session_state.get("xf_raw")
                    _pos_ids = st.session_state.cohort_ids_used or set()
                    if _xf_raw is not None and sel_feat_d in _xf_raw.columns:
                        _fd_ser = pd.to_numeric(_xf_raw[sel_feat_d], errors="coerce")
                        if _dual and st.session_state.cohort_b_ids_used:
                            _all_shown = _pos_ids | (st.session_state.cohort_b_ids_used or set())
                            _fd_ser = _fd_ser[_fd_ser.index.isin(_all_shown)]
                        pos_v = _fd_ser[_fd_ser.index.isin(_pos_ids)].dropna()
                        neg_v = _fd_ser[~_fd_ser.index.isin(_pos_ids)].dropna()
                        fd    = _fd_ser
                    else:
                        _raw_pkl = _load_pkl(pickle_file)
                        if "Pregnancy ID" not in _raw_pkl.columns:
                            _raw_pkl = _raw_pkl.reset_index().rename(columns={"index": "Pregnancy ID"})
                        _raw_pkl["_pos"] = _raw_pkl["Pregnancy ID"].astype(str).isin(_pos_ids)
                        if _dual and st.session_state.cohort_b_ids_used:
                            _all_shown = _pos_ids | (st.session_state.cohort_b_ids_used or set())
                            _raw_pkl = _raw_pkl[_raw_pkl["Pregnancy ID"].astype(str).isin(_all_shown)]
                        fd    = pd.to_numeric(_raw_pkl.get(sel_feat_d), errors="coerce")
                        pos_v = fd[_raw_pkl["_pos"]].dropna() if fd is not None else pd.Series(dtype=float)
                        neg_v = fd[~_raw_pkl["_pos"]].dropna() if fd is not None else pd.Series(dtype=float)
                    if len(pos_v) > 0 or len(neg_v) > 0:
                        is_bin = set(fd.dropna().unique()).issubset({0, 1, 0.0, 1.0})
                        # For continuous features, exclude values at the full-column
                        # median — DB_6 filled missing values with fillna(median)
                        # before saving the pickle, so these are imputed, not real.
                        if not is_bin:
                            _full_col = pd.to_numeric(
                                _load_pkl(pickle_file).get(sel_feat_d), errors="coerce")
                            if _full_col is not None:
                                _fill_val = _full_col.median()
                                pos_v = pos_v[pos_v != _fill_val]
                                neg_v = neg_v[neg_v != _fill_val]
                        if is_bin:
                            pr = pos_v.mean() if len(pos_v) else 0
                            nr = neg_v.mean() if len(neg_v) else 0
                            fig3 = go.Figure([
                                go.Bar(name=_pos_label, x=["Prevalence"], y=[pr],
                                       marker_color="#e85c30"),
                                go.Bar(name=_neg_label, x=["Prevalence"], y=[nr],
                                       marker_color="#3078e8"),
                            ])
                            fig3.update_layout(barmode="group", title=sel_feat_d,
                                               yaxis_tickformat=".1%", height=340)
                            st.plotly_chart(fig3, use_container_width=True)
                            dc1, dc2, dc3 = st.columns(3)
                            dc1.metric(_pos_label,  f"{pr:.1%}")
                            dc2.metric(_neg_label,  f"{nr:.1%}")
                            if pr > 0 and nr > 0 and pr < 1 and nr < 1:
                                _or = (pr / (1 - pr)) / (nr / (1 - nr))
                                dc3.metric("Odds ratio", f"{_or:.2f}")
                            else:
                                dc3.metric("Δ", f"{pr - nr:+.1%}")
                        else:
                            fig3 = go.Figure([
                                go.Histogram(x=pos_v, name=_pos_label,  opacity=0.6,
                                             marker_color="#e85c30",
                                             histnorm="probability density"),
                                go.Histogram(x=neg_v, name=_neg_label, opacity=0.5,
                                             marker_color="#3078e8",
                                             histnorm="probability density"),
                            ])
                            fig3.update_layout(barmode="overlay", title=sel_feat_d, height=360)
                            st.plotly_chart(fig3, use_container_width=True)
                            dc1, dc2, dc3 = st.columns(3)
                            dc1.metric(f"{_pos_label} median",  f"{pos_v.median():.3g}")
                            dc2.metric(f"{_neg_label} median",  f"{neg_v.median():.3g}")
                            if len(pos_v) > 1 and len(neg_v) > 1:
                                from scipy.stats import mannwhitneyu as _mwu
                                _u_stat, _ = _mwu(pos_v, neg_v, alternative="two-sided")
                                _rbc = 1 - 2 * _u_stat / (len(pos_v) * len(neg_v))
                                dc3.metric("Rank-biserial r", f"{_rbc:.3f}",
                                           help="Effect size: |r| > 0.3 = medium, > 0.5 = large")
                    else:
                        st.info(f"'{sel_feat_d}' not found in the data file.")
        except Exception:
            st.warning(traceback.format_exc())

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "🧬 **PeriMiner** · Maxim Seferovic · "
    "[seferovi@bcm.edu](mailto:seferovi@bcm.edu) · "
    "Found a bug or have an idea? Send an email."
)
