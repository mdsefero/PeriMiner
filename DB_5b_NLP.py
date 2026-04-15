# NLP processing of free-filled columns in peribank database for later reassembly/data mining.
# Free-filled columns are "details" of conditions. Medications are also free-filled and
# dealt with separately (DB_5a_meds.py).
#
# NEW APPROACH (March 2026):
#   Medical concepts are extracted from raw cell text by Claude (DB_3_claude_extract.py)
#   and cached in claude_extraction_cache.json.  This module looks up each cell in the
#   cache, applies UMLS normalization (token_to_concept.json), then booleanizes.
#   spaCy is retained as a fallback for any cells not present in the cache (e.g. new data).
#
# Last updated March 2026, Maxim Seferovic, seferovi@bcm.edu

import hashlib
import re
import os
import json
import pandas as pd
from collections import Counter
from multiprocessing import Pool

# ── Lazy spaCy load ────────────────────────────────────────────────────────
# spaCy is only needed for the fallback path (cells absent from the Claude cache).
# Delaying the load means DB_3 and DB_4 can import clean_text() from this module
# without triggering a 10-second model load.
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_sci_lg")
    return _nlp


# ── UMLS concept normalization lookup ──────────────────────────────────────
# Two tiers loaded from umls_overrides.json:
#   pre_nlp:      substituted as-is on extracted concept strings (abbrev/synonym fixing).
#   override_nlp: hard-replace specific output tokens (last resort).
#
# token_to_concept.json (from DB_4) feeds into pre_nlp; manual overrides merge on top.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _build_re(mapping):
    """Build a longest-match-first regex from a dict's keys."""
    if not mapping:
        return None
    keys = sorted(mapping.keys(), key=len, reverse=True)
    return re.compile(
        r'\b(' + '|'.join(re.escape(k) for k in keys) + r')\b',
        re.IGNORECASE,
    )


# Load generated UMLS map (pre_nlp tier)
_PRE_NLP_MAP = {}
_concept_map_path = os.path.join(_SCRIPT_DIR, "token_to_concept.json")
if os.path.exists(_concept_map_path):
    with open(_concept_map_path, "r", encoding="utf-8") as _f:
        _PRE_NLP_MAP = json.load(_f)

# Load manual overrides (both tiers)
_OVERRIDE_NLP_MAP = {}
_overrides_path = os.path.join(_SCRIPT_DIR, "umls_overrides.json")
if os.path.exists(_overrides_path):
    with open(_overrides_path, "r", encoding="utf-8") as _f:
        _overrides = json.load(_f)
    _PRE_NLP_MAP.update(_overrides.get("pre_nlp", {}))
    _OVERRIDE_NLP_MAP = _overrides.get("override_nlp", {})

_PRE_NLP_RE = _build_re(_PRE_NLP_MAP)
_OVERRIDE_NLP_RE = _build_re(_OVERRIDE_NLP_MAP)

# Abbreviation-only map: used by DB_4 when rebuilding the UMLS map to avoid
# circular normalization (old canonical names must not be re-submitted to UMLS).
_ABBREV_ONLY_MAP = {}
_ABBREV_ONLY_RE  = None
if os.path.exists(_overrides_path):
    with open(_overrides_path, "r", encoding="utf-8") as _f:
        _abbrev_data = json.load(_f)
    _ABBREV_ONLY_MAP = _abbrev_data.get("pre_nlp", {})
    _ABBREV_ONLY_RE  = _build_re(_ABBREV_ONLY_MAP)

# Minimum pregnancies a concept must appear in to survive booleanization
feat_stringency = 10


# ── Text helpers ───────────────────────────────────────────────────────────

def clean_text(text):
    text = text.replace('\t', ' ')  # PeriBank multi-select uses tabs as delimiters
    text = re.sub(r'\[.*?\]', '', text).lower()
    remove = ('h/o', 'hx', 'h/x', 's/p', 'w/', '?', '~')
    for i in remove:
        text = text.replace(i, '')
    text = re.sub(r'(\w)\s*/\s*(\w)', r'\1, \2', text)
    replace = ('*', '+', '-', ')', '(', '/', ';', ':')
    for i in replace:
        text = text.replace(i, ' ')
    text = text.replace('d & c', 'd&c').replace('pst', 'post')
    return ' '.join(text.split())


_CANCER_ORGANS = frozenset([
    'breast', 'colon', 'lung', 'prostate', 'ovarian', 'ovary', 'cervical',
    'cervix', 'uterine', 'uterus', 'endometrial', 'thyroid', 'skin', 'liver',
    'pancreatic', 'pancreas', 'kidney', 'renal', 'bladder', 'gastric',
    'stomach', 'colorectal', 'rectal', 'bone', 'brain', 'esophageal',
    'melanoma', 'leukemia', 'lymphoma', 'hematologic',
])


def expand_cancer_context(cell, col_name):
    """For cancer columns, expand bare organ names to '[organ] cancer'."""
    if 'cancer' not in col_name.lower():
        return cell
    parts = re.split(r'\s*,\s*', cell.strip())
    out = []
    for p in parts:
        words = p.strip().lower().split()
        if len(words) == 1 and words[0] in _CANCER_ORGANS:
            out.append(p.strip() + ' cancer')
        else:
            out.append(p.strip())
    return ', '.join(out)


def normalize_concepts(text):
    """Apply pre-NLP UMLS map + overrides.  Maps concept strings to canonical names."""
    if not _PRE_NLP_RE or not text:
        return text
    return _PRE_NLP_RE.sub(lambda m: _PRE_NLP_MAP[m.group(0).lower()], text)


def expand_abbreviations_only(text):
    """Apply ONLY hand-curated abbreviation overrides (used by DB_4 during map rebuild)."""
    if not _ABBREV_ONLY_RE or not text:
        return text
    return _ABBREV_ONLY_RE.sub(lambda m: _ABBREV_ONLY_MAP[m.group(0).lower()], text)


def override_nlp_tokens(token_csv):
    """Post-extraction: hard-replace tokens that NLP or Claude mishandles."""
    if not _OVERRIDE_NLP_RE or not token_csv:
        return token_csv
    tokens = token_csv.split(',')
    out = []
    for t in tokens:
        replaced = _OVERRIDE_NLP_RE.sub(
            lambda m: _OVERRIDE_NLP_MAP[m.group(0).lower()], t)
        out.append(replaced)
    return ','.join(out)


# ── spaCy fallback (for cells absent from the Claude cache) ───────────────

def process_text(col):
    """spaCy noun-chunk extraction — used only as cache fallback."""
    doc = _get_nlp()(col)
    covered = set()
    phrases = []
    for chunk in doc.noun_chunks:
        covered.update(range(chunk.start, chunk.end))
        phrases.append(chunk.text)
    for token in doc:
        if (token.i not in covered
                and not token.is_stop
                and not token.is_punct
                and len(token.lemma_) >= 3):
            phrases.append(token.lemma_)
    return override_nlp_tokens(','.join(set(phrases)))


def process_text_batch(texts):
    """Batch spaCy processing — used only as cache fallback."""
    results = []
    for doc in _get_nlp().pipe(texts, batch_size=64, disable=["ner"]):
        covered = set()
        phrases = []
        for chunk in doc.noun_chunks:
            covered.update(range(chunk.start, chunk.end))
            phrases.append(chunk.text)
        for token in doc:
            if (token.i not in covered
                    and not token.is_stop
                    and not token.is_punct
                    and len(token.lemma_) >= 3):
                phrases.append(token.lemma_)
        results.append(override_nlp_tokens(','.join(set(phrases))))
    return results


def apply_multiprocessing_text(df):
    """spaCy multiprocessing extraction — only called if no cache is available."""
    n_cpu = os.cpu_count() or 1
    with Pool(processes=n_cpu) as pool:
        for col in df.columns:
            print(f"\t{col}...")
            texts = (df[col]
                     .apply(clean_text)
                     .apply(lambda x: expand_cancer_context(x, col))
                     .apply(normalize_concepts)
                     .tolist())
            batch_size = max(1, -(-len(texts) // n_cpu))
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            results = pool.map(process_text_batch, batches)
            df[col] = [item for sublist in results for item in sublist]
    return df


# ── Claude extraction cache lookup ────────────────────────────────────────

def _cache_key(cell_text: str) -> str:
    """MD5 of stripped+lowercased cell — must match the key used in DB_3."""
    return hashlib.md5(cell_text.strip().lower().encode("utf-8")).hexdigest()


def load_extraction_cache() -> dict:
    """Load claude_extraction_cache.json.  Returns empty dict if absent."""
    cache_path = os.path.join(_SCRIPT_DIR, "claude_extraction_cache.json")
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_cache_extraction(df: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """
    Replace each cell with a comma-separated list of UMLS-normalized concept strings.

    For cells present in the Claude cache: look up extracted concepts, apply
    token_to_concept.json normalization, apply override_nlp post-overrides.

    For cells absent from the cache (new data since last DB_3 run): fall back to
    minimal clean_text() without spaCy — concepts pass through as raw cleaned text.
    A warning is printed so you know to re-run DB_3 when there are many misses.
    """
    grand_hits   = 0
    grand_misses = 0

    for col in df.columns:
        col_cache    = cache.get(col, {})
        results      = []
        total_hits   = 0
        total_misses = 0

        for cell in df[col]:
            if not cell.strip():
                results.append('')
                continue

            h        = _cache_key(cell)
            concepts = col_cache.get(h)

            if concepts is not None:
                # Apply UMLS normalization to each extracted concept
                normalized = []
                for c in concepts:
                    normed = normalize_concepts(c)
                    # normalize_concepts may produce comma-separated tokens for
                    # multi-token canonical names; split and re-join cleanly
                    for part in normed.split(','):
                        part = part.strip()
                        if part:
                            normalized.append(part)
                token_csv = ','.join(dict.fromkeys(normalized))  # dedup, preserve order
                results.append(override_nlp_tokens(token_csv))
                total_hits += 1
            else:
                # Fallback: basic clean without spaCy
                cleaned = normalize_concepts(
                    expand_cancer_context(clean_text(cell), col))
                results.append(cleaned)
                total_misses += 1

        print(f"\t{col}: {total_hits} cache hits, {total_misses} fallback")
        df[col] = results
        grand_hits   += total_hits
        grand_misses += total_misses

    if grand_misses > 0:
        print(f"\n  WARNING: {grand_misses} cells used spaCy fallback (not in cache).")
        print(f"  Re-run DB_3_claude_extract.py to cache new data, then re-run DB_4.")

    return df


# ── Boolean conversion ─────────────────────────────────────────────────────

def split_dataframe(df, num_chunks):
    num_chunks  = min(num_chunks, len(df.columns))
    column_chunks = [df.columns[i::num_chunks] for i in range(num_chunks)]
    return [df[chunk] for chunk in column_chunks if len(chunk) > 0]


def process_boolean(partition_df):
    new_dfs = []
    partition_df = partition_df.fillna('')
    min_count = feat_stringency
    for col in partition_df.columns:
        col_norm = (partition_df[col]
                    .str.lower()
                    .str.replace(r'\s*,\s*', ',', regex=True)
                    .str.strip(','))
        token_counts = Counter()
        for cell in col_norm:
            if cell:
                token_counts.update(set(cell.split(',')))
        surviving = {t for t, c in token_counts.items() if t and c >= min_count}
        n_before  = len(token_counts)
        if not surviving:
            print(f"\t{col}  ({n_before} unique → 0 survive, skipped)")
            continue
        col_norm = col_norm.apply(
            lambda cell, s=surviving:
                ','.join(t for t in cell.split(',') if t in s) if cell else '')
        result = col_norm.str.get_dummies(sep=',')
        result = result.loc[:, result.columns != '']
        result.columns = [f"{col}_{c}" for c in result.columns]
        new_dfs.append(result)
        print(f"\t{col}  ({n_before} unique → {len(surviving)} survive)")
    if not new_dfs:
        return pd.DataFrame(index=partition_df.index)
    return pd.concat(new_dfs, axis=1)


def apply_multiprocessing_boolean(df):
    num_processes = min(os.cpu_count() or 1, len(df.columns))
    df_chunks     = split_dataframe(df, num_processes)
    with Pool(num_processes) as pool:
        result_chunks = pool.map(process_boolean, df_chunks)
    return pd.concat(result_chunks, axis=1)


def filter_words(col, df, rows):
    return (df[col] == True).sum() < feat_stringency


def process_partition(partition, rows):
    droplist = []
    for col in partition.columns:
        if filter_words(col, partition, rows):
            droplist.append(col)
    partition.drop(columns=droplist, inplace=True)
    return partition


def parallel_filter_words(df):
    start = len(df.columns)
    rows  = len(df)
    ncpu  = os.cpu_count() or 1
    partitions = [df.iloc[:, i::ncpu] for i in range(ncpu)]
    with Pool(processes=ncpu) as pool:
        processed_partitions = pool.starmap(
            process_partition, [(p, rows) for p in partitions])
    df = pd.concat(processed_partitions, axis=1)
    print(f"{start - len(df.columns)} tokens removed, {len(df.columns)} remain")
    return df


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv("PBDBfinal_details.csv", sep="|",
                     index_col='Pregnancy ID', dtype=str)
    df.fillna('', inplace=True)

    cache = load_extraction_cache()

    if cache:
        print("Applying Claude-extracted concepts (cache lookup + UMLS normalization)...")
        df = apply_cache_extraction(df, cache)
    else:
        print("WARNING: claude_extraction_cache.json not found.")
        print("  Run DB_3_claude_extract.py first for best results.")
        print("  Falling back to spaCy NLP (original approach)...")
        df = apply_multiprocessing_text(df)

    print("Processing to Boolean...")
    df = apply_multiprocessing_boolean(df)
    print("Filtering rare tokens...")
    df = parallel_filter_words(df)
    df.to_csv("PBDBfinal_details_tok.csv", sep='|', index=True)


if __name__ == '__main__':
    main()
