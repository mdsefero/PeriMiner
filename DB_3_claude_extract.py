#!/usr/bin/env python3
"""DB_3_claude_extract.py — Extract medical concepts from free-text columns via Claude Batch API.

One-time (or incremental) script that:
1. Reads PBDBfinal_details.csv.
2. For each detail column, deduplicates non-empty cells and batches them (BATCH_SIZE per request).
3. Submits all batches to the Anthropic Batch API (50% cheaper than real-time, async).
4. Polls for completion, downloads results.
5. Writes claude_extraction_cache.json: {col: {cell_hash: ["concept", ...]}}

RESUMABLE: Existing cache entries are never re-submitted. The script saves in-flight batch IDs
to claude_batch_state.json so that if polling is interrupted (crash, Ctrl-C), you can resume
with --poll-only and skip re-submission entirely.

Cache keys are MD5 hashes of the lowercased cell text — no PHI is stored as dict keys.

COST:
  claude-haiku-4-5-20251001 + Batch API: ~$0.50–2.00 for a 58k-row dataset.
  claude-sonnet-4-6          + Batch API: ~$5–10. Use for higher extraction quality.

REQUIRES:
    pip install anthropic python-dotenv pandas

USAGE:
    python DB_3_claude_extract.py                         # full run
    python DB_3_claude_extract.py --batch-size 75         # larger batches
    python DB_3_claude_extract.py --poll-only             # resume after interruption
    python DB_3_claude_extract.py --dry-run               # count cells, estimate cost, exit

Last updated: March 2026, Maxim Seferovic, seferovi@bcm.edu
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

import pandas as pd
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from DB_5b_NLP import clean_text, expand_cancer_context

load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

CACHE_PATH       = os.path.join(SCRIPT_DIR, "claude_extraction_cache.json")
BATCH_STATE_PATH = os.path.join(SCRIPT_DIR, "claude_batch_state.json")

# Anthropic Batch API hard limit per submission call
_MAX_REQUESTS_PER_SUBMISSION = 10_000

# ── System prompt sent once per batch request ──────────────────────────────
_SYSTEM_PROMPT = """\
This data has been de-identified. You are extracting medical concepts only.

You extract medical concepts from perinatal clinical database free-text fields.
Return ONLY a JSON object mapping cell index strings to lists of normalized concept strings.

Rules:
- Expand abbreviations to full medical terms:
    HLD → hyperlipidemia
    URI → upper respiratory infection
    STD → sexually transmitted infection
    GBS → group b streptococcus
    hx / h/o / s/p → omit the prefix entirely, just name the condition
    D&C → dilation and curettage
    Any ICD code in brackets (e.g. [282.5A]) → omit the code, keep the condition name
- Each cell may contain multiple conditions separated by slash / comma / semicolon:
    extract each as a separate concept
- Return lowercase multi-word canonical medical concept names
    (e.g. "breast cancer", "sickle cell trait", "gestational diabetes mellitus")
- Include: diseases, syndromes, infections, procedures, congenital anomalies,
    mental health conditions, pregnancy complications, family history items
- When the column name contains "Cancer", treat bare organ/tissue names
    (e.g. "breast", "lung", "colon", "cervical") as "[organ] cancer"
- Exclude: purely vague terms (unknown, other, unspecified, N/A, none),
    single anatomical words without clinical context UNLESS the column name
    implies a specific disease category (e.g. Cancer, Mental health)
- Format: {"0": ["concept", ...], "1": [...], ...}
- Omit indices where no medical concepts are present
- Return ONLY the JSON object, no explanation, no markdown fences"""


# ── Cache key ──────────────────────────────────────────────────────────────

def _cache_key(cell_text: str) -> str:
    """MD5 of stripped+lowercased cell text — used as a PHI-free cache dict key."""
    return hashlib.md5(cell_text.strip().lower().encode("utf-8")).hexdigest()


# ── Cache / state I/O ──────────────────────────────────────────────────────

def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def _load_batch_state() -> dict:
    """Returns {batch_id: {custom_id: [col, [hash, hash, ...]]}}"""
    if os.path.exists(BATCH_STATE_PATH):
        with open(BATCH_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_batch_state(state: dict):
    """state: {batch_id: {custom_id: [col, [hash, ...]]}} — only hashes, no PHI."""
    with open(BATCH_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ── Build requests ─────────────────────────────────────────────────────────

def build_requests(df: pd.DataFrame, cache: dict, batch_size: int, model: str):
    """
    Iterate columns, collect unique uncached cell texts, batch them.

    Returns:
        api_requests  — list of Anthropic Batch API request dicts
        state_entries — list of (batch_id_placeholder, custom_id, col, [hash, ...])
                        (batch_id filled in after submission)
    """
    api_requests = []
    state_entries = []  # (custom_id, col, [hash, ...])

    for col_idx, col in enumerate(df.columns):
        col_cache = cache.get(col, {})
        cached_hashes = set(col_cache.keys())

        # Collect unique new cells for this column
        unique_cells: dict[str, str] = {}  # hash -> cleaned_text
        for cell in df[col]:
            if not cell.strip():
                continue
            cleaned = clean_text(cell)
            if not cleaned:
                continue
            # For cancer detail columns, expand bare organ names before sending
            # to Claude (e.g. "breast" → "breast cancer") so the extraction
            # cache stores the correct concept, not just the organ name.
            cleaned = expand_cancer_context(cleaned, col)
            # Strip inline ages/dates that add no NLP value
            cleaned = re.sub(
                r"\b(at age \d+|\d{1,2}/\d{1,2}/\d{2,4}|\d+ weeks? gestation)\b",
                "", cleaned, flags=re.I,
            ).strip()
            h = _cache_key(cell)
            if h not in cached_hashes and h not in unique_cells:
                unique_cells[h] = cleaned

        if not unique_cells:
            print(f"  {col}: fully cached ({len(col_cache)} unique cells), skipping")
            continue

        cell_items = list(unique_cells.items())  # [(hash, cleaned_text), ...]
        n_batches = -(-len(cell_items) // batch_size)  # ceiling division
        print(f"  {col}: {len(cell_items)} new unique cells → {n_batches} request(s)")

        for batch_i in range(n_batches):
            chunk = cell_items[batch_i * batch_size : (batch_i + 1) * batch_size]
            custom_id = f"c{col_idx:03d}_b{batch_i:05d}"

            # User message: column header + numbered cell texts
            lines = [f'Column: "{col}"\n']
            for i, (_h, text) in enumerate(chunk):
                lines.append(f"{i}: {text}")

            api_requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": model,
                    "max_tokens": 2048,
                    "system": _SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": "\n".join(lines)}],
                },
            })
            # Store only hashes (no PHI) for the state file
            state_entries.append((custom_id, col, [h for h, _ in chunk]))

    return api_requests, state_entries


# ── Submit ─────────────────────────────────────────────────────────────────

def submit_batches(client, api_requests: list, state_entries: list) -> dict:
    """
    Submit api_requests to Anthropic Batch API in chunks of _MAX_REQUESTS_PER_SUBMISSION.
    Returns state dict: {batch_id: {custom_id: [col, [hash, ...]]}}
    """
    # Build a lookup from custom_id to (col, hashes)
    custom_id_meta = {e[0]: (e[1], e[2]) for e in state_entries}

    state = {}  # batch_id -> {custom_id: [col, [hash, ...]]}
    total = len(api_requests)

    for start in range(0, total, _MAX_REQUESTS_PER_SUBMISSION):
        chunk = api_requests[start : start + _MAX_REQUESTS_PER_SUBMISSION]
        end   = min(start + _MAX_REQUESTS_PER_SUBMISSION, total)
        print(f"  Submitting requests {start+1}–{end} of {total} ...")

        batch = client.messages.batches.create(requests=chunk)
        print(f"  Batch ID: {batch.id}  status: {batch.processing_status}")

        batch_meta = {}
        for req in chunk:
            cid = req["custom_id"]
            col, hashes = custom_id_meta[cid]
            batch_meta[cid] = [col, hashes]

        state[batch.id] = batch_meta

    return state


# ── Poll ───────────────────────────────────────────────────────────────────

def poll_batches(client, state: dict, poll_interval: int):
    """Block until all batches in state have processing_status == 'ended'."""
    pending = set(state.keys())
    print(f"\nPolling {len(pending)} batch(es) every {poll_interval}s — Ctrl-C safe "
          f"(state saved to {os.path.basename(BATCH_STATE_PATH)}) ...")

    while pending:
        time.sleep(poll_interval)
        still_pending = set()
        for batch_id in sorted(pending):
            b = client.messages.batches.retrieve(batch_id)
            rc = b.request_counts
            print(f"  {batch_id}: {b.processing_status:<12s} "
                  f"processing={rc.processing}  succeeded={rc.succeeded}  "
                  f"errored={rc.errored}")
            if b.processing_status != "ended":
                still_pending.add(batch_id)
        pending = still_pending

    print("All batches complete.")


# ── Download & parse results ───────────────────────────────────────────────

def download_results(client, state: dict, cache: dict) -> dict:
    """
    Download results from all batches, parse JSON responses, update cache in-place.
    Returns the updated cache.
    """
    n_cells = 0
    n_concepts = 0
    n_errors = 0

    for batch_id, id_to_meta in state.items():
        print(f"\nDownloading results: {batch_id} ...")

        for result in client.messages.batches.results(batch_id):
            cid = result.custom_id
            if cid not in id_to_meta:
                continue

            col, hashes = id_to_meta[cid]

            if result.result.type != "succeeded":
                print(f"  WARN: {cid} — {result.result.type}")
                n_errors += 1
                continue

            raw = result.result.message.content[0].text.strip()
            # Strip markdown code fences if Claude wrapped its response
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw.rstrip())

            try:
                extracted: dict = json.loads(raw)
            except json.JSONDecodeError:
                print(f"  WARN: {cid} — unparseable JSON (first 200 chars): {raw[:200]}")
                n_errors += 1
                continue

            col_cache = cache.setdefault(col, {})
            for i, h in enumerate(hashes):
                concepts = extracted.get(str(i), [])
                if isinstance(concepts, list):
                    clean = [c.lower().strip() for c in concepts
                             if isinstance(c, str) and c.strip()]
                    col_cache[h] = clean
                    n_concepts += len(clean)
                else:
                    col_cache[h] = []
                n_cells += 1

    print(f"\nDownload complete: {n_cells} cells cached, "
          f"{n_concepts} concepts extracted, {n_errors} errors")
    return cache


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract medical concepts from detail columns via Claude Batch API."
    )
    parser.add_argument("--input", default="PBDBfinal_details.csv",
                        help="Input details CSV (default: PBDBfinal_details.csv)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Cells per API request (default: 50)")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between status polls (default: 60)")
    parser.add_argument("--poll-only", action="store_true",
                        help="Skip submission — poll+download from saved batch state")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count new cells and estimate cost, then exit without submitting")
    args = parser.parse_args()

    # ── API client ──────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        print("ERROR: ANTHROPIC_API_KEY not set. Edit .env and add your key.")
        sys.exit(1)

    model = os.environ.get("CLAUDE_EXTRACT_MODEL", "claude-haiku-4-5-20251001")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # ── Load existing cache ─────────────────────────────────────────────────
    cache = _load_cache()
    total_cached = sum(len(v) for v in cache.values())
    if total_cached:
        print(f"Loaded existing cache: {total_cached} cell entries across "
              f"{len(cache)} columns")

    # ── Poll-only mode: resume from saved state ─────────────────────────────
    if args.poll_only:
        state = _load_batch_state()
        if not state:
            print("No saved batch state found. Run without --poll-only first.")
            sys.exit(1)
        print(f"Resuming poll for {len(state)} batch(es) ...")
        poll_batches(client, state, args.poll_interval)
        cache = download_results(client, state, cache)
        _save_cache(cache)
        print(f"Cache saved → {CACHE_PATH}")
        if os.path.exists(BATCH_STATE_PATH):
            os.remove(BATCH_STATE_PATH)
        return

    # ── Read input CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(SCRIPT_DIR, args.input)
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run DB_2_clean.py first.")
        sys.exit(1)

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path, sep="|", index_col="Pregnancy ID", dtype=str)
    df.fillna("", inplace=True)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # ── Prune stale columns from cache ─────────────────────────────────────
    # Columns renamed by DB_1 or rerouted to meds by DB_2 leave orphan cache
    # entries that pollute DB_4's concept counts.  Drop them here.
    stale_cols = set(cache.keys()) - set(df.columns)
    if stale_cols:
        for col in stale_cols:
            del cache[col]
        _save_cache(cache)
        print(f"  Pruned {len(stale_cols)} stale column(s) from cache: "
              f"{', '.join(sorted(stale_cols))}")

    # ── Build requests ──────────────────────────────────────────────────────
    print("\nBuilding batch requests (already-cached cells are skipped) ...")
    api_requests, state_entries = build_requests(df, cache, args.batch_size, model)

    if not api_requests:
        print("\nAll cells already cached. Nothing to submit.")
        print("Next step: run DB_4_build_umls_map.py")
        return

    n_requests = len(api_requests)
    print(f"\nRequests to submit: {n_requests:,}  "
          f"(~{n_requests * args.batch_size:,} cells)")

    # ── Cost estimate ───────────────────────────────────────────────────────
    # Rough heuristic: avg 250 input tokens / request, avg 200 output tokens / request
    # Haiku Batch API pricing: $0.04/1M in, $0.20/1M out
    # Sonnet Batch API pricing: $1.50/1M in, $7.50/1M out
    est_in_tok  = n_requests * 250
    est_out_tok = n_requests * 200
    if "haiku" in model:
        est_cost = est_in_tok / 1e6 * 0.04 + est_out_tok / 1e6 * 0.20
    else:
        est_cost = est_in_tok / 1e6 * 1.50 + est_out_tok / 1e6 * 7.50
    print(f"Estimated cost ({model}): ~${est_cost:.2f}  "
          f"(rough; actual depends on cell lengths)")

    if args.dry_run:
        print("\n--dry-run: exiting without submitting.")
        return

    # ── Submit ──────────────────────────────────────────────────────────────
    print(f"\nSubmitting to Anthropic Batch API (model: {model}) ...")
    state = submit_batches(client, api_requests, state_entries)
    _save_batch_state(state)
    print(f"Batch state saved → {BATCH_STATE_PATH}  "
          f"(use --poll-only to resume if this session is interrupted)")

    # ── Poll ────────────────────────────────────────────────────────────────
    poll_batches(client, state, args.poll_interval)

    # ── Download & save ─────────────────────────────────────────────────────
    cache = download_results(client, state, cache)
    _save_cache(cache)

    n_cols = len(cache)
    n_cells = sum(len(v) for v in cache.values())
    n_concepts = sum(len(c) for col_cache in cache.values()
                     for c in col_cache.values() if c)
    print(f"\nCache saved → {CACHE_PATH}")
    print(f"  {n_cols} columns | {n_cells:,} unique cells | {n_concepts:,} concepts")

    # Clean up ephemeral batch state
    if os.path.exists(BATCH_STATE_PATH):
        os.remove(BATCH_STATE_PATH)

    print("\nNext step: run DB_4_build_umls_map.py")


if __name__ == "__main__":
    main()
