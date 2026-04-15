#!/usr/bin/env python3
"""DB_4_build_umls_map.py — Build UMLS canonical-name map from Claude-extracted concepts.

Reads claude_extraction_cache.json (produced by DB_3_claude_extract.py), collects every
unique medical concept string Claude extracted, and runs scispaCy UMLS entity linking to
map each concept to its canonical UMLS name.  Because the input is already clean medical
terminology (not raw free-text), UMLS hit rates are substantially higher than before.

Workflow:
1. Load claude_extraction_cache.json — concepts are clean strings like "breast cancer",
   "hyperlipidemia", "sickle cell trait" (not raw tokens like "HLD", "breast", "sickle").
2. Merge manual pre_nlp overrides from umls_overrides.json (overrides always win).
3. Run scispaCy EntityLinker (UMLS) on every unique concept not already overridden.
4. Write token_to_concept.json for use by DB_5b_NLP.py at runtime.
5. Write unmapped_concepts_audit.tsv — concepts UMLS could not map, with source columns.

Unmapped concepts pass through unchanged in DB_5b (they become boolean features using
their Claude-extracted name, e.g. "arthritis_nec").  Use the audit file to add entries to
umls_overrides.json["pre_nlp"] for any that need explicit canonical names.

First run downloads ~500 MB UMLS knowledge base to ~/.scispacy/ (cached permanently).

Requires:
    pip install scispacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

Usage:
    python DB_4_build_umls_map.py
    python DB_4_build_umls_map.py --threshold 0.85
    python DB_4_build_umls_map.py --cache claude_extraction_cache.json

Last updated: March 2026, Maxim Seferovic, seferovi@bcm.edu
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict

import spacy
from scispacy.linking import EntityLinker  # noqa: F401 — registers scispacy_linker factory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Generic UMLS targets that destroy clinical signal ─────────────────────
# When UMLS maps a specific concept to one of these, the mapping is discarded
# so the concept passes through DB_5b using its clean Claude-extracted name.
BLOCKED_TARGETS = frozenset({
    "pregnancy",
    "abnormal",
    "maternal",
    "removal technique",
    "positive",
    "expression negative",
    "no risk",
    "newborn encounter admission source",
    "severe (severity modifier)",
    "third trimester onset",
    "therapeutic procedure",
    "implantation procedure",
    "complication",
    "disease",
    "treated with",
    "follow up status",
    "thickened",
    "immune status",
    "immunization",
    "conception",
})

# ── Perinatal-relevant UMLS Semantic Type Unique Identifiers (TUIs) ────────
RELEVANT_TUIS = {
    "T047",  # Disease or Syndrome
    "T048",  # Mental or Behavioral Dysfunction
    "T184",  # Sign or Symptom
    "T033",  # Finding
    "T019",  # Congenital Abnormality
    "T046",  # Pathologic Function
    "T037",  # Injury or Poisoning
    "T061",  # Therapeutic or Preventive Procedure
    "T060",  # Diagnostic Procedure
    "T059",  # Laboratory Procedure
    "T034",  # Laboratory or Test Result
    "T121",  # Pharmacologic Substance
    "T200",  # Clinical Drug
    "T191",  # Neoplastic Process
    "T020",  # Acquired Abnormality
    "T190",  # Anatomical Abnormality
    "T039",  # Physiologic Function
    "T040",  # Organism Function
    "T042",  # Organ or Tissue Function
}


def load_concepts_from_cache(cache_path: str):
    """Read claude_extraction_cache.json, collect unique concept strings.

    Returns:
        concept_counts:  Counter — concept string → number of unique cells it appeared in
        concept_columns: dict    — concept string → set of source column names
    """
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    concept_counts  = Counter()
    concept_columns = defaultdict(set)

    for col, col_cache in cache.items():
        for _hash, concepts in col_cache.items():
            for concept in concepts:
                concept_counts[concept] += 1
                concept_columns[concept].add(col)

    n_cols  = len(cache)
    n_cells = sum(len(v) for v in cache.values())
    print(f"  Cache: {n_cols} columns, {n_cells:,} unique cells, "
          f"{len(concept_counts)} unique concepts")
    return concept_counts, concept_columns


def load_overrides(path: str = "umls_overrides.json"):
    """Load pre_nlp overrides; return empty dict if absent.
    Only pre_nlp entries are merged into token_to_concept.json.
    override_nlp entries are applied post-extraction at DB_4b runtime."""
    full_path = os.path.join(SCRIPT_DIR, path)
    if not os.path.exists(full_path):
        print(f"  No overrides file ({path})")
        return {}
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    overrides = data.get("pre_nlp", {})
    print(f"  Loaded {len(overrides)} pre_nlp overrides from {path}")
    n_hard = len(data.get("override_nlp", {}))
    if n_hard:
        print(f"  ({n_hard} override_nlp entries — applied at DB_4b runtime)")
    return overrides


def _extract_best_candidate(doc, linker, relevant_tuis):
    """Extract best TUI-valid UMLS match from a doc's entities.

    Returns (canonical_name, score, was_tui_filtered).
    """
    best_name  = None
    best_score = 0.0
    filtered   = False
    for ent in doc.ents:
        for cui, score in (ent._.kb_ents or []):
            obj = linker.kb.cui_to_entity.get(cui)
            if obj is None:
                continue
            if set(obj.types) & relevant_tuis:
                if score > best_score:
                    best_score = score
                    best_name  = obj.canonical_name
                break
            else:
                filtered = True
    return best_name, best_score, filtered


def _extract_longest_span(doc, linker, relevant_tuis, concept_text):
    """Extract UMLS match preferring the longest NER span, then name overlap.

    Prevents short modifier spans ("abnormal", "severe") from beating longer
    clinically-specific spans ("pap smear", "preeclampsia").
    Returns (canonical_name, score, was_tui_filtered).
    """
    _STOP = frozenset({'of', 'the', 'a', 'an', 'in', 'on', 'for', 'to', 'by',
                        'and', 'or', 'with', 'as', 'at', 'from', 'is', 'was'})
    concept_words = set(concept_text.lower().split()) - _STOP

    candidates = []
    filtered   = False
    for ent in doc.ents:
        for cui, score in (ent._.kb_ents or []):
            obj = linker.kb.cui_to_entity.get(cui)
            if obj is None:
                continue
            if set(obj.types) & relevant_tuis:
                canonical = obj.canonical_name.lower().replace("-", " ")
                canon_words = set(canonical.split()) - _STOP
                overlap = len(concept_words & canon_words) if concept_words else 0
                candidates.append((len(ent), overlap, score, canonical))
                break
            else:
                filtered = True

    if not candidates:
        return None, 0.0, filtered

    # Rank by: (1) span token length, (2) word overlap with input, (3) UMLS score
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return candidates[0][3], candidates[0][2], filtered


def link_entities(concepts: set, threshold: float = 0.80, batch_size: int = 256):
    """Run UMLS entity linking with two-pass approach for clean concept strings.

    The input concepts are already clean medical terms from Claude extraction,
    not raw free-text.  The old single-pass approach relied on scispaCy NER to
    detect entity spans, then picked the highest-scoring UMLS match across all
    spans.  This caused NER span fragmentation: "severe preeclampsia" was split
    into ["severe", "preeclampsia"], and "severe" → "severe (severity modifier)"
    (score 0.95) beat "preeclampsia" → "pre-eclampsia" (score 0.90).  Result:
    ~900 concepts collapsed into generic modifier sinks like "abnormal" (179
    tokens), "maternal" (156), "removal technique" (128).

    Fix — two-pass linking:

    Pass 1 (full-text): Bypass NER; treat the entire concept string as a single
    entity span and let the UMLS linker match against the full phrase.  This
    gives UMLS the complete context ("severe preeclampsia" matches "severe
    pre-eclampsia" directly) and eliminates span fragmentation.

    Pass 2 (NER fallback): For concepts unmapped in Pass 1 (typically short or
    ambiguous terms where full-text matching underperforms), fall back to NER-
    detected spans but prefer the LONGEST span and highest word overlap with the
    input concept, preventing generic modifiers from winning.

    Returns:
        mapping:      dict — concept → canonical_concept_name (lowercase)
        tui_filtered: set  — concepts matched UMLS but excluded by semantic type filter
    """
    from spacy.tokens import Span

    print("Loading en_core_sci_lg + scispaCy EntityLinker (UMLS) ...")
    print("  (First run downloads ~500 MB knowledge base to ~/.scispacy/)")
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe(
        "scispacy_linker",
        config={
            "resolve_abbreviations": True,
            "linker_name": "umls",
            "threshold": threshold,
        },
    )
    linker = nlp.get_pipe("scispacy_linker")

    concept_list = list(concepts)

    # Normalize hyphens → spaces for consistent UMLS matching.
    # "ehlers-danlos" and "ehlers danlos" should get the same result.
    norm_to_originals = {}  # normalized_string → [original_string, ...]
    normalized_list = []
    for c in concept_list:
        norm = c.replace("-", " ")
        if norm not in norm_to_originals:
            norm_to_originals[norm] = []
            normalized_list.append(norm)
        norm_to_originals[norm].append(c)

    n_deduped = len(concept_list) - len(normalized_list)
    if n_deduped:
        print(f"  Hyphen normalization: {len(concept_list)} concepts → "
              f"{len(normalized_list)} unique (merged {n_deduped} hyphen variants)")

    mapping      = {}
    tui_filtered = set()
    n_total      = len(normalized_list)
    n_mapped_full = 0
    n_mapped_ner  = 0
    n_self        = 0
    n_filtered    = 0

    print(f"Linking {n_total} concepts (threshold={threshold}) ...")

    # ══ PASS 1: Full-text matching (bypass NER) ══════════════════════════════
    # Process through the pipeline WITHOUT NER, then set a single full-text
    # entity span and run the linker.  This prevents span fragmentation.
    print("  Pass 1: Full-text matching (bypass NER) ...")
    t0 = time.time()
    unmapped_pass1 = []

    for i in range(0, n_total, batch_size):
        batch = normalized_list[i : i + batch_size]
        # Run pipeline without NER and linker — gets tok2vec/tagger/parser output
        docs = list(nlp.pipe(batch, batch_size=batch_size,
                             disable=["ner", "scispacy_linker"]))

        for concept_text, doc in zip(batch, docs):
            if len(doc) == 0:
                unmapped_pass1.append(concept_text)
                continue
            # Set single entity covering the full concept text
            doc.ents = [Span(doc, 0, len(doc), label="ENTITY")]
            linker(doc)

            name, score, was_filtered = _extract_best_candidate(
                doc, linker, RELEVANT_TUIS)
            if name:
                canonical = name.lower().replace("-", " ")
                if len(canonical) >= 3:
                    mapping[concept_text] = canonical
                    n_mapped_full += 1
                    if canonical == concept_text:
                        n_self += 1
                    continue
            # Full-text didn't produce a usable match — try NER in pass 2
            unmapped_pass1.append(concept_text)

        done = min(i + batch_size, n_total)
        if done % 1000 < batch_size or done == n_total:
            elapsed = time.time() - t0
            rate    = done / elapsed if elapsed > 0 else 0
            print(f"    {done}/{n_total}  ({rate:.0f}/s)  "
                  f"full-text mapped={n_mapped_full}")

    print(f"  Pass 1 complete: {n_mapped_full} mapped, "
          f"{len(unmapped_pass1)} remaining for NER fallback")

    # ══ PASS 2: NER-based fallback (longest span wins) ═══════════════════════
    # For concepts that failed full-text matching, use NER-detected spans but
    # prefer the longest span to avoid generic modifier sinks.
    if unmapped_pass1:
        print(f"  Pass 2: NER fallback for {len(unmapped_pass1)} concepts ...")
        t1 = time.time()

        for i in range(0, len(unmapped_pass1), batch_size):
            batch = unmapped_pass1[i : i + batch_size]
            docs  = list(nlp.pipe(batch, batch_size=batch_size))

            for concept_text, doc in zip(batch, docs):
                if not doc.ents:
                    continue

                name, score, was_filtered = _extract_longest_span(
                    doc, linker, RELEVANT_TUIS, concept_text)
                if name and len(name) >= 3:
                    mapping[concept_text] = name
                    n_mapped_ner += 1
                    if name == concept_text:
                        n_self += 1
                elif was_filtered:
                    tui_filtered.add(concept_text)
                    n_filtered += 1

            done = min(i + batch_size, len(unmapped_pass1))
            if done % 500 < batch_size or done == len(unmapped_pass1):
                elapsed = time.time() - t1
                rate    = done / elapsed if elapsed > 0 else 0
                print(f"    {done}/{len(unmapped_pass1)}  ({rate:.0f}/s)  "
                      f"NER mapped={n_mapped_ner}  TUI-filtered={n_filtered}")

    n_mapped = n_mapped_full + n_mapped_ner
    print(f"Linking complete: {n_mapped} mapped "
          f"(full-text={n_mapped_full}, NER={n_mapped_ner}, self={n_self}), "
          f"{n_filtered} TUI-filtered, "
          f"{n_total - n_mapped - n_filtered} no match")

    # Expand mapping from normalized keys back to all original concept strings
    expanded = {}
    for norm_key, canonical in mapping.items():
        for orig in norm_to_originals.get(norm_key, [norm_key]):
            expanded[orig] = canonical
    mapping = expanded

    return mapping, tui_filtered


def _is_likely_medication(concept: str, columns: set) -> bool:
    """Heuristic: concept is probably a medication (handled by DB_5a, not DB_5b)."""
    _MED_COL_PATTERNS = ('Meds', 'COMBINED LIST', 'Medication', 'medication')
    return any(pat in col for col in columns for pat in _MED_COL_PATTERNS)


def write_unmapped_audit(unmapped: set, concept_counts: Counter,
                         concept_columns: dict, tui_filtered: set,
                         output_path: str):
    """Write TSV audit of concepts UMLS did not map, sorted by frequency.

    Columns: concept | count | reason | likely_med | source_columns

    likely_med = True for concepts that only appear in medication columns
    (handled by DB_5a, expected to be unmapped here).
    """
    sorted_concepts = sorted(unmapped, key=lambda c: concept_counts[c], reverse=True)

    n_clinical = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["concept", "count", "reason", "likely_med", "source_columns"])
        for concept in sorted_concepts:
            cols     = concept_columns.get(concept, set())
            cols_str = ", ".join(sorted(cols))
            med      = _is_likely_medication(concept, cols)
            reason   = ("tui_filtered" if concept in tui_filtered else "no_umls_match")
            writer.writerow([concept, concept_counts[concept], reason, med, cols_str])
            if not med:
                n_clinical += 1

    print(f"\nWrote unmapped concepts audit: {output_path}")
    print(f"  {len(sorted_concepts)} unmapped ({n_clinical} clinical, "
          f"{len(sorted_concepts) - n_clinical} likely medications)")
    print(f"  Review clinical concepts and add to umls_overrides.json[\"pre_nlp\"]")


def main():
    parser = argparse.ArgumentParser(
        description="Build UMLS concept map from Claude extraction cache."
    )
    parser.add_argument("--cache",     default="claude_extraction_cache.json",
                        help="Claude extraction cache (default: claude_extraction_cache.json)")
    parser.add_argument("--output",    default="token_to_concept.json",
                        help="Output map file (default: token_to_concept.json)")
    parser.add_argument("--overrides", default="umls_overrides.json",
                        help="Manual overrides JSON (default: umls_overrides.json)")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="UMLS linking confidence threshold (default: 0.80)")
    parser.add_argument("--min-count", type=int, default=1,
                        help="Min unique-cell count to attempt UMLS linking (default: 1)")
    parser.add_argument("--audit",     default="unmapped_concepts_audit.tsv",
                        help="Unmapped concepts audit TSV (default: unmapped_concepts_audit.tsv)")
    args = parser.parse_args()

    cache_path = os.path.join(SCRIPT_DIR, args.cache)
    if not os.path.exists(cache_path):
        print(f"ERROR: {cache_path} not found. Run DB_3_claude_extract.py first.")
        sys.exit(1)

    # Step 1: Load concepts from Claude extraction cache
    print(f"Reading {cache_path} ...")
    concept_counts, concept_columns = load_concepts_from_cache(cache_path)

    # Step 2: Load manual overrides (these win over UMLS)
    overrides = load_overrides(args.overrides)
    already_overridden = set()
    for k in overrides:
        already_overridden.add(k.lower())
        already_overridden.add(k.lower().replace("-", " "))

    # Step 3: UMLS entity linking on concepts not already overridden
    to_link = {c for c, n in concept_counts.items()
               if n >= args.min_count and c.lower() not in already_overridden
               and c.lower().replace("-", " ") not in already_overridden}
    print(f"  {len(to_link)} concepts to link via UMLS "
          f"({len(already_overridden)} already covered by overrides)")

    umls_mapping, tui_filtered = link_entities(to_link, threshold=args.threshold)

    # Step 3b: Filter blocked generic sinks (concepts pass through DB_5b as-is)
    blocked_set = set()
    for concept in list(umls_mapping):
        if umls_mapping[concept] in BLOCKED_TARGETS:
            blocked_set.add(concept)
            del umls_mapping[concept]
    if blocked_set:
        print(f"  Blocked {len(blocked_set)} mappings to generic sinks")

    # Step 3c: Strip identity mappings (source == target is a no-op in DB_5b)
    identity_set = set()
    for concept in list(umls_mapping):
        if concept.lower() == umls_mapping[concept].lower():
            identity_set.add(concept)
            del umls_mapping[concept]
    if identity_set:
        print(f"  Stripped {len(identity_set)} identity mappings")

    # Step 4: Merge — overrides win, expanded to cover hyphenated variants
    # Override keys are hyphen-free ("ehlers danlos") but cache concepts may
    # have hyphens ("ehlers-danlos").  Expand overrides so both forms map.
    expanded_overrides = {}
    for k, v in overrides.items():
        expanded_overrides[k] = v
        hyphenated_variant = k  # original already in map
        # Check if any concept in the cache is a hyphenated form of this override
        for c in concept_counts:
            if c.lower().replace("-", " ") == k.lower() and c.lower() != k.lower():
                expanded_overrides[c] = v
    final_mapping = {**umls_mapping, **expanded_overrides}

    # Step 5: Write token_to_concept.json
    out_path = os.path.join(SCRIPT_DIR, args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_mapping, f, indent=2, sort_keys=True)
    print(f"\nWrote {len(final_mapping)} mappings → {out_path}")
    print(f"  UMLS-linked:      {len(umls_mapping)}")
    print(f"  Manual overrides: {len(overrides)}")

    # Step 6: Unmapped audit
    # Exclude blocked/identity — those pass through DB_5b correctly
    unmapped = to_link - set(final_mapping.keys()) - blocked_set - identity_set
    audit_path = os.path.join(SCRIPT_DIR, args.audit)
    write_unmapped_audit(unmapped, concept_counts, concept_columns,
                         tui_filtered, audit_path)
    if unmapped:
        top = sorted(unmapped, key=lambda c: concept_counts[c], reverse=True)[:20]
        print("\nTop 20 unmapped concepts:")
        for c in top:
            reason = "TUI-filtered" if c in tui_filtered else "no match"
            print(f"  {concept_counts[c]:>6d}  {c:<40s}  [{reason}]")


if __name__ == "__main__":
    main()
