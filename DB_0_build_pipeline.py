#!/usr/bin/env python3
"""DB_0_build_pipeline.py — PeriMiner full-pipeline orchestrator.

Runs the DB pipeline in dependency order, with DB_5a and DB_5b in parallel
(they are independent: different inputs, different outputs).

Dependency graph:
    DB_1 → DB_2 → DB_3 (Claude concept extraction, one-time, ~$1)
                → DB_4 (UMLS canonical-name map)
                → ┬─ DB_5a (meds fuzzy-match + boolean) ─┬→ DB_6 → pickle
                  └─ DB_5b (NLP free-text → boolean)    ─┘

NOTE on DB_3:
    DB_3 submits cells to the Anthropic Batch API and polls until complete (minutes
    to hours depending on dataset size).  It is fully resumable — re-running it costs
    nothing for already-cached cells.  Requires ANTHROPIC_API_KEY in .env.

Usage:
    python DB_0_build_pipeline.py                  # run entire pipeline
    python DB_0_build_pipeline.py --from DB_2      # skip DB_1, start at DB_2
    python DB_0_build_pipeline.py --from DB_3      # re-run Claude extraction onward
    python DB_0_build_pipeline.py --from DB_4      # re-run UMLS map + NLP (cache exists)
    python DB_0_build_pipeline.py --from DB_5      # skip DB_1-4, run DB_5a+DB_5b in parallel
    python DB_0_build_pipeline.py --from DB_6      # just reassemble (DB_5a+DB_5b already done)
    python DB_0_build_pipeline.py --dry-run        # print what would run without executing

Last updated: March 2026, Maxim Seferovic, seferovi@bcm.edu
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import timedelta

# ── Use the same Python interpreter that launched this script ──────────────
PYTHON = sys.executable


# ── Project directory (where all DB_ scripts and data files live) ──────────
# Using abspath(__file__) means subprocess calls always resolve relative paths
# against the project folder, regardless of where the user invokes this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Pipeline step definitions ──────────────────────────────────────────────
# parallel_group: steps with the same non-None group value are launched together
# and we wait for all of them before proceeding.
STEPS = [
    {
        "name":           "DB_1",
        "script":         "DB_1_recreate.py",
        "outputs":        ["PBDBfinal.txt"],
        "parallel_group": None,
    },
    {
        "name":           "DB_2",
        "script":         "DB_2_clean.py",
        "outputs":        [
            "PBDBfinal_cleaned.csv",
            "PBDBfinal_meds.csv",
            "PBDBfinal_details.csv",
        ],
        "parallel_group": None,
    },
    {
        "name":           "DB_3",
        "script":         "DB_3_claude_extract.py",
        "outputs":        ["claude_extraction_cache.json"],
        "parallel_group": None,
    },
    {
        "name":           "DB_4",
        "script":         "DB_4_build_umls_map.py",
        "outputs":        ["token_to_concept.json"],
        "parallel_group": None,
    },
    {
        "name":           "DB_5a",
        "script":         "DB_5a_meds.py",
        "outputs":        ["PBDBfinal_meds_dictcorrect_bool.csv"],
        "parallel_group": "DB_5",
    },
    {
        "name":           "DB_5b",
        "script":         "DB_5b_NLP.py",
        "outputs":        ["PBDBfinal_details_tok.csv"],
        "parallel_group": "DB_5",
    },
    {
        "name":           "DB_6",
        "script":         "DB_6_reassemble_forML.py",
        "outputs":        ["PBDBfinal_ready_forML_IHCP_paper3.pkl"],
        "parallel_group": None,
    },
]

# Map step name → index for --from argument validation
STEP_START_MAP = {
    "DB_1": 0,
    "DB_2": 1,
    "DB_3": 2,   # Claude concept extraction (one-time, Batch API)
    "DB_4": 3,   # UMLS canonical-name map
    "DB_5": 4,   # DB_5a (meds) + DB_5b (NLP) in parallel
    "DB_6": 6,   # reassemble for ML
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _fmt(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def run_sequential(step: dict, dry_run: bool) -> None:
    """Run a single step synchronously; abort on failure."""
    cmd = [PYTHON, "-u", step["script"]]  # -u = unbuffered; ensures print() output
                                           # appears even if the child crashes
    print(f"\n>>> {step['name']}: {' '.join(cmd)}")
    if dry_run:
        return
    t0 = time.time()
    result = subprocess.run(cmd, check=False, cwd=SCRIPT_DIR)
    elapsed = _fmt(time.time() - t0)
    if result.returncode != 0:
        print(f"\n  ✗ {step['name']} FAILED after {elapsed} "
              f"(exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n  ✓ {step['name']} done in {elapsed}")


def run_parallel_group(steps: list, dry_run: bool) -> None:
    """Launch all steps in the group simultaneously; wait for all; abort on failure."""
    names = " + ".join(s["name"] for s in steps)
    cmds  = [[PYTHON, "-u", s["script"]] for s in steps]  # -u = unbuffered
    print(f"\n>>> {names} (running in parallel)")
    for cmd in cmds:
        print(f"    {' '.join(cmd)}")
    if dry_run:
        return

    t0    = time.time()
    procs = [subprocess.Popen(cmd, cwd=SCRIPT_DIR) for cmd in cmds]

    # Wait for all; collect return codes
    return_codes = [p.wait() for p in procs]
    elapsed      = _fmt(time.time() - t0)

    failed = [steps[i]["name"] for i, rc in enumerate(return_codes) if rc != 0]
    if failed:
        # Kill any still-running siblings (shouldn't be any after .wait(), but be safe)
        for p in procs:
            try:
                p.kill()
            except OSError:
                pass
        print(f"\n  ✗ {', '.join(failed)} FAILED after {elapsed} (wall time)")
        sys.exit(1)

    # Report individual timings (wall-clock only — subprocesses don't return elapsed)
    print(f"\n  ✓ {names} done — wall time {elapsed} "
          f"(ran in parallel; slowest determines total)")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PeriMiner pipeline orchestrator — builds the ML-ready pickle "
                    "from raw PeriBank exports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--from",
        dest="start_from",
        choices=list(STEP_START_MAP.keys()),
        default="DB_1",
        metavar="STEP",
        help="Start pipeline from this step "
             "(choices: DB_1 DB_2 DB_3 DB_4 DB_5 DB_6). "
             "Earlier steps are skipped entirely.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing anything.",
    )
    args = parser.parse_args()

    start_idx = STEP_START_MAP[args.start_from]
    active    = STEPS[start_idx:]

    print("=" * 60)
    print("  PeriMiner Pipeline Build")
    if args.dry_run:
        print("  [DRY RUN — nothing will be executed]")
    if args.start_from != "DB_1":
        print(f"  Starting from: {args.start_from}")
    print("=" * 60)

    wall_start    = time.time()
    seen_groups   = set()

    i = 0
    while i < len(active):
        step  = active[i]
        group = step["parallel_group"]

        if group is not None and group not in seen_groups:
            # Collect all steps in this parallel group
            group_steps = [s for s in active if s["parallel_group"] == group]
            seen_groups.add(group)
            run_parallel_group(group_steps, args.dry_run)
            # Skip past all steps in this group
            i += len(group_steps)
        elif group is not None:
            # Already handled as part of the group above
            i += 1
        else:
            run_sequential(step, args.dry_run)
            i += 1

    total = _fmt(time.time() - wall_start)
    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"  [DRY RUN complete — no files were created]")
    else:
        print(f"  Build complete in {total}")
        # Report final pickle if it exists
        pkl = next((s["outputs"][0] for s in STEPS
                    if s["name"] == "DB_6" and s["outputs"]), None)
        if pkl and os.path.exists(pkl):
            size_mb = os.path.getsize(pkl) / 1024 ** 2
            print(f"  Pickle: {pkl}  ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
