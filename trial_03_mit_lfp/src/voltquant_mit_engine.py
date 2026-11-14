"""VoltQuant MIT Engine: Unified CLI for Trial 3 (MIT LFP Battery Cohort)

Provides a unified command-line interface for:
  1. Ingestion & automated imputation (m01)
  2. Reproduction of Severson et al. (Nature Energy 2019) baselines (m02)
  3. Comprehensive classical regression benchmark panel (m03)
  4. Physics-Guided Hybrid Quantum-Classical (PG-HQML) training & evaluation (m04)
  5. Publication figure generation (generate_mit_figures)
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"


def print_banner():
    banner = r"""
================================================================================
   _    __      _  __  ____                    __ 
  | |  / /___  / |/ /_/ __ \__  ______ _____  / /_
  | | / / __ \/    / / / / / / / / __ `/ __ \/ __/
  | |/ / /_/ / /| / / /_/ / /_/ / /_/ / / / / /_  
  |___/\____/_/ |_/  \___\_\__,_/\__,_/_/ /_/\__/   (Trial 3: MIT LFP Cohort)
================================================================================
  Early Battery Cycle-Life Prediction (Cycles 2-100) on 124 Commercial LFP Cells
"""
    print(banner)


def cmd_ingest():
    from m01_ingest_and_impute import ingest_and_impute
    ingest_and_impute()


def cmd_severson():
    from m02_reproduce_severson_baselines import run_severson_replication
    run_severson_replication()


def cmd_classical():
    from m03_classical_benchmark_panel import run_classical_benchmarks
    run_classical_benchmarks()


def cmd_hybrid():
    from m04_hybrid_pg_hqml import run_hybrid_benchmark
    run_hybrid_benchmark()


def cmd_figures():
    from generate_mit_figures import main as gen_figs
    gen_figs()


def cmd_all():
    print_banner()
    t0 = time.time()
    print(">>> STEP 1: INGESTION & IMPUTATION <<<")
    cmd_ingest()
    print("\n>>> STEP 2: REPRODUCING SEVERSON 2019 BASELINES <<<")
    cmd_severson()
    print("\n>>> STEP 3: CLASSICAL ML BENCHMARKS <<<")
    cmd_classical()
    print("\n>>> STEP 4: HYBRID PG-HQML EVALUATION <<<")
    cmd_hybrid()
    print("\n>>> STEP 5: GENERATING PUBLICATION FIGURES <<<")
    cmd_figures()
    elapsed = time.time() - t0
    print(f"\n[DONE] Full Trial 3 MIT pipeline completed in {elapsed:.1f}s.")


def main():
    parser = argparse.ArgumentParser(description="VoltQuant MIT LFP Unified CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    subparsers.add_parser("ingest", help="Ingest clean/raw data and impute b1c18 missing values")
    subparsers.add_parser("severson", help="Replicate Severson Nature Energy 2019 Table 1 baselines")
    subparsers.add_parser("classical", help="Run 5-model classical regression benchmark panel")
    subparsers.add_parser("hybrid", help="Train and evaluate Physics-Guided Hybrid Quantum-Classical model")
    subparsers.add_parser("figures", help="Generate publication-ready figures")
    subparsers.add_parser("all", help="Execute complete pipeline end-to-end")

    args = parser.parse_args()

    if args.command is None:
        print_banner()
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "ingest": cmd_ingest,
        "severson": cmd_severson,
        "classical": cmd_classical,
        "hybrid": cmd_hybrid,
        "figures": cmd_figures,
        "all": cmd_all
    }

    if args.command in dispatch:
        print_banner()
        dispatch[args.command]()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

