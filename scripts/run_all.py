"""Run the full Phase-1 experiment and write tables, figures and results.json.

Usage:
    python scripts/run_all.py                    # 300 Monte Carlo realizations
    python scripts/run_all.py --realizations 50  # quicker run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voi_tinyml.experiments import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results", help="output directory (default: results)")
    ap.add_argument("--realizations", type=int, default=300, help="Monte Carlo link realizations")
    args = ap.parse_args()
    run(args.out, args.realizations)
