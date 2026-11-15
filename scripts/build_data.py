"""Rebuild data/processed/ from the raw FIRMS archives in data/raw/.

Usage:  python scripts/build_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from firewatch import harmonize, ingest  # noqa: E402

if __name__ == "__main__":
    ingest.run()
    _, v = harmonize.run()
    for k, m in v["metrics"].items():
        print(f"{k:16s} cell-month R2={m['cell_month']['r2']:.3f}  "
              f"monthly-total R2={m['domain_month']['r2']:.3f}  "
              f"yearly-total bias={m['domain_year']['bias_pct']:+.1f}%")
