"""Verify published clipped manuscript tables against immutable source records."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "paper" / "audit"
PUBLISHED = ROOT / "paper" / "data"


def main():
    subprocess.run([sys.executable, str(AUDIT / "audit_results.py")], check=True)
    for name in (
        "pointwise_clipped.csv",
        "per_run_summary.csv",
        "aggregate_summary.csv",
        "runtime_summary.json",
        "signed_error_components_raw.csv",
    ):
        rebuilt = (AUDIT / "results" / name).read_bytes()
        published = (PUBLISHED / name).read_bytes()
        if rebuilt != published:
            raise RuntimeError(f"Published paper/data/{name} differs from verified audit")
    print("Published clipped tables match the verified source-data audit.")


if __name__ == "__main__":
    main()
