"""Refresh the raw OWID snapshots used by the experiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carbon_intensity_workflow import audit_sources, download_sources


if __name__ == "__main__":
    files = download_sources(force=True)
    provenance = audit_sources()
    print(f"Downloaded {len(files)} files")
    print(f"Target years: {provenance['target_year_range']}")
    print(f"Target rows: {provenance['target_shape'][0]}")
