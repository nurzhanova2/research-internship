"""Download external macro data into data/external/raw/ and write data/external/manifest.csv.

FHFA HPI (3-digit ZIP and state, quarterly), BLS LAUS state unemployment (monthly, SA),
Freddie Mac PMMS (weekly 30-year rate). Then parse each file once to check it loads.

    uv run python scripts/download_external.py
"""

import sys

from xai_ews.data import external


def main() -> int:
    external.download_all()
    for name, loader in [("hpi_zip3", external.load_hpi_zip3), ("hpi_state", external.load_hpi_state),
                         ("pmms", external.load_pmms), ("unemployment", external.load_unemployment)]:
        df = loader()
        time_cols = [c for c in ("year", "month") if c in df.columns]
        span = f"{df[time_cols[0]].min()}..{df[time_cols[0]].max()}" if time_cols else ""
        print(f"{name}: {df.height:,} rows, {span}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
