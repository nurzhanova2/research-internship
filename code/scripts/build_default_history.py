"""First quarter-end snapshot at which a loan has 90+ DPD in its trailing 12 months (dpd_max_12m >= 3),
over the panel and the full train. Snapshots are quarterly and the window is 12 months, so every
90+ episode before t is seen; the Y1 population drops a loan from that snapshot on
(labels.early_default.exclude_after_default).

    uv run python scripts/build_default_history.py
"""

import sys

import polars as pl

from xai_ews.paths import DEFAULT_HISTORY, PANEL_DIR, TRAIN_FULL_DIR


def main() -> int:
    parts = [pl.scan_parquet(d / "vintage=*" / "*.parquet", hive_partitioning=True)
             .filter(pl.col("dpd_max_12m") >= 3).select("loan_identifier", "m")
             for d in (PANEL_DIR, TRAIN_FULL_DIR)]
    hist = (pl.concat(parts).group_by("loan_identifier").agg(pl.col("m").min().alias("first_default_m"))
            .sort("loan_identifier").collect(engine="streaming"))
    hist.write_parquet(DEFAULT_HISTORY)
    print(f"{hist.height:,} loans with a 90+ DPD episode -> {DEFAULT_HISTORY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
