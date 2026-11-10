"""Build the loan x observation-date panel from data/interim/<vintage>/ into
data/processed/panel/vintage=<Q>/part.parquet (hive-partitioned; read with
pl.scan_parquet(PANEL_DIR, hive_partitioning=True)).

Loans of a vintage are processed in batches of about --batch-rows monthly rows to stay within
8 GB of RAM.

--train-full builds the train observations from the train vintages ingested at
100 % (data/interim/train_full/) into data/processed/train_full/vintage=<Q>/: only rows with
split_main == "train" (holdout loans excluded), months after train end + horizon are not read,
and each row gets a deterministic draw `u_neg` for negative subsampling
(xai_ews.data.split.train_sample). The exposure norm (mean ln(1 + UPB) on train, mean 1) is
fitted on this full train population and written to data/processed/panel/exposure_norm.json.

    uv run python scripts/build_panel.py --vintages 2024Q1
    uv run python scripts/build_panel.py --all
    uv run python scripts/build_panel.py --train-full
"""

import argparse
import json
import resource
import sys
import time

import polars as pl

from xai_ews.config import data_config
from xai_ews.data.panel import (
    Macro,
    build_vintage_panel,
    load_macro,
    month_index,
    scan_panel,
    scan_train_full,
    ym_to_index,
)
from xai_ews.data.split import add_negative_draw, assign_splits
from xai_ews.paths import INTERIM_DIR, INTERIM_TRAIN_FULL_DIR, PANEL_DIR, TRAIN_FULL_DIR

ID = "loan_identifier"


def build_one(vintage: str, macro: Macro, cfg: dict, batch_rows: int,
              train_full: bool = False) -> dict:
    t0 = time.time()
    src = (INTERIM_TRAIN_FULL_DIR if train_full else INTERIM_DIR) / vintage
    dst = TRAIN_FULL_DIR if train_full else PANEL_DIR
    loans = pl.read_parquet(src / "loans.parquet")
    perf_lf = pl.scan_parquet(src / "performance.parquet")
    if train_full:  # labels of the last train date end at train end + horizon
        last = ym_to_index(cfg["split"]["train"]["end"]) + cfg["labels"]["early_default"]["horizon_months"]
        perf_lf = perf_lf.filter(month_index("monthly_reporting_period") <= last)
    # Loan row counts -> contiguous batches of loans.
    counts = perf_lf.group_by(ID).len().collect().sort(ID)
    counts = counts.with_columns((pl.col("len").cum_sum() // batch_rows).alias("batch"))

    parts, stats = [], {}
    for b in range(int(counts["batch"].max()) + 1):
        ids = counts.filter(pl.col("batch") == b)[ID]
        perf = perf_lf.filter(pl.col(ID).is_in(ids.implode())).collect()
        panel, s = build_vintage_panel(perf, loans.filter(pl.col(ID).is_in(ids.implode())),
                                       macro, cfg)
        panel = assign_splits(panel, cfg)
        if train_full:
            panel = add_negative_draw(panel.filter(pl.col("split_main") == "train"), cfg)
        parts.append(panel)
        for k, v in s.items():
            stats[k] = stats.get(k, 0) + v
        del perf, panel

    out = dst / f"vintage={vintage}"
    out.mkdir(parents=True, exist_ok=True)
    panel = pl.concat(parts, how="vertical_relaxed").drop("vintage")  # the partition key
    panel.write_parquet(out / "part.parquet", compression="zstd")
    stats.update(vintage=vintage, loans=loans.height, batches=b + 1, rows_written=panel.height,
                 bytes=(out / "part.parquet").stat().st_size, seconds=round(time.time() - t0, 1),
                 peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20))
    (dst / "stats").mkdir(exist_ok=True)
    (dst / "stats" / f"{vintage}.json").write_text(json.dumps(stats, indent=2))
    return stats


def write_exposure_norm() -> dict:
    """Fitted on the full train population (train_full) when it exists, else on panel train."""
    full = any(TRAIN_FULL_DIR.glob("vintage=*"))
    train = (scan_train_full() if full else scan_panel()).filter(pl.col("split_main") == "train") \
        .select("upb")
    p99 = train.select(pl.col("upb").quantile(0.99)).collect().item()
    norm = train.select(
        pl.col("upb").log1p().mean().alias("mean_log1p_upb"),
        pl.col("upb").clip(None, p99).mean().alias("mean_upb_capped_p99"),
        pl.len().alias("train_rows"),
    ).collect().row(0, named=True)
    norm["upb_p99"] = p99
    norm["fitted_on"] = "train_full (100 % of train vintages)" if full else "panel train (15 %)"
    (PANEL_DIR / "exposure_norm.json").write_text(json.dumps(norm, indent=2))
    return norm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vintages", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-rows", type=int, default=1_500_000)
    parser.add_argument("--train-full", action="store_true")
    args = parser.parse_args()

    cfg = data_config()
    vintages = cfg["raw"]["vintages"] if args.all else args.vintages
    if args.train_full:
        vintages = cfg["panel"]["train_full"]["vintages"]
    src_root = INTERIM_TRAIN_FULL_DIR if args.train_full else INTERIM_DIR
    dst_root = TRAIN_FULL_DIR if args.train_full else PANEL_DIR
    if not vintages:
        parser.error("give --vintages or --all")
    macro = load_macro()
    for v in vintages:
        if not (src_root / v / "ingest_stats.json").exists():
            print(f"{v}: not ingested, skipping", flush=True)
            continue
        if not args.overwrite and (dst_root / "stats" / f"{v}.json").exists():
            print(f"{v}: panel exists, skipping", flush=True)
            continue
        s = build_one(v, macro, cfg, args.batch_rows, args.train_full)
        print(f"{v}: {s['rows_written']:,} rows from {s['loans']:,} loans, "
              f"{s['bytes'] / 2**20:.0f} MB, {s['seconds']}s, peak RSS {s['peak_rss_mb']} MB",
              flush=True)
    if any(PANEL_DIR.glob("vintage=*")):
        print("exposure norm:", write_exposure_norm())
    return 0


if __name__ == "__main__":
    sys.exit(main())
