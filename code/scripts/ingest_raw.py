"""Stream raw quarterly zips into sampled Parquet tables in data/interim/<quarter>/.

One quarter at a time, never unzipping to disk; see xai_ews.data.ingest for the outputs.
Quarters that already have ingest_stats.json (complete run) are skipped unless --overwrite.

    uv run python scripts/ingest_raw.py --quarters 2024Q1
    uv run python scripts/ingest_raw.py --all
    uv run python scripts/ingest_raw.py --quarters 2024Q1 --max-blocks 2 --out-root /tmp/x
    uv run python scripts/ingest_raw.py --train-full     # train vintages at 100 % -> interim/train_full
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from xai_ews.config import data_config
from xai_ews.data.ingest import BLOCK_BYTES, ingest_quarter
from xai_ews.paths import INTERIM_DIR, INTERIM_TRAIN_FULL_DIR, RAW_DIR


def is_done(out_root: Path, quarter: str) -> bool:
    stats = out_root / quarter / "ingest_stats.json"
    return stats.exists() and json.loads(stats.read_text()).get("complete", False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarters", nargs="*", default=[])
    parser.add_argument("--all", action="store_true", help="every vintage in configs/data.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-blocks", type=int, help="debug: stop after N blocks")
    parser.add_argument("--block-mb", type=int, default=BLOCK_BYTES // 2**20)
    parser.add_argument("--out-root", type=Path, default=INTERIM_DIR)
    parser.add_argument("--sample-rate", type=float, help="override panel.loan_sample_rate")
    parser.add_argument("--train-full", action="store_true",
                        help="panel.train_full vintages at its sample rate into interim/train_full")
    args = parser.parse_args()

    cfg = data_config()
    quarters = cfg["raw"]["vintages"] if args.all else args.quarters
    if args.train_full:
        quarters = cfg["panel"]["train_full"]["vintages"]
        args.sample_rate = cfg["panel"]["train_full"]["sample_rate"]
        args.out_root = INTERIM_TRAIN_FULL_DIR
    # Smallest archives first: early failures surface quickly.
    quarters = sorted(quarters, key=lambda q: (RAW_DIR / q / f"{q}.zip").stat().st_size)
    if not quarters:
        parser.error("give --quarters or --all")

    for q in quarters:
        if not args.overwrite and args.max_blocks is None and is_done(args.out_root, q):
            print(f"{q}: already ingested, skipping", flush=True)
            continue
        free_gb = shutil.disk_usage(args.out_root.parent if not args.out_root.exists()
                                    else args.out_root).free / 2**30
        if free_gb < 3:
            print(f"stopping before {q}: only {free_gb:.1f} GB free", file=sys.stderr)
            return 1
        stats = ingest_quarter(q, args.out_root, args.block_mb * 2**20, args.max_blocks,
                               args.sample_rate)
        size_mb = sum(stats["bytes"].values()) / 2**20
        print(f"{q}: {stats['rows_read']:,} rows, {stats['loans_sampled']:,}/"
              f"{stats['loans_total']:,} loans, {size_mb:.0f} MB, {stats['seconds']}s, "
              f"peak RSS {stats['peak_rss_mb']} MB, free {free_gb:.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
