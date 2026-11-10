"""Does the subsampled full train fit in 8 GB?

For each negatives rate r and label, loads train_sample(scan_train_full(), ...) with all
candidate features and builds a CatBoost Pool, each in a fresh subprocess under /usr/bin/time -l.
The reported figure is macOS "peak memory footprint": RSS understates use because macOS
compresses memory (r = 0.5: RSS 1.9 GB, footprint 9.4 GB). Writes
reports/data_audit/train_memory.csv.

    uv run python scripts/check_train_memory.py
"""

import json
import re
import resource
import subprocess
import sys
import time

import polars as pl

from xai_ews.config import data_config
from xai_ews.paths import DATA_AUDIT_DIR


def one(label: str, r: float) -> dict:
    from catboost import Pool

    from xai_ews.data.features import candidate_features, categorical_features
    from xai_ews.data.panel import scan_train_full
    from xai_ews.data.split import train_sample

    t0 = time.time()
    feats = candidate_features()
    cats = [c for c in categorical_features() if c in feats]
    df = train_sample(scan_train_full(), label, r, feats).collect()
    rss_polars = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    x = df.select(
        *[pl.col(c).cast(pl.String).fill_null("NA") for c in cats],
        *[pl.col(c).cast(pl.Float32) for c in feats if c not in cats],
    ).to_pandas()
    pool = Pool(x, df[label].to_numpy(), cat_features=cats, weight=df["sample_weight"].to_numpy())
    return {"label": label, "r": r, "rows": df.height, "events": int(df[label].sum()),
            "weighted_rows": float(df["sample_weight"].sum()), "pool_rows": pool.num_row(),
            "peak_rss_mb_after_polars": round(rss_polars),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),
            "seconds": round(time.time() - t0, 1)}


def main() -> int:
    if len(sys.argv) == 3:
        print(json.dumps(one(sys.argv[1], float(sys.argv[2]))))
        return 0
    cfg = data_config()["panel"]["y2_negative_subsampling"]
    rows = []
    for r in sorted({cfg["negatives_rate"], *cfg["negatives_rate_sensitivity"]}):
        for label in ("y1", "y2"):
            proc = subprocess.run(["/usr/bin/time", "-l", sys.executable, __file__, label, str(r)],
                                  capture_output=True, text=True, check=True)
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            foot = re.search(r"(\d+)\s+peak memory footprint", proc.stderr)
            row["peak_footprint_mb"] = round(int(foot.group(1)) / 2**20) if foot else None
            rows.append(row)
            print(rows[-1], flush=True)
    pl.DataFrame(rows).write_csv(DATA_AUDIT_DIR / "train_memory.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
