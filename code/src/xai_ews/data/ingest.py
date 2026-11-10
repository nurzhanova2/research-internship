"""Stream one raw quarterly zip into sampled, typed Parquet tables in data/interim/<quarter>/.

The raw member is never unzipped to disk and never handed to polars as a stream (polars buffers
the whole multi-GB member). Instead ~100 MB byte blocks are read from the zip, cut at the last
newline, and each block is parsed on its own with only the needed columns.

Outputs per quarter:
    loans.parquet          one row per sampled loan: origination (static) fields, taken from the
                           loan's first record, plus the two hash draws used for sampling/holdout
    performance.parquet    monthly records of sampled loans, dynamic fields only, in file order
                           (grouped by loan, months increasing — asserted while reading)
    credit_events.parquet  termination records with a default zero-balance code (02/03/09/15) for
                           ALL loans, not only sampled ones: LGD segments need more liquidations
                           than a 15 % sample gives; static fields included for MTM LTV
    ingest_stats.json      row/loan counts, zero-balance code counts, order and data checks,
                           timing, peak memory, output sizes

Loans are sampled deterministically: u = blake2b(seed:salt:loan_identifier) / 2^64 ~ U[0, 1),
a loan is kept iff u_sample < loan_sample_rate. The holdout draw uses another salt, so it is
independent of the sampling draw.
"""

import hashlib
import io
import json
import resource
import time
import zipfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from xai_ews.config import data_config
from xai_ews.data.schema import DELIMITER, column_names
from xai_ews.paths import INTERIM_DIR, RAW_DIR

BLOCK_BYTES = 100 * 1024 * 1024
ID = "loan_identifier"
PERIOD = "monthly_reporting_period"
ZBC = "zero_balance_code"

# Origination characteristics: one value per loan (checked, counts in ingest_stats.json).
STATIC = {
    "channel": pl.String,
    "seller_name": pl.String,
    "original_interest_rate": pl.Float64,
    "original_upb": pl.Float64,
    "original_loan_term": pl.Int16,
    "origination_date": pl.Date,
    "first_payment_date": pl.Date,
    "original_loan_to_value_ratio_ltv": pl.Int16,
    "original_combined_loan_to_value_ratio_cltv": pl.Int16,
    "number_of_borrowers": pl.Int8,
    "debt_to_income_dti": pl.Int16,
    "borrower_credit_score_at_origination": pl.Int16,
    "co_borrower_credit_score_at_origination": pl.Int16,
    "first_time_home_buyer_indicator": pl.String,
    "loan_purpose": pl.String,
    "property_type": pl.String,
    "number_of_units": pl.Int8,
    "occupancy_status": pl.String,
    "property_state": pl.String,
    "metropolitan_statistical_area_msa_or_metropolitan_statistical_division_area_msda": pl.String,
    "zip_code_short": pl.String,
    "mortgage_insurance_percentage": pl.Float32,
    "amortization_type": pl.String,
    "prepayment_penalty_indicator": pl.String,
    "interest_only_loan_indicator": pl.String,
    "mortgage_insurance_type": pl.String,
    "special_eligibility_program": pl.String,
    "relocation_mortgage_indicator": pl.String,
    "property_valuation_method": pl.String,
    "high_balance_loan_indicator": pl.String,
    "high_loan_to_value_hltv_refinance_option_indicator": pl.String,
}

# Fields that change month to month, including termination and loss fields (label/LGD only).
DYNAMIC = {
    "servicer_name": pl.String,
    "current_interest_rate": pl.Float64,
    "current_actual_upb": pl.Float64,
    "loan_age": pl.Int16,
    "remaining_months_to_legal_maturity": pl.Int16,
    "remaining_months_to_maturity": pl.Int16,
    "maturity_date": pl.Date,
    "current_loan_delinquency_status": pl.Int16,  # "XX" (unknown) -> null
    "modification_flag": pl.String,
    ZBC: pl.String,
    "zero_balance_effective_date": pl.Date,
    "upb_at_the_time_of_removal": pl.Float64,
    "last_paid_installment_date": pl.Date,
    "foreclosure_date": pl.Date,
    "disposition_date": pl.Date,
    "foreclosure_costs": pl.Float64,
    "property_preservation_and_repair_costs": pl.Float64,
    "asset_recovery_costs": pl.Float64,
    "miscellaneous_holding_expenses_and_credits": pl.Float64,
    "associated_taxes_for_holding_property": pl.Float64,
    "net_sales_proceeds": pl.Float64,
    "credit_enhancement_proceeds": pl.Float64,
    "repurchase_make_whole_proceeds": pl.Float64,
    "other_foreclosure_proceeds": pl.Float64,
    "modification_related_non_interest_bearing_upb": pl.Float64,
    "principal_forgiveness_amount": pl.Float64,
    "foreclosure_principal_write_off_amount": pl.Float64,
    "servicing_activity_indicator": pl.String,
    "borrower_assistance_plan": pl.String,
    "alternative_delinquency_resolution": pl.String,
    "alternative_delinquency_resolution_count": pl.Int16,
    "total_deferral_amount": pl.Float64,
    "payment_deferral_modification_event_indicator": pl.String,
}

KEYS = {ID: pl.String, PERIOD: pl.Date}
SCHEMA = KEYS | STATIC | DYNAMIC


def uniform_draw(ids: list[str], seed: int, salt: str) -> np.ndarray:
    """Deterministic U[0, 1) per loan id, stable across library versions and machines."""
    out = np.empty(len(ids), dtype=np.float64)
    prefix = f"{seed}:{salt}:".encode()
    for i, loan_id in enumerate(ids):
        digest = hashlib.blake2b(prefix + loan_id.encode(), digest_size=8).digest()
        out[i] = int.from_bytes(digest, "big") / 2**64
    return out


def iter_blocks(zip_path: Path, block_bytes: int = BLOCK_BYTES) -> Iterator[bytes]:
    """Yield blocks of whole lines from the single CSV member of a raw zip."""
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        if len(members) != 1:
            raise ValueError(f"{zip_path}: expected one member, got {[m.filename for m in members]}")
        tail = b""
        with zf.open(members[0]) as f:
            while chunk := f.read(block_bytes):
                data = tail + chunk
                cut = data.rfind(b"\n") + 1
                if cut == 0:
                    tail = data
                    continue
                tail = data[cut:]
                yield data[:cut]
        if tail.strip():
            yield tail + b"\n"


def parse_block(block: bytes, columns: list[str] | None = None) -> pl.DataFrame:
    """Parse one block of raw lines; every column is read as a string."""
    names = column_names()
    wanted = set(columns or SCHEMA)
    positions = [i for i, name in enumerate(names) if name in wanted]
    df = pl.read_csv(
        io.BytesIO(block),
        separator=DELIMITER,
        has_header=False,
        columns=positions,
        infer_schema=False,
        quote_char=None,
    )
    # Without a header polars (2.x) names columns column_0..column_112 by 0-based position.
    return df.rename({f"column_{i}": names[i] for i in positions})


def _mmyyyy_to_date(col: str) -> pl.Expr:
    return pl.concat_str(pl.lit("01"), pl.col(col)).str.to_date("%d%m%Y", strict=False)


def cast_types(df: pl.DataFrame) -> pl.DataFrame:
    """Raw strings -> typed columns; empty strings and 'XX' become null."""
    exprs = []
    for name, dtype in SCHEMA.items():
        if name not in df.columns:
            continue
        col = pl.col(name).replace("", None)
        if dtype == pl.Date:
            exprs.append(_mmyyyy_to_date(name).alias(name))
        elif dtype == pl.String:
            exprs.append(col.alias(name))
        else:
            exprs.append(col.cast(dtype, strict=False).alias(name))
    return df.with_columns(exprs)


class OrderCheck:
    """Assert that rows come grouped by loan with strictly increasing months within a loan.

    Grouping across blocks: a block may only start with the loan the previous block ended on
    (a loan straddling the boundary) or with a loan never seen before.
    """

    def __init__(self) -> None:
        self.seen: set[int] = set()
        self.last_id: str | None = None
        self.last_period = None
        self.gap_rows = 0

    def update(self, ids: pl.Series, periods: pl.Series) -> None:
        df = pl.DataFrame({"id": ids, "p": periods})
        new_run = df["id"] != df["id"].shift(1)
        same = ~new_run.fill_null(True)
        runs = int(new_run.fill_null(True).sum())
        uniq = df["id"].n_unique()
        assert runs == uniq, f"loans not grouped within block: {runs} runs, {uniq} loans"

        step = (df["p"].dt.year().cast(pl.Int32) * 12 + df["p"].dt.month().cast(pl.Int32)).diff()
        bad = (same & (step <= 0)).sum()
        assert bad == 0, f"{bad} rows with non-increasing month within a loan"
        self.gap_rows += int((same & (step > 1)).sum())

        first_id, first_p = df["id"][0], df["p"][0]
        if first_id == self.last_id:
            assert first_p > self.last_period, f"month order broken across blocks for {first_id}"
        else:
            assert int(first_id) not in self.seen, f"loan {first_id} reappears after other loans"
        block_ids = df["id"].unique().cast(pl.Int64).to_list()
        self.seen.update(block_ids)
        self.last_id, self.last_period = df["id"][-1], df["p"][-1]


def _static_changes(df: pl.DataFrame) -> dict[str, int]:
    """Number of loans whose 'static' field takes more than one non-null value in this block."""
    counts = df.group_by(ID).agg(pl.col(c).drop_nulls().n_unique() for c in STATIC)
    return {c: int((counts[c] > 1).sum()) for c in STATIC}


def ingest_quarter(
    quarter: str,
    out_root: Path = INTERIM_DIR,
    block_bytes: int = BLOCK_BYTES,
    max_blocks: int | None = None,
    sample_rate: float | None = None,
) -> dict:
    """sample_rate overrides panel.loan_sample_rate (1.0 for the full train vintages). The
    hash draw is the same, so a 15 % ingest is a subset of a full one."""
    cfg = data_config()
    rate = cfg["panel"]["loan_sample_rate"] if sample_rate is None else sample_rate
    seed = cfg["panel"]["sample_seed"]
    default_codes = set(cfg["labels"]["early_default"]["default_zero_balance_codes"])

    zip_path = RAW_DIR / quarter / f"{quarter}.zip"
    out_dir = out_root / quarter
    out_dir.mkdir(parents=True, exist_ok=True)
    perf_tmp = out_dir / "performance.parquet.tmp"

    t0 = time.time()
    check = OrderCheck()
    zbc_counts: Counter = Counter()
    static_changes: Counter = Counter()
    loans_parts, events_parts = [], []
    rows_read = rows_kept = 0
    writer = None
    dyn_cols = [ID, PERIOD, *DYNAMIC]

    try:
        for i, block in enumerate(iter_blocks(zip_path, block_bytes)):
            if max_blocks is not None and i >= max_blocks:
                break
            raw = parse_block(block)
            rows_read += raw.height
            df = cast_types(raw)
            del raw
            check.update(df[ID], df[PERIOD])

            zbc = df[ZBC].drop_nulls()
            zbc_counts.update(dict(zbc.value_counts().iter_rows()))
            events = df.filter(pl.col(ZBC).is_in(list(default_codes)))
            if events.height:
                events_parts.append(events)

            ids = df[ID].unique(maintain_order=True)
            u = uniform_draw(ids.to_list(), seed, "sample")
            draws = pl.DataFrame({ID: ids, "u_sample": u}).filter(pl.col("u_sample") < rate)
            kept = df.join(draws.select(ID), on=ID, how="semi")
            rows_kept += kept.height
            if kept.height == 0:
                continue

            static_changes.update(_static_changes(kept))
            first = kept.unique(subset=ID, keep="first", maintain_order=True)
            loans_parts.append(first.select(ID, PERIOD, *STATIC).join(draws, on=ID))

            table = kept.select(dyn_cols).to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(perf_tmp, table.schema, compression="zstd")
            writer.write_table(table)
            print(f"{quarter} block {i}: {rows_read:,} rows read, {rows_kept:,} kept, "
                  f"{time.time() - t0:.0f}s", flush=True)
    finally:
        if writer is not None:
            writer.close()

    # A loan straddling a block boundary appears in both blocks; keep its first record.
    loans = (
        pl.concat(loans_parts)
        .unique(subset=ID, keep="first", maintain_order=True)
        .rename({PERIOD: "first_period"})
    )
    loans = loans.with_columns(
        pl.lit(quarter).alias("vintage"),
        pl.Series("u_holdout", uniform_draw(loans[ID].to_list(), seed, "holdout")),
    )
    loans.write_parquet(out_dir / "loans.parquet", compression="zstd")
    perf_tmp.replace(out_dir / "performance.parquet")
    events = (pl.concat(events_parts) if events_parts else df.clear()).with_columns(
        pl.lit(quarter).alias("vintage")
    )
    events.write_parquet(out_dir / "credit_events.parquet", compression="zstd")

    sizes = {p.name: p.stat().st_size for p in out_dir.glob("*.parquet")}
    stats = {
        "quarter": quarter,
        "sample_rate": rate,
        "seed": seed,
        "rows_read": rows_read,
        "rows_kept": rows_kept,
        "loans_total": len(check.seen),
        "loans_sampled": loans.height,
        "zero_balance_codes_all_loans": dict(sorted(zbc_counts.items())),
        "credit_events_all_loans": events.height,
        "gap_rows_month_skipped": check.gap_rows,
        "static_field_changes_within_loan": {k: v for k, v in static_changes.items() if v},
        "order_check": "pass",
        "complete": max_blocks is None,
        "seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),
        "bytes": sizes,
    }
    with open(out_dir / "ingest_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats
