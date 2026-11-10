"""External macro data: FHFA HPI, BLS LAUS state unemployment, Freddie Mac PMMS.

Raw downloads go to data/external/raw/ and are listed in data/external/manifest.csv (url, bytes,
sha256, download date). The loaders below turn them into tidy polars frames keyed by month or
quarter. Sources (checked 2026-10-24):

- FHFA all-transactions HPI, quarterly, NSA: 3-digit ZIP (xlsx) and state (csv). FHFA does not
  publish a monthly state index; the monthly purchase-only index exists only for census divisions.
- BLS LAUS seasonally adjusted state unemployment rate, series LASST{fips}0000000000003, via the
  public API v1 (download.bls.gov rejects scripted requests with 403).
- Freddie Mac PMMS weekly history (30-year fixed), from freddiemac.com: the primary source.
  FRED (which mirrors it) did not respond from this machine.
"""

import csv
import hashlib
import json
import time
import urllib.request
from datetime import date
from pathlib import Path

import polars as pl

from xai_ews.paths import EXTERNAL_DIR

RAW = EXTERNAL_DIR / "raw"
MANIFEST = EXTERNAL_DIR / "manifest.csv"
MANIFEST_FIELDS = ["file", "source", "url", "bytes", "sha256", "downloaded"]
USER_AGENT = "Mozilla/5.0 (xai-ews thesis research script)"

FHFA = "https://www.fhfa.gov/hpi/download/quarterly_datasets/"
FILES = {
    "hpi_at_3zip.xlsx": ("FHFA HPI all-transactions, 3-digit ZIP, quarterly NSA",
                         FHFA + "hpi_at_3zip.xlsx"),
    "hpi_at_state.csv": ("FHFA HPI all-transactions, state, quarterly NSA",
                         FHFA + "hpi_at_state.csv"),
    "hpi_at_pr.xlsx": ("FHFA HPI all-transactions, Puerto Rico, quarterly NSA (developmental)",
                       FHFA + "hpi_at_pr.xlsx"),
    "hpi_at_us_and_census.csv": ("FHFA HPI all-transactions, USA and census divisions, quarterly NSA",
                                 FHFA + "hpi_at_us_and_census.csv"),
    "PMMS_history.csv": ("Freddie Mac Primary Mortgage Market Survey, weekly",
                         "https://www.freddiemac.com/pmms/docs/PMMS_history.csv"),
}
BLS_API = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
BLS_US_SERIES = "LNS14000000"  # national unemployment rate, SA (CPS)
BLS_FIRST_YEAR = 2000

STATE_FIPS = {
    "AL": 1, "AK": 2, "AZ": 4, "AR": 5, "CA": 6, "CO": 8, "CT": 9, "DE": 10, "DC": 11, "FL": 12,
    "GA": 13, "HI": 15, "ID": 16, "IL": 17, "IN": 18, "IA": 19, "KS": 20, "KY": 21, "LA": 22,
    "ME": 23, "MD": 24, "MA": 25, "MI": 26, "MN": 27, "MS": 28, "MO": 29, "MT": 30, "NE": 31,
    "NV": 32, "NH": 33, "NJ": 34, "NM": 35, "NY": 36, "NC": 37, "ND": 38, "OH": 39, "OK": 40,
    "OR": 41, "PA": 42, "RI": 44, "SC": 45, "SD": 46, "TN": 47, "TX": 48, "UT": 49, "VT": 50,
    "VA": 51, "WA": 53, "WV": 54, "WI": 55, "WY": 56, "PR": 72,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_to_manifest(records: list[dict]) -> None:
    """Replace or append manifest rows for the given files."""
    rows = []
    if MANIFEST.exists():
        with open(MANIFEST, newline="") as f:
            rows = list(csv.DictReader(f))
    names = {r["file"] for r in records}
    rows = [r for r in rows if r["file"] not in names] + records
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _record(name: str, source: str, url: str) -> dict:
    path = RAW / name
    return {"file": f"raw/{name}", "source": source, "url": url, "bytes": path.stat().st_size,
            "sha256": _sha256(path), "downloaded": date.today().isoformat()}


def _fetch(url: str, data: bytes | None = None, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _bls_series_id(state: str) -> str:
    return f"LASST{STATE_FIPS[state]:02d}0000000000003"


def download_bls(last_year: int, series: list[str] | None = None, prefix: str = "bls_laus") -> list[Path]:
    """API v1 allows 25 series and 10 years per request; one JSON file per request."""
    all_ids = series or [_bls_series_id(s) for s in sorted(STATE_FIPS)]
    paths = []
    for i in range(0, len(all_ids), 25):
        ids = all_ids[i : i + 25]
        for start in range(BLS_FIRST_YEAR, last_year + 1, 10):
            end = min(start + 9, last_year)
            body = json.dumps({"seriesid": ids, "startyear": str(start), "endyear": str(end)})
            raw = _fetch(BLS_API, body.encode(), {"Content-Type": "application/json"})
            reply = json.loads(raw)
            if reply.get("status") != "REQUEST_SUCCEEDED":
                raise RuntimeError(f"BLS API: {reply.get('status')} {reply.get('message')}")
            path = RAW / f"{prefix}_{i // 25}_{start}_{end}.json"
            path.write_bytes(raw)
            paths.append(path)
            time.sleep(1)
    return paths


def download_all(last_year: int | None = None) -> list[dict]:
    RAW.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (source, url) in FILES.items():
        (RAW / name).write_bytes(_fetch(url))
        records.append(_record(name, source, url))
        print(f"{name}: {records[-1]['bytes']:,} bytes", flush=True)
    year = last_year or date.today().year
    for path in download_bls(year):
        records.append(_record(path.name, "BLS LAUS state unemployment rate, SA (API v1)", BLS_API))
        print(f"{path.name}: {records[-1]['bytes']:,} bytes", flush=True)
    for path in download_bls(year, [BLS_US_SERIES], "bls_us"):
        records.append(_record(path.name, f"BLS CPS US unemployment rate {BLS_US_SERIES}, SA (API v1)",
                               BLS_API))
        print(f"{path.name}: {records[-1]['bytes']:,} bytes", flush=True)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(records)
    return records


def load_hpi_zip3() -> pl.DataFrame:
    """zip3 (3-char string), year, quarter, hpi, native (False = FHFA 'Balance of State' fill)."""
    import pandas as pd

    df = pd.read_excel(RAW / "hpi_at_3zip.xlsx", header=None, skiprows=5,
                       names=["zip3", "year", "quarter", "hpi", "index_type"])
    return pl.from_pandas(df).select(
        pl.col("zip3").cast(pl.Int32).cast(pl.String).str.zfill(3),
        pl.col("year").cast(pl.Int16),
        pl.col("quarter").cast(pl.Int8),
        pl.col("hpi").cast(pl.Float64, strict=False),
        pl.col("index_type").str.starts_with("Native").alias("native"),
    ).drop_nulls("hpi")


def load_hpi_state() -> pl.DataFrame:
    """state, year, quarter, hpi; the state file has no Puerto Rico, FHFA publishes it apart."""
    import pandas as pd

    states = pl.read_csv(
        RAW / "hpi_at_state.csv", has_header=False,
        new_columns=["state", "year", "quarter", "hpi"],
        schema_overrides={"year": pl.Int16, "quarter": pl.Int8, "hpi": pl.Float64},
        null_values=["-", ""],
    )
    pr = pd.read_excel(RAW / "hpi_at_pr.xlsx", header=None, skiprows=5,
                       names=["state", "year", "quarter", "hpi"])
    pr = pl.from_pandas(pr).select(pl.col("state").cast(pl.String), pl.col("year").cast(pl.Int16),
                                   pl.col("quarter").cast(pl.Int8),
                                   pl.col("hpi").cast(pl.Float64, strict=False))
    return pl.concat([states, pr]).drop_nulls("hpi")


def load_pmms() -> pl.DataFrame:
    """Monthly mean of the weekly 30-year fixed rate: month (Date, 1st of month), pmms30."""
    weekly = pl.read_csv(RAW / "PMMS_history.csv", columns=["date", "pmms30"],
                         schema_overrides={"date": pl.String, "pmms30": pl.String})
    return (
        weekly.select(
            pl.col("date").str.to_date("%m/%d/%Y").dt.truncate("1mo").alias("month"),
            pl.col("pmms30").str.strip_chars().cast(pl.Float64, strict=False),
        )
        .drop_nulls()
        .group_by("month").agg(pl.col("pmms30").mean())
        .sort("month")
    )


def load_hpi_us() -> pl.DataFrame:
    """year, quarter, hpi for the USA (all-transactions)."""
    df = pl.read_csv(RAW / "hpi_at_us_and_census.csv", has_header=False,
                     new_columns=["area", "year", "quarter", "hpi"],
                     schema_overrides={"year": pl.Int16, "quarter": pl.Int8, "hpi": pl.Float64},
                     null_values=["-", ""])
    return df.filter(pl.col("area") == "USA").drop("area").drop_nulls("hpi")


def load_unemployment() -> pl.DataFrame:
    """state, month (Date), unemployment rate (%), seasonally adjusted; state 'US' = national."""
    fips_to_state = {_bls_series_id(s): s for s in STATE_FIPS} | {BLS_US_SERIES: "US"}
    rows = []
    for path in sorted([*RAW.glob("bls_laus_*.json"), *RAW.glob("bls_us_*.json")]):
        for series in json.loads(path.read_text())["Results"]["series"]:
            state = fips_to_state[series["seriesID"]]
            for obs in series["data"]:
                if obs["period"].startswith("M") and obs["period"] != "M13" and obs["value"] != "-":
                    rows.append((state, date(int(obs["year"]), int(obs["period"][1:]), 1),
                                 float(obs["value"])))
    return (pl.DataFrame(rows, schema=["state", "month", "unemployment"], orient="row")
            .unique(["state", "month"]).sort("state", "month"))
