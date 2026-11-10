"""Natural-disaster exclusion rule.

An observation (loan, month t) is excluded from validation, test and stress evaluation if a FEMA
Major Disaster with assistance to individuals began in (t - 3 months, t + 12 months] and the
loan's 3-digit ZIP is affected. The rule uses only external events, never loan outcomes.

    declarations  OpenFEMA Disaster Declarations Summaries (v2): declarationType "DR" with
                  iaProgramDeclared or ihProgramDeclared (IA was replaced by the Individuals and
                  Households Program in 2002-2008; the IA flag is always 0 after 2008);
                  all incident types; date = incidentBeginDate; designated areas = counties
                  (tribal areas with county code 000 have no county and are not mapped).
                  COVID-19 declarations (incidentType "Biological") are not applied inside the
                  COVID stress scenario: the pandemic is the scenario itself.
    ZIP3 share    Census 2010 ZCTA-to-county relationship file (the HUD USPS
                  crosswalk planned at first is not reachable from outside the US). HUPT = housing
                  units of the ZCTA inside the county. A ZIP3 is affected if >= 50 % of its housing
                  units (summed over its ZCTAs) lie in declared counties. ZCTAs approximate
                  residential ZIP codes; county codes are those of 2010 (later changes, e.g. the
                  Connecticut planning regions of 2022, are reported as unmatched).
    territories   the Virgin Islands, Guam and the Northern Mariana Islands have no ZCTAs: their
                  ZIP3 counts as affected when any of their counties is declared.

Flags per (zip3, month m) are written to data/processed/disaster_flags.parquet; models join them
on zip_code_short and m.
"""

import numpy as np
import polars as pl

from xai_ews.data.panel import ym_to_index
from xai_ews.paths import EXTERNAL_DIR

RAW = EXTERNAL_DIR / "raw"
FEMA_FILE = RAW / "DisasterDeclarationsSummaries.csv"
FEMA_URL = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries.csv"
ZCTA_FILE = RAW / "zcta_county_rel_10.txt"
ZCTA_URL = "https://www2.census.gov/geo/docs/maps-data/data/rel/zcta_county_rel_10.txt"
TERRITORY_ZIP3 = {"78": ["008"], "66": ["969"], "69": ["969"]}   # VI, GU, MP: no ZCTAs
WINDOW = (-2, 12)                               # incident month in [m - 2, m + 12] = (t-3m, t+12m]
SHARE_MIN = 0.5
HURRICANE_TYPES = ("Hurricane", "Tropical Storm", "Typhoon")
BIOLOGICAL = "Biological"


def month_of(date_col: str) -> pl.Expr:
    d = pl.col(date_col).str.slice(0, 10).str.to_date()
    return d.dt.year().cast(pl.Int32) * 12 + d.dt.month().cast(pl.Int32) - 1


def load_declarations(path=FEMA_FILE) -> pl.DataFrame:
    """One row per (disaster, county): disaster, state, incident_type, m_incident, year, county
    (5-digit FIPS), tribal (no county)."""
    d = pl.read_csv(path, infer_schema=False)
    d = d.filter((pl.col("declarationType") == "DR")
                 & ((pl.col("iaProgramDeclared") == "1") | (pl.col("ihProgramDeclared") == "1")))
    return d.select(
        pl.col("disasterNumber").cast(pl.Int32).alias("disaster"),
        pl.col("state"),
        pl.col("incidentType").alias("incident_type"),
        month_of("incidentBeginDate").alias("m_incident"),
        pl.col("incidentBeginDate").str.slice(0, 4).cast(pl.Int32).alias("year"),
        (pl.col("fipsStateCode").str.zfill(2) + pl.col("fipsCountyCode").str.zfill(3)).alias("county"),
        (pl.col("fipsCountyCode") == "000").alias("tribal"),
    ).unique()


def zip3_county_shares(path=ZCTA_FILE) -> pl.DataFrame:
    """zip3, county, hu (housing units of the ZIP3 in the county), share = hu / housing units of
    the ZIP3; territories without ZCTAs get one row per county with share 1."""
    rel = pl.read_csv(path, infer_schema=False)
    rel = rel.select(pl.col("ZCTA5").str.zfill(5).str.slice(0, 3).alias("zip3"),
                     pl.col("GEOID").str.zfill(5).alias("county"), pl.col("HUPT").cast(pl.Float64).alias("hu"))
    g = rel.group_by("zip3", "county").agg(pl.col("hu").sum())
    g = g.with_columns((pl.col("hu") / pl.col("hu").sum().over("zip3")).fill_nan(0.0).alias("share"))
    return g


def territory_rows(decl: pl.DataFrame) -> pl.DataFrame:
    """zip3, county, share 1 for every declared county of a territory without ZCTAs."""
    t = decl.filter(pl.col("county").str.slice(0, 2).is_in(list(TERRITORY_ZIP3))).select("county").unique()
    rows = [{"zip3": z, "county": c, "hu": np.nan, "share": 1.0}
            for c in t["county"].to_list() for z in TERRITORY_ZIP3[c[:2]]]
    return pl.DataFrame(rows, schema={"zip3": pl.String, "county": pl.String, "hu": pl.Float64,
                                      "share": pl.Float64})


def affected_zip3(decl: pl.DataFrame, shares: pl.DataFrame) -> pl.DataFrame:
    """Per (disaster, zip3): share of the ZIP3's housing units in the disaster's declared counties
    (`share`) and whether any part of the ZIP3 touches them (`any_overlap`)."""
    sh = pl.concat([shares, territory_rows(decl)])
    j = decl.filter(~pl.col("tribal")).join(sh, on="county", how="inner")
    return j.group_by("disaster", "incident_type", "m_incident", "zip3").agg(
        pl.col("share").sum(), (pl.col("share") > 0).any().alias("any_overlap"))


def disaster_flags(hits: pl.DataFrame, months: list[int]) -> pl.DataFrame:
    """Flags per (zip3, m) for the observation months `months`.

    dis_main       any incident type except Biological, ZIP3 share >= 50 %
    dis_bio        Biological (COVID-19) only, share >= 50 %; applied outside the COVID scenario
    dis_hurricane  sensitivity: hurricanes, tropical storms, typhoons only, share >= 50 %
    dis_any        sensitivity: any incident type except Biological, any overlap of the ZIP3
    """
    m = pl.DataFrame({"m": months}, schema={"m": pl.Int32})
    h = hits.join(m, how="cross").filter(
        pl.col("m_incident").is_between(pl.col("m") + WINDOW[0], pl.col("m") + WINDOW[1]))
    bio = pl.col("incident_type") == BIOLOGICAL
    major = pl.col("share") >= SHARE_MIN - 1e-9
    return h.group_by("zip3", "m").agg(
        (major & ~bio).any().alias("dis_main"),
        (major & bio).any().alias("dis_bio"),
        (major & pl.col("incident_type").is_in(HURRICANE_TYPES)).any().alias("dis_hurricane"),
        (pl.col("any_overlap") & ~bio).any().alias("dis_any"),
    ).sort("zip3", "m")


def exclusion_expr(variant: str = "main") -> pl.Expr:
    """Boolean 'exclude this row' after joining the flags (missing flags = not affected).

    variant: main, all (sensitivity: keep every row), hurricane, any_overlap."""
    f = lambda c: pl.col(c).fill_null(False)
    covid = pl.col("stress").fill_null("") == "covid"
    bio = f("dis_bio") & ~covid
    if variant == "main":
        return f("dis_main") | bio
    if variant == "all":
        return f("dis_main") & False
    if variant == "hurricane":
        return f("dis_hurricane")
    if variant == "any_overlap":
        return f("dis_any") | bio
    raise ValueError(variant)


def observation_months(first: str = "2006-03", last: str = "2026-03") -> list[int]:
    return list(range(ym_to_index(first), ym_to_index(last) + 1))


def build_flags(write: bool = True) -> dict:
    """Declarations x relationship file -> flags per (zip3, m), written to DISASTER_FLAGS; the FEMA
    and Census files are added to data/external/manifest.csv. Returns tables for reporting."""
    from datetime import date

    from xai_ews.data.external import _sha256, add_to_manifest
    from xai_ews.paths import DISASTER_FLAGS

    decl = load_declarations()
    shares = zip3_county_shares()
    hits = affected_zip3(decl, shares)
    flags = disaster_flags(hits, observation_months())
    known = pl.concat([shares.select("county"), territory_rows(decl).select("county")]).unique()
    unmatched = decl.filter(~pl.col("tribal")).join(known, on="county", how="anti")
    if write:
        flags.write_parquet(DISASTER_FLAGS)
        recs = []
        for f, src, url in [(FEMA_FILE, "OpenFEMA Disaster Declarations Summaries v2", FEMA_URL),
                            (ZCTA_FILE, "Census 2010 ZCTA to county relationship file (via Colab)", ZCTA_URL)]:
            recs.append({"file": f"raw/{f.name}", "source": src, "url": url, "bytes": f.stat().st_size,
                         "sha256": _sha256(f), "downloaded": date.fromtimestamp(f.stat().st_mtime).isoformat()})
        add_to_manifest(recs)
    return {"declarations": decl, "shares": shares, "hits": hits, "flags": flags, "unmatched_counties": unmatched}
