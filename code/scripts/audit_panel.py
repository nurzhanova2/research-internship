"""Stage-1 data audit: event counts, prepayment / forbearance shares, Y2 positives in train,
liquidations per LGD segment, feature distributions and PSI train -> validation / test.

Writes CSVs to reports/data_audit/ and a short summary to reports/data_audit/audit_summary.json
(the narrative report reads from these files).

    uv run python scripts/audit_panel.py
"""

import json
import sys

import polars as pl

from xai_ews.config import data_config
from xai_ews.data.audit import (
    event_table,
    liquidations_with_segments,
    psi_categorical,
    psi_numeric,
    truncated_histories,
    truncation_summary,
)
from xai_ews.data.features import candidate_features, categorical_features
from xai_ews.data.lgd import lgd_variants
from xai_ews.data.panel import load_macro, month_index, scan_panel, scan_train_full, ym_to_index
from xai_ews.data.split import vintage_index
from xai_ews.paths import DATA_AUDIT_DIR, INTERIM_DIR, PANEL_DIR, TRAIN_FULL_DIR

SAMPLE_ROWS = 600_000  # per split, for distributions and PSI


def _year() -> pl.Expr:
    return pl.col("obs_date").dt.year().alias("obs_year")


def split_sample(lf: pl.LazyFrame, split: str, cols: list[str]) -> pl.DataFrame:
    """Loan-level deterministic subsample of one split (frac of the holdout draw)."""
    part = lf.filter(pl.col("split_main") == split)
    n = part.select(pl.len()).collect().item()
    frac = min(1.0, SAMPLE_ROWS / max(n, 1))
    return part.filter((pl.col("u_holdout") * 1000) % 1 < frac).select(cols).collect()


def main() -> int:
    cfg = data_config()
    out = DATA_AUDIT_DIR
    out.mkdir(parents=True, exist_ok=True)
    lf = scan_panel().filter(pl.col("pop_y1"))   # Y1 population: loans current at t
    summary: dict = {}

    # 1. Events by sample and year
    t = event_table(lf.with_columns(_year()), ["split_main", "obs_year"])
    t.write_csv(out / "events_by_split_year.csv")
    by_split = event_table(lf, ["split_main"])
    by_split.write_csv(out / "events_by_split.csv")
    test = lf.filter(pl.col("split_main") == "test")
    pl.concat([
        event_table(test.with_columns(pl.lit("holdout_loan").alias("subset"),
                                      pl.col("holdout_loan").cast(pl.String).alias("value")),
                    ["subset", "value"]),
        event_table(test.with_columns(pl.lit("new_cohort").alias("subset"),
                                      pl.col("new_cohort").cast(pl.String).alias("value")),
                    ["subset", "value"]),
    ]).write_csv(out / "events_test_subsets.csv")
    event_table(lf.filter(pl.col("split_vintage").is_not_null()), ["split_vintage"]) \
        .write_csv(out / "events_by_vintage_split.csv")

    # Validation by date, hurricane states vs the rest
    hur_states = cfg["split"]["validation_exclusions"]["hurricanes_2017"]["states"]
    val = lf.filter(pl.col("split_main") == "validation").with_columns(
        pl.col("obs_date").cast(pl.String).str.slice(0, 7).alias("obs_month"),
        pl.when(pl.col("property_state").is_in(hur_states)).then(pl.lit("TX/FL/PR/VI"))
        .otherwise(pl.lit("other states")).alias("state_group"))
    event_table(val, ["obs_month", "state_group"]).write_csv(out / "events_validation_hurricanes.csv")
    with_without = pl.concat([
        event_table(val.with_columns(pl.lit("all").alias("validation_set")), ["validation_set"]),
        event_table(val.filter(~pl.col("val_hurricane_2017"))
                    .with_columns(pl.lit("without_hurricane_rows").alias("validation_set")),
                    ["validation_set"]),
    ])
    with_without.write_csv(out / "events_validation_with_without_hurricanes.csv")
    summary["validation_hurricanes_2017"] = with_without.select(
        "validation_set", "n_obs", "y1_events", "y1_event_loans", "y1_rate", "y2_events",
        "y2_event_loans", "y2_rate").to_dicts()
    stress = event_table(lf.filter(pl.col("stress").is_not_null()), ["stress"])
    covid_no_fb = event_table(
        lf.filter((pl.col("stress") == "covid") & (pl.col("n_assistance_12m") == 0))
        .with_columns(pl.lit("covid_without_assistance").alias("stress")), ["stress"])
    pl.concat([stress, covid_no_fb]).write_csv(out / "events_by_stress.csv")
    event_table(lf, ["vintage"]).write_csv(out / "events_by_vintage.csv")

    train = by_split.filter(pl.col("split_main") == "train").row(0, named=True)
    trig = cfg["panel"]["y2_negative_subsampling"]["trigger_min_train_events"]
    summary["train"] = {k: train[k] for k in ("n_obs", "n_loans", "y1_events", "y1_event_loans",
                                              "y1_rate", "n_obs_y2_pop", "y2_events",
                                              "y2_event_loans", "y2_rate", "y2_strict_events",
                                              "y2_strict_event_loans", "prepaid_in_window_share")}
    summary["y2_train_events_vs_trigger_15pct"] = {
        "events": train["y2_events"], "event_loans": train["y2_event_loans"], "trigger": trig,
        "fallback_needed": train["y2_event_loans"] < trig}
    summary["splits"] = by_split.select("split_main", "n_obs", "y1_events", "y2_events",
                                        "y1_rate", "y2_rate").to_dicts()

    # 2. Liquidations per LGD segment (all loans, not only the 15 % sample)
    events = pl.concat([pl.read_parquet(p) for p in sorted(INTERIM_DIR.glob("*/credit_events.parquet"))],
                       how="vertical_relaxed")
    liq = liquidations_with_segments(events, load_macro(), cfg)
    # LGD only from liquidations completed (zero balance and, where reported,
    # disposition) by the end of the train period; GFC vintages are stress-only, so they are
    # shown for context and not used.
    s = cfg["split"]
    train_end = ym_to_index(s["train"]["end"])
    vi = pl.col("vintage").map_elements(vintage_index, return_dtype=pl.Int32)
    liq = liq.with_columns(vi.alias("_vi"), month_index("disposition_date").alias("_disp"))
    done_by_train_end = (pl.col("m") <= train_end) & (pl.col("_disp").fill_null(0) <= train_end)
    train_vint = pl.col("_vi").is_between(vintage_index(s["train"]["vintages"]["from"]),
                                          vintage_index(s["train"]["vintages"]["to"]))
    gfc = pl.col("_vi") <= vintage_index(s["stress"]["gfc"]["vintages"]["to"])
    liq = liq.with_columns(
        pl.when(train_vint & done_by_train_end).then(pl.lit("train_period"))
        .when(gfc & done_by_train_end).then(pl.lit("gfc_vintages_by_train_end"))
        .when(gfc).then(pl.lit("gfc_vintages_later"))
        .otherwise(pl.lit("post_2012_later")).alias("period"))
    nmin = cfg["exposure"]["lgd"]["min_liquidations_per_segment"]
    seg = (liq.group_by("period", "judicial", "mtm_ltv_bucket")
              .agg(pl.len().alias("liquidations"),
                   pl.col("has_loss_data").sum().alias("with_loss_data"),
                   pl.col("lgd").clip(-1, 2).mean().alias("lgd_mean_clipped"),
                   pl.col("lgd").median().alias("lgd_median"))
              .with_columns((pl.col("liquidations") < nmin).alias("below_n_min"))
              .sort("period", "judicial", "mtm_ltv_bucket"))
    seg.write_csv(out / "lgd_segments.csv")
    liq.group_by("period", "zero_balance_code").len().sort("period", "zero_balance_code") \
        .write_csv(out / "liquidations_by_code.csv")
    tp = seg.filter(pl.col("period") == "train_period")
    summary["lgd_train_period"] = {
        "liquidations": int(tp["liquidations"].sum()),
        "with_loss_data": int(tp["with_loss_data"].sum()),
        "segments": tp.height,
        "segments_below_n_min": int(tp["below_n_min"].sum()),
        "n_min": nmin,
    }

    variants = lgd_variants(liq, cfg)
    pl.concat([v.with_columns(pl.lit(k).alias("variant")) for k, v in variants.items()],
              how="diagonal_relaxed").write_csv(out / "lgd_variants.csv")
    base = variants["base"]
    summary["lgd_base_all_vintages"] = {
        "definition": "gross of MI, discounted at note rate, segment by MTM LTV at default",
        "liquidations": int(base["liquidations"].sum()), "segments": base.height,
        "segments_below_n_min": int(base["below_n_min"].sum()),
        **{f"{k}_{stat}": variants[k][f"lgd_{stat}"][0]
           for k in ("constant", "constant_after_mi", "constant_undiscounted")
           for stat in ("mean", "median")}}

    # 2b. Histories that stop before the data end without a zero-balance code
    end = ym_to_index(cfg["raw"]["data_end"])
    hist = pl.concat([truncated_histories(pl.scan_parquet(p), end)
                      for p in sorted(INTERIM_DIR.glob("2*/performance.parquet"))])
    trunc = truncation_summary(hist)
    trunc.write_csv(out / "truncated_histories.csv")
    summary["truncated_histories"] = trunc.to_dicts()

    # 2c. Full train population
    if any(TRAIN_FULL_DIR.glob("vintage=*")):
        tf_lf = scan_train_full().filter(pl.col("pop_y1"))
        tf = event_table(tf_lf, ["split_main"])
        tf.write_csv(out / "events_train_full.csv")
        event_table(tf_lf.with_columns(_year()), ["split_main", "obs_year"]) \
            .write_csv(out / "events_train_full_by_year.csv")
        tf_row = tf.row(0, named=True)
        summary["y2_threshold_unique_loans_train_full"] = {
            "y1_events": tf_row["y1_events"], "y1_event_loans": tf_row["y1_event_loans"],
            "y2_events": tf_row["y2_events"], "y2_event_loans": tf_row["y2_event_loans"],
            "y2_strict_event_loans": tf_row["y2_strict_event_loans"], "trigger": trig,
            "y2_below_trigger": tf_row["y2_event_loans"] < trig}
        r = cfg["panel"]["y2_negative_subsampling"]
        rows = []
        for rate in [r["negatives_rate"], *r["negatives_rate_sensitivity"]]:
            for label, pop in (("y1", "pop_y1"), ("y2", "pop_y2")):
                n = scan_train_full().filter(
                    pl.col(pop) & pl.col(label).is_not_null()
                    & ((pl.col(label) == 1) | (pl.col("u_neg") < rate))).select(
                    pl.len(), pl.col(label).sum()).collect().row(0)
                rows.append({"r": rate, "label": label, "rows": n[0], "events": n[1]})
        pl.DataFrame(rows).write_csv(out / "train_full_subsamples.csv")
        summary["train_full"] = {**{k: tf_row[k] for k in
                                    ("n_obs", "n_loans", "y1_events", "y1_event_loans", "y2_events",
                                     "y2_event_loans", "y2_strict_events")},
                                 "subsamples": rows}

    # 3. Feature distributions and PSI
    feats = candidate_features()
    cats = set(categorical_features())
    cols = [*feats, "u_holdout"]
    tr = split_sample(lf, "train", cols)
    rows_dist, rows_psi = [], []
    for other in ("validation", "test"):
        ot = split_sample(lf, other, cols)
        for f in feats:
            if f in cats:
                psi = psi_categorical(tr[f], ot[f])
            else:
                psi = psi_numeric(tr[f], ot[f], cfg["audit"]["psi_bins"])
            rows_psi.append({"feature": f, "compare": f"train->{other}", "psi": psi})
        for f in feats:
            if f in cats:
                continue
            for name, d in (("train", tr), (other, ot)):
                if other == "test" and name == "train":
                    continue
                x = d[f].cast(pl.Float64)
                q = {f"p{int(p * 100)}": x.quantile(p) for p in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}
                rows_dist.append({"feature": f, "split": name, "mean": x.mean(),
                                  "null_share": x.null_count() / max(x.len(), 1), **q})
        del ot
    psi = pl.DataFrame(rows_psi).pivot(on="compare", index="feature", values="psi") \
        .sort("train->validation", descending=True)
    psi.write_csv(out / "psi_train_vs_other.csv")
    pl.DataFrame(rows_dist).write_csv(out / "feature_distributions.csv")
    summary["psi_validation_over_0.25"] = psi.filter(pl.col("train->validation") > 0.25) \
        .select("feature", "train->validation").to_dicts()
    summary["psi_validation_0.1_0.25"] = psi.filter(
        pl.col("train->validation").is_between(0.1, 0.25)).select("feature").to_series().to_list()

    sizes = {p.parent.name: p.stat().st_size for p in PANEL_DIR.glob("vintage=*/part.parquet")}
    summary["panel_bytes"] = sum(sizes.values())
    summary["panel_rows"] = lf.select(pl.len()).collect().item()
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
