import copy
import csv
from datetime import date

import numpy as np
import polars as pl
import pytest
from test_panel_labels import CLEAN, loan, loans_table

from xai_ews.config import data_config
from xai_ews.data.audit import psi_categorical, psi_numeric
from xai_ews.data.features import COLUMNS, candidate_features
from xai_ews.data.panel import build_vintage_panel, ym_to_index
from xai_ews.data.split import assign_splits, check_embargo
from xai_ews.paths import LEAKAGE_REGISTER_V02


def rows(obs: list[str], vintage: str, u: float, state: str = "OH") -> pl.DataFrame:
    return pl.DataFrame({"m": [ym_to_index(o) for o in obs], "vintage": [vintage] * len(obs),
                         "u_holdout": [u] * len(obs), "property_state": [state] * len(obs)})


def test_main_split_windows_vintages_and_holdout():
    cfg = data_config()
    df = assign_splits(pl.concat([
        rows(["2013-03", "2015-06", "2015-09", "2016-06", "2017-03", "2018-03", "2018-12"], "2012Q1", 0.9),
        rows(["2013-03", "2016-06", "2018-03"], "2012Q1", 0.1),   # holdout loan
        rows(["2016-06"], "2016Q4", 0.9),                          # vintage too new for validation
        rows(["2018-06"], "2016Q1", 0.9),
    ]), cfg)
    assert df["split_main"].to_list() == [
        "train", "train", None, "validation", "validation", "test", "test",
        None, None, "test",
        None,
        "test",
    ]
    assert df["new_cohort"].to_list()[-1] is True
    assert df["split_vintage"].to_list()[-1] == "train"


def test_vintage_split_is_pre_covid_and_2019_goes_to_stress():
    cfg = data_config()
    df = assign_splits(pl.concat([rows(["2018-12", "2019-03"], "2016Q2", 0.5),
                                  rows(["2018-12"], "2018Q1", 0.5),
                                  rows(["2019-12"], "2019Q2", 0.5)]), cfg)
    assert df["split_vintage"].to_list() == ["train", None, "validation", None]
    assert df["stress"].to_list()[-1] == "covid"


def test_hurricane_flag_only_in_validation_states_and_dates():
    cfg = data_config()
    df = assign_splits(pl.concat([
        rows(["2016-12", "2017-03", "2016-09"], "2014Q1", 0.9, "TX"),
        rows(["2017-03"], "2014Q1", 0.9, "PR"),
        rows(["2017-03"], "2014Q1", 0.9, "OH"),
        rows(["2017-03"], "2014Q1", 0.1, "FL"),      # holdout loan: not in validation
    ]), cfg)
    assert df["val_hurricane_2017"].to_list() == [True, True, False, True, False, False]


def test_stress_scenarios():
    cfg = data_config()
    df = assign_splits(pl.concat([rows(["2020-06"], "2019Q1", 0.5), rows(["2008-06"], "2006Q1", 0.5),
                                  rows(["2023-03"], "2022Q1", 0.5), rows(["2016-06"], "2012Q1", 0.5)]),
                       cfg)
    assert df["stress"].to_list() == ["covid", "gfc", "rates", None]


def test_embargo_is_enforced():
    cfg = copy.deepcopy(data_config())
    check_embargo(cfg)
    cfg["split"]["validation"]["start"] = "2016-03"
    with pytest.raises(ValueError):
        check_embargo(cfg)


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = pl.Series("x", rng.normal(size=20000))
    assert psi_numeric(a, pl.Series("x", rng.normal(size=20000))) < 0.01
    assert psi_numeric(a, pl.Series("x", rng.normal(1.0, 1, size=20000))) > 0.25
    c = pl.Series("c", ["a"] * 500 + ["b"] * 500)
    assert psi_categorical(c, c) < 1e-9
    assert psi_categorical(c, pl.Series("c", ["a"] * 900 + ["b"] * 100)) > 0.25


def test_every_panel_column_has_a_role():
    cfg = copy.deepcopy(data_config())
    cfg["raw"]["data_end"] = "2016-12"
    lt = loans_table(["a"]).with_columns(pl.lit("2012Q3").alias("vintage"),
                                         pl.lit(0.5).alias("u_holdout"), pl.lit(0.1).alias("u_sample"),
                                         pl.lit(date(2013, 1, 1)).alias("first_period"))
    panel, _ = build_vintage_panel(loan("a", CLEAN), lt, None, cfg)
    macro_cols = {c for c, v in COLUMNS.items() if v.group == "macro" or c == "hpi_t_state_fallback"}
    split_cols = {c for c, v in COLUMNS.items() if v.group == "split" and c != "u_holdout"}
    # synthetic loans table holds only the schedule fields, so static columns are absent here
    assert set(panel.columns) <= set(COLUMNS)
    missing = set(COLUMNS) - set(panel.columns) - macro_cols - split_cols
    assert all(COLUMNS[c].temporal == "static_origination" or c == "vintage" for c in missing), missing


def test_no_future_or_termination_field_is_a_feature():
    for name in candidate_features():
        assert COLUMNS[name].temporal not in ("future_window", "post_termination"), name
    with open(LEAKAGE_REGISTER_V02, newline="") as f:
        for r in csv.DictReader(f):
            if r["v02_usage_decision"] == "candidate_feature":
                assert r["temporal_type"] not in ("future_window", "post_termination"), r["field_name"]


def test_merge_small_lgd_segments():
    from xai_ews.data.lgd import merge_small

    assert merge_small([300, 250, 400], 200) == [0, 1, 2]
    assert merge_small([50, 120, 300], 200) == [0, 0, 0]          # 170 < 200: all three merge
    assert merge_small([150, 120, 300], 200) == [0, 0, 1]
    assert merge_small([300, 50, 20], 200) == [0, 0, 0]          # small tail joins previous
    assert merge_small([10, 20], 200) == [0, 0]                  # total below n_min: one group


def test_negative_subsample_keeps_all_positives_weights_and_nests():
    from xai_ews.data.split import add_negative_draw, train_sample

    n = 20000
    df = pl.DataFrame({"loan_identifier": [f"{i:012d}" for i in range(n)], "m": [24180] * n,
                       "pop_y1": [True] * n, "y1": pl.Series([1 if i % 100 == 0 else 0
                                                              for i in range(n)], dtype=pl.Int8),
                       "x": list(range(n))})
    df = add_negative_draw(df, data_config())
    s2 = train_sample(df.lazy(), "y1", 0.2, ["x"]).collect()
    s1 = train_sample(df.lazy(), "y1", 0.1, ["x"]).collect()
    assert s2["y1"].sum() == df["y1"].sum()                       # every positive kept
    assert set(s1["x"]) <= set(s2["x"])                           # r = 0.1 nested in r = 0.2
    neg = s2.filter(pl.col("y1") == 0)
    assert (neg["sample_weight"] == 5.0).all()
    assert (s2.filter(pl.col("y1") == 1)["sample_weight"] == 1.0).all()
    assert abs(neg.height / (n - df["y1"].sum()) - 0.2) < 0.02


def test_lgd_gross_discounted_and_sensitivities():
    from xai_ews.data.audit import liquidations_with_segments
    from xai_ews.data.panel import Macro

    ev = pl.DataFrame({
        "loan_identifier": ["a"], "monthly_reporting_period": [date(2012, 6, 1)],
        "upb_at_the_time_of_removal": [100_000.0], "foreclosure_costs": [10_000.0],
        "net_sales_proceeds": [80_000.0], "credit_enhancement_proceeds": [15_000.0],
        "last_paid_installment_date": [date(2011, 6, 1)], "disposition_date": [date(2012, 6, 1)],
        "current_interest_rate": [None], "original_interest_rate": [6.0],
        "origination_date": [date(2006, 1, 1)], "original_loan_to_value_ratio_ltv": [80],
        "original_upb": [100_000.0], "zip_code_short": ["100"], "property_state": ["NY"],
    }, schema_overrides={"current_interest_rate": pl.Float64})
    for c in ("property_preservation_and_repair_costs", "asset_recovery_costs",
              "miscellaneous_holding_expenses_and_credits", "associated_taxes_for_holding_property",
              "repurchase_make_whole_proceeds", "other_foreclosure_proceeds"):
        ev = ev.with_columns(pl.lit(None, pl.Float64).alias(c))
    hpi = pl.DataFrame({"zip3": ["100"] * 30, "year": [2006 + i // 4 for i in range(30)],
                        "quarter": [i % 4 + 1 for i in range(30)], "hpi": [100.0] * 30})
    macro = Macro(hpi_zip3=hpi, hpi_state=hpi.rename({"zip3": "state"}).head(0),
                  unemployment=pl.DataFrame(), pmms=pl.DataFrame())
    row = liquidations_with_segments(ev, macro, data_config()).row(0, named=True)
    df = 1.005 ** -12                                     # 12 months at 6 % / 12
    assert row["workout_months"] == 12
    assert row["lgd"] == pytest.approx((100_000 - df * 70_000) / 100_000)          # 0.3407
    assert row["lgd_net_mi"] == pytest.approx((100_000 - df * 85_000) / 100_000)   # 0.1994
    assert row["lgd_undisc"] == pytest.approx(0.30)
    assert row["mtm_ltv_at_default"] == pytest.approx(80.0) and row["judicial"]
