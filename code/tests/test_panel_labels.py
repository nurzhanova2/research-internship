"""Label and history logic of the panel on synthetic loan histories.

Each loan starts in 2013-01 with loan_age 6, so 2013-03 (age 8) is a snapshot month and
2013-03 + k months is index k + 2 in the history lists below.
"""

import copy
from datetime import date

import polars as pl
import pytest

from xai_ews.config import data_config
from xai_ews.data.panel import (
    LabelSpec,
    add_history,
    add_labels,
    build_vintage_panel,
    prepare_monthly,
    select_snapshots,
)

START = (2013, 1)
T0 = date(2013, 3, 1)  # first snapshot month
ORIG_UPB = 200_000.0


def _months(n: int, start=START) -> list[date]:
    y, m = start
    out = []
    for _ in range(n):
        out.append(date(y, m, 1))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def loan(loan_id: str, dpd: list, *, exit_code: str | None = None, upb: list | None = None,
         age0: int = 6, start=START, assist: list | None = None) -> pl.DataFrame:
    """Monthly records; if exit_code is given one more record with that zero-balance code
    (UPB 0) is appended after the last month in `dpd`."""
    n = len(dpd) + (1 if exit_code else 0)
    upb = upb or [ORIG_UPB * (1 - 0.002 * (age0 + i)) for i in range(len(dpd))]
    rows = {
        "loan_identifier": [loan_id] * n,
        "monthly_reporting_period": _months(n, start),
        "current_loan_delinquency_status": [*dpd, None] if exit_code else dpd,
        "current_actual_upb": [*upb, 0.0] if exit_code else upb,
        "loan_age": [age0 + i for i in range(n)],
        "zero_balance_code": [None] * len(dpd) + ([exit_code] if exit_code else []),
        "current_interest_rate": [4.0] * n,
        "remaining_months_to_maturity": [360 - age0 - i for i in range(n)],
        "remaining_months_to_legal_maturity": [360 - age0 - i for i in range(n)],
        "servicer_name": ["Other"] * n,
        "modification_flag": ["N"] * n,
        "borrower_assistance_plan": (assist or ["7"] * len(dpd)) + (["7"] if exit_code else []),
    }
    return pl.DataFrame(rows, schema_overrides={"current_loan_delinquency_status": pl.Int16,
                                                "loan_age": pl.Int16,
                                                "remaining_months_to_maturity": pl.Int16,
                                                "remaining_months_to_legal_maturity": pl.Int16})


def loans_table(ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "loan_identifier": ids,
        "original_upb": [ORIG_UPB] * len(ids),
        "original_interest_rate": [4.0] * len(ids),
        "original_loan_term": pl.Series([360] * len(ids), dtype=pl.Int16),
        "first_payment_date": [date(2012, 8, 1)] * len(ids),
        "origination_date": [date(2012, 7, 1)] * len(ids),
    })


@pytest.fixture
def cfg():
    c = copy.deepcopy(data_config())
    c["raw"]["data_end"] = "2016-12"
    return c


def run(perf: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Monthly frame with history and labels (before snapshot selection)."""
    ids = perf["loan_identifier"].unique().to_list()
    monthly = prepare_monthly(perf, loans_table(ids))
    return add_labels(add_history(monthly, cfg), LabelSpec.from_config(cfg))


def at(df: pl.DataFrame, loan_id: str, when: date = T0) -> dict:
    m = when.year * 12 + when.month - 1
    rows = df.filter((pl.col("loan_identifier") == loan_id) & (pl.col("m") == m))
    assert rows.height == 1
    return rows.row(0, named=True)


CLEAN = [0] * 40


def test_y1_window_includes_t_plus_12_but_not_t_plus_13(cfg):
    # T0 is index 2; t + k is index 2 + k.
    hit12 = CLEAN[:14] + [3] + CLEAN[:20]
    hit13 = CLEAN[:15] + [3] + CLEAN[:20]
    hit1 = CLEAN[:3] + [3] + CLEAN[:30]
    df = run(pl.concat([loan("a", hit12), loan("b", hit13), loan("c", hit1)]), cfg)
    assert at(df, "a")["y1"] == 1 and at(df, "a")["y1_month"] == 12
    assert at(df, "b")["y1"] == 0
    assert at(df, "c")["y1"] == 1 and at(df, "c")["y1_month"] == 1


def test_event_at_t_itself_is_not_in_the_window(cfg):
    dpd = CLEAN[:2] + [1, 0] + CLEAN[:30]   # 30 DPD at t only, clean afterwards
    df = run(loan("a", dpd), cfg)
    assert at(df, "a")["y1"] == 0 and at(df, "a")["y2"] == 0


def test_prepayment_inside_window_gives_label_zero_and_is_flagged(cfg):
    df = run(loan("a", CLEAN[:6], exit_code="01"), cfg)  # prepaid at t + 4
    row = at(df, "a")
    assert row["y1"] == 0 and row["y2"] == 0
    assert row["prepaid_in_window"] and row["exit_code_in_window"] == "01"


def test_liquidation_before_90_dpd_counts_as_default(cfg):
    dpd = CLEAN[:4] + [1, 2]                  # 60 DPD at t + 3
    df = run(loan("a", dpd, exit_code="03"), cfg)  # short sale at t + 4
    row = at(df, "a")
    assert row["y1"] == 1 and row["y1_month"] == 4
    assert row["y2"] == 1 and row["y2_month"] == 3


def test_default_exit_without_delinquency_is_y1_not_y2(cfg):
    df = run(loan("a", CLEAN[:5], exit_code="09"), cfg)  # REO at t + 3, never 60+
    row = at(df, "a")
    assert row["y1"] == 1 and row["y2"] == 0


def test_non_credit_exit_is_label_zero(cfg):
    df = run(loan("a", CLEAN[:5], exit_code="06"), cfg)  # repurchase
    row = at(df, "a")
    assert row["y1"] == 0 and not row["prepaid_in_window"]


def test_window_beyond_data_end_is_dropped(cfg):
    cfg["raw"]["data_end"] = "2014-02"         # t + 12 = 2014-03 > data end
    df = run(loan("a", CLEAN[:20]), cfg)
    assert at(df, "a")["y1"] is None
    cfg["raw"]["data_end"] = "2014-03"
    df = run(loan("a", CLEAN[:20]), cfg)
    assert at(df, "a")["y1"] == 0


def test_censored_history_is_dropped_unless_event_seen(cfg):
    df = run(pl.concat([loan("a", CLEAN[:10]),                     # stops at t + 7, no code
                        loan("b", CLEAN[:7] + [3, 3, 3])]), cfg)    # 90+ at t + 5, then stops
    assert at(df, "a")["y1"] is None
    assert at(df, "b")["y1"] == 1


def test_y2_population_needs_12_clean_months(cfg):
    recent = [0, 0, 0, 0, 0, 1, 0] + CLEAN[:30]   # 30 DPD at t + 3; snapshot t + 9 is 6 months after
    old = [1] + CLEAN[:40]                          # 30 DPD in 2013-01
    df = run(pl.concat([loan("a", recent), loan("b", old)]), cfg)
    snaps = select_snapshots(df, cfg, LabelSpec.from_config(cfg))

    def pop(loan_id, when):
        return snaps.filter((pl.col("loan_identifier") == loan_id)
                            & (pl.col("m") == when.year * 12 + when.month - 1))["pop_y2"][0]

    assert not pop("a", date(2013, 9, 1))     # 30+ in 2013-06, within 12 months
    assert not pop("a", date(2014, 3, 1))     # 9 months later: still within 12
    assert pop("a", date(2014, 9, 1))         # 15 months later: clean
    assert not pop("b", date(2013, 12, 1))    # 2013-01 is 11 months before
    assert pop("b", date(2014, 3, 1))         # 14 months before


def test_y2_strict_window(cfg):
    early = CLEAN[:5] + [2] + CLEAN[:30]      # 60+ at t + 3
    late = CLEAN[:9] + [2] + CLEAN[:30]       # 60+ at t + 7
    df = run(pl.concat([loan("a", early), loan("b", late), loan("c", CLEAN)]), cfg)
    assert at(df, "a")["y2"] == 1 and at(df, "a")["y2_strict"] is None
    assert at(df, "b")["y2"] == 1 and at(df, "b")["y2_strict"] == 1
    assert at(df, "c")["y2_strict"] == 0


def test_masked_upb_is_reconstructed_from_schedule(cfg):
    upb = [0.0, 0.0, 199_000.0] + [198_000.0] * 30
    df = run(loan("a", CLEAN[:33], upb=upb, age0=4), cfg)
    first = df.filter(pl.col("loan_identifier") == "a").sort("m").head(3)
    assert first["upb_masked"].to_list() == [True, True, False]
    # 4 payments of 954.83 on a 4 % 30-year 200k loan, ~289 principal each: 198,841.57
    assert first["upb"][0] == pytest.approx(198_841.57, abs=0.01)
    assert first["upb"][2] == 199_000.0


def test_missing_month_is_inserted_so_windows_count_months(cfg):
    perf = loan("a", CLEAN[:3] + [3] + CLEAN[:30])
    perf = perf.filter(pl.col("monthly_reporting_period") != date(2013, 2, 1))  # gap before t
    perf = perf.filter(pl.col("monthly_reporting_period") != date(2013, 5, 1))  # gap in window
    df = run(perf, cfg)
    assert df["inserted_month"].sum() == 2
    assert at(df, "a")["y1"] == 1 and at(df, "a")["y1_month"] == 1


def test_trailing_history_counts(cfg):
    dpd = [0, 1, 2, 1, 0, 0, 0, 0] + CLEAN[:30]  # snapshot 2013-06 is index 5
    df = run(loan("a", dpd), cfg)
    row = at(df, "a", date(2013, 6, 1))
    assert row["dpd_max_6m"] == 2 and row["n_30p_6m"] == 3
    assert row["n_cure_12m"] == 1 and row["months_since_30p"] == 2 and row["ever_30p"] == 1


def test_snapshots_are_quarter_end_months_with_min_age(cfg):
    perf = loan("a", CLEAN, age0=2)               # age 6 reached in 2013-05
    panel, _ = build_vintage_panel(perf, loans_table(["a"]).with_columns(
        pl.lit("2012Q3").alias("vintage"), pl.lit(0.5).alias("u_holdout"),
        pl.lit(0.1).alias("u_sample"), pl.lit(date(2013, 1, 1)).alias("first_period")), None, cfg)
    assert {d.month for d in panel["obs_date"]} <= {3, 6, 9, 12}
    assert panel["obs_date"].min() == date(2013, 6, 1)
    assert (panel["age"] >= 6).all()
