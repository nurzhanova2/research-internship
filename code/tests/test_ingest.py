import zipfile
from datetime import date

import polars as pl
import pytest

from xai_ews.data.ingest import OrderCheck, cast_types, iter_blocks, parse_block, uniform_draw
from xai_ews.data.schema import N_FIELDS, column_names

NAMES = column_names()


def raw_line(loan_id: str, period: str, **fields) -> str:
    values = [""] * N_FIELDS
    values[NAMES.index("loan_identifier")] = loan_id
    values[NAMES.index("monthly_reporting_period")] = period
    for name, value in fields.items():
        values[NAMES.index(name)] = value
    return "|".join(values)


def write_zip(tmp_path, lines: list[str]):
    path = tmp_path / "Q.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Q.csv", "\n".join(lines) + "\n")
    return path


def test_uniform_draw_is_deterministic_and_salt_dependent():
    ids = [f"{i:012d}" for i in range(5000)]
    a = uniform_draw(ids, 1, "sample")
    assert (a == uniform_draw(ids, 1, "sample")).all()
    assert not (a == uniform_draw(ids, 1, "holdout")).all()
    assert not (a == uniform_draw(ids, 2, "sample")).all()
    assert ((a >= 0) & (a < 1)).all()
    assert abs((a < 0.15).mean() - 0.15) < 0.02


def test_blocks_cut_at_line_ends_and_keep_every_row(tmp_path):
    lines = [raw_line(f"{i // 3:012d}", f"{i % 3 + 1:02d}2020") for i in range(300)]
    path = write_zip(tmp_path, lines)
    blocks = list(iter_blocks(path, block_bytes=1000))  # far smaller than the data
    assert len(blocks) > 5
    assert all(b.endswith(b"\n") for b in blocks)
    df = pl.concat([parse_block(b) for b in blocks])
    assert df.height == 300
    assert df["loan_identifier"].to_list() == [f"{i // 3:012d}" for i in range(300)]


def test_cast_types_parses_dates_status_and_blanks(tmp_path):
    line = raw_line("000000000001", "032015", current_loan_delinquency_status="XX",
                    current_actual_upb="", original_upb="150000.00", loan_age="7",
                    origination_date="082014")
    df = cast_types(parse_block((line + "\n").encode()))
    row = df.row(0, named=True)
    assert row["monthly_reporting_period"] == date(2015, 3, 1)
    assert row["origination_date"] == date(2014, 8, 1)
    assert row["current_loan_delinquency_status"] is None
    assert row["current_actual_upb"] is None
    assert row["original_upb"] == 150000.0 and row["loan_age"] == 7


def _check(ids, months):
    return pl.Series(ids), pl.Series([date(2020, m, 1) for m in months])


def test_order_check_accepts_grouped_rows_and_straddling_loan():
    c = OrderCheck()
    c.update(*_check(["1", "1", "2", "2"], [1, 2, 1, 2]))
    c.update(*_check(["2", "3"], [3, 1]))  # loan 2 continues across the block boundary


def test_order_check_rejects_interleaved_loans():
    with pytest.raises(AssertionError):
        OrderCheck().update(*_check(["1", "2", "1"], [1, 1, 2]))
    c = OrderCheck()
    c.update(*_check(["1", "2"], [1, 1]))
    with pytest.raises(AssertionError):
        c.update(*_check(["1"], [2]))      # loan 1 reappears in a later block


def test_order_check_rejects_unsorted_months_and_counts_gaps():
    with pytest.raises(AssertionError):
        OrderCheck().update(*_check(["1", "1"], [2, 1]))
    c = OrderCheck()
    c.update(*_check(["1", "1", "1"], [1, 2, 5]))
    assert c.gap_rows == 1
