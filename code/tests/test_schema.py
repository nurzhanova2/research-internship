import csv

from xai_ews.data.schema import N_FIELDS, column_names, load_fields, load_leakage_register
from xai_ews.paths import DICTIONARY_DIR


def test_dictionary_has_all_positions():
    names = column_names()
    assert len(names) == N_FIELDS
    assert len(set(names)) == N_FIELDS
    assert names[1] == "loan_identifier"
    assert names[2] == "monthly_reporting_period"


def test_all_quarter_dictionaries_share_one_layout():
    reference = column_names()
    for path in DICTIONARY_DIR.glob("*field_dictionary.csv"):
        assert column_names(path) == reference, path.name


def test_leakage_register_covers_every_field_and_excludes_termination_fields():
    register = load_leakage_register()
    assert set(register) == {f.name for f in load_fields()}
    for field in ("zero_balance_code", "foreclosure_date", "net_sales_proceeds"):
        assert register[field] == "label_only_or_exclude"
    with open(DICTIONARY_DIR / "fannie_2008q1_leakage_register_v01.csv", newline="") as f:
        decisions = {r["v01_usage_decision"] for r in csv.DictReader(f)}
    assert "candidate_feature" in decisions
