"""Leakage register v02: v01 with corrected temporal types plus every derived panel column.

v01 (data/dictionaries/fannie_2008q1_leakage_register_v01.csv) stays untouched. v02 adds:
- temporal_type for each raw field (v01 called current UPB, loan age, remaining months and the
  current rate "static"; they are reported for month t);
- v02_usage_decision and the panel column the field feeds;
- one row per derived panel column, from xai_ews.data.features.

    uv run python scripts/build_leakage_register_v02.py
"""

import csv
import sys

from xai_ews.data.features import COLUMNS
from xai_ews.data.ingest import DYNAMIC, STATIC
from xai_ews.paths import LEAKAGE_REGISTER, LEAKAGE_REGISTER_V02

# Raw fields whose v02 reading differs from v01 or needs a note: (temporal, decision, panel, note)
RAW_OVERRIDES = {
    "current_interest_rate": ("dynamic_at_t", "candidate_feature", "current_interest_rate, rate_change",
                              "v01 said static; value reported for month t"),
    "current_actual_upb": ("dynamic_at_t", "source_only", "upb (+ derived)",
                           "v01 said static; masked as 0 for young loans, reconstructed in `upb`"),
    "loan_age": ("dynamic_at_t", "candidate_feature", "age", "v01 said static"),
    "remaining_months_to_legal_maturity": ("dynamic_at_t", "candidate_feature", "same",
                                           "v01 said static; changes on modification"),
    "remaining_months_to_maturity": ("dynamic_at_t", "candidate_feature", "same",
                                     "v01 said static; falls faster with curtailments"),
    "maturity_date": ("dynamic_at_t", "not_used", "", ("changes on modification; covered by "
                      "remaining months")),
    "servicer_name": ("dynamic_at_t", "candidate_feature", "servicer_name",
                      "servicing transfers; drift risk"),
    "seller_name": ("static_origination", "candidate_feature", "seller_name",
                    "value of the first record; renamed in later records for some loans"),
    "origination_date": ("static_origination", "exclude_time_proxy", "vintage, sato",
                         "date: encodes calendar time"),
    "first_payment_date": ("static_origination", "exclude_time_proxy", "age fallback", "date"),
    "current_loan_delinquency_status": ("dynamic_at_t", "population_filter",
                                        "Y1 population (DPD 0 at t); labels (> t)",
                                        ("month t defines the population, future months the label; "
                                         "delinquency history is not a feature (exclude_delinquency)")),
    "loan_payment_history": ("trailing_to_t", "not_ingested", "",
                             "24-month string; redundant with the full monthly history"),
    "modification_flag": ("dynamic_at_t", "candidate_feature", "modified_now, ever_modified", ""),
    "total_principal_current": ("dynamic_at_t", "not_ingested", "",
                                ("UPB change at t; not post-termination (v01 too strict), "
                                "derived from UPB instead")),
    "zero_balance_code": ("post_termination", "label_only", ("y1, exit_code_in_window, "
                          "prepaid_in_window"), "default codes 02/03/09/15"),
    "borrower_assistance_plan": ("dynamic_at_t", "report_only",
                                 "forbearance_now, assistance_now, n_assistance_12m",
                                 "workout; COVID stress split; not a feature"),
    "mortgage_insurance_type": ("static_origination", "candidate_feature", "same",
                                "origination field; v01 pending review"),
    "high_balance_loan_indicator": ("static_origination", "candidate_feature", "same",
                                    "origination field; v01 pending review"),
    "property_valuation_method": ("static_origination", "candidate_feature", "same",
                                  "origination field; v01 pending review"),
    "relocation_mortgage_indicator": ("static_origination", "candidate_feature", "same",
                                      "origination field; v01 pending review"),
    "special_eligibility_program": ("static_origination", "candidate_feature", "same",
                                    "origination field; v01 pending review"),
    "high_loan_to_value_hltv_refinance_option_indicator": (
        "static_origination", "candidate_feature", "same", "origination field; v01 pending review"),
}

DEFAULT_BY_V01 = {
    "exclude_identifier": ("identifier", "identifier"),
    "index_only": ("index", "index"),
    "label_only_or_exclude": ("post_termination", "label_only"),
    "exclude_initial_model": ("dynamic_at_t", "report_only"),
    "exclude_pending_review": ("unknown", "exclude_pending_review"),
    "candidate_feature": ("static_origination", "candidate_feature"),
    "candidate_feature_conditional": ("trailing_to_t", "candidate_feature_conditional"),
}

FIELDS = ["field_name", "kind", "temporal_type", "v01_usage_decision", "v02_usage_decision",
          "ingested", "panel_columns", "sources", "group", "note"]


def main() -> int:
    ingested = set(STATIC) | set(DYNAMIC) | {"loan_identifier", "monthly_reporting_period"}
    rows = []
    with open(LEAKAGE_REGISTER, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name, v01 = r["field_name"], r["v01_usage_decision"]
            temporal, decision = DEFAULT_BY_V01[v01]
            panel, note = (name if name in COLUMNS else ""), ""
            if name in RAW_OVERRIDES:
                temporal, decision, panel, note = RAW_OVERRIDES[name]
                panel = name if panel == "same" else panel
            elif v01 == "label_only_or_exclude" and name in ingested:
                note = "kept in interim data for LGD / labels; dropped from the panel"
            if name not in ingested and decision == "candidate_feature":
                decision = "not_ingested"
            rows.append({"field_name": name, "kind": "raw", "temporal_type": temporal,
                         "v01_usage_decision": v01, "v02_usage_decision": decision,
                         "ingested": name in ingested, "panel_columns": panel, "sources": "",
                         "group": "", "note": note})

    raw_names = {r["field_name"] for r in rows}
    for c in COLUMNS.values():
        if c.name in raw_names:  # raw field kept as is: annotate the raw row instead
            row = next(r for r in rows if r["field_name"] == c.name)
            row["group"] = c.group
            if row["v02_usage_decision"] != c.decision and c.name not in RAW_OVERRIDES:
                row["v02_usage_decision"] = c.decision
            row["note"] = row["note"] or c.note
            continue
        rows.append({"field_name": c.name, "kind": "derived", "temporal_type": c.temporal,
                     "v01_usage_decision": "", "v02_usage_decision": c.decision, "ingested": "",
                     "panel_columns": c.name, "sources": c.sources, "group": c.group,
                     "note": c.note})

    with open(LEAKAGE_REGISTER_V02, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    future = [r["field_name"] for r in rows if r["temporal_type"] in ("future_window", "post_termination")
              and r["v02_usage_decision"] == "candidate_feature"]
    assert not future, f"future/post-termination fields marked as features: {future}"
    n_cf = sum(r["v02_usage_decision"] == "candidate_feature" for r in rows)
    print(f"{len(rows)} rows ({len(raw_names)} raw), {n_cf} candidate features -> {LEAKAGE_REGISTER_V02}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
