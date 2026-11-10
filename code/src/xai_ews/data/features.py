"""Role of every panel column: model feature, label, split, report-only, ...

This is the single list the leakage register v02, the data audit and the leakage check read.
Decisions:
    candidate_feature   may enter models (stage 2 selects among them)
    label_only          outcome over (t, t+12]; never a feature
    population_filter   defines a population (Y1: DPD at t); not a feature
    exclude_delinquency past or current delinquency status: Y1 is predicted for loans current
                        at t, before any delinquency is visible, so these are not features
    split_only          sample assignment
    report_only         workout / forbearance (exclude_initial_model in v01) and data-quality
                        flags; reported, not modelled
    exclude_time_proxy  market level or date that mostly encodes calendar time
    identifier / index  keys
Temporal types:
    static_origination  known at origination, constant for the loan
    dynamic_at_t        value reported for month t
    trailing_to_t       computed from months t-11..t (or t-5..t, or all months <= t)
    macro_lagged        external series at t minus its publication lag (configs/data.yaml)
    future_window       uses months t+1..t+12: labels only
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    name: str
    group: str
    temporal: str
    decision: str
    sources: str
    note: str = ""


S, D, T, MAC, FUT = "static_origination", "dynamic_at_t", "trailing_to_t", "macro_lagged", "future_window"
CF = "candidate_feature"
XD = "exclude_delinquency"

_COLUMNS = [
    # keys
    Column("loan_identifier", "key", "identifier", "identifier", "loan_identifier"),
    Column("m", "key", "index", "index", "monthly_reporting_period", "month index 12*year+month-1"),
    Column("obs_date", "key", "index", "index", "monthly_reporting_period"),
    Column("vintage", "key", S, "index", "file quarter", "partition key"),
    Column("origination_date", "key", S, "exclude_time_proxy", "origination_date"),
    Column("first_payment_date", "key", S, "exclude_time_proxy", "first_payment_date"),
    Column("u_holdout", "split", S, "split_only", "loan_identifier", "hash draw for the holdout"),
    # origination: borrower
    Column("borrower_credit_score_at_origination", "borrower", S, CF, "same"),
    Column("co_borrower_credit_score_at_origination", "borrower", S, CF, "same"),
    Column("debt_to_income_dti", "borrower", S, CF, "same"),
    Column("number_of_borrowers", "borrower", S, CF, "same"),
    Column("first_time_home_buyer_indicator", "borrower", S, CF, "same"),
    # origination: loan
    Column("original_interest_rate", "loan", S, CF, "same"),
    Column("original_upb", "exposure", S, CF, "same"),
    Column("original_loan_term", "loan", S, CF, "same"),
    Column("original_loan_to_value_ratio_ltv", "loan", S, CF, "same"),
    Column("original_combined_loan_to_value_ratio_cltv", "loan", S, CF, "same"),
    Column("loan_purpose", "loan", S, CF, "same"),
    Column("channel", "loan", S, CF, "same"),
    Column("seller_name", "loan", S, CF, "same",
           "first record; renamed for ~2-3 % of loans over time; drift risk, check PSI"),
    Column("property_type", "loan", S, CF, "same"),
    Column("number_of_units", "loan", S, CF, "same"),
    Column("occupancy_status", "loan", S, CF, "same"),
    Column("mortgage_insurance_percentage", "loan", S, CF, "same"),
    Column("mortgage_insurance_type", "loan", S, CF, "same", "v01 pending review; origination field"),
    Column("amortization_type", "loan", S, CF, "same", "FRM only in this dataset"),
    Column("prepayment_penalty_indicator", "loan", S, CF, "same"),
    Column("interest_only_loan_indicator", "loan", S, CF, "same"),
    Column("high_balance_loan_indicator", "loan", S, CF, "same", "v01 pending review; origination field"),
    Column("property_valuation_method", "loan", S, CF, "same", "v01 pending review; origination field"),
    Column("relocation_mortgage_indicator", "loan", S, CF, "same", "v01 pending review"),
    Column("special_eligibility_program", "loan", S, CF, "same", "v01 pending review"),
    Column("high_loan_to_value_hltv_refinance_option_indicator", "loan", S, CF, "same",
           "v01 pending review"),
    # geography
    Column("property_state", "geography", S, CF, "same"),
    Column("metropolitan_statistical_area_msa_or_metropolitan_statistical_division_area_msda",
           "geography", S, CF, "same", "first record; MSA codes are re-delineated over time"),
    Column("zip_code_short", "geography", S, CF, "same"),
    # state at t
    Column("servicer_name", "state_at_t", D, CF, "servicer_name", "drift risk, check PSI"),
    Column("current_interest_rate", "state_at_t", D, CF, "current_interest_rate"),
    Column("age", "state_at_t", D, CF, "loan_age; first_payment_date if missing"),
    Column("remaining_months_to_maturity", "state_at_t", D, CF, "same"),
    Column("remaining_months_to_legal_maturity", "state_at_t", D, CF, "same"),
    Column("dpd", "state_at_t", D, "population_filter", "current_loan_delinquency_status",
           "0..2 in the panel files, 0 in the Y1 population"),
    Column("upb", "exposure", D, CF, "current_actual_upb; schedule if masked", "EAD at t"),
    Column("upb_to_orig", "exposure", D, CF, "upb, original_upb"),
    Column("sched_upb", "exposure", D, "report_only", "original terms", "schedule balance"),
    Column("upb_vs_schedule", "behaviour", D, CF, "upb, sched_upb", "<0: paying ahead"),
    Column("rate_change", "behaviour", D, CF, "current - original rate", "modification signal"),
    Column("modified_now", "behaviour", D, CF, "modification_flag"),
    Column("ever_modified", "behaviour", T, CF, "modification_flag, months <= t"),
    # trailing behaviour
    *[Column(f"dpd_max_{w}m", "behaviour", T, XD, "delinquency status") for w in (6, 12)],
    *[Column(f"n_30p_{w}m", "behaviour", T, XD, "delinquency status") for w in (6, 12)],
    *[Column(f"n_curtail_{w}m", "behaviour", T, CF, "upb, schedule",
             "months with principal above schedule (partial prepayment)") for w in (6, 12)],
    *[Column(f"upb_chg_{w}m", "behaviour", T, CF, "upb") for w in (6, 12)],
    Column("dpd_mean_12m", "behaviour", T, XD, "delinquency status"),
    Column("n_60p_12m", "behaviour", T, XD, "delinquency status"),
    Column("n_cure_12m", "behaviour", T, XD, "delinquency status", "30+ -> current transitions"),
    Column("n_dpd_unknown_12m", "behaviour", T, CF, "delinquency status 'XX'"),
    Column("n_upb_increase_12m", "behaviour", T, CF, "upb", "capitalisation, e.g. modification"),
    Column("excess_principal_12m", "behaviour", T, CF, "upb, schedule",
           "principal above schedule over 12 months / original UPB"),
    Column("months_since_30p", "behaviour", T, XD, "delinquency status, all months <= t",
           "null if never 30+"),
    Column("ever_30p", "behaviour", T, XD, "delinquency status, all months <= t"),
    Column("dpd_max_prev12", "population", T, "population_filter", "delinquency status t-12..t-1"),
    Column("months_observed_12m", "quality", T, "report_only", "reporting months"),
    Column("upb_masked", "quality", D, "report_only", "current_actual_upb",
           "UPB reconstructed from schedule"),
    # workout / forbearance (exclude_initial_model)
    Column("forbearance_now", "workout", D, "report_only", "borrower_assistance_plan = F"),
    Column("assistance_now", "workout", D, "report_only", "borrower_assistance_plan in F/R/T"),
    Column("n_assistance_12m", "workout", T, "report_only", "borrower_assistance_plan"),
    # macro
    Column("mtm_ltv", "macro", MAC, CF, "original LTV, upb, FHFA HPI (zip3, state fallback)"),
    Column("hpi_chg_since_orig", "macro", MAC, CF, "FHFA HPI"),
    Column("hpi_chg_12m", "macro", MAC, CF, "FHFA HPI"),
    Column("unemployment", "macro", MAC, CF, "BLS LAUS state, SA"),
    Column("unemployment_chg_6m", "macro", MAC, CF, "BLS LAUS"),
    Column("unemployment_chg_12m", "macro", MAC, CF, "BLS LAUS"),
    Column("unemployment_rel_us", "macro", MAC, CF, "state - US unemployment (BLS)",
           "relative version"),
    Column("unemployment_chg_12m_rel_us", "macro", MAC, CF, "state - US 12-month change",
           "relative version"),
    Column("hpi_chg_12m_rel_us", "macro", MAC, CF, "local - US HPI 12-month change",
           "relative version"),
    Column("hpi_chg_since_orig_rel_us", "macro", MAC, CF, "local - US HPI change since origination",
           "relative version"),
    Column("rate_spread_pmms", "macro", MAC, CF, "current rate - PMMS 30y at t"),
    Column("sato", "macro", S, CF, "original rate - PMMS 30y at origination month"),
    Column("hpi_t", "macro", MAC, "exclude_time_proxy", "FHFA HPI level"),
    Column("hpi_t_state_fallback", "quality", MAC, "report_only", "no zip3 index, state used"),
    Column("pmms30_t", "macro", MAC, "exclude_time_proxy", "PMMS level"),
    # labels
    *[Column(c, "label", FUT, "label_only", "delinquency status / zero balance code, t+1..t+12")
      for c in ("y1", "y2", "y2_strict", "y1_month", "y2_month", "exit_month",
                "exit_code_in_window", "prepaid_in_window")],
    Column("pop_y1", "population", D, "population_filter", "dpd at t"),
    Column("pop_y2", "population", T, "population_filter", "dpd at t, t-12..t-1"),
    # split
    *[Column(c, "split", "index", "split_only", "obs date, vintage, u_holdout")
      for c in ("split_main", "holdout_loan", "new_cohort", "split_vintage", "stress",
                "val_hurricane_2017")],
]

COLUMNS: dict[str, Column] = {c.name: c for c in _COLUMNS}

STATIC_GROUPS = {"borrower", "loan", "geography"}


def candidate_features(static_only: bool = False) -> list[str]:
    """Feature candidates; static_only=True gives the origination + geography set (H3 baseline)."""
    return [c.name for c in _COLUMNS if c.decision == CF
            and (not static_only or (c.temporal == S and c.group in STATIC_GROUPS))]


def categorical_features() -> list[str]:
    numeric_codes = {"number_of_borrowers", "number_of_units"}
    return [n for n in candidate_features() if n not in numeric_codes and (
        n.endswith(("_indicator", "_name", "_type", "_method", "_program", "_status"))
        or n in {"channel", "loan_purpose", "property_state", "zip_code_short",
                 "metropolitan_statistical_area_msa_or_metropolitan_statistical_division_area_msda"})]
