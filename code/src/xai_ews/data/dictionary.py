"""Data dictionary of the candidate features: data type, unit and allowed values.

One row per candidate feature with its meaning (from `xai_ews.features.rationale`), whether it
is static (fixed at origination) or changes over time, the kind of value, the unit and where it
comes from. Notebooks show this table next to the data; it is not used by any model.

Kinds:
    categorical   a code or a name, no order (state, seller, loan purpose)
    binary        two values: Y/N in the source data or a 0/1 flag built by us
    integer       a whole number: a score, a count of months, months, a percentage given in whole points
    continuous    a real number: an amount in dollars, a rate, a ratio or a share
`model_input` says how CatBoost receives the column (`categorical_features()`): "category" or
"number"; binary Y/N fields are passed as categories, 0/1 flags as numbers.
"""

import pandas as pd

from xai_ews.data.features import COLUMNS, candidate_features, categorical_features

CAT, BIN, INT, NUM = "categorical", "binary", "integer", "continuous"

KIND_RU = {CAT: "категориальный", BIN: "бинарный", INT: "целое", NUM: "вещественное"}

# feature: (kind, unit EN, unit RU, values / range)
_SPEC = {
    # borrower at origination
    "borrower_credit_score_at_origination": (INT, "points", "баллы", "300-850"),
    "co_borrower_credit_score_at_origination": (INT, "points", "баллы", "300-850; empty = no co-borrower"),
    "debt_to_income_dti": (INT, "% of monthly income", "% месячного дохода", ""),
    "number_of_borrowers": (INT, "persons", "человек", ""),
    "first_time_home_buyer_indicator": (BIN, "code", "код", "Y yes, N no, empty = unknown"),
    # loan at origination
    "original_interest_rate": (NUM, "% per year", "% годовых", ""),
    "original_upb": (NUM, "US dollars", "долл. США", ""),
    "original_loan_term": (INT, "months", "мес.", ""),
    "original_loan_to_value_ratio_ltv": (INT, "% of property value", "% стоимости жилья", ""),
    "original_combined_loan_to_value_ratio_cltv": (INT, "% of property value", "% стоимости жилья", "all liens"),
    "loan_purpose": (CAT, "code", "код", "P purchase, R refinance, C cash-out refinance, U refinance not specified"),
    "channel": (CAT, "code", "код", "R retail, B broker, C correspondent"),
    "seller_name": (CAT, "name", "название", "top 20 sellers + Other"),
    "property_type": (CAT, "code", "код", "SF single-family, PU planned unit, CO condo, CP co-op, MH manufactured"),
    "number_of_units": (INT, "dwelling units", "квартир", "1-4"),
    "occupancy_status": (CAT, "code", "код", "P principal residence, S second home, I investor"),
    "mortgage_insurance_percentage": (NUM, "% of loan covered", "% долга под страховкой", "empty = no MI"),
    "mortgage_insurance_type": (CAT, "code", "код", "1 borrower-paid, 2 lender-paid, 3 enterprise-paid, empty = no MI"),
    "amortization_type": (CAT, "code", "код", "FRM fixed rate (only value in the data)"),
    "prepayment_penalty_indicator": (BIN, "code", "код", "Y yes, N no"),
    "interest_only_loan_indicator": (BIN, "code", "код", "Y yes, N no"),
    "high_balance_loan_indicator": (BIN, "code", "код", "Y yes, N no"),
    "property_valuation_method": (CAT, "code", "код", "A appraisal, W waiver, C/P waiver + data collection, R targeted refi; empty in early vintages"),
    "relocation_mortgage_indicator": (BIN, "code", "код", "Y yes, N no"),
    "special_eligibility_program": (CAT, "code", "код", "H HomeReady, F HFA Preferred, R RefiNow, 7 none"),
    "high_loan_to_value_hltv_refinance_option_indicator": (BIN, "code", "код", "Y yes, N no"),
    # geography of the property
    "property_state": (CAT, "code", "код", "2-letter code: 50 states, DC, PR, VI, GU"),
    "metropolitan_statistical_area_msa_or_metropolitan_statistical_division_area_msda":
        (CAT, "code", "код", "5-digit MSA code, 00000 = outside a metro area"),
    "zip_code_short": (CAT, "code", "код", "first 3 digits of the ZIP code (ZIP3)"),
    # state at t
    "servicer_name": (CAT, "name", "название", "top 20 servicers + Other"),
    "current_interest_rate": (NUM, "% per year", "% годовых", ""),
    "age": (INT, "months", "мес.", ">= 6 in the panel"),
    "remaining_months_to_maturity": (INT, "months", "мес.", ""),
    "remaining_months_to_legal_maturity": (INT, "months", "мес.", ""),
    "upb": (NUM, "US dollars", "долл. США", "outstanding balance at t"),
    "upb_to_orig": (NUM, "share of original amount", "доля исходной суммы", "0-1"),
    "upb_vs_schedule": (NUM, "share (0.05 = 5 % above schedule)", "доля (0,05 = на 5 % выше графика)", "< 0 paying ahead"),
    "rate_change": (NUM, "percentage points", "п.п.", "< 0 rate cut (modification)"),
    "modified_now": (BIN, "flag", "флаг", "0/1"),
    # behaviour over months <= t
    "ever_modified": (BIN, "flag", "флаг", "0/1"),
    "n_curtail_6m": (INT, "months", "мес.", "0-6"),
    "n_curtail_12m": (INT, "months", "мес.", "0-12"),
    "upb_chg_6m": (NUM, "share (-0.02 = balance fell 2 %)", "доля (−0,02 = долг снизился на 2 %)", "usually < 0"),
    "upb_chg_12m": (NUM, "share (-0.02 = balance fell 2 %)", "доля (−0,02 = долг снизился на 2 %)", "usually < 0"),
    "n_dpd_unknown_12m": (INT, "months", "мес.", "0-12"),
    "n_upb_increase_12m": (INT, "months", "мес.", "0-12"),
    "excess_principal_12m": (NUM, "share of original amount", "доля исходной суммы", ">= 0"),
    # macro (lagged by the publication delay)
    "mtm_ltv": (NUM, "% of current property value", "% текущей стоимости жилья", ""),
    "hpi_chg_since_orig": (NUM, "share (0.10 = +10 %)", "доля (0,10 = +10 %)", ""),
    "hpi_chg_12m": (NUM, "share (0.10 = +10 %)", "доля (0,10 = +10 %)", ""),
    "unemployment": (NUM, "% of labour force", "% рабочей силы", ""),
    "unemployment_chg_6m": (NUM, "percentage points", "п.п.", ""),
    "unemployment_chg_12m": (NUM, "percentage points", "п.п.", ""),
    "unemployment_rel_us": (NUM, "percentage points", "п.п.", "state minus US"),
    "unemployment_chg_12m_rel_us": (NUM, "percentage points", "п.п.", "state minus US"),
    "hpi_chg_12m_rel_us": (NUM, "share points (0.02 = 2 p.p. faster than US)", "доли (0,02 = на 2 п.п. быстрее США)", "local minus US"),
    "hpi_chg_since_orig_rel_us": (NUM, "share points (0.02 = 2 p.p. faster than US)", "доли (0,02 = на 2 п.п. быстрее США)", "local minus US"),
    "rate_spread_pmms": (NUM, "percentage points", "п.п.", "loan rate minus market rate"),
    "sato": (NUM, "percentage points", "п.п.", "loan rate minus market rate at origination"),
}

# reported by Fannie Mae under another name (or reconstructed only where masked)
_RAW_RENAMED = {"servicer_name", "current_interest_rate", "age", "upb"}

# macro feature name prefix -> external series it is built from (joined to the loan's location)
_MACRO_SOURCE = {"hpi": "FHFA HPI", "mtm_ltv": "FHFA HPI", "unemployment": "BLS LAUS",
                 "rate_spread": "Freddie Mac PMMS", "sato": "Freddie Mac PMMS"}


def _source(name: str) -> str:
    c = COLUMNS[name]
    if c.group == "macro":
        found = [v for k, v in _MACRO_SOURCE.items() if name.startswith(k)]
        return f"{found[0]} + Fannie Mae"
    if c.sources == "same" or name in _RAW_RENAMED:
        return "Fannie Mae"
    return "computed from Fannie Mae"


def feature_dictionary(features: list[str] | None = None) -> pd.DataFrame:
    """Kind, unit, values and source per feature; default = all candidates in candidate_features() order."""
    from xai_ews.features.rationale import RATIONALE

    cats = set(categorical_features())
    rows = []
    for f in features or candidate_features():
        kind, unit_en, unit_ru, values = _SPEC[f]
        c = COLUMNS[f]
        rows.append({
            "feature": f,
            "meaning_en": RATIONALE[f].meaning_en,
            "meaning_ru": RATIONALE[f].meaning_ru,
            "group": c.group,
            "static": c.temporal == "static_origination",
            "kind": kind,
            "kind_ru": KIND_RU[kind],
            "unit_en": unit_en,
            "unit_ru": unit_ru,
            "values": values,
            "model_input": "category" if f in cats else "number",
            "source": _source(f),
        })
    return pd.DataFrame(rows)
