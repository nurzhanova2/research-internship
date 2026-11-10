"""Economic rationale and EXPECTED sign of every candidate feature (stage 2, step 1).

Written before any association with Y1 / Y2 was computed and not changed afterwards: a
mismatch between the observed and the expected sign is a reason to investigate (leakage,
interaction, data error), not to rewrite the expectation.

Sign convention (risk = probability of the Y1 / Y2 event):
    "+"     higher value -> higher risk
    "-"     higher value -> lower risk
    "cat"   categorical: level `risky` is expected to have a higher event rate than `safe`
    "none"  no prior direction (exposure size, geography, institution names; methodology 4)
"""

from dataclasses import dataclass

import pandas as pd

from xai_ews.data.features import COLUMNS, candidate_features


@dataclass(frozen=True)
class Rationale:
    feature: str
    sign: str
    meaning_en: str
    meaning_ru: str
    why_en: str
    why_ru: str
    risky: str | None = None
    safe: str | None = None


R = Rationale
_RATIONALE = [
    # borrower at origination
    R("borrower_credit_score_at_origination", "-", "FICO score of the borrower at origination",
      "кредитный рейтинг (FICO) заёмщика при выдаче",
      "summarises past repayment behaviour; the classic strongest predictor",
      "отражает историю платежей; классический сильнейший предиктор"),
    R("co_borrower_credit_score_at_origination", "-", "FICO score of the co-borrower (empty if none)",
      "рейтинг созаёмщика (пусто, если его нет)",
      "a second reliable payer lowers risk", "второй надёжный плательщик снижает риск"),
    R("debt_to_income_dti", "+", "monthly debt payments / income at origination, %",
      "долговая нагрузка: платежи по долгам / доход, %",
      "less income left after debt service, less buffer against shocks",
      "меньше свободного дохода — меньше запас прочности при шоке"),
    R("number_of_borrowers", "-", "number of borrowers on the loan", "число заёмщиков по кредиту",
      "two incomes diversify the risk of job loss", "два дохода снижают риск потери платёжеспособности"),
    R("first_time_home_buyer_indicator", "cat", "first-time home buyer (Y/N)",
      "покупает жильё впервые (Y/N)", "less equity and no ownership experience",
      "меньше накоплений и нет опыта владения жильём", risky="Y", safe="N"),
    # loan at origination
    R("original_interest_rate", "+", "note rate at origination, %", "ставка по кредиту при выдаче, %",
      "risk-based pricing plus a higher payment burden",
      "ставка включает надбавку за риск и увеличивает платёж"),
    R("original_upb", "none", "loan amount at origination", "сумма кредита при выдаче",
      "exposure: enters via the loss weight and EL, no prior sign (methodology 4)",
      "экспозиция: учитывается через вес ω и EL, знак заранее не задаётся (методология, п. 4)"),
    R("original_loan_term", "+", "contractual term, months", "срок кредита по договору, мес.",
      "15-year borrowers are stronger; 30-year loans amortise slowly",
      "заёмщики на 15 лет обычно сильнее; 30-летний кредит медленнее гасится"),
    R("original_loan_to_value_ratio_ltv", "+", "loan / property value at origination, %",
      "кредит / стоимость жилья при выдаче (LTV), %",
      "less own equity, more incentive to default when prices fall",
      "меньше собственных средств, больше стимул к дефолту при падении цен"),
    R("original_combined_loan_to_value_ratio_cltv", "+", "all liens / property value, %",
      "все кредиты под этот залог / стоимость жилья (CLTV), %",
      "as LTV, including second liens", "как LTV, но с учётом вторых закладных"),
    R("loan_purpose", "cat", "purpose: P purchase, R rate-term refi, C cash-out refi",
      "цель: P покупка, R рефинансирование, C рефинансирование с выдачей наличных",
      "cash-out refinancing extracts equity and signals liquidity need",
      "cash-out уменьшает долю собственника и говорит о потребности в деньгах", risky="C", safe="P"),
    R("channel", "cat", "origination channel: R retail, B broker, C correspondent",
      "канал выдачи: R розница, B брокер, C корреспондент",
      "third-party origination has weaker incentives for underwriting quality",
      "у посредников слабее стимулы к качеству андеррайтинга", risky="B", safe="R"),
    R("seller_name", "none", "institution that sold the loan to Fannie Mae", "продавец кредита в Fannie Mae",
      "no prior; drift risk", "знак не задаётся; риск дрейфа"),
    R("property_type", "cat", "SF single-family, PU planned unit, CO condo, CP co-op, MH manufactured",
      "тип объекта: SF дом, PU таунхаус в посёлке, CO кондоминиум, CP кооператив, MH сборный дом",
      "manufactured housing loses value faster", "сборные дома быстрее теряют стоимость",
      risky="MH", safe="SF"),
    R("number_of_units", "+", "number of dwelling units (1-4)", "число квартир в объекте (1–4)",
      "multi-unit properties depend on rental income", "многоквартирные объекты зависят от аренды"),
    R("occupancy_status", "cat", "P principal residence, S second home, I investor",
      "P основное жильё, S второе жильё, I инвестиционное",
      "investors default strategically more easily", "инвестору проще отказаться от платежей",
      risky="I", safe="P"),
    R("mortgage_insurance_percentage", "+", "mortgage insurance coverage, %", "покрытие ипотечного страхования, %",
      "insurance is required for LTV > 80, so it proxies for high LTV",
      "страховка обязательна при LTV > 80, поэтому это прокси высокого LTV"),
    R("mortgage_insurance_type", "cat", "1 borrower-paid, 2 lender-paid, empty = no insurance",
      "1 платит заёмщик, 2 платит кредитор, пусто — страховки нет",
      "insured loans are high-LTV loans", "застрахованы именно кредиты с высоким LTV",
      risky="1", safe="NA"),
    R("amortization_type", "cat", "FRM fixed rate / ARM adjustable", "FRM фиксированная / ARM плавающая ставка",
      "payment shocks on ARMs", "у ARM возможен рост платежа", risky="ARM", safe="FRM"),
    R("prepayment_penalty_indicator", "cat", "prepayment penalty (Y/N)", "штраф за досрочное погашение (Y/N)",
      "typical of riskier products", "характерен для более рискованных продуктов", risky="Y", safe="N"),
    R("interest_only_loan_indicator", "cat", "interest-only period (Y/N)", "период выплаты только процентов (Y/N)",
      "no amortisation, payment shock later", "нет погашения долга, позже рост платежа", risky="Y", safe="N"),
    R("high_balance_loan_indicator", "none", "loan above the standard conforming limit (Y/N)",
      "кредит выше стандартного лимита (Y/N)",
      "high-cost areas; wealthier borrowers but larger loans: no prior",
      "дорогие регионы: заёмщики богаче, но кредиты больше — знак не задаётся"),
    R("property_valuation_method", "none", "A appraisal, W waiver, ...", "способ оценки: A оценщик, W без оценки, …",
      "no prior", "знак не задаётся"),
    R("relocation_mortgage_indicator", "cat", "relocation mortgage (employer-assisted move)",
      "ипотека при переезде по работе",
      "employer support and a stable job", "поддержка работодателя и стабильная работа",
      risky="N", safe="Y"),
    R("special_eligibility_program", "cat", "affordable programs: H HomeReady, F HFA, 7 none",
      "льготные программы: H HomeReady, F HFA, 7 нет",
      "affordable programs target lower incomes", "программы рассчитаны на низкие доходы",
      risky="H", safe="7"),
    R("high_loan_to_value_hltv_refinance_option_indicator", "cat", "high-LTV refinance option (Y/N)",
      "рефинансирование при высоком LTV (Y/N)", "the borrower is underwater",
      "заёмщик должен больше, чем стоит жильё", risky="Y", safe="N"),
    # geography
    R("property_state", "none", "state of the property", "штат объекта",
      "local markets and foreclosure law differ; no direction", "разные рынки и законы об изъятии; направления нет"),
    R("metropolitan_statistical_area_msa_or_metropolitan_statistical_division_area_msda", "none",
      "metropolitan area code (00000 = outside)", "код агломерации (00000 — вне агломерации)",
      "local labour and housing market", "местный рынок труда и жилья"),
    R("zip_code_short", "none", "first 3 digits of the ZIP code", "первые 3 цифры почтового индекса",
      "finest geography available", "самая подробная география в данных"),
    # state at t
    R("servicer_name", "none", "current servicer", "текущий сервисер (обслуживающая организация)",
      "no prior; servicing transfers concentrate on troubled loans; drift risk",
      "знак не задаётся; проблемные кредиты чаще передаются другим сервисерам; риск дрейфа"),
    R("current_interest_rate", "+", "note rate at t", "ставка на дату t",
      "as the original rate; differs only after modification", "как ставка при выдаче; отличается только после модификации"),
    R("age", "+", "loan age at t, months", "возраст кредита на t, мес.",
      "seasoning: default hazard rises over the first 3-5 years (train ages 6-42 months)",
      "сезонность: риск дефолта растёт первые 3–5 лет (в train возраст 6–42 мес.)"),
    R("remaining_months_to_maturity", "+", "months left to maturity at t", "остаток срока на t, мес.",
      "more debt left, slower amortisation", "больше долга впереди, медленнее накапливается доля собственника"),
    R("remaining_months_to_legal_maturity", "+", "months left to the legal maturity date",
      "остаток срока до юридической даты погашения, мес.", "as remaining months", "как остаток срока"),
    R("upb", "none", "outstanding balance (ОД) at t", "остаток долга (ОД) на t",
      "exposure: no prior sign (methodology 4)", "экспозиция: знак заранее не задаётся (методология, п. 4)"),
    R("upb_to_orig", "none", "UPB at t / original UPB", "ОД на t / ОД при выдаче",
      "exposure: no prior sign (methodology 4)", "экспозиция: знак заранее не задаётся (методология, п. 4)"),
    # behaviour
    R("upb_vs_schedule", "+", "UPB / scheduled UPB - 1 (< 0: paying ahead)",
      "ОД / плановый ОД − 1 (< 0 — платит быстрее графика)",
      "behind schedule = missed principal", "отставание от графика = недоплаченный долг"),
    R("rate_change", "-", "current - original rate (cut = modification)",
      "текущая ставка − ставка при выдаче (снижение = модификация)",
      "rate cuts come from loss-mitigation modifications", "снижение ставки бывает при реструктуризации"),
    R("modified_now", "+", "modification flag at t", "флаг модификации на t",
      "modified loans re-default often", "реструктурированные кредиты часто снова уходят в просрочку"),
    R("ever_modified", "+", "ever modified up to t", "модифицировался когда-либо до t",
      "as modified_now", "как modified_now"),
    R("n_curtail_6m", "-", "months with a partial prepayment in 6 months",
      "число месяцев с частичным досрочным погашением за 6 мес.",
      "spare liquidity", "есть свободные деньги"),
    R("n_curtail_12m", "-", "months with a partial prepayment in 12 months",
      "число месяцев с частичным досрочным погашением за 12 мес.", "spare liquidity", "есть свободные деньги"),
    R("upb_chg_6m", "+", "UPB_t / UPB_(t-6) - 1 (usually < 0)", "ОД_t / ОД_(t−6) − 1 (обычно < 0)",
      "no principal paid -> change near 0", "если долг не гасится, изменение близко к 0"),
    R("upb_chg_12m", "+", "UPB_t / UPB_(t-12) - 1", "ОД_t / ОД_(t−12) − 1", "as upb_chg_6m", "как upb_chg_6m"),
    R("n_dpd_unknown_12m", "none", "months with unknown status (XX) in 12 months",
      "число месяцев с неизвестным статусом (XX) за 12 мес.", "data quality; no prior", "качество данных; знак не задаётся"),
    R("n_upb_increase_12m", "+", "months with a UPB increase in 12 months", "число месяцев с ростом ОД за 12 мес.",
      "capitalised arrears (modification)", "капитализация долга (реструктуризация)"),
    R("excess_principal_12m", "-", "principal paid above schedule over 12 months / original UPB",
      "сверхплановое погашение за 12 мес. / ОД при выдаче", "spare liquidity", "есть свободные деньги"),
    # macro / collateral
    R("mtm_ltv", "+", "mark-to-market LTV: UPB / (value indexed by FHFA HPI)",
      "текущий LTV: ОД / стоимость жилья, пересчитанная по индексу цен FHFA",
      "negative equity is the main trigger of strategic default", "отрицательный капитал — главный триггер дефолта"),
    R("hpi_chg_since_orig", "-", "local house-price change since origination", "изменение цен на жильё с даты выдачи",
      "rising prices build equity", "рост цен увеличивает долю собственника"),
    R("hpi_chg_12m", "-", "local house-price change over 12 months", "изменение цен на жильё за 12 мес.",
      "falling prices destroy equity", "падение цен уменьшает долю собственника"),
    R("unemployment", "+", "state unemployment rate", "безработица в штате", "income shocks", "риск потери дохода"),
    R("unemployment_chg_6m", "+", "change of state unemployment over 6 months", "изменение безработицы за 6 мес.",
      "worsening labour market", "ухудшение рынка труда"),
    R("unemployment_chg_12m", "+", "change of state unemployment over 12 months", "изменение безработицы за 12 мес.",
      "worsening labour market", "ухудшение рынка труда"),
    R("unemployment_rel_us", "+", "state - US unemployment (relative)",
      "безработица штата минус США (относительная)", "local labour market weaker than the nation",
      "местный рынок труда слабее среднего"),
    R("unemployment_chg_12m_rel_us", "+", "state - US 12-month change of unemployment",
      "изменение безработицы штата минус изменение по США", "local worsening", "местное ухудшение"),
    R("hpi_chg_12m_rel_us", "-", "local - US 12-month house-price change", "изменение цен локально минус по США",
      "local price weakness", "местная слабость цен"),
    R("hpi_chg_since_orig_rel_us", "-", "local - US house-price change since origination",
      "изменение цен с выдачи локально минус по США", "local equity build-up", "местный рост доли собственника"),
    R("rate_spread_pmms", "+", "current rate - market 30y rate (PMMS) at t", "текущая ставка − рыночная (PMMS) на t",
      "not refinancing although rates are lower: constrained borrower", "не рефинансируется при более низких ставках — ограниченные возможности"),
    R("sato", "+", "spread at origination: rate - PMMS at origination", "спред при выдаче: ставка − PMMS на дату выдачи",
      "the lender priced extra risk", "кредитор заложил надбавку за риск"),
]

RATIONALE: dict[str, Rationale] = {r.feature: r for r in _RATIONALE}


def rationale_table() -> pd.DataFrame:
    """One row per candidate feature, in candidate_features() order; `static` marks the H3 set."""
    static = set(candidate_features(static_only=True))
    rows = []
    for f in candidate_features():
        r = RATIONALE[f]
        rows.append({"feature": f, "group": COLUMNS[f].group, "temporal": COLUMNS[f].temporal,
                     "static_h3": f in static, "expected_sign": r.sign,
                     "risky_level": r.risky, "safe_level": r.safe,
                     "meaning_en": r.meaning_en, "meaning_ru": r.meaning_ru,
                     "why_en": r.why_en, "why_ru": r.why_ru})
    return pd.DataFrame(rows)
