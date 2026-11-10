<div align="center">

# Explainable early warning of mortgage credit risk

### Research internship: data, code, figures and report

**Language:** [Русский](README_ru.md) · [English](README.md)

Zarina Nurzhanova
Astana IT University, Astana, Kazakhstan · Applied Data Analytics (7M06103)

[![status — research internship 2026](https://img.shields.io/badge/status-research%20internship%202026-f9a825)](Неделя_10_Отчет/Отчет_по_исследовательской_практике_Нуржанова.docx)
[![data — Fannie Mae, 2006–2025](https://img.shields.io/badge/data-Fannie%20Mae%2C%202006%E2%80%932025-1565c0)](Неделя_04_Данные/Описание_данных.md)
[![task — first default within 12 months](https://img.shields.io/badge/task-first%20default%20within%2012%20months-6a1b9a)](Неделя_01_Организационный_этап/Цели_и_задачи.md)
[![model — CatBoost · SHAP](https://img.shields.io/badge/model-CatBoost%20%C2%B7%20SHAP-455a64)](Неделя_07_Базовые_модели/Результаты_моделирования.md)
[![code — MIT](https://img.shields.io/badge/code-MIT-455a64)](LICENSE)
[![data — CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-455a64)](LICENSE-DATA)

</div>

> [!IMPORTANT]
> This repository contains the materials of the research internship (base: Smart City Research and Innovation Center, 7 September – 14 November 2026) for the master's thesis *Development of an explainable AI model for early detection of credit risk and hidden deterioration of borrowers*. It is organised by the weeks of the internship schedule: each week holds its documents, executed notebooks, result tables and figures. The documents are in Russian.

## About

The study predicts the **first default** (90+ days past due or a default exit) within 12 months for mortgage loans that are current at the observation date, using open Fannie Mae loan-level data with house prices, unemployment and market mortgage rates:

```text
Fannie Mae loan performance → loan × quarter panel → first-default label → 59 candidate features → 18 selected
→ scorecard / CatBoost / XGBoost / LightGBM / MLP → calibration and alert list → frozen test protocol → SHAP and LIME
```

The panel has 40.6 million quarterly snapshots of a 15 % loan sample. Samples are split by time: train 2013–2015, validation 2016–2017, test 2018; stress periods 2007–2009, COVID 2019–2021 and rising rates 2022–2025. The event rate is 0.12–0.21 %, so models are compared by AUC, PR-AUC, calibration and the precision and recall of the alert list (calibrated PD ≥ 1 %).

> [!WARNING]
> The test set was opened a second time, under a new protocol frozen before the opening (the first opening was in an earlier two-task setting); this is named as a limitation. Alert lead time is measured on quarterly snapshots and is a coarse measure. The models are research results, not a production credit model.

## Main results

Test 2018: 3,758,886 observations, 7,824 defaults; alert list = calibrated PD ≥ 1 %.

| Model | AUC [95 % CI] | Loans on the list | Defaults caught | Loss caught |
|---|---:|---:|---:|---:|
| Scorecard B0 | 0.825 [0.819; 0.831] | 3.3 % | 22.6 % | 20.9 % |
| MLP M2 | 0.834 [0.828; 0.839] | 3.7 % | 27.0 % | 27.1 % |
| **CatBoost M1** | **0.838** [0.832; 0.844] | 3.7 % | 27.6 % | 27.2 % |
| XGBoost | 0.839 [0.833; 0.845] | 4.2 % | 30.2 % | 30.6 % |
| LightGBM | 0.839 [0.833; 0.845] | 3.9 % | 29.1 % | 29.0 % |

- **H3 confirmed:** dynamic features add +0.015 AUC [0.013; 0.017] and raise alerts for 36 % of future defaults instead of 26 %; median lead time 10 months.
- **H4 partial:** weighting by the loan balance keeps calibration but does not raise the captured loss (−0.5 pp). **H2** holds with a small effect.
- **Explanations:** CatBoost's built-in SHAP is not additive for depthwise trees deeper than 2, so exact permutation SHAP is used; reason codes are dominated by FICO and the co-borrower; LIME agrees on the top feature but is less stable (local R² 0.16).

Sources: [`comparison_table.csv`](Неделя_07_Базовые_модели/tables/test/comparison_table.csv), [`alert_capture.csv`](Неделя_07_Базовые_модели/tables/test/alert_capture.csv), [`h3_verdict.csv`](Неделя_07_Базовые_модели/tables/test/h3_verdict.csv), [`h4_verdict.csv`](Неделя_07_Базовые_модели/tables/test/h4_verdict.csv).

## Quick start

### 1. Find the material of a week

| Week | Activity | Folder contents |
|---|---|---|
| [01](Неделя_01_Организационный_этап) | Organisational stage | schedule plan, aims, objectives and hypotheses |
| [02](Неделя_02_Обзор_литературы) | Analytical stage | literature review and reference list |
| [03](Неделя_03_Сравнение_подходов) | Analytical stage | comparison table of credit-risk models and XAI methods |
| [04](Неделя_04_Данные) | Data collection | data description, data audit report, dictionary of 59 features, audit notebook and tables |
| [05](Неделя_05_Очистка_и_признаки) | Data preparation | feature selection 59 → 18: description, report, 10 notebooks, tables |
| [06](Неделя_06_Разведочный_анализ) | Exploratory analysis | EDA notebook, 7 figures, tables |
| [07](Неделя_07_Базовые_модели) | Modelling | models and test results, test protocol, 11 notebooks, metric tables, figure and its script |
| [08](Неделя_08_Объяснимый_ИИ) | Modelling (XAI) | SHAP and LIME notebook, 5 figures, tables |
| [09](Неделя_09_Аналитическая_записка) | Analysis of results | analytical note with interim conclusions and plan |
| [10](Неделя_10_Отчет) | Final stage | research internship report (.docx) |

### 2. Find the data behind a table or figure of the report

| Report | Figure | Data |
|---|---|---|
| Table 3 (samples) | — | [`events_by_split.csv`](Неделя_04_Данные/tables/events_by_split.csv), [`events_train_full.csv`](Неделя_04_Данные/tables/events_train_full.csv) |
| Table 4 (feature selection) | — | [`feature_selection_table.csv`](Неделя_05_Очистка_и_признаки/tables/feature_selection_table.csv), [`ablation.csv`](Неделя_05_Очистка_и_признаки/tables/ablation.csv) |
| Table 5 (test metrics) | — | [`comparison_table.csv`](Неделя_07_Базовые_модели/tables/test/comparison_table.csv), [`comparison_pairwise.csv`](Неделя_07_Базовые_модели/tables/test/comparison_pairwise.csv) |
| Table 6 (alert list) | — | [`alert_capture.csv`](Неделя_07_Базовые_модели/tables/test/alert_capture.csv) |
| Table 7 (hypotheses) | — | [`h2_verdict.csv`](Неделя_07_Базовые_модели/tables/h2_verdict.csv), [`h3_verdict.csv`](Неделя_07_Базовые_модели/tables/test/h3_verdict.csv), [`h4_verdict.csv`](Неделя_07_Базовые_модели/tables/test/h4_verdict.csv) |
| Figure 1 | [`01_sample.png`](Неделя_06_Разведочный_анализ/figures/01_sample.png) | [`eda_by_date.csv`](Неделя_06_Разведочный_анализ/tables/eda_by_date.csv), [`eda_by_vintage.csv`](Неделя_06_Разведочный_анализ/tables/eda_by_vintage.csv) |
| Figure 2 | [`02_missing.png`](Неделя_06_Разведочный_анализ/figures/02_missing.png) | [`eda_missing.csv`](Неделя_06_Разведочный_анализ/tables/eda_missing.csv) |
| Figure 3 | [`04_distributions.png`](Неделя_06_Разведочный_анализ/figures/04_distributions.png) | [`eda_distributions.csv`](Неделя_06_Разведочный_анализ/tables/eda_distributions.csv) |
| Figure 4 | [`05_category_rates.png`](Неделя_06_Разведочный_анализ/figures/05_category_rates.png) | [`eda_category_rates.csv`](Неделя_06_Разведочный_анализ/tables/eda_category_rates.csv) |
| Figure 5 | [`06_upb_masking.png`](Неделя_06_Разведочный_анализ/figures/06_upb_masking.png) | [`eda_upb_masking.csv`](Неделя_06_Разведочный_анализ/tables/eda_upb_masking.csv) |
| Figure 6 | [`fig_models.png`](Неделя_07_Базовые_модели/figures/fig_models.png) | [`comparison_table.csv`](Неделя_07_Базовые_модели/tables/test/comparison_table.csv), [`stress_metrics.csv`](Неделя_07_Базовые_модели/tables/stress/stress_metrics.csv) |
| Figure 7 | [`01_shap_global.png`](Неделя_08_Объяснимый_ИИ/figures/01_shap_global.png) | [`shap_global.csv`](Неделя_08_Объяснимый_ИИ/tables/shap_global.csv) |
| Figure 8 | [`02_shap_waterfall_caught.png`](Неделя_08_Объяснимый_ИИ/figures/02_shap_waterfall_caught.png) | [`local_reason_codes.csv`](Неделя_08_Объяснимый_ИИ/tables/local_reason_codes.csv) |
| Figure 9 | [`04_reason_codes.png`](Неделя_08_Объяснимый_ИИ/figures/04_reason_codes.png) | [`reason_code_frequency.csv`](Неделя_08_Объяснимый_ИИ/tables/reason_code_frequency.csv) |
| Figures A.1–A.3 | [`03_outliers.png`](Неделя_06_Разведочный_анализ/figures/03_outliers.png), [`07_woe_fico.png`](Неделя_06_Разведочный_анализ/figures/07_woe_fico.png), [`03_shap_vs_lime_case.png`](Неделя_08_Объяснимый_ИИ/figures/03_shap_vs_lime_case.png) | [`eda_outliers.csv`](Неделя_06_Разведочный_анализ/tables/eda_outliers.csv), [`eda_woe_fico.csv`](Неделя_06_Разведочный_анализ/tables/eda_woe_fico.csv), [`shap_vs_lime.csv`](Неделя_08_Объяснимый_ИИ/tables/shap_vs_lime.csv) |

### 3. Reproduce the results

- **Requirements.** Python 3.11 and [uv](https://docs.astral.sh/uv/); the environment is pinned by `code/uv.lock`.
- **Without downloads.** The tests and the model-comparison figure use only the files in this repository:

```bash
cd code && uv sync && uv run pytest
cd .. && uv run --project code python Неделя_07_Базовые_модели/code/fig_models.py
```

- **From the original data.** The notebooks need the Fannie Mae Single-Family Loan Performance Data (free after registration on the Fannie Mae website) and the public FHFA, BLS and Freddie Mac series, which are not distributed here. The scripts in `code/scripts/` load the raw files (`ingest_raw.py`), build the panel (`build_panel.py`, `build_default_history.py`), audit it (`audit_panel.py`, `leakage_check.py`), tune and fit the models (`tune.py`, `fit_models.py`); the notebooks then reproduce every table and figure.

## Structure

```text
.
├── Неделя_01_Организационный_этап/   # schedule plan, aims and objectives
├── Неделя_02_Обзор_литературы/       # literature review, reference list
├── Неделя_03_Сравнение_подходов/     # comparison of models and XAI methods
├── Неделя_04_Данные/                 # data description and audit
├── Неделя_05_Очистка_и_признаки/     # cleaning and feature selection
├── Неделя_06_Разведочный_анализ/     # exploratory data analysis
├── Неделя_07_Базовые_модели/         # baseline models, test, stress tests
├── Неделя_08_Объяснимый_ИИ/          # SHAP and LIME
├── Неделя_09_Аналитическая_записка/  # analytical note
├── Неделя_10_Отчет/                  # internship report
│     each week: notebooks/ · tables/ · figures/ where present
├── code/                             # xai_ews package, scripts, configs, tests
├── LICENSE                           # MIT, code
└── LICENSE-DATA                      # CC BY 4.0, data, figures and documents
```

## Sources

- Fannie Mae, *Single-Family Loan Performance Data*;
- FHFA House Price Index; U.S. Bureau of Labor Statistics, Local Area Unemployment Statistics; Freddie Mac Primary Mortgage Market Survey;
- OpenFEMA Disaster Declarations (exclusion of observations hit by natural disasters);
- the literature of the review — [`Список_литературы.md`](Неделя_02_Обзор_литературы/Список_литературы.md).

## Citation

Nurzhanova Z. *Research internship report: development of an explainable AI model for early detection of credit risk and hidden deterioration of borrowers.* Astana IT University, School of Artificial Intelligence and Data Science, 2026.

## License

- **Code** in `code/`, the week scripts and the code cells of the notebooks: [MIT License](LICENSE).
- **Data, figures, documents and the report**: [Creative Commons Attribution 4.0 International](LICENSE-DATA).

The result tables are derived from public Fannie Mae, FHFA, BLS, Freddie Mac and OpenFEMA data, which keep their own terms of use and are not distributed here. The license covers the author's processing, analysis and results; when using a value, cite the primary source together with this repository.
