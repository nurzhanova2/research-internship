# Протокол открытия test (этап 3, задача Y1)

**Статус: ЗАМОРОЖЕН 2026-11-03 10:54** — второе открытие test (раздел 0). Протокол пишется до открытия test; пробный прогон кода 03_6 / 03_6b сделан на validation 2026-11-03 (все ячейки выполнены, результаты — `reports/modeling/dry_run_validation/`; в нём калибраторы оцениваются и на половине B, поэтому цифры — только проверка кода). Test открывается один раз ноутбуками `notebooks/03_6_test_evaluation.ipynb` и `notebooks/03_6b_test_sensitivity.ipynb`; их первая ячейка сверяет sha256 из раздела 12. Любое изменение после просмотра test помечается как post-hoc и записывается в раздел 13 и в журнал решений.

Источники правил: гипотезы — [03_goal_tasks_hypotheses.md](../../docs/proposal/ru/03_goal_tasks_hypotheses.md); методология — [05_methodology.md](../../docs/proposal/ru/05_methodology.md); решения 19, 24, 30, 32–40, 46–50, 58–63 — [08_open_questions.md](../../docs/proposal/ru/08_open_questions.md).

## 0. Это второе открытие test

Test уже открывался 2026-10-26 по предыдущему протоколу (две задачи Y1 и Y2, популяция Y1 — DPD < 3, 25 признаков, тревога — топ-10 % портфеля). Тот протокол и все его результаты сохранены без изменений в `reports/archive/stage3_two_tasks_2026-10-27/`. После этого постановка изменилась по указаниям руководителя и решениям автора, принятым на train и validation (решения 58–63): осталась одна задача Y1; популяция — кредиты без просрочки на дату t и без дефолта в прошлом; признаки просрочки исключены; заново отобраны 18 признаков; тревога — откалиброванная PD ≥ 0,01. Все модели этого протокола обучены заново.

Поэтому настоящее открытие — **повторная оценка на тех же датах test**, а не первая встреча с ними. В отчёте и статьях это называется ограничением: автор видел результаты первого открытия (на другой постановке), и нельзя полностью исключить, что это косвенно повлияло на решения 58–63. Описательная доля повторных дефолтов в test (13,6 %, из аудита этапа 1) упомянута в решении 62; на выбор признаков, моделей и порогов результаты test не влияли — они выбраны по train и validation (этап 2, ноутбуки 03_0–03_5, 03_2b).

## 1. Выборки

| Выборка | Что входит | Роль |
| --- | --- | --- |
| train | полный train (решение 24): t 2013-03…2015-06, поколения 2012Q1, 2012Q3, 2014Q1–Q4 загружены на 100 %; все события + отрицательные с `u_neg` < r, r = 0,2, вес строки ω(ОД) × 1/r (решение 35); отложенные 20 % кредитов не входят. 2 294 292 строки, 14 062 события | обучение всех моделей |
| validation, половина A | t 2016-06…2017-03, 15 %-я хэш-выборка, без отложенных кредитов и без наблюдений по правилу бедствий; половины по ID кредита, стратифицированы по наличию события Y1, seed 20261025 (решение 48). 853 029 строк, 1 695 событий | подбор гиперпараметров, ранняя остановка, выбор C скоринговой карты, проверка H2 |
| validation, половина B | то же; 1 695 событий | калибровка Платта; больше ни для чего |
| test | t 2018-03…2018-12, 15 %-я хэш-выборка, **все** кредиты (включая отложенные 20 %), без наблюдений по правилу бедствий | таблица сравнения, H3, H4 |
| стресс | кризис 2007-03…2010-12, COVID 2019-03…2021-12, рост ставок 2022-03…2025-03 | только оценка (решение 7) |

**Задача Y1** (методология 2.1–2.2, решения 58, 60, 62): событие — 90+ DPD или выход с кодом 02/03/09/15 в (t, t + 12]; досрочное погашение — 0. Популяция — DPD = 0 на t и ни одного эпизода 90+ DPD до t (только первый дефолт; `default_history.parquet`, сужение при чтении).

## 2. Правило исключения стихийных бедствий (решения 46, 49, 50)

- Декларации OpenFEMA Disaster Declarations Summaries v2: тип `DR`, `iaProgramDeclared` или `ihProgramDeclared`, все типы инцидентов, дата — `incidentBeginDate`; территории племён (код округа 000) не сопоставляются.
- Наблюдение (кредит, t) исключается, если месяц начала инцидента ∈ [t − 2, t + 12] и ZIP3 объекта затронут: ≥ 50 % жилых единиц ZIP3 в объявленных округах (Census 2010 ZCTA–county, HUPT). Виргинские острова и Гуам: ZIP3 затронут при объявлении любого их округа.
- Декларации COVID-19 (Biological) не применяются внутри стресс-сценария COVID.
- Одинаково для validation, test и стресс-сценариев; к train не применяется. Флаги — `data/processed/disaster_flags.parquet` (ноутбук 03_0).
- Основной результат — без затронутых наблюдений. Чувствительность: все наблюдения; только ураганы, тропические штормы и тайфуны; ZIP3 затронут при любом пересечении. Доли исключённых (`reports/modeling/disaster_exclusions.csv`): validation 15,1 % наблюдений (47,6 % событий — ураганы 2017 г.), test 8,6 % наблюдений (9,7 % событий).

## 3. Модели

Признаки — основной набор из **18 признаков** `configs/features.yaml → selected.features` (11 статических, 7 динамических; 5 категориальных); наборы H3 — `static_h3` (16) и `h3_full` (23). Категории — целые коды, `seller_name` — топ-20 по train + «прочие».

| Модель | Описание | Итоговый прогноз |
| --- | --- | --- |
| B0 | логистическая регрессия на WoE (корзины этапа 2), L2, C подобрано по log-loss на A | одна модель |
| M1 | CatBoost, вес ω × 1/r | **среднее логитов 5 фолдовых моделей** GroupKFold по ID кредита, каждая на 4/5 train с подобранными параметрами и числом деревьев (решение 19); out-of-fold логиты train сохраняются для этапа 4 |
| M2 | MLP, вес ω × 1/r | **среднее логитов 5 моделей** с 5 seed лучшей конфигурации; отдельно — среднее ± std метрик по seed |
| M1-H4 (`m1_y1_noomega`) | CatBoost **без** ω (вес 1/r), гиперпараметры и число деревьев — от M1 | как M1 |
| M1-H3-static, M1-H3-full | CatBoost на `static_h3` / `h3_full`, подобраны **отдельно** с тем же бюджетом | как M1 |
| XGBoost, LightGBM (`xgb_y1`, `lgb_y1`) | сравнительные модели: тот же протокол, что у M1 (18 признаков, 40 попыток, ω × 1/r, 5 фолдов, Платт на B); ноутбук 03_2b | как M1; **описательно**, не входят в решающие правила |

**Устройство.** Модели CatBoost этапа 3 (подбор и итоговые) обучены на GPU в Google Colab (`notebooks/colab_stage3_catboost.ipynb`, пакет данных без test — `scripts/make_colab_bundle.py`); применяются на CPU. MLP, B0, XGBoost, LightGBM и одиночные модели стресс-тестов обучены локально. XGBoost и LightGBM считаются в отдельном процессе (`scripts/score_gbdt.py`): их среда OpenMP зависает в одном процессе с PyTorch; на результаты это не влияет.

**Фиксированные настройки CatBoost** (не подбираются): Logloss; `grow_policy = Depthwise`; `bootstrap_type = Bernoulli`; `max_ctr_complexity = 1`; не больше 3 000 деревьев, ранняя остановка через 100 деревьев без улучшения.

**MLP:** квантильное преобразование числовых признаков к нормальному (параметры по train), пропуски → 0 + индикатор, эмбеддинги категорий, n × (Linear → ReLU → Dropout), AdamW, BCE с весом ω × 1/r, ранняя остановка по log-loss на A (терпение 3 эпохи, не больше 40 эпох), MPS.

## 4. Подбор гиперпараметров (решение 48)

- Optuna TPE (seed 20261025), **40 попыток** на модель × набор; журналы — `data/processed/models/optuna.db`, выгрузка — `reports/modeling/tuning_*.csv`.
- Метрика выбора — log-loss на половине A с весом ω(ОД) (без 1/r); AUC на A — для контроля.
- CatBoost, XGBoost, LightGBM: глубина 4–10; learning_rate 0,02–0,3 (лог.); L2 1–30 (лог.); минимум строк в листе 1–2000 (лог.); subsample 0,5–1. MLP: слоёв 2–4; ширина 64/128/256/512; dropout 0–0,5; lr 10⁻⁴–10⁻² (лог.); weight decay 10⁻⁷–10⁻² (лог.); батч 1024/2048/4096/8192; эмбеддинг 4/8/16.
- У всех трёх подборов CatBoost лучшая глубина — 4, нижняя граница диапазона; оптимум плоский (30–40 из 40 попыток в пределах 1 % от лучшего log-loss). Диапазон не расширяется задним числом; в отчёте это описывается.
- Не подбираются: признаки и корзины WoE, функция потерь, ω, r = 0,2, калибратор, порог тревоги.
- H4: модели с ω и без ω — одни и те же гиперпараметры (подобранные для модели с ω). H3: статический и полный наборы — отдельный подбор.
- Проверки чувствительности — без нового подбора (допущение, называется в отчёте).

## 5. Калибровка

Platt: PD = sigmoid(a + b · logit), без весов, на половине B (решение 30) — для каждой модели и варианта отдельно. Isotonic на половине B — чувствительность. На test калибратор не переобучается.

## 6. Метрики и интервалы

- Дискриминация: AUC, Gini, KS, PR-AUC; **AUC внутри групп FICO** < 680 (с пропусками) / 680–739 / ≥ 740 (решение 63). Калибровка: Brier, log-loss, ECE (10 равных по числу корзин), ECE / доля событий, calibration slope и intercept. Экспозиция: AUC, взвешенный по ОД. Список тревог: доля строк под тревогой, точность, доля пойманных событий, доля пойманного EL. Стабильность: PSI скора train → test; разброс по seed (M2).
- 95 %-е интервалы — **кластерный bootstrap по ID кредита**, B = 1000, seed 20261025.
- DeLong — на **одном случайном снимке на кредит** (seed 20261025); чувствительность — на всех снимках.
- Сравнения в таблице: M1 − B0, M1 − M2, M2 − B0 по ΔAUC (DeLong), поправка Холма по трём сравнениям. Отдельно и описательно: M1 − XGBoost, M1 − LightGBM (Холм по двум).

## 7. EL, сигнал тревоги, пойманный EL (решения 59, 61)

- LGD — сегментный по решению 34 (`reports/data_audit/lgd_variants.csv`, вариант `base`): корзина **текущего** MTM LTV на t × судебный штат; при пропуске LTV — средний LGD своей группы штатов. EAD = ОД на t. EL = PD (откалиброванная) × LGD × EAD.
- **Тревога:** кредит под тревогой на дату t, если откалиброванная PD модели ≥ **0,01** (`configs/data.yaml → alert.pd_threshold`); чувствительность — 0,02 и 0,005. Доля портфеля под тревогой не задаётся — она результат.
- **Доля пойманного EL** = Σ по строкам под тревогой (y × LGD × EAD) / Σ по всем строкам (y × LGD × EAD), суммы по всем датам; y — фактическое событие Y1. **Доля пойманных событий** = Σ y под тревогой / Σ y.

## 8. Lead time

- Событие кредита: месяц E = t + смещение первого события по положительному снимку test.
- Окно снимков для test: 2017-06 … 2019-09 (все кредиты); для validation (настройка в 03_4): 2015-09 … 2017-12 (без отложенных кредитов).
- Тревога: первый квартальный снимок s ∈ [E − 12, E − 1], на котором кредит в популяции Y1 и его откалиброванная PD ≥ 0,01 (0,02 и 0,005 — чувствительность). Снимки вне выборки test оцениваются по признакам на дату s без использования меток; наблюдения по правилу бедствий не используются.
- Lead time = E − s (мес.); кредит без тревоги — «пропущен».
- Основная величина — **медиана lead time по кредитам с тревогой**, 95 % CI — bootstrap по кредитам с событием (B = 1000). Дополнительно: доля кредитов с тревогой, доля с lead time ≥ 6 мес., медиана с пропущенными = 0.

## 9. Решающие правила

| Гипотеза | Где | Правило |
| --- | --- | --- |
| H2 | train + validation (ноутбук 03_5) — **проверена до этого протокола, на test не перепроверяется** | LR test (логистическая регрессия на WoE 18 признаков с взаимодействиями «признак × сегмент» против без них, df = 36), p < 0,05 **и** Δlog-loss на validation > 0 (Platt на B, оценка на A). Сегменты: FICO < 680 / 680–739 / ≥ 740; чувствительность — MTM LTV ≤ 80 / 80–95 / > 95. Результат: подтверждена (p ≈ 7·10⁻⁶⁷; Δlog-loss +0,000031 [0,000008; 0,000056]), эффект мал; по LTV не выполняется |
| H3 | test, Y1 | DeLong: M1-H3-full против M1-H3-static, ΔAUC > 0 при p < 0,05; **и** нижняя граница 95 % CI медианного lead time M1-H3-full ≥ 3 мес. (тревога PD ≥ 0,01) |
| H4 | test, Y1 | M1 (с ω) против M1-H4 (без ω), обе откалиброваны Платтом на B, у каждой свой список тревог (PD ≥ 0,01): 95 % CI Δ доли пойманного EL выше 0; **и** \|Δslope\| ≤ 0,1 и \|Δintercept\| ≤ 0,1 (логит-шкала) на test. Абсолютная калибровка — описательно (решение 38) |

Описательно для H3 (записано до открытия test, решающее правило не меняется): на validation медианный lead time статической модели тоже 10 мес. (03_4) — квартальные снимки делают медиану почти одинаковой у всех моделей. Поэтому дополнительно приводятся разности «полная − статическая» доли событий с тревогой и медианного lead time на одних и тех же кредитах с событием, с 95 % CI (bootstrap по кредитам), для всех трёх порогов.

Ожидания по validation (половина A; для сравнения после открытия, не правило): M1 AUC 0,863, M2 0,858, B0 0,852, XGBoost 0,863, LightGBM 0,863; H3 ΔAUC +0,014 [0,009; 0,018], доля кредитов с тревогой 37 % против 25 %; H4 — частично (Δ доли EL +0,003 [−0,010; +0,016], калибровка без изменений).

Вердикт: «подтверждена» — выполнены все части правила; «частично» — выполнена часть; «отвергнута» — не выполнена ни одна.

## 10. Проверки чувствительности (ноутбук 03_6b; на гиперпараметрах основных моделей, без нового подбора)

1. r = 0,1 (M1).
2. Линейный вес ω = ОД / mean, обрезка на p99 (M1; для H4).
3. Без досрочно погашенных в окне (`prepaid_in_window`): M1 обучается и оценивается без них.
4. Правило бедствий: все наблюдения; только ураганы; любое пересечение ZIP3 (модели, H3, H4).
5. Отложенные 20 % кредитов (loan-disjoint): test только по `holdout_loan`.
6. Новые когорты (поколения 2015–2018): test только по `new_cohort`.
7. Калибровка isotonic вместо Platt (H4).
8. LGD: после страхования, без дисконтирования, только поколения train, постоянный (H4).
9. Порог тревоги 0,02 и 0,005 (H3, H4); DeLong на всех снимках (H3).
10. PSI скора train → test для B0, M1, M2, XGBoost, LightGBM.
11. r = 0,5 — **не проводится** (решение автора 2026-10-26: все события уже в train, прореживаются только отрицательные; на validation r = 0,1 и r = 0,2 дают одинаковый AUC). В отчёте указывается как непроведённая проверка.

## 11. Стресс-тесты (ноутбук 03_7; только оценка, без переобучения и перекалибровки)

Кризис 2007–2010, COVID (с forbearance и без — по `forbearance_now`), рост ставок 2022+: метрики раздела 6 и список тревог (PD ≥ 0,01) для B0, M1, M2. Разбиение по поколениям до COVID (решение 39): M1 с основными гиперпараметрами обучается на `split_vintage = train` (поколения 2014–2017) и оценивается на поколении 2018, t ≤ 2019-02. COVID, вариант «отобранные + безработица» (решение 42): одиночный CatBoost на 18 + `unemployment_rel_us`, `unemployment_chg_12m_rel_us` против такого же на 18 признаках. Знак эффекта `occupancy_status` в стрессе (решение 45) — описательно (SHAP по категориям и группам FICO).

Технические уточнения: (1) в стресс-сценариях до 13 млн строк, поэтому интервалы — кластерный bootstrap с **B = 200**; прогнозы считаются по одной дате наблюдения; (2) модели для разбиения по поколениям и для варианта с безработицей — **одна модель CatBoost на CPU** (без фолдов) с основными гиперпараметрами и числом деревьев (`scripts/fit_stress_models.py`: `vintage_y1`, `unemp_y1_18`, `unemp_y1_20`); (3) к train разбиения по поколениям правило бедствий не применяется, к оценке — применяется; (4) COVID без forbearance — без наблюдений с `forbearance_now` на дату t.

## 12. Контрольные суммы (sha256)

Модели: b0_y1, m1_y1, m2_y1, m1_y1_noomega, h3_static_y1, h3_full_y1, m1_y1_linear, m1_y1_r01, m1_y1_noprepaid, xgb_y1, lgb_y1, vintage_y1, unemp_y1_18, unemp_y1_20; плюс конфиги, флаги бедствий, половины validation, таблица LGD (79 файлов).

| Файл | sha256 |
| --- | --- |
| `configs/data.yaml` | `9d6ffca9e8ba04b81cf3cecd913174baaec17f48c5d29c772d33c139a1807221` |
| `configs/features.yaml` | `c5e54749e9c2951ebf7af727e071757843868f1f3a9971f94c011a7ce8c26faf` |
| `data/processed/disaster_flags.parquet` | `8f553e69661e6715e34125cd9f9a78d3931d38547c043d31000bae9471b4fe1f` |
| `data/processed/validation_halves.parquet` | `92330a7a29ad58274c1a8b54043ae2524bcc4d4b505b1fb07a25fd3d2b292208` |
| `reports/data_audit/lgd_variants.csv` | `55f9194e10af874715cb67cd6d7548f43ee4f7f7277eed8091a3e23fb6907415` |
| `data/processed/models/b0_y1.pkl` | `7fd16898e0f047009d3be7d79210ceaabf48daee734666c39e658960c0a45eb2` |
| `data/processed/models/m1_y1/encoder.pkl` | `1f029e1c36d4b284fac08ff46ba9824883ebd37f294a1637d1b71dc719ffddff` |
| `data/processed/models/m1_y1/fold_0.cbm` | `bb1f0c6616df7a59fcefa30017b9da3a68501a005cfca2796cf213be6117b631` |
| `data/processed/models/m1_y1/fold_1.cbm` | `fb76ac567c39439d0a39f29009c71f4be7460c1c087c3d71e35d37e563f9b0dc` |
| `data/processed/models/m1_y1/fold_2.cbm` | `a36632121e5eaf9b71b11d96bb78e5052e2ef9b2dc740d20ab42b550a660f83b` |
| `data/processed/models/m1_y1/fold_3.cbm` | `8fd8c92f283f96493ee594fb9feb22cf459b1814707f72cc75df00446dbf1792` |
| `data/processed/models/m1_y1/fold_4.cbm` | `28953f23cf9692e68e3e57c1dbe1ee7b1392aec80710817a03bb0d00c9a59dfa` |
| `data/processed/models/m1_y1/meta.json` | `9b7c51f5db9ad6f13f240af12324ea3523fd658d0375bd71711d7672b714e4cb` |
| `data/processed/models/m2_y1/encoder.pkl` | `f1df112b68751e35a45de7e6af7d0dadf19c7b393d0d8c499b64aea39ac2f328` |
| `data/processed/models/m2_y1/meta.json` | `7da87ba670cb2707cf2d2873c6797643a2c3681d6bb27fc2b58c68ab1cef5e90` |
| `data/processed/models/m2_y1/seed_0.pt` | `0b95e61e803792bd9cfa099fdc018c8d075fe9d093c15f6f505fe56575ff541e` |
| `data/processed/models/m2_y1/seed_1.pt` | `afa758d8b965eea2e2bfcf747dd804ddd0ef87b69133be66dfb2a8234248bc1e` |
| `data/processed/models/m2_y1/seed_2.pt` | `460baa6e99ee42df37ecc8fae7a4bee254ec47bf5270d23b06406d39b88b461b` |
| `data/processed/models/m2_y1/seed_3.pt` | `7411e5fed806c34bde9fc0edccaf111f34bf5280f7970c909eb02249e00262e5` |
| `data/processed/models/m2_y1/seed_4.pt` | `8d8df6de85b54b9dcd3f4c12a1e3c55fd8a1a498d22abf7973436a85e0a22a4f` |
| `data/processed/models/m1_y1_noomega/encoder.pkl` | `1f029e1c36d4b284fac08ff46ba9824883ebd37f294a1637d1b71dc719ffddff` |
| `data/processed/models/m1_y1_noomega/fold_0.cbm` | `36fd75e28e5430dad56f7fa11c9b6f3cff435e1dc62664ac825c4df6b92b8b74` |
| `data/processed/models/m1_y1_noomega/fold_1.cbm` | `59926f6fb7358d1c4c0f6e0103b1e5968854de5a2b484ae74b144fe0acc857e6` |
| `data/processed/models/m1_y1_noomega/fold_2.cbm` | `6590859b2d29390668642d95fcf709f01682d2214af05b24f6297d5a4c7188f9` |
| `data/processed/models/m1_y1_noomega/fold_3.cbm` | `b4dc16887acc987ce61c768f6ad03d5ffb30dc012f8e98b02435e106d850844e` |
| `data/processed/models/m1_y1_noomega/fold_4.cbm` | `50ca9d4eedfeef61f2a6a5e047e7b1aa5c76a65581f8f11ab38e64b046bcd90f` |
| `data/processed/models/m1_y1_noomega/meta.json` | `6967a898ea5da09106e8d80dbcacef4db4aee7f2e724d090f2582bce02b71cd3` |
| `data/processed/models/h3_static_y1/encoder.pkl` | `65b4561048da49653db2662b513b62a55cd9d8127f1fba7a6d87799014fbb4b1` |
| `data/processed/models/h3_static_y1/fold_0.cbm` | `89427ad620c1c23b25d9c34de12f740ecefd97af77d2408a628db939a04dea90` |
| `data/processed/models/h3_static_y1/fold_1.cbm` | `fbfa3d9fd19ecfcb503cd442fc2add8948f7aa8b0a0f543d2f8d03f1018a8c70` |
| `data/processed/models/h3_static_y1/fold_2.cbm` | `2273db7095fd3f611144f951710f012ae4c98e5cc33fcb89e5320ac5d498f840` |
| `data/processed/models/h3_static_y1/fold_3.cbm` | `92ecaf696b1e2ebc9ca9873d241e5f294781509e7142e3dd8bb49d250c2fcf4e` |
| `data/processed/models/h3_static_y1/fold_4.cbm` | `745695914a1d04156e6adb3af544e4f179fe499d28a318ef29b1d573b8514269` |
| `data/processed/models/h3_static_y1/meta.json` | `30a39401d7f8ddf41c309a035740b88fafffd0d0920dfd1668015014df0c0e37` |
| `data/processed/models/h3_full_y1/encoder.pkl` | `f4d61670ea56d62458cace5d91f39fafdc70e871f4d11a96f1cdd4cec367bcc0` |
| `data/processed/models/h3_full_y1/fold_0.cbm` | `3dbae6e8d11fa8257e4e088ad4db0560da19c06404b92bbcd86bd52a5c024ab0` |
| `data/processed/models/h3_full_y1/fold_1.cbm` | `387b3cdd72e9613871200b9a91544d2b779440b146409836cf5b9c5f95b40c8c` |
| `data/processed/models/h3_full_y1/fold_2.cbm` | `e197df4d620cd5380e05d586a9c653eaba3dfaa0efce31fcc0e0f00cdda9f0e6` |
| `data/processed/models/h3_full_y1/fold_3.cbm` | `d76fd4e193bb063e824cba38a43ea0008fc9830750c8a419c01c3ab89a7c447c` |
| `data/processed/models/h3_full_y1/fold_4.cbm` | `2df83d3b016c3f56e17818c878e7ab26909e8a30b61d241e84c6637c86c270fe` |
| `data/processed/models/h3_full_y1/meta.json` | `3f52cb8153c363445451e7930410c694c6560371beab2daee189f5983f395456` |
| `data/processed/models/m1_y1_linear/encoder.pkl` | `1f029e1c36d4b284fac08ff46ba9824883ebd37f294a1637d1b71dc719ffddff` |
| `data/processed/models/m1_y1_linear/fold_0.cbm` | `c8d761984c1d53148588b6a9332b8bd3f7d25ae17d1b60a7ca294cc0028ce7fe` |
| `data/processed/models/m1_y1_linear/fold_1.cbm` | `3711b6f4ef0c071187cfa1f5b9d3c5518c4acc93fd6bf664c2c75758f7c6b1a0` |
| `data/processed/models/m1_y1_linear/fold_2.cbm` | `e468e77d03239cb7562f8b9faba85feceeb09658e5cef37e4b07317bf9c17d86` |
| `data/processed/models/m1_y1_linear/fold_3.cbm` | `7c09b43a257b8c6ab72015d5582823a750d129bc712e32d937aafb3fc6be3f05` |
| `data/processed/models/m1_y1_linear/fold_4.cbm` | `2b864e56f2f2a419b250e5cb35719d8e55850ffc788a1fe1391f900541a0e097` |
| `data/processed/models/m1_y1_linear/meta.json` | `8f1703df067e4d83c75e2bdbb79863eaa18fc6f3119cca2f663bf62649901aa6` |
| `data/processed/models/m1_y1_r01/encoder.pkl` | `85ad5e3a75854687bb92467d1ca83bebdd999611ad3ba5750467e0f03294da74` |
| `data/processed/models/m1_y1_r01/fold_0.cbm` | `4ec68b73e49f4da7bbb6d8a5924a46e920a2075ad536536c4bafe8032b467c35` |
| `data/processed/models/m1_y1_r01/fold_1.cbm` | `8423693efbf8883534fa4501f57dddd85d2d0673e2e60cd6b75f5a410b24d514` |
| `data/processed/models/m1_y1_r01/fold_2.cbm` | `033f6661d5711d831fa6ae31344b1b24c02e256c21832ef5b05f9f6c88f2716b` |
| `data/processed/models/m1_y1_r01/fold_3.cbm` | `026d1a48ed81204e753e304cf5474634e9dfb07abecae6a742deccdfdf913ff9` |
| `data/processed/models/m1_y1_r01/fold_4.cbm` | `1baf3aea334661045bcd0584c5ed32819ef9bca7f33cf3a6ddfd321f437a1193` |
| `data/processed/models/m1_y1_r01/meta.json` | `0b8437abe73611035b863f4384756d5cec6ccc767b22f4b6d36ca0966ac20342` |
| `data/processed/models/m1_y1_noprepaid/encoder.pkl` | `01f6169d2d435295e37a2c0838da2288d057d7b8bccb44d43fe5dcabf8a23948` |
| `data/processed/models/m1_y1_noprepaid/fold_0.cbm` | `ec1e847db95ecc9d37f66960ca054e68da437be1ea115725ffe96ed3d46bab3d` |
| `data/processed/models/m1_y1_noprepaid/fold_1.cbm` | `24adee4eb670b8d56d69f0da2a23080f0c486bbb611c16dd5e7e12648824c959` |
| `data/processed/models/m1_y1_noprepaid/fold_2.cbm` | `cf751738226713611800e47d811363d22702a35252cfda350bf8da7bb44cd9d6` |
| `data/processed/models/m1_y1_noprepaid/fold_3.cbm` | `5290c27e98c3dc20f5c809fa04464fd950f247ef4f8ecc2386296a7ce1436860` |
| `data/processed/models/m1_y1_noprepaid/fold_4.cbm` | `d3a22ff570960545f724dc6e929c608d8894a5e234d0b4c44bb1bf965ebb332f` |
| `data/processed/models/m1_y1_noprepaid/meta.json` | `a3e9520f21769528c9047db1ce270142f923ba00c7e5af717bb5993b236f43dd` |
| `data/processed/models/xgb_y1/encoder.pkl` | `1f029e1c36d4b284fac08ff46ba9824883ebd37f294a1637d1b71dc719ffddff` |
| `data/processed/models/xgb_y1/fold_0.model` | `497a45387ada0b7bac1a0a5055e079ba4882fdf8086cae47d2e2831d069c50d3` |
| `data/processed/models/xgb_y1/fold_1.model` | `3ecfb7b05a31b91e7cd59db202c50594ba4f71279bc01cf2cfa787d1629e2f28` |
| `data/processed/models/xgb_y1/fold_2.model` | `b21871eefe33e2887db2a8d288ac51f5a88935d7c8144e1d9d5b51710fac0601` |
| `data/processed/models/xgb_y1/fold_3.model` | `71834e72cc16baee73c6a5028865f90fe8fbc0dddbe9543d7ddb81dd0459986b` |
| `data/processed/models/xgb_y1/fold_4.model` | `7c3f638d1d7273c978e36ff2063ca8c4daac16ec6fa6bb30a844c930492ad6ac` |
| `data/processed/models/xgb_y1/meta.json` | `47e509611f79dbe6df0e32d087adf02c6978e7da9aad14ea14e70de32a6b32f7` |
| `data/processed/models/lgb_y1/encoder.pkl` | `1f029e1c36d4b284fac08ff46ba9824883ebd37f294a1637d1b71dc719ffddff` |
| `data/processed/models/lgb_y1/fold_0.model` | `d2ac07d71d751cc1a22f2d7a34c14af858edaa84173bcc08a72999947d9b2ad2` |
| `data/processed/models/lgb_y1/fold_1.model` | `64f0a490e7a4b0dcb966464bcce37bd1fcce9e8884afeeb315b0a10217ceb16e` |
| `data/processed/models/lgb_y1/fold_2.model` | `42add4ab8d7e798ae8aedf4221298d96598d3a07ebf2cba96c34bcd52863117d` |
| `data/processed/models/lgb_y1/fold_3.model` | `e7bb4b99b5917962f26d42cb055a82b415665e00f192bc719ce1dc9b8a96d3df` |
| `data/processed/models/lgb_y1/fold_4.model` | `c9618119fdfb22958d25f74dd35b8c3167308104b34c0e230464d407936387ac` |
| `data/processed/models/lgb_y1/meta.json` | `274b413dfe63bda5b96aff68545c27f4a89946a9d2a4472c6fcb931298019984` |
| `data/processed/models/vintage_y1.pkl` | `ffb00d5eb38c05ede676f7bf247e2d949fb801e13557ebf3d8ff37c3922a2603` |
| `data/processed/models/unemp_y1_18.pkl` | `570ad550f07b3b021f1e02dab9550ba90785b6d4adc0ad8ddde62bc4a5f3cc20` |
| `data/processed/models/unemp_y1_20.pkl` | `a6cedd443a1462295b75c3412eaf3f9833a6e1cd0bebd80ea1c1a3018b112f3b` |

## 13. Журнал изменений после открытия test (post-hoc)

| Дата | Что | Влияние на результаты |
| --- | --- | --- |
| 2026-11-03 | Ноутбуки 03_6, 03_6b, 03_7 выполнены один раз (10:55–12:04) без ошибок и без изменений кода; post-hoc расчётов нет. После запуска заполнены только текстовые ячейки «Как читать» / «Итог», текст 03_7 переписан под Y1 и 18 признаков; из строки статуса убран номер решения (он печатается в выводе первой ячейки), в выводе 03_6 / 03_6b исправлена та же строка. | Нет: числа и выводы не менялись. |
| 2026-11-10 | **Post hoc, описательно:** шкала представления риска (`xai_ews.evaluation.risk_scale`, ноутбук `notebooks/03_8_risk_scale.ipynb`): относительный риск (PD / средняя PD портфеля на дату), процентиль в портфеле на дату, класс риска 1–10 на заранее заданных границах PD (0,05; 0,1; 0,2; 0,35; 0,5; 1; 2; 3,5; 5 %; порог тревоги 1 % открывает класс 7), баллы. По сохранённым прогнозам test из 03_6; модели и калибраторы не менялись. Результат: `test/risk_scale_grades.csv`, `test/risk_scale_examples.csv` (решение 66). | Нет: шкалы монотонны по PD, ранжирование, метрики и вердикты не меняются. |
