"""Stage-3 helpers: metrics, calibration, EL capture, lead time, tests, disaster rule, models."""

import numpy as np
import polars as pl
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from xai_ews.data.disasters import affected_zip3, disaster_flags, exclusion_expr, zip3_county_shares
from xai_ews.evaluation.el import (
    alert_mask,
    alert_threshold,
    captured_el_share,
    captured_event_share,
)
from xai_ews.evaluation.leadtime import event_months, lead_times, summarise
from xai_ews.evaluation.metrics import (
    all_metrics,
    calibration_slope_intercept,
    ece,
    ks,
    log_loss,
    sigmoid,
)
from xai_ews.evaluation.stats import cluster_bootstrap, holm, lr_test, weighted_auc_fn
from xai_ews.models.calibration import Platt
from xai_ews.models.catboost_model import group_folds


def calibrated_data(n=200_000, seed=0, a=-4.0, b=1.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = sigmoid(a + b * x)
    return x, p, (rng.random(n) < p).astype(int)


# ---- metrics -------------------------------------------------------------------------------

def test_ks_matches_two_sample_ks():
    rng = np.random.default_rng(1)
    s = np.r_[rng.normal(1, 1, 3000), rng.normal(0, 1, 7000)]
    y = np.r_[np.ones(3000), np.zeros(7000)].astype(int)
    assert ks(y, s) == pytest.approx(stats.ks_2samp(s[y == 1], s[y == 0]).statistic, abs=1e-12)


def test_calibrated_pd_has_slope_one_intercept_zero_and_small_ece():
    _, p, y = calibrated_data()
    slope, intercept = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.05) and intercept == pytest.approx(0.0, abs=0.05)
    assert ece(y, p) < 0.003
    # PD too extreme (logit x 2) -> slope ~ 0.5; PD too high -> negative intercept
    z = np.log(p / (1 - p))
    assert calibration_slope_intercept(y, sigmoid(2 * z))[0] == pytest.approx(0.5, abs=0.05)
    assert calibration_slope_intercept(y, sigmoid(z + 0.5))[1] == pytest.approx(-0.5, abs=0.05)


def test_all_metrics_and_weighted_log_loss():
    _, p, y = calibrated_data(50_000)
    m = all_metrics(y, p)
    assert m["auc"] == pytest.approx(roc_auc_score(y, p)) and m["gini"] == pytest.approx(2 * m["auc"] - 1)
    w = np.where(y == 1, 1.0, 2.0)
    yy = np.r_[y, y[y == 0]]
    pp = np.r_[p, p[y == 0]]
    assert log_loss(y, p, w) == pytest.approx(log_loss(yy, pp))   # weight 2 = row twice


# ---- calibration ---------------------------------------------------------------------------

def test_platt_recovers_the_true_map():
    x, _, y = calibrated_data(a=-3.0, b=0.7)
    cal = Platt().fit(x, y)
    assert cal.a == pytest.approx(-3.0, abs=0.05) and cal.b == pytest.approx(0.7, abs=0.03)


# ---- EL and alerts -------------------------------------------------------------------------

def test_alert_mask_is_pd_at_or_above_the_threshold():
    pd_ = np.array([0.5, 0.01, 0.0099, 0.002, 0.03])
    assert alert_mask(pd_, 0.01).tolist() == [True, True, False, False, True]
    assert alert_mask(pd_).tolist() == alert_mask(pd_, alert_threshold()).tolist()


def test_captured_el_share_counts_realised_loss_on_the_list():
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    lgd = np.full(10, 0.5)
    ead = np.array([100, 1, 300, 1, 100, 1, 1, 1, 1, 1], float)
    pd_ = np.array([0.09, 0.08, 0.001, 0, 0.005, 0, 0, 0, 0, 0])   # PD >= 0.01: rows 0 and 1
    al = alert_mask(pd_, 0.01)
    assert captured_el_share(y, al, lgd, ead) == pytest.approx(100 / 500)
    assert captured_event_share(y, al) == pytest.approx(1 / 3)


# ---- lead time -----------------------------------------------------------------------------

def test_lead_time_takes_first_alert_in_the_12_months_before_the_event():
    # loan a: event at month 112 (from t = 100, offset 12); alerted at 97 (too early), 100 and 103
    rows = pl.DataFrame({"loan_identifier": ["a", "a"], "m": [100, 103], "y1": [1, 1],
                         "y1_month": [12, 9]})
    ev = event_months(rows, "y1", "y1_month")
    assert ev["E"].to_list() == [112]
    scored = pl.DataFrame({"loan_identifier": ["a", "b", "a", "b", "a", "b"],
                           "m": [97, 97, 100, 100, 103, 103], "pd": [0.2, 0.001, 0.2, 0.001, 0.2, 0.001]})
    lt = lead_times(ev, scored, threshold=0.01)
    assert lt["lead"].to_list() == [12]
    s = summarise(lt)
    assert s["alert_rate"] == 1.0 and s["median_lead"] == 12 and s["share_ge6"] == 1.0
    missed = lead_times(ev, scored.with_columns(pl.col("pd") / 100), threshold=0.01)
    assert missed["lead"].to_list() == [None] and summarise(missed)["median_lead_missed0"] == 0


# ---- statistical tests ---------------------------------------------------------------------

def test_cluster_bootstrap_auc_point_and_ci():
    _, p, y = calibrated_data(20_000)
    loans = np.arange(20_000) // 4
    res = cluster_bootstrap({"auc": weighted_auc_fn(y, p)}, loans, b=200).set_index("metric")
    assert res.loc["auc", "value"] == pytest.approx(roc_auc_score(y, p))
    assert res.loc["auc", "lo"] < res.loc["auc", "value"] < res.loc["auc", "hi"]


def test_lr_test_and_holm():
    r = lr_test(-100.0, -103.0, 2)
    assert r["lr_stat"] == pytest.approx(6.0) and r["p"] == pytest.approx(np.exp(-3.0))
    assert holm([0.01, 0.02, 0.04]).tolist() == pytest.approx([0.03, 0.04, 0.04])


def test_group_folds_keep_loans_together():
    g = np.repeat(np.arange(1000), 3)
    f = group_folds(g, 5)
    assert set(f) == set(range(5))
    assert all(len(set(f[g == k])) == 1 for k in range(0, 1000, 37))


# ---- disaster rule -------------------------------------------------------------------------

def test_zip3_housing_shares_and_territories(tmp_path):
    # ZIP3 100: ZCTA 10001 has 60 housing units in A and 40 in B, ZCTA 10002 has 300 in B
    f = tmp_path / "rel.txt"
    f.write_text("ZCTA5,STATE,COUNTY,GEOID,HUPT\n10001,01,001,01001,60\n10001,01,002,01002,40\n"
                 "10002,01,002,01002,300\n")
    sh = zip3_county_shares(f)
    got = {r["county"]: r["share"] for r in sh.iter_rows(named=True)}
    assert got == pytest.approx({"01001": 0.15, "01002": 0.85})
    decl = pl.DataFrame({"disaster": [1, 2], "incident_type": ["Flood", "Hurricane"],
                         "m_incident": [100, 100], "county": ["01001", "78010"], "tribal": [False, False]},
                        schema_overrides={"m_incident": pl.Int32})
    hits = affected_zip3(decl, sh).sort("disaster")
    assert hits["zip3"].to_list() == ["100", "008"]          # VI has no ZCTAs: share 1
    assert hits["share"].to_list() == pytest.approx([0.15, 1.0]) and hits["any_overlap"].all()


def test_disaster_window_is_t_minus_3_to_t_plus_12_months():
    hits = pl.DataFrame({"disaster": [1], "incident_type": ["Flood"], "m_incident": [100],
                         "zip3": ["123"], "share": [0.6], "any_overlap": [True]},
                        schema_overrides={"m_incident": pl.Int32})
    f = disaster_flags(hits, list(range(80, 110)))
    months = sorted(f.filter(pl.col("dis_main"))["m"].to_list())
    assert months == list(range(88, 103))        # incident m in [m - 2, m + 12]
    small = disaster_flags(hits.with_columns(pl.lit(0.3).alias("share")), [100])
    assert not small["dis_main"][0] and small["dis_any"][0]


def test_biological_declarations_are_not_applied_in_the_covid_scenario():
    df = pl.DataFrame({"dis_main": [False, False, None], "dis_bio": [True, True, None],
                       "dis_hurricane": [False, False, None], "dis_any": [False, False, None],
                       "stress": ["covid", None, None]})
    assert df.select(exclusion_expr("main"))[:, 0].to_list() == [False, True, False]
    assert df.select(exclusion_expr("all"))[:, 0].to_list() == [False, False, False]


# ---- MLP -----------------------------------------------------------------------------------

def test_mlp_learns_a_simple_signal_on_cpu(monkeypatch):
    import torch

    from xai_ews.models import mlp

    monkeypatch.setattr(mlp, "device", lambda: torch.device("cpu"))
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.normal(size=(n, 3)).astype(np.float32)
    c = rng.integers(0, 4, size=(n, 1))
    y = (rng.random(n) < sigmoid(-2 + 1.5 * x[:, 0] + 0.8 * (c[:, 0] == 3))).astype(int)
    pre = mlp.Preprocessor(n_quantiles=200, subsample=n).fit(x)
    xn = pre.transform(x)
    cfg = mlp.Config(n_layers=2, width=32, lr=3e-3, batch_size=512, emb_dim=4)
    ev = (xn[:5000], c[:5000], y[:5000], np.ones(5000))
    model, hist = mlp.train(xn, c, y, np.ones(n), [4], cfg, eval_data=ev, max_epochs=5)
    z = mlp.predict_logit(model, xn[:5000], c[:5000])
    assert roc_auc_score(y[:5000], z) > 0.75 and hist["best_epoch"] >= 0


# ---- H2 -----------------------------------------------------------------------------------

def test_segment_codes_bands_and_missing():
    from xai_ews.evaluation.heterogeneity import segment_codes
    assert segment_codes([679, 680, 739, 740, np.nan], "fico").tolist() == [0, 1, 1, 2, 0]
    assert segment_codes([80, 80.5, 95, 96, np.nan], "mtm_ltv").tolist() == [0, 1, 1, 2, 0]


def test_auc_within_segments_removes_the_between_band_part():
    from xai_ews.evaluation.heterogeneity import auc_within_segments
    rng = np.random.default_rng(8)
    n = 60_000
    fico = rng.choice([650.0, 700.0, 780.0, np.nan], n)
    band = np.where(np.isnan(fico) | (fico < 680), 0, np.where(fico < 740, 1, 2))
    y = (rng.random(n) < np.array([0.06, 0.03, 0.01])[band]).astype(int)
    score = -band + 1e-9 * rng.normal(size=n)                 # orders the bands only, random inside
    r = auc_within_segments(y, score, fico).set_index("segment")
    assert r["rows"].sum() == n and r.loc["<680", "rows"] == int((band == 0).sum())
    assert roc_auc_score(y, score) > 0.6 and np.allclose(r["auc"], 0.5, atol=0.03)


def test_lr_test_finds_a_planted_interaction_and_not_a_missing_one():
    from xai_ews.evaluation.heterogeneity import lr_interaction_test
    rng = np.random.default_rng(3)
    n = 60_000
    x = rng.normal(size=(n, 2)).astype(np.float32)
    seg = rng.integers(0, 3, n).astype(np.int8)
    slope = np.where(seg == 2, 2.0, 0.5)                     # feature 0 matters more in segment 2
    y = (rng.random(n) < sigmoid(-2 + slope * x[:, 0] + 0.5 * x[:, 1])).astype(int)
    r = lr_interaction_test(x, seg, y, 3)
    assert r["df"] == 4 and r["p"] < 1e-10
    y0 = (rng.random(n) < sigmoid(-2 + 0.8 * x[:, 0] + 0.5 * x[:, 1])).astype(int)
    assert lr_interaction_test(x, seg, y0, 3)["p"] > 0.001


def test_weighted_ks_matches_ks_and_psi_is_zero_for_same_distribution():
    from xai_ews.evaluation.stats import score_psi, weighted_ks_fn
    _, p, y = calibrated_data(20_000)
    w = np.random.default_rng(1).integers(0, 3, len(y)).astype(float)
    assert weighted_ks_fn(y, p)(w) == pytest.approx(ks(y, p, w), abs=1e-12)
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=50_000), rng.normal(size=50_000)
    assert score_psi(a, b) < 0.002 and score_psi(a, b + 1) > 0.5


def test_metrics_table_and_pairwise_run_and_agree_with_point_metrics():
    from xai_ews.evaluation.compare import metrics_table, pairwise
    _, p, y = calibrated_data(20_000)
    rng = np.random.default_rng(5)
    noisy = sigmoid(np.log(p / (1 - p)) + rng.normal(0, 1, len(p)))
    loans = np.arange(len(y)) // 2
    lgd, ead = np.full(len(y), 0.4), rng.uniform(1e5, 3e5, len(y))
    t = metrics_table(y, {"good": p, "noisy": noisy}, loans, lgd, ead, threshold=0.05, b=100)
    assert t.loc["good", "auc"] > t.loc["noisy", "auc"]
    assert t.loc["good", "auc_lo"] < t.loc["good", "auc"] < t.loc["good", "auc_hi"]
    assert t.loc["good", "el_share_alert"] == pytest.approx(captured_el_share(y, alert_mask(p, 0.05), lgd, ead))
    assert t.loc["good", "alert_recall"] == pytest.approx(captured_event_share(y, alert_mask(p, 0.05)))
    pw = pairwise(y, {"good": p, "noisy": noisy}, loans, [("good", "noisy")], b=100)
    assert pw.loc[0, "delta_auc"] > 0 and pw.loc[0, "lo"] > 0 and pw.loc[0, "delong_p"] < 0.05


def test_check_frozen_accepts_matching_hashes_and_rejects_changes(tmp_path, monkeypatch):
    from xai_ews.evaluation import protocol
    f = tmp_path / "model.bin"
    f.write_bytes(b"abc")
    doc = tmp_path / "protocol.md"
    monkeypatch.setattr(protocol, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(protocol, "PROTOCOL", doc)
    table = f"| `model.bin` | `{protocol.sha256(f)}` |"
    doc.write_text(f"**Статус: ЧЕРНОВИК**\n{table}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not frozen"):
        protocol.check_frozen()
    doc.write_text(f"**Статус: ЗАМОРОЖЕН 2026-10-26**\n{table}\n", encoding="utf-8")
    assert "1 files verified" in protocol.check_frozen()
    f.write_bytes(b"abd")
    with pytest.raises(RuntimeError, match="differ"):
        protocol.check_frozen()


def test_paired_lead_time_bootstrap():
    from xai_ews.evaluation.leadtime import paired_bootstrap
    ids = [str(i) for i in range(400)]
    a = pl.DataFrame({"loan_identifier": ids, "lead": [9.0] * 300 + [None] * 100})
    b = pl.DataFrame({"loan_identifier": ids, "lead": [6.0] * 200 + [None] * 200})
    r = paired_bootstrap(a, b, b=200)
    assert r.loc["alert_rate", "delta"] == pytest.approx(0.25) and r.loc["median_lead", "delta"] == 3.0
    assert r.loc["alert_rate", "lo"] > 0


def test_h4_check_detects_better_loss_capture():
    from xai_ews.evaluation.compare import h4_check
    _, p, y = calibrated_data(40_000)
    rng = np.random.default_rng(7)
    noisy = sigmoid(np.log(p / (1 - p)) + rng.normal(0, 1.5, len(p)))
    d = pl.DataFrame({"loan_identifier": np.arange(len(y)) // 2, "m": np.arange(len(y)) % 4, "y": y,
                      "upb": rng.uniform(1e5, 3e5, len(y)), "lgd": np.full(len(y), 0.4), "a": p, "b": noisy})
    r = h4_check(d, "y", "a", "b", b=100)
    assert r["delta"] > 0 and r["el_part"] and r["calibration_part"] is False


def test_newton_calibration_matches_statsmodels():
    import statsmodels.api as sm
    rng = np.random.default_rng(11)
    z = rng.normal(-4, 1.2, 50_000)
    y = (rng.random(len(z)) < sigmoid(0.3 + 0.8 * z)).astype(float)
    slope, intercept = calibration_slope_intercept(y, sigmoid(z))
    ref_s = sm.GLM(y, sm.add_constant(z), family=sm.families.Binomial()).fit().params[1]
    ref_i = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=z).fit().params[0]
    assert slope == pytest.approx(ref_s, abs=1e-8) and intercept == pytest.approx(ref_i, abs=1e-8)


_GBDT_ROUNDTRIP = """
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from xai_ews.evaluation.metrics import sigmoid
from xai_ews.models import gbdt
lib, path = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(4)
n = 6000
X = pd.DataFrame({"x": rng.normal(size=n), "c": rng.integers(0, 3, n)})
y = (rng.random(n) < sigmoid(-2 + 1.5 * X["x"] + 0.7 * (X["c"] == 2))).astype(int)
X = gbdt.as_frame(X, ["c"], {"c": 4})
p = {"depth": 3, "learning_rate": 0.1, "l2": 1.0, "min_leaf": 20, "subsample": 0.9}
m = gbdt.fit(lib, X[:4000], y[:4000], np.ones(4000), p, eval_set=(X[4000:], y[4000:], np.ones(2000)))
z = m.logit(X[4000:])
assert roc_auc_score(y[4000:], z) > 0.75 and m.n_trees >= 1
m.save(path)
z2 = gbdt.Model.load(lib, path, m.n_trees).logit(X[4000:])
assert np.allclose(z, z2, atol=1e-5)
"""


@pytest.mark.parametrize("lib", ["xgboost", "lightgbm"])
def test_gbdt_fit_save_load_roundtrip(lib, tmp_path):
    # separate process: XGBoost/LightGBM and PyTorch (loaded by the MLP tests) segfault in one process on macOS (OpenMP)
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-c", _GBDT_ROUNDTRIP, lib, str(tmp_path / "m.model")],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr[-2000:]
