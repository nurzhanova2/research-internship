"""Stage-2 helpers: WoE / IV, univariate statistics, PSI, redundancy, DeLong, bootstrap, Boruta."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from xai_ews.config import feature_config
from xai_ews.data.dictionary import feature_dictionary
from xai_ews.data.features import candidate_features
from xai_ews.features.data import CategoryEncoder, subsample
from xai_ews.features.evaluation import cluster_bootstrap_auc, delong_test, one_snapshot_per_loan
from xai_ews.features.rationale import RATIONALE, rationale_table
from xai_ews.features.redundancy import (
    correlated_pairs,
    greedy_decorrelate,
    mrmr,
    mutual_info,
    vif_from_corr,
    vif_iterative,
    weighted_corr,
)
from xai_ews.features.stability import macro_stable, psi_binned, sign_check
from xai_ews.features.univariate import (
    bh_adjust,
    chi2_test,
    cramers_v,
    direction,
    ks_statistic,
    weighted_auc,
)
from xai_ews.features.woe import WoEBinner, bin_table, fit_woe, woe_iv

CFG = feature_config()["binning"]


def logistic_data(n=20000, seed=0, beta=1.5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-3 + beta * x)))
    return x, (rng.random(n) < p).astype(int)


# ---- rationale -----------------------------------------------------------------------------

def test_rationale_covers_every_candidate_and_signs_are_valid():
    assert set(RATIONALE) == set(candidate_features())
    t = rationale_table()
    assert len(t) == 59 and t["static_h3"].sum() == 28
    assert set(t["expected_sign"]) <= {"+", "-", "cat", "none"}
    cats = t[t["expected_sign"] == "cat"]
    assert cats["risky_level"].notna().all() and cats["safe_level"].notna().all()


def test_data_dictionary_covers_every_candidate_and_agrees_with_model_input():
    d = feature_dictionary()
    assert list(d["feature"]) == candidate_features()
    assert set(d["kind"]) <= {"categorical", "binary", "integer", "continuous"}
    assert d["unit_en"].str.len().gt(0).all() and d["unit_ru"].str.len().gt(0).all()
    # names and codes go to CatBoost as categories; numbers never do
    assert (d.loc[d["kind"] == "categorical", "model_input"] == "category").all()
    assert (d.loc[d["kind"].isin(["integer", "continuous"]), "model_input"] == "number").all()
    sel = feature_config()["selected"]["features"]
    assert list(feature_dictionary(sel)["feature"]) == sel


# ---- WoE / IV ------------------------------------------------------------------------------

def test_woe_iv_formula_matches_hand_computation():
    good, bad = np.array([60.0, 40.0]), np.array([10.0, 30.0])
    woe, iv = woe_iv(good, bad, smoothing=0.0)
    g, b = good / 100, bad / 40
    assert np.allclose(woe, np.log(g / b))
    assert iv == pytest.approx(np.sum((g - b) * np.log(g / b)))


def test_numeric_binning_recovers_monotone_risk_and_respects_limits():
    x, y = logistic_data()
    b = fit_woe(x, y, np.ones_like(x), "x", False, CFG)
    assert 2 <= b.n_bins <= CFG["max_bins"]
    assert np.all(np.diff(b.woe) < 0)                      # risk rises with x -> WoE falls
    assert b.iv > 0.3
    t = bin_table(b, x, y, np.ones_like(x))
    assert (t["share"] >= CFG["min_bin_share"] - 1e-9).all()
    assert (t["events"] >= CFG["min_bin_events"]).all()
    assert t["iv_part"].sum() == pytest.approx(b.iv, rel=0.02)


def test_noise_feature_has_tiny_iv_and_missing_gets_own_bin():
    rng = np.random.default_rng(1)
    x = rng.normal(size=20000)
    y = (rng.random(20000) < 0.05).astype(int)
    x[:2000] = np.nan
    b = fit_woe(x, y, np.ones_like(x), "noise", False, CFG)
    assert b.iv < 0.02
    assert b.labels[b.missing_bin] == "missing"
    assert b.transform(np.array([np.nan]))[0] == pytest.approx(b.woe[b.missing_bin])


def test_weights_reconstruct_population_woe():
    """Subsampling negatives at r with weight 1/r gives the full-population bins' WoE."""
    x, y = logistic_data(n=200000, seed=2)
    full = fit_woe(x, y, np.ones_like(x), "x", False, CFG)
    rng = np.random.default_rng(3)
    keep = (y == 1) | (rng.random(len(y)) < 0.2)
    w = np.where(y[keep] == 1, 1.0, 5.0)
    sub = fit_woe(x[keep], y[keep], w, "x", False, CFG)
    assert sub.iv == pytest.approx(full.iv, rel=0.1)


def test_categorical_binning_groups_and_handles_unseen():
    rng = np.random.default_rng(4)
    cats = np.array(["A", "B", "C", "D"])
    x = cats[rng.integers(0, 4, 40000)]
    rate = {"A": 0.01, "B": 0.01, "C": 0.08, "D": 0.08}
    y = (rng.random(40000) < np.vectorize(rate.get)(x)).astype(int)
    x[:20] = "Z"                                          # rare -> pooled
    b = fit_woe(x, y, np.ones(len(x)), "c", True, CFG)
    assert b.bin_index(np.array(["A"]))[0] != b.bin_index(np.array(["C"]))[0]
    assert b.transform(np.array(["never_seen"]))[0] == pytest.approx(b.woe[b.default_bin])
    restored = WoEBinner.from_dict(b.to_dict())
    assert np.allclose(restored.transform(x), b.transform(x))


# ---- univariate statistics -----------------------------------------------------------------

def test_weighted_auc_and_ks_match_reference_implementations():
    x, y = logistic_data(seed=5)
    xr = np.round(x, 1)                                   # ties
    w = np.random.default_rng(6).uniform(0.5, 2, len(x))
    assert weighted_auc(y, xr, w) == pytest.approx(roc_auc_score(y, xr, sample_weight=w))
    assert ks_statistic(y, x) == pytest.approx(stats.ks_2samp(x[y == 1], x[y == 0]).statistic)
    assert direction(y, x) > 0.2 and direction(y, -x) < -0.2


def test_chi2_cramers_v_and_bh():
    y = np.array([0] * 50 + [1] * 50)
    assert cramers_v(y, y) == pytest.approx(1.0)
    rng = np.random.default_rng(7)
    a = rng.integers(0, 3, 30000)
    _stat, p, dof = chi2_test(a, rng.integers(0, 2, 30000))
    assert dof == 2 and p > 0.001
    pv = np.array([0.01, 0.04, 0.03, 0.2, np.nan])
    assert np.allclose(bh_adjust(pv)[:4], multipletests(pv[:4], method="fdr_bh")[1])


# ---- stability -----------------------------------------------------------------------------

def test_psi_zero_for_same_distribution_and_large_for_shift():
    x, y = logistic_data(seed=8)
    b = fit_woe(x, y, np.ones_like(x), "x", False, CFG)
    rng = np.random.default_rng(9)
    assert psi_binned(b, x, None, rng.normal(size=20000)) < 0.01
    assert psi_binned(b, x, None, rng.normal(1.5, 1, 20000)) > 0.25


def test_macro_rule_and_sign_check():
    ok = pd.DataFrame({"feature": "u", "iv": [0.05, 0.04, 0.03], "direction": [0.1, 0.08, 0.05]})
    flip = ok.assign(direction=[0.1, -0.05, 0.05])
    assert macro_stable(ok, 0.02, 0.01)["stable"]
    assert not macro_stable(flip, 0.02, 0.01)["stable"]
    assert not macro_stable(ok.assign(iv=[0.05, 0.01, 0.03]), 0.02, 0.01)["stable"]
    assert sign_check("-", -0.2, 0.01) == "match"
    assert sign_check("+", -0.2, 0.01) == "mismatch"
    assert sign_check("+", 0.004, 0.01) == "weak"
    assert sign_check("none", 0.3, 0.01) == "no prior"


# ---- redundancy ----------------------------------------------------------------------------

def test_corr_vif_and_greedy_decorrelation():
    rng = np.random.default_rng(10)
    a = rng.normal(size=5000)
    X = np.c_[a, a + 0.1 * rng.normal(size=5000), rng.normal(size=5000)]
    c = weighted_corr(X)
    assert np.allclose(c, np.corrcoef(X.T))
    v = vif_from_corr(c)
    assert v[0] > 10 and v[2] == pytest.approx(1.0, abs=0.05)
    corr = pd.DataFrame(c, index=list("abc"), columns=list("abc"))
    strength = pd.Series({"a": 0.3, "b": 0.5, "c": 0.1})
    kept, log = greedy_decorrelate(strength, [corr], 0.7)
    assert kept == ["b", "c"] and log[0]["feature"] == "a"
    kept, _ = greedy_decorrelate(strength, [corr], 0.7, prefer={"b": "a"})
    assert "a" in kept and "b" not in kept
    kept, vlog = vif_iterative(corr, 10, strength)
    assert vlog[0]["feature"] == "a"
    assert list(correlated_pairs(corr, 0.7)[["a", "b"]].iloc[0]) == ["a", "b"]


def test_mutual_information_and_mrmr_prefers_non_redundant():
    rng = np.random.default_rng(11)
    n = 20000
    y = rng.integers(0, 2, n)
    f1 = np.where(rng.random(n) < 0.8, y, 1 - y)
    f2 = f1.copy()                                       # exact copy of f1
    f3 = np.where(rng.random(n) < 0.7, y, 1 - y)         # weaker, independent noise
    assert mutual_info(f1, f1) == pytest.approx(stats.entropy(np.bincount(f1) / n))
    assert mutual_info(f1, rng.integers(0, 2, n)) < 1e-3
    codes = np.c_[f1, f2, f3]
    names = ["f1", "f2", "f3"]
    rel = pd.Series([mutual_info(c, y) for c in codes.T], index=names)
    pairs = pd.DataFrame([[mutual_info(a, b) for b in codes.T] for a in codes.T], index=names, columns=names)
    order = mrmr(rel, pairs)["feature"].tolist()
    assert order[0] in {"f1", "f2"} and order[1] == "f3"


# ---- model comparison ----------------------------------------------------------------------

def test_one_snapshot_per_loan():
    ids = np.array(["a", "a", "b", "c", "c", "c"])
    idx = one_snapshot_per_loan(ids, seed=1)
    assert sorted(ids[idx]) == ["a", "b", "c"]


def test_delong_matches_auc_and_detects_difference():
    x, y = logistic_data(n=5000, seed=12)
    rng = np.random.default_rng(13)
    good, bad = x + 0.3 * rng.normal(size=5000), x + 3 * rng.normal(size=5000)
    r = delong_test(y, good, bad)
    assert r["auc_1"] == pytest.approx(roc_auc_score(y, good))
    assert r["auc_2"] == pytest.approx(roc_auc_score(y, bad))
    assert r["p"] < 0.001 and r["delta"] > 0
    assert delong_test(y, good, good)["p"] == pytest.approx(1.0)


def test_delong_variance_matches_bootstrap():
    x, y = logistic_data(n=3000, seed=14)
    rng = np.random.default_rng(15)
    s1, s2 = x + rng.normal(size=3000), x + 1.5 * rng.normal(size=3000)
    se = delong_test(y, s1, s2)["se"]
    d = []
    for _ in range(400):
        i = rng.integers(0, 3000, 3000)
        d.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    assert se == pytest.approx(np.std(d), rel=0.2)


def test_cluster_bootstrap_auc_point_and_ci():
    x, y = logistic_data(n=4000, seed=16)
    loans = np.repeat(np.arange(1000), 4)
    rng = np.random.default_rng(17)
    s1, s2 = x + 0.5 * rng.normal(size=4000), x + 2 * rng.normal(size=4000)
    t = cluster_bootstrap_auc(y, {"a": s1, "b": s2}, loans, b=200, reference="b").set_index("model")
    assert t.loc["a", "auc"] == pytest.approx(roc_auc_score(y, s1))
    assert t.loc["a", "auc_lo"] < t.loc["a", "auc"] < t.loc["a", "auc_hi"]
    assert t.loc["a", "delta_lo"] > 0


# ---- data helpers --------------------------------------------------------------------------

def test_category_encoder_top_n_and_unseen():
    df = pl.DataFrame({"s": ["x", "x", "x", "y", "y", "z"], "sample_weight": [1.0] * 6})
    enc = CategoryEncoder(top_n={"s": 2}).fit(df, ["s"])
    out = enc.transform(pl.DataFrame({"s": ["x", "y", "z", "new"]}))["s"].to_list()
    assert out[0] == 0 and out[1] == 1 and out[2] == out[3] == 2
    assert enc.group(df)["s"].to_list()[-1] == CategoryEncoder.OTHER


def test_subsample_keeps_positives_and_population_weight():
    df = pl.DataFrame({"y": [1] * 10 + [0] * 990, "sample_weight": [1.0] * 10 + [5.0] * 990})
    s = subsample(df, "y", 200, seed=0)
    assert s.height == 200 and s["y"].sum() == 10
    assert s["sample_weight"].sum() == pytest.approx(df["sample_weight"].sum())


def test_boruta_confirms_signal_and_rejects_noise():
    from xai_ews.features.importance import boruta
    rng = np.random.default_rng(18)
    n = 4000
    X = pd.DataFrame({"sig": rng.normal(size=n), "noise": rng.normal(size=n)})
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1 + 2 * X["sig"])))).astype(int)
    res = boruta(X, y, None, [], {"iterations": 60, "depth": 3, "learning_rate": 0.2}, n_iter=12,
                 alpha=0.05).set_index("feature")
    assert res.loc["sig", "boruta_decision"] == "confirmed"
    assert res.loc["noise", "boruta_decision"] != "confirmed"


# ---- selection output ----------------------------------------------------------------------

def test_write_features_yaml_keeps_rules_and_replaces_generated_part(tmp_path):
    import yaml

    from xai_ews.features.selection import MARKER, write_features_yaml
    p = tmp_path / "features.yaml"
    p.write_text(f"rules:\n  iv_min: 0.02   # comment kept\n\n{MARKER} ---\nold: 1\n", encoding="utf-8")
    write_features_yaml({"selected": ["a", "b"]}, p)
    write_features_yaml({"selected": ["c"]}, p)
    text = p.read_text(encoding="utf-8")
    assert "# comment kept" in text and text.count(MARKER) == 1
    assert yaml.safe_load(text) == {"rules": {"iv_min": 0.02}, "selected": ["c"]}


def test_selection_table_reports_first_failed_step():
    from xai_ews.features.selection import selection_table
    rat = pd.DataFrame({"feature": ["a", "b", "c"], "group": "loan", "static_h3": True, "expected_sign": "+",
                        "risky_level": None, "safe_level": None})
    uni = pd.DataFrame([{"feature": f, "task": t, "iv_train": iv, "iv_val": iv, "ks_train": 0.1, "ks_val": 0.1,
                         "direction_train": 0.1, "direction_val": 0.1}
                        for f, iv in [("a", 0.3), ("b", 0.01), ("c", 0.2)] for t in ["y1", "y2"]])
    stab = pd.DataFrame({"feature": ["a", "b", "c"], "psi_max": [0.01, 0.01, 0.4], "psi_max_with_hurricanes": 0.0,
                         "stability_rule": "psi", "pass_stability": [True, True, False],
                         "stability_reason": ["", "", "PSI 0.40 > 0.25"], "pass_sign": True, "sign_status": ""})
    red = pd.DataFrame({"feature": ["a"], "max_corr_partner": [None], "max_abs_corr": [0.1], "pass_redundancy": [True],
                        "redundancy_reason": [""], "static_h3_set": [True]})
    imp = pd.DataFrame({"feature": ["a"], "pass_importance": [True]})
    t = selection_table(rat, uni, stab, red, imp, pd.DataFrame({"y1": [1.2]}, index=["a"])).set_index("feature")
    assert t.loc["a", "decision"] == "keep"
    assert t.loc["b", "dropped_at_step"] == "2 univariate"
    assert t.loc["c", "reason"] == "PSI 0.40 > 0.25"
    t9 = selection_table(rat, uni, stab, red, imp, pd.DataFrame({"y1": [1.2]}, index=["a"]),
                         ablation_drop={"a": "group harms Y2"}).set_index("feature")
    assert t9.loc["a", "dropped_at_step"] == "9 ablation" and t9.loc["a", "reason"] == "group harms Y2"


def test_compare_models_reference_has_zero_delta_columns():
    from xai_ews.features.evaluation import compare_models
    x, y = logistic_data(n=3000, seed=19)
    rng = np.random.default_rng(20)
    base = pl.DataFrame({"loan_identifier": np.repeat(np.arange(1000), 3).astype(str), "m": np.tile([0, 1, 2], 1000),
                         "y1": y, "val_hurricane_2017": rng.random(3000) < 0.05})
    preds = {"ref": base.with_columns(pl.Series("logit", x + rng.normal(size=3000))),
             "noisy": base.with_columns(pl.Series("logit", x + 3 * rng.normal(size=3000)))}
    t = compare_models(preds, "y1", "ref", b=100).set_index("model")
    assert np.isnan(t.loc["ref", "delong_p"]) and t.loc["noisy", "delta_auc"] < 0
    loans_main = base.filter(~pl.col("val_hurricane_2017"))["loan_identifier"].n_unique()
    assert t.attrs["counts"]["one_snapshot_rows"] == loans_main


def test_rare_significant_bin_stays_separate_and_rare_plain_bin_merges():
    """dpd-like feature: 0.3 % of rows with a 20 % event rate vs 0.1 % elsewhere."""
    rng = np.random.default_rng(21)
    n = 200000
    x = np.zeros(n)
    x[:600] = 1                                           # rare, risky
    x[600:1200] = 2                                       # rare, ordinary
    x[1200:] = rng.integers(3, 6, n - 1200)
    p = np.where(x == 1, 0.2, 0.01)
    y = (rng.random(n) < p).astype(int)
    cfg = {**CFG, "min_bin_share": 0.005, "rare_bin_rate_ratio": 2.0}
    b = fit_woe(x, y, np.ones(n), "dpd_like", False, cfg)
    i1, i2, i3 = b.bin_index(np.array([1.0, 2.0, 3.0]))
    assert i1 not in (i2, i3)
    off = fit_woe(x, y, np.ones(n), "dpd_like", False, {**cfg, "rare_bin_rate_ratio": None})
    assert off.bin_index(np.array([1.0]))[0] == off.bin_index(np.array([2.0]))[0]


def test_rare_risky_missing_values_get_own_bin():
    rng = np.random.default_rng(22)
    n = 200000
    x = rng.normal(size=n)
    x[:400] = np.nan                                      # 0.2 % missing, 25 % event rate
    y = (rng.random(n) < np.where(np.isnan(x), 0.25, 0.01)).astype(int)
    cfg = {**CFG, "min_bin_share": 0.005, "rare_bin_rate_ratio": 2.0}
    b = fit_woe(x, y, np.ones(n), "m", False, cfg)
    assert b.labels[b.missing_bin] == "missing"
    off = fit_woe(x, y, np.ones(n), "m", False, {**cfg, "rare_bin_rate_ratio": None})
    assert off.labels[off.missing_bin] != "missing"


def test_select_per_task_keeps_feature_needed_by_one_task_only():
    """a and b are duplicates in task 1; in task 2 only b is informative and not redundant."""
    from xai_ews.features.redundancy import select_per_task
    idx = ["a", "b", "c"]
    c1 = pd.DataFrame([[1, .9, 0], [.9, 1, 0], [0, 0, 1]], index=idx, columns=idx, dtype=float)
    c2 = pd.DataFrame([[1, .9, 0], [.9, 1, .1], [0, .1, 1]], index=idx, columns=idx, dtype=float)
    iv = pd.DataFrame({"t1": [0.5, 0.3, 0.1], "t2": [0.0, 0.2, 0.1]}, index=idx)
    union, kept, log = select_per_task({"t1": c1, "t2": c2}, iv, idx, 0.02, 0.7, 10)
    assert kept["t1"] == ["a", "c"] and kept["t2"] == ["b", "c"]
    assert union == ["a", "b", "c"]
    assert (log["feature"] == "b").sum() == 1


def test_select_per_task_removes_pair_redundant_in_every_task():
    """Different winners per task (a in t1, b in t2) but a ~ b in both: only the stronger stays."""
    from xai_ews.features.redundancy import select_per_task
    idx = ["a", "b"]
    c = pd.DataFrame([[1, .9], [.9, 1]], index=idx, columns=idx, dtype=float)
    iv = pd.DataFrame({"t1": [0.30, 0.20], "t2": [0.10, 0.25]}, index=idx)
    union, _kept, log = select_per_task({"t1": c, "t2": c}, iv, idx, 0.02, 0.7, 10)
    assert union == ["a"] and (log["rule"] == "union").any()



def test_confusion_at_threshold_counts_and_f1():
    from xai_ews.features.evaluation import confusion_at_threshold
    logit = np.log(np.array([0.9, 0.6, 0.4, 0.2, 0.7, 0.1]) / (1 - np.array([0.9, 0.6, 0.4, 0.2, 0.7, 0.1])))
    p = pl.DataFrame({"logit": logit, "y": [1, 1, 1, 0, 0, 0]})
    r = confusion_at_threshold(p, "y")                     # alerts: 0.9, 0.6, 0.7
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 2)
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3)
    assert r["accuracy"] == pytest.approx(4 / 6)
    assert r["balanced_accuracy"] == pytest.approx((2 / 3 + 2 / 3) / 2)
