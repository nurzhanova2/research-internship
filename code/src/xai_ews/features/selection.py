"""Final feature-selection table and configs/features.yaml output (stage 2).

The decision of every candidate is the FIRST step it fails, in the order the notebooks apply
them: 2 univariate IV -> 6 stability -> 7 sign -> 4-5 redundancy -> 8 model importance -> 9 ablation
(a whole group whose removal significantly raises validation AUC).
"""

import pandas as pd
import yaml

from xai_ews.paths import CONFIGS_DIR

MARKER = "# --- written by 02_6_ablation_and_final.ipynb"


def _wide(uni: pd.DataFrame, col: str) -> pd.DataFrame:
    w = uni.pivot(index="feature", columns="task", values=col)
    return w.rename(columns={t: f"{col}_{t}" for t in w.columns})


def selection_table(rat: pd.DataFrame, uni: pd.DataFrame, stab: pd.DataFrame, red: pd.DataFrame,
                    imp: pd.DataFrame, vif_final: pd.DataFrame,
                    ablation_drop: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per candidate: expected / observed sign, IV, KS, PSI, VIF, importance, decision, reason."""
    t = rat[["feature", "group", "static_h3", "expected_sign", "risky_level", "safe_level"]].set_index("feature")
    for col in ["iv_train", "iv_val", "ks_train", "ks_val", "direction_train", "direction_val"]:
        t = t.join(_wide(uni, col))
    t = t.join(stab.set_index("feature")[["psi_max", "psi_max_with_hurricanes", "stability_rule",
                                          "pass_stability", "stability_reason", "pass_sign", "sign_status"]])
    t = t.join(red.set_index("feature")[["max_corr_partner", "max_abs_corr", "pass_redundancy",
                                         "redundancy_reason", "static_h3_set"]])
    t = t.join(imp.set_index("feature")[[c for c in imp.columns if c.startswith(("shap_share_y", "perm_auc_drop_y", "boruta_decision"))]
                                        + ["pass_importance"]])
    t = t.join(vif_final.add_prefix("vif_final_"))
    iv_cols = [c for c in t.columns if c.startswith("iv_train_")]
    t["pass_iv"] = t[iv_cols].max(axis=1) >= 0.02
    iv_text = lambda r: ", ".join(f"{c.removeprefix('iv_train_').upper()} {r[c]:.3f}" for c in iv_cols)
    steps = [("2 univariate", "pass_iv", lambda r: f"IV < 0.02 ({iv_text(r)})"),
             ("6 stability", "pass_stability", lambda r: r.stability_reason),
             ("7 sign", "pass_sign", lambda r: r.sign_status),
             ("4-5 redundancy", "pass_redundancy", lambda r: r.redundancy_reason),
             ("8 importance", "pass_importance", lambda r: "Boruta rejected and permutation <= 0")]
    decision, step, reason = [], [], []
    for _, r in t.iterrows():
        for name, col, why in steps:
            if pd.isna(r[col]) or not bool(r[col]):
                decision.append("drop"); step.append(name); reason.append(why(r))
                break
        else:
            if ablation_drop and r.name in ablation_drop:
                decision.append("drop"); step.append("9 ablation"); reason.append(ablation_drop[r.name])
                continue
            decision.append("keep"); step.append("")
            reason.append(r["sign_status"] if isinstance(r["sign_status"], str) and r["sign_status"] else "")
    t["decision"], t["dropped_at_step"], t["reason"] = decision, step, reason
    return t.reset_index()


def write_features_yaml(generated: dict, path=None) -> None:
    """Replace everything after MARKER in configs/features.yaml with `generated`; rules stay intact."""
    path = CONFIGS_DIR / "features.yaml" if path is None else path
    text = path.read_text(encoding="utf-8")
    head = text.split(MARKER)[0].rstrip() + "\n\n"
    body = yaml.safe_dump(generated, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(head + MARKER + " " + "-" * 20 + "\n" + body, encoding="utf-8")
