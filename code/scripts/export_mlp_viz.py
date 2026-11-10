"""Export the trained M2 MLP (seed 0) for the 3D page reports/modeling/mlp_3d/index.html:

    uv run python scripts/export_mlp_viz.py [n_loans]

Per task (Y1, Y2): input nodes (numeric features, missing-value indicators, embedding
dimensions of the categorical features), hidden layer sizes, the K strongest incoming weights of
every neuron, the weight norm of every input, and the activations of n_loans validation loans
(half with an event, half without, spread evenly over the PD range) with their feature values and
calibrated PD. Labels are in Russian and English. Output: reports/modeling/mlp_3d/data.js
(window.M2_DATA = {...}), so the page opens from disk without a server.
"""

import json
import pickle
import sys

import numpy as np
import polars as pl
import torch

from xai_ews.data.panel import ym_to_index
from xai_ews.models import mlp, registry
from xai_ews.models.data import arrays, load_eval
from xai_ews.paths import MODELING_DIR, MODELS_DIR

K_IN = 4           # strongest incoming connections drawn per hidden neuron
K_OUT = 24         # strongest connections drawn into the output neuron
DATE = "2016-09"   # validation date the loans come from
OUT = MODELING_DIR / "mlp_3d" / "data.js"

LABELS = {  # feature: (ru, en)
    "borrower_credit_score_at_origination": ("FICO заёмщика", "Borrower FICO"),
    "co_borrower_credit_score_at_origination": ("FICO созаёмщика", "Co-borrower FICO"),
    "debt_to_income_dti": ("DTI, %", "DTI, %"),
    "first_time_home_buyer_indicator": ("Первый покупатель", "First-time buyer"),
    "original_interest_rate": ("Ставка при выдаче, %", "Rate at origination, %"),
    "original_upb": ("Сумма при выдаче, $", "Original balance, $"),
    "loan_purpose": ("Цель кредита", "Loan purpose"),
    "seller_name": ("Продавец", "Seller"),
    "property_type": ("Тип объекта", "Property type"),
    "occupancy_status": ("Использование жилья", "Occupancy"),
    "mortgage_insurance_percentage": ("Страхование, %", "Mortgage insurance, %"),
    "remaining_months_to_maturity": ("Остаток срока, мес.", "Remaining term, mo."),
    "dpd": ("Просрочка сейчас, мес.", "Current delinquency, mo."),
    "upb_vs_schedule": ("Долг против графика", "Balance vs schedule"),
    "ever_modified": ("Был модифицирован", "Ever modified"),
    "n_curtail_6m": ("Досрочные доплаты, 6 мес.", "Curtailments, 6 mo."),
    "upb_chg_6m": ("Изменение долга, 6 мес.", "Balance change, 6 mo."),
    "upb_chg_12m": ("Изменение долга, 12 мес.", "Balance change, 12 mo."),
    "n_60p_12m": ("Месяцев 60+, 12 мес.", "Months 60+ DPD, 12 mo."),
    "n_upb_increase_12m": ("Рост долга, 12 мес.", "Balance increases, 12 mo."),
    "excess_principal_12m": ("Сверхплановое погашение, 12 мес.", "Excess principal, 12 mo."),
    "months_since_30p": ("Месяцев с просрочки 30+", "Months since 30+ DPD"),
    "mtm_ltv": ("Текущий LTV, %", "Current LTV, %"),
    "hpi_chg_12m_rel_us": ("Цены жилья за год к США", "House prices 1y vs US"),
    "hpi_chg_since_orig_rel_us": ("Цены жилья с выдачи к США", "House prices since orig. vs US"),
}
CAT_VALUES = {  # code: (ru, en)
    "loan_purpose": {"P": ("покупка", "purchase"), "C": ("рефинансирование с выводом", "cash-out refinance"),
                     "R": ("рефинансирование", "refinance"), "U": ("рефинансирование", "refinance")},
    "occupancy_status": {"P": ("своё жильё", "primary residence"), "S": ("второе жильё", "second home"),
                         "I": ("инвестиционное", "investment")},
    "property_type": {"SF": ("дом на одну семью", "single-family home"), "PU": ("дом в посёлке", "planned unit"),
                      "CO": ("квартира в кондоминиуме", "condominium"), "CP": ("кооператив", "co-op"),
                      "MH": ("сборный дом", "manufactured home")},
    "first_time_home_buyer_indicator": {"Y": ("да", "yes"), "N": ("нет", "no")},
}
MISSING = {"months_since_30p": ("не было", "never"),
           "co_borrower_credit_score_at_origination": ("нет созаёмщика", "no co-borrower")}
SIGNED = ("hpi_chg_12m_rel_us", "hpi_chg_since_orig_rel_us", "upb_chg_6m", "upb_chg_12m", "upb_vs_schedule")
BANDS = [(0.05, ("высокий риск", "high risk")), (0.01, ("повышенный риск", "elevated risk")),
         (0.001, ("средний риск", "moderate risk")), (0.0, ("низкий риск", "low risk"))]


def fmt(f: str, v, lang: int) -> str:
    """Display value of feature f; lang 0 = ru (decimal comma, thin-space thousands), 1 = en."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return MISSING.get(f, ("нет данных", "n/a"))[lang]
    if f in CAT_VALUES:
        return CAT_VALUES[f].get(str(v), (str(v), str(v)))[lang]
    if f == "seller_name":
        return str(v).title()[:40]
    if f == "original_upb":
        s = f"{v:,.0f}"
        return s.replace(",", " ") if lang == 0 else s
    s = f"{v:+.3f}" if f in SIGNED else (f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}")
    return s.replace(".", ",") if lang == 0 else s


def export(task: str, n_loans: int) -> dict:
    name = f"m2_{task}"
    meta = registry.meta(name)
    enc, pre = pickle.loads((MODELS_DIR / name / "encoder.pkl").read_bytes())
    cfg = mlp.Config(**meta["params"])
    model = mlp.MLP(meta["n_num"], meta["cardinalities"], cfg)
    model.load_state_dict(torch.load(MODELS_DIR / name / "seed_0.pt", map_location="cpu"))
    model.eval()

    feats, cats = meta["features"], meta["categorical"]
    num = [f for f in feats if f not in cats]
    inputs = [{"kind": "num", "ru": LABELS[f][0], "en": LABELS[f][1]} for f in num]
    inputs += [{"kind": "missing", "ru": "пропуск: " + LABELS[num[i]][0], "en": "missing: " + LABELS[num[i]][1]}
               for i in pre.missing_cols]
    for c in cats:
        inputs += [{"kind": "emb", "ru": f"{LABELS[c][0]} · вектор {k + 1}", "en": f"{LABELS[c][1]} · dim {k + 1}"}
                   for k in range(cfg.emb_dim)]

    linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    w1 = linears[0].weight.detach().numpy()
    in_norm = np.linalg.norm(w1, axis=0)
    layers = []
    for li, lin in enumerate(linears):
        w = lin.weight.detach().numpy()
        k = K_OUT if li == len(linears) - 1 else K_IN
        top = np.argsort(-np.abs(w), axis=1)[:, :k]
        layers.append({"size": int(w.shape[0]),
                       "edges": [[int(j), round(float(w[i, j]), 4)] for i in range(w.shape[0]) for j in top[i]],
                       "k": k, "w_scale": round(float(np.percentile(np.abs(w), 99.5)), 4)})

    df = load_eval("validation", task, feats, cats, where=pl.col("m") == ym_to_index(DATE))
    df = df.filter(pl.col("half") == "A")
    xn, xc = arrays(df, feats, cats, enc)
    xn = pre.transform(xn)
    with torch.no_grad():
        h = torch.cat([torch.as_tensor(xn), *[e(torch.as_tensor(xc[:, i])) for i, e in enumerate(model.emb)]], dim=1)
        acts = [h.numpy().copy()]
        for m in model.net:
            h = m(h)
            if isinstance(m, torch.nn.ReLU):
                acts.append(h.numpy().copy())
        logit = h.squeeze(1).numpy()
    pd_ = registry.calibrator(name).predict(logit)
    y = df[task].to_numpy()

    def spread(mask, n):
        # n rows spread evenly over the PD ranks of the rows in mask
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(pd_[idx])]
        return [int(idx[int((k + 0.5) / n * len(idx))]) for k in range(n)]

    rows = spread(y == 1, n_loans // 2) + spread(y == 0, n_loans - n_loans // 2)
    rows = sorted(set(rows), key=lambda r: -pd_[r])
    scale = [float(np.percentile(np.abs(a), 99)) or 1.0 for a in acts]
    examples = []
    for n, r in enumerate(rows, 1):
        row = df.row(r, named=True)
        band = next(b for lo, b in BANDS if pd_[r] >= lo)
        examples.append({
            "title": {"ru": f"№ {n} · {band[0]}", "en": f"#{n} · {band[1]}"}, "event": bool(y[r] == 1),
            "pd": round(float(pd_[r]), 6), "logit": round(float(logit[r]), 3),
            "features": [{"ru": [LABELS[f][0], fmt(f, row[f], 0)], "en": [LABELS[f][1], fmt(f, row[f], 1)]} for f in feats],
            "acts": [[round(float(v) / scale[li], 2) for v in a[r]] for li, a in enumerate(acts)],
        })
    return {"task": task, "inputs": inputs, "in_norm": [round(float(v), 4) for v in in_norm / in_norm.max()],
            "layers": layers, "examples": examples, "params": int(sum(p.numel() for p in model.parameters())),
            "config": {"n_layers": cfg.n_layers, "width": cfg.width, "dropout": round(cfg.dropout, 2),
                       "lr": cfg.lr, "batch": cfg.batch_size, "emb_dim": cfg.emb_dim},
            "n_rows": int(df.height), "n_events": int(y.sum()), "date": DATE}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    data = {t: export(t, n) for t in ("y1", "y2")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.M2_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    for t, d in data.items():
        ev = sum(e["event"] for e in d["examples"])
        print(t, "loans", len(d["examples"]), "with event", ev, "| PD range",
              f"{d['examples'][-1]['pd']:.5f}..{d['examples'][0]['pd']:.4f}", "| params", d["params"])
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
