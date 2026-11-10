"""Рисунок сравнения моделей: AUC на test 2018 с 95 % интервалом и AUC в стресс-периодах.

Запуск из папки недели 7:  python code/fig_models.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WEEK = Path(__file__).resolve().parents[1]
test = pd.read_csv(WEEK / "tables/test/comparison_table.csv").set_index("model")
stress = pd.read_csv(WEEK / "tables/stress/stress_metrics.csv")
stress = stress[stress["variant"] == "all"]

names = {"b0_y1": "Скоринговая\nкарта B0", "m2_y1": "Нейросеть\nM2", "m1_y1": "CatBoost\nM1", "xgb_y1": "XGBoost", "lgb_y1": "LightGBM"}
order = ["b0_y1", "m2_y1", "m1_y1", "xgb_y1", "lgb_y1"]
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))

x = np.arange(len(order))
auc = test.loc[order, "auc"]
err = [auc - test.loc[order, "auc_lo"], test.loc[order, "auc_hi"] - auc]
ax[0].errorbar(x, auc, yerr=err, fmt="o", capsize=5, color="C0", ms=7)
ax[0].set_xticks(x, [names[m] for m in order], fontsize=9)
ax[0].set_ylabel("AUC")
ax[0].set_title("Test 2018: AUC и 95 % доверительный интервал", fontsize=11)

periods = [("test 2018", None), ("кризис\n2007–2009", "gfc"), ("COVID\n2019–2021", "covid"), ("рост ставок\n2022–2025", "rates")]
for k, (m, label, color) in enumerate([("b0_y1", "карта B0", "C7"), ("m1_y1", "CatBoost M1", "C0"), ("m2_y1", "нейросеть M2", "C1")]):
    vals = [test.loc[m, "auc"]] + [stress[(stress["scenario"] == sc) & (stress["model"] == m)]["auc"].iloc[0] for _, sc in periods[1:]]
    ax[1].bar(np.arange(4) + (k - 1) * 0.26, vals, width=0.26, label=label, color=color)
ax[1].set_xticks(np.arange(4), [p for p, _ in periods], fontsize=9)
ax[1].set_ylim(0.7, 0.86)
ax[1].set_ylabel("AUC")
ax[1].set_title("AUC на test и в стресс-периодах", fontsize=11)
ax[1].legend(fontsize=8)

plt.tight_layout()
fig.savefig(WEEK / "figures/fig_models.png", dpi=150, bbox_inches="tight")
print("saved", WEEK / "figures/fig_models.png")
