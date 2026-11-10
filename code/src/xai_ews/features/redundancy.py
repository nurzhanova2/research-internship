"""Redundancy between features (stage 2, steps 4-5).

Correlation and VIF are computed on WoE-transformed values: every feature (numeric or
categorical, with missing values) becomes one number on the risk scale, and the scorecard and
the network see the features in a comparable form. Mutual information and mRMR work on the
train WoE bins.

    VIF_j = 1 / (1 - R_j^2) = j-th diagonal element of the inverse correlation matrix
    MI(A, B) = sum p(a, b) ln(p(a, b) / (p(a) p(b)))
    mRMR (MID form): next = argmax_f  MI(f, y) - mean_{s in selected} MI(f, s)
"""

import numpy as np
import pandas as pd


def weighted_corr(X: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted Pearson correlation of the columns of X; constant columns give NaN rows."""
    X = np.asarray(X, dtype=np.float64)
    w = np.ones(len(X)) if w is None else np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    Xc = X - w @ X
    cov = (Xc * w[:, None]).T @ Xc
    sd = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(sd, sd)
    corr[sd == 0, :] = np.nan
    corr[:, sd == 0] = np.nan
    return corr


def vif_from_corr(corr: np.ndarray) -> np.ndarray:
    """VIF of every column from the correlation matrix (pseudo-inverse if singular)."""
    return np.diag(np.linalg.pinv(np.asarray(corr, dtype=np.float64)))


def vif_iterative(corr: pd.DataFrame, vif_max: float, strength: pd.Series) -> tuple[list[str], list[dict]]:
    """Drop features one by one while some VIF > vif_max. Among the features above the limit the
    weakest one (smallest strength) goes first. Returns (kept, removal log)."""
    kept = list(corr.index)
    log = []
    while len(kept) > 1:
        v = pd.Series(vif_from_corr(corr.loc[kept, kept].to_numpy()), index=kept)
        over = v[v > vif_max]
        if over.empty:
            break
        drop = strength.reindex(over.index).idxmin()
        log.append({"feature": drop, "vif": float(v[drop]), "max_vif_before": float(v.max())})
        kept.remove(drop)
    return kept, log


def mutual_info(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> float:
    """Mutual information (nats) of two discrete arrays of non-negative integer codes."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    w = np.ones(len(a)) if w is None else np.asarray(w, dtype=np.float64)
    nb = b.max() + 1
    joint = np.bincount(a * nb + b, weights=w, minlength=(a.max() + 1) * nb).reshape(-1, nb)
    p = joint / joint.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p / (pa * pb)), 0.0)
    return float(max(terms.sum(), 0.0))


def mi_matrix(codes: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Pairwise MI of the columns of an integer code matrix."""
    k = codes.shape[1]
    out = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            out[i, j] = out[j, i] = mutual_info(codes[:, i], codes[:, j], w)
    return out


def mrmr(relevance: pd.Series, mi_pairs: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    """Greedy mRMR order. relevance = MI(feature, y); mi_pairs = feature x feature MI."""
    left = list(relevance.index)
    k = len(left) if k is None else k
    chosen, rows = [], []
    while left and len(chosen) < k:
        red = (mi_pairs.loc[left, chosen].mean(axis=1) if chosen
               else pd.Series(0.0, index=left))
        score = relevance[left] - red
        f = score.idxmax()
        rows.append({"rank": len(chosen) + 1, "feature": f, "relevance": float(relevance[f]),
                     "redundancy": float(red[f]), "mrmr_score": float(score[f])})
        chosen.append(f)
        left.remove(f)
    return pd.DataFrame(rows)


def correlated_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Pairs with |rho| > threshold (upper triangle)."""
    c = corr.to_numpy()
    i, j = np.triu_indices_from(c, k=1)
    m = np.abs(c[i, j]) > threshold
    return pd.DataFrame({"a": corr.index[i[m]], "b": corr.index[j[m]], "rho": c[i, j][m]}) \
        .sort_values("rho", key=np.abs, ascending=False).reset_index(drop=True)


def greedy_decorrelate(strength: pd.Series, corrs: list[pd.DataFrame], threshold: float,
                       prefer: dict[str, str] | None = None) -> tuple[list[str], list[dict]]:
    """Go through features from strongest to weakest and keep one only if its |rho| with every
    already kept feature is <= threshold in every correlation matrix (NaN = not comparable,
    e.g. a feature constant in that task's population). `prefer` maps a feature to the one that
    should win against it regardless of strength (interpretability override)."""
    prefer = prefer or {}
    order = list(strength.sort_values(ascending=False).index)
    for loser, winner in prefer.items():         # move preferred features in front of their rivals
        if loser in order and winner in order and order.index(winner) > order.index(loser):
            order.remove(winner)
            order.insert(order.index(loser), winner)
    kept, log = [], []
    for f in order:
        clash = None
        for c in corrs:
            if f not in c.index:
                continue
            for g in kept:
                if g in c.index and np.abs(np.nan_to_num(c.loc[f, g])) > threshold:
                    clash = (g, float(c.loc[f, g]))
                    break
            if clash:
                break
        if clash:
            log.append({"feature": f, "kept_instead": clash[0], "rho": clash[1]})
        else:
            kept.append(f)
    return kept, log


def select_per_task(corrs: dict[str, pd.DataFrame], iv: pd.DataFrame, candidates: list[str],
                    iv_min: float, corr_max: float, vif_max: float,
                    prefer: dict[str, str] | None = None) -> tuple[list[str], dict[str, list[str]], pd.DataFrame]:
    """Redundancy filter run separately for each task, on that task's population.

    For task t only the candidates that are informative for t (IV_t >= iv_min, not constant in
    t's population) take part; greedy |rho| filter by IV_t, then iterative VIF. A feature stays in
    the common set if it survives in at least one task where it is needed.
    Returns (kept union in candidate order, kept per task, log of removals per task).
    """
    prefer = prefer or {}
    kept_t, log = {}, []
    for t, c in corrs.items():
        need = [f for f in candidates if f in c.index and iv.loc[f, t] >= iv_min]
        s = iv.loc[need, t]
        pref = {a: b for a, b in prefer.items() if a in need and b in need}
        k, lg = greedy_decorrelate(s, [c], corr_max, prefer=pref)
        log += [{**r, "task": t, "rule": "corr"} for r in lg]
        k2, vlg = vif_iterative(c.loc[k, k], vif_max, s)
        log += [{"feature": r["feature"], "task": t, "rule": "vif", "vif": r["vif"]} for r in vlg]
        kept_t[t] = k2
    union = [f for f in candidates if any(f in k for k in kept_t.values())]
    # Different winners per task can leave a pair that is redundant in EVERY task where both
    # are needed (e.g. remaining term vs remaining legal term): keep only the stronger one.
    strength = iv.loc[union].max(axis=1)
    winners = set(prefer.values())
    for f in sorted(union, key=lambda x: strength[x]):
        for g in union:
            if g == f or strength[g] < strength[f] and g not in winners:
                continue
            need_f = [t for t, c in corrs.items() if f in c.index and iv.loc[f, t] >= iv_min]
            dup_everywhere = all(g in corrs[t].index and iv.loc[g, t] >= iv_min
                                 and abs(np.nan_to_num(corrs[t].loc[f, g])) > corr_max for t in need_f)
            if need_f and dup_everywhere and prefer.get(g) != f:
                both = need_f
                log.append({"feature": f, "task": "all", "rule": "union", "kept_instead": g,
                            "rho": float(max(abs(corrs[t].loc[f, g]) for t in both))})
                union.remove(f)
                for k in kept_t.values():
                    if f in k:
                        k.remove(f)
                break
    return union, kept_t, pd.DataFrame(log)
