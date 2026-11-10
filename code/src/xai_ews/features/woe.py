"""WoE binning and Information Value (stage 2, step 2).

WoE (weight of evidence) of bin i = ln(share of goods in bin i / share of bads in bin i), with
goods = y 0, bads = y 1 and shares weighted by the sample weight (1/r for subsampled negatives).
Positive WoE = safer than average, negative = riskier. IV = sum_i (good_i - bad_i) * WoE_i.

Binning (fitted on train only):
    numeric      weighted quantile pre-bins (each distinct value if there are few), then
                 adjacent bins are merged until every bin has >= min_bin_share of the weight and
                 >= min_bin_events positives, and at most max_bins remain (ChiMerge: the adjacent
                 pair with the smallest chi2 goes first). Exception: a rare significant bin
                 (>= min_bin_events positives, event rate >= rare_bin_rate_ratio times that of its
                 neighbours, e.g. modified loans) stays separate below min_bin_share. Missing values get
                 their own bin if they are common enough or rare significant, otherwise they join
                 the bin with the closest event rate.
    categorical  rare categories pooled into "__rare__", categories ordered by WoE, then the
                 same merge of neighbours in that order. Unseen categories -> "__rare__" bin
                 (or WoE 0 if there is none).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RARE = "__rare__"


@dataclass
class WoEBinner:
    feature: str
    kind: str                                   # "numeric" | "categorical"
    edges: list[float] = field(default_factory=list)       # bin j: edges[j-1] < x <= edges[j]
    cat_to_bin: dict[str, int] = field(default_factory=dict)
    missing_bin: int | None = None              # numeric: bin of NaN; None if unseen on train
    default_bin: int | None = None              # categorical: bin of unseen categories
    woe: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    iv: float = 0.0

    @property
    def n_bins(self) -> int:
        return len(self.woe)

    def bin_index(self, x) -> np.ndarray:
        """Bin per value; -1 = not seen on train (WoE 0)."""
        if self.kind == "numeric":
            x = np.asarray(x, dtype=np.float64)
            idx = np.searchsorted(np.asarray(self.edges, dtype=np.float64), x, side="left")
            nan = np.isnan(x)
            idx[nan] = -1 if self.missing_bin is None else self.missing_bin
            return idx.astype(np.int32)
        s = pd.Series(np.asarray(x, dtype=object))
        default = -1 if self.default_bin is None else self.default_bin
        return s.map(self.cat_to_bin).fillna(default).to_numpy().astype(np.int32)

    def transform(self, x) -> np.ndarray:
        return np.append(np.asarray(self.woe, dtype=np.float32), np.float32(0.0))[self.bin_index(x)]

    def to_dict(self) -> dict:
        return {"feature": self.feature, "kind": self.kind, "edges": list(map(float, self.edges)),
                "cat_to_bin": self.cat_to_bin, "missing_bin": self.missing_bin,
                "default_bin": self.default_bin, "woe": list(map(float, self.woe)),
                "labels": self.labels, "iv": float(self.iv)}

    @classmethod
    def from_dict(cls, d: dict) -> "WoEBinner":
        return cls(**d)


def woe_iv(good: np.ndarray, bad: np.ndarray, smoothing: float = 0.5) -> tuple[np.ndarray, float]:
    """WoE per bin and total IV from (weighted) good / bad counts per bin."""
    good = np.asarray(good, dtype=np.float64)
    bad = np.asarray(bad, dtype=np.float64)
    k = len(good)
    g = (good + smoothing) / (good.sum() + smoothing * k)
    b = (bad + smoothing) / (bad.sum() + smoothing * k)
    woe = np.log(g / b)
    iv = float(np.sum((good / max(good.sum(), 1e-12) - bad / max(bad.sum(), 1e-12)) * woe))
    return woe, iv


def weighted_quantiles(x: np.ndarray, w: np.ndarray, levels: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    cw = np.cumsum(w[order])
    pos = np.searchsorted(cw, levels * cw[-1], side="left")
    return x[order][np.clip(pos, 0, len(x) - 1)]


def _chi2_pair(g1, b1, g2, b2) -> float:
    n = g1 + b1 + g2 + b2
    obs = np.array([[g1, b1], [g2, b2]])
    exp = np.outer([g1 + b1, g2 + b2], [g1 + g2, b1 + b2]) / max(n, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum(np.where(exp > 0, (obs - exp) ** 2 / exp, 0.0)))


def _rate_ratio(a: float, b: float) -> float:
    lo, hi = min(a, b), max(a, b)
    return hi / max(lo, 1e-12)


def merge_adjacent(good, bad, n_bad, min_share: float, min_events: int, max_bins: int,
                   rare_ratio: float | None = None) -> list[list[int]]:
    """Groups of consecutive pre-bins after merging; returns the pre-bin indices of each group.

    A bin below `min_share` is merged into its neighbour with the closer event rate, unless it is
    a rare significant bin (`rare_ratio` set): >= min_events positives and an event rate at least
    `rare_ratio` times different from both neighbours (e.g. modified loans among all loans).
    """
    groups = [[i] for i in range(len(good))]
    g, b, nb = (list(map(float, a)) for a in (good, bad, n_bad))
    total = sum(g) + sum(b)

    def merge(i: int, j: int) -> None:          # j = i + 1
        groups[i] += groups.pop(j)
        g[i] += g.pop(j)
        b[i] += b.pop(j)
        nb[i] += nb.pop(j)

    while len(groups) > 1:
        share = (np.array(g) + np.array(b)) / max(total, 1e-12)
        rate = np.array(b) / np.maximum(np.array(g) + np.array(b), 1e-12)
        events = np.array(nb)
        small = share < min_share
        if rare_ratio is not None:
            for j in np.flatnonzero(small & (events >= min_events)):
                nbrs = [k for k in (j - 1, j + 1) if 0 <= k < len(groups)]
                if min(_rate_ratio(rate[j], rate[k]) for k in nbrs) >= rare_ratio:
                    small[j] = False
        viol = small | (events < min_events)
        if viol.any():
            j = int(np.argmin(np.where(viol, share, np.inf)))
            if j == 0:
                merge(0, 1)
            elif j == len(groups) - 1:
                merge(j - 1, j)
            else:
                left_closer = abs(rate[j] - rate[j - 1]) <= abs(rate[j] - rate[j + 1])
                merge(j - 1, j) if left_closer else merge(j, j + 1)
        elif len(groups) > max_bins:
            chi = [_chi2_pair(g[i], b[i], g[i + 1], b[i + 1]) for i in range(len(groups) - 1)]
            i = int(np.argmin(chi))
            merge(i, i + 1)
        else:
            break
    return groups


def _bin_counts(idx: np.ndarray, y: np.ndarray, w: np.ndarray, k: int):
    good = np.bincount(idx, weights=w * (y == 0), minlength=k)
    bad = np.bincount(idx, weights=w * (y == 1), minlength=k)
    n_bad = np.bincount(idx, weights=(y == 1).astype(float), minlength=k)
    return good, bad, n_bad


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def fit_numeric(x, y, w, feature: str, cfg: dict) -> WoEBinner:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    w = np.asarray(w, dtype=np.float64)
    nan = np.isnan(x)
    xv, yv, wv = x[~nan], y[~nan], w[~nan]
    binner = WoEBinner(feature, "numeric")
    if len(xv) == 0:
        good, bad = np.array([w[y == 0].sum()]), np.array([w[y == 1].sum()])
        binner.missing_bin, binner.labels = 0, ["missing"]
        binner.woe, binner.iv = [0.0], 0.0
        return binner
    uniq = np.unique(xv)
    if len(uniq) <= cfg["prebins"]:
        edges = uniq[:-1]
    else:
        levels = np.arange(1, cfg["prebins"]) / cfg["prebins"]
        edges = np.unique(weighted_quantiles(xv, wv, levels))
        edges = edges[edges < uniq[-1]]
    idx = np.searchsorted(edges, xv, side="left")
    good, bad, n_bad = _bin_counts(idx, yv, wv, len(edges) + 1)
    # empty pre-bins (possible with weighted quantiles) are dropped before merging
    keep = (good + bad) > 0
    edges_full = np.append(edges, np.inf)[keep]
    good, bad, n_bad = good[keep], bad[keep], n_bad[keep]
    total_w = w.sum()
    min_share = cfg["min_bin_share"] * total_w / max(wv.sum(), 1e-12)
    rare_ratio = cfg.get("rare_bin_rate_ratio")
    groups = merge_adjacent(good, bad, n_bad, min_share, cfg["min_bin_events"], cfg["max_bins"], rare_ratio)
    upper = [edges_full[grp[-1]] for grp in groups]
    binner.edges = [float(e) for e in upper[:-1]]
    gb = [(good[grp].sum(), bad[grp].sum()) for grp in groups]
    lo = [-np.inf, *binner.edges]
    binner.labels = [f"({_fmt(a)}, {_fmt(b)}]" for a, b in zip(lo, [*binner.edges, np.inf])]
    if nan.any():
        mg, mb = w[nan & (y == 0)].sum(), w[nan & (y == 1)].sum()
        rates = np.array([bb / max(a + bb, 1e-12) for a, bb in gb])
        rate_m = mb / max(mg + mb, 1e-12)
        closest = int(np.argmin(np.abs(rates - rate_m)))
        separate = (mg + mb) / total_w >= cfg["min_bin_share"] or (
            rare_ratio is not None and (nan & (y == 1)).sum() >= cfg["min_bin_events"]
            and _rate_ratio(rate_m, rates[closest]) >= rare_ratio)
        if separate:
            gb.append((mg, mb))
            binner.labels.append("missing")
            binner.missing_bin = len(gb) - 1
        else:
            # rare, unremarkable missing values join the bin with the closest event rate
            # (the largest bin when the rare-bin rule is off: the stage-2 sensitivity variant)
            to = closest if rare_ratio is not None else int(np.argmax([a + bb for a, bb in gb]))
            gb[to] = (gb[to][0] + mg, gb[to][1] + mb)
            binner.missing_bin = to
            binner.labels[to] += " + missing"
    woe, iv = woe_iv(*zip(*gb), smoothing=cfg["smoothing"])
    binner.woe, binner.iv = [float(v) for v in woe], iv
    return binner


def fit_categorical(x, y, w, feature: str, cfg: dict) -> WoEBinner:
    y = np.asarray(y)
    w = np.asarray(w, dtype=np.float64)
    codes, uniques = pd.factorize(pd.Series(np.asarray(x, dtype=object)), sort=False)
    k = len(uniques)
    good, bad, n_bad = _bin_counts(codes, y, w, k)
    share = (good + bad) / w.sum()
    rare = share < cfg["rare_category_share"]
    names = [str(u) for u in uniques]
    # pool rare categories into one group
    group_of = np.arange(k)
    if rare.sum() > 1 or (rare.any() and k > 1):
        kept = np.flatnonzero(~rare)
        group_of = np.full(k, len(kept))
        group_of[kept] = np.arange(len(kept))
        group_names = [[names[i]] for i in kept] + [[RARE, *[names[i] for i in np.flatnonzero(rare)]]]
    else:
        group_names = [[n] for n in names]
    m = len(group_names)
    gg = np.bincount(group_of, weights=good, minlength=m)
    gbad = np.bincount(group_of, weights=bad, minlength=m)
    gnb = np.bincount(group_of, weights=n_bad, minlength=m)
    order = np.argsort(woe_iv(gg, gbad, cfg["smoothing"])[0], kind="stable")   # riskiest first
    gg, gbad, gnb = gg[order], gbad[order], gnb[order]
    group_names = [group_names[i] for i in order]
    # pre-group long tails into at most `prebins` runs of similar WoE
    if m > cfg["prebins"]:
        cum = np.cumsum(gg + gbad) / (gg + gbad).sum()
        run = np.minimum((cum - (gg + gbad) / (gg + gbad).sum() / 2) * cfg["prebins"],
                         cfg["prebins"] - 1).astype(int)
        _, run = np.unique(run, return_inverse=True)
        r = run.max() + 1
        gg, gbad, gnb = (np.bincount(run, weights=a, minlength=r) for a in (gg, gbad, gnb))
        group_names = [[c for i in np.flatnonzero(run == j) for c in group_names[i]] for j in range(r)]
    groups = merge_adjacent(gg, gbad, gnb, cfg["min_bin_share"], cfg["min_bin_events"], cfg["max_bins"],
                            cfg.get("rare_bin_rate_ratio"))
    binner = WoEBinner(feature, "categorical")
    gb, labels = [], []
    for j, grp in enumerate(groups):
        members = [c for i in grp for c in group_names[i]]
        for c in members:
            if c == RARE:
                binner.default_bin = j
            else:
                binner.cat_to_bin[c] = j
        gb.append((gg[grp].sum(), gbad[grp].sum()))
        shown = [c for c in members if c != RARE]
        lab = ", ".join(shown[:6]) + (f", … (+{len(shown) - 6})" if len(shown) > 6 else "")
        labels.append(lab + (" + rare" if RARE in members else ""))
    woe, iv = woe_iv(*zip(*gb), smoothing=cfg["smoothing"])
    binner.woe, binner.iv, binner.labels = [float(v) for v in woe], iv, labels
    return binner


def fit_woe(x, y, w, feature: str, categorical: bool, cfg: dict) -> WoEBinner:
    return (fit_categorical if categorical else fit_numeric)(x, y, w, feature, cfg)


def bin_table(binner: WoEBinner, x, y, w) -> pd.DataFrame:
    """Per-bin counts on any sample with the binner's (train) bins: rows, events, weighted share,
    event rate, WoE, IV contribution recomputed on this sample."""
    idx = binner.bin_index(x)
    k = binner.n_bins + 1                       # last slot: unseen on train
    idx = np.where(idx < 0, k - 1, idx)
    y = np.asarray(y)
    w = np.asarray(w, dtype=np.float64)
    good, bad, n_bad = _bin_counts(idx, y, w, k)
    n = np.bincount(idx, minlength=k)
    labels = [*binner.labels, "unseen"]
    keep = np.arange(k) < binner.n_bins
    keep[-1] = n[-1] > 0
    G, B = good.sum(), bad.sum()
    df = pd.DataFrame({"feature": binner.feature, "bin": np.arange(k), "label": labels,
                       "rows": n, "events": n_bad.astype(int),
                       "share": (good + bad) / max(G + B, 1e-12),
                       "event_rate": bad / np.maximum(good + bad, 1e-12),
                       "woe_train": [*binner.woe, 0.0]})[keep]
    gs, bs = df.index, df.index
    g_sh = good[gs] / max(G, 1e-12)
    b_sh = bad[bs] / max(B, 1e-12)
    df["iv_part"] = (g_sh - b_sh) * df["woe_train"].to_numpy()
    return df.reset_index(drop=True)


def save_binners(binners: dict[str, WoEBinner], path) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({f: b.to_dict() for f, b in binners.items()}))


def load_binners(path) -> dict[str, WoEBinner]:
    import json
    return {f: WoEBinner.from_dict(d) for f, d in json.loads(path.read_text()).items()}
