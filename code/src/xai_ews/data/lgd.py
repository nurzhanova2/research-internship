"""Segment LGD: realised loss rates of liquidations completed by the end
of the train period, from ALL vintages including 2006-2008 (downturn LGD in the Basel IRB sense;
crisis vintages are not used for PD training). LGD is gross of mortgage insurance and discounted
at the note rate from the last paid instalment to disposition (see audit.liquidations_with_segments).
Segments = MTM LTV at default x judicial state (when applied to live loans: current MTM LTV at
t); a segment with fewer than n_min liquidations is merged with its LTV neighbour within the
same judicial group. LGD is floored at 0 (liquidations at a gain count as no loss).

Variants:
    base              gross of MI, discounted, all vintages
    after_mi          sensitivity: after mortgage insurance (credit_enhancement_proceeds deducted)
    undiscounted      sensitivity: gross of MI, no discounting
    train_vintages    sensitivity: base definition, only train vintages (2012Q1-2014Q4)
    constant          sensitivity: one LGD for the whole portfolio (base definition and sample)
"""

import itertools

import polars as pl

from xai_ews.data.panel import month_index, ym_to_index
from xai_ews.data.split import vintage_index


def completed_by_train_end(liq: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Zero balance and, where reported, disposition no later than the last train month."""
    end = ym_to_index(cfg["split"]["train"]["end"])
    disp = month_index("disposition_date").fill_null(0)
    return liq.filter((pl.col("m") <= end) & (disp <= end) & pl.col("has_loss_data"))


def bucket_order(cfg: dict) -> list[str]:
    edges = cfg["lgd_segments"]["mtm_ltv_buckets"]
    return [f"<{edges[0]}"] + [f"{a}-{b}" for a, b in itertools.pairwise(edges)] \
        + [f"{edges[-1]}+"]


def merge_small(counts: list[int], n_min: int) -> list[int]:
    """Group ids for ordered buckets so that each group has >= n_min (if the total allows):
    small buckets join the next one; a small last group joins the previous one."""
    groups, current, acc = [], 0, 0
    for n in counts:
        groups.append(current)
        acc += n
        if acc >= n_min:
            current, acc = current + 1, 0
    if acc < n_min and current > 0 and current in groups:  # small or empty trailing group
        groups = [g if g < current else current - 1 for g in groups]
    return groups


def segment_lgd(liq: pl.DataFrame, cfg: dict, lgd_col: str = "lgd") -> pl.DataFrame:
    n_min = cfg["exposure"]["lgd"]["min_liquidations_per_segment"]
    order = bucket_order(cfg)
    df = liq.filter(pl.col("mtm_ltv_bucket") != "unknown").with_columns(
        pl.col(lgd_col).alias("lgd"), pl.col(lgd_col).clip(0, None).alias("lgd0"))
    rows = []
    for judicial in (False, True):
        part = df.filter(pl.col("judicial") == judicial)
        counts = [part.filter(pl.col("mtm_ltv_bucket") == b).height for b in order]
        for g in sorted(set(groups := merge_small(counts, n_min))):
            buckets = [b for b, gb in zip(order, groups, strict=True) if gb == g]
            sel = part.filter(pl.col("mtm_ltv_bucket").is_in(buckets))
            rows.append({
                "judicial": judicial,
                "segment": f"{'J' if judicial else 'NJ'}:{buckets[0].split('-')[0]}"
                           + (f"..{buckets[-1]}" if len(buckets) > 1 else f"|{buckets[0]}"),
                "mtm_ltv_buckets": ",".join(buckets),
                "liquidations": sel.height,
                "lgd_mean": sel["lgd0"].mean(),
                "lgd_median": sel["lgd0"].median(),
                "share_gain_clipped": (sel["lgd"] < 0).mean(),
                "median_workout_months": sel["workout_months"].median(),
                "below_n_min": sel.height < n_min,
            })
    return pl.DataFrame(rows)


def _constant(done: pl.DataFrame, lgd_col: str) -> pl.DataFrame:
    lgd0 = done[lgd_col].clip(0, None)
    return pl.DataFrame({"judicial": [None], "segment": ["all"], "mtm_ltv_buckets": ["all"],
                         "liquidations": [done.height], "lgd_mean": [lgd0.mean()],
                         "lgd_median": [lgd0.median()],
                         "share_gain_clipped": [(done[lgd_col] < 0).mean()],
                         "median_workout_months": [done["workout_months"].median()],
                         "below_n_min": [False]},
                        schema_overrides={"judicial": pl.Boolean})


def lgd_variants(liq: pl.DataFrame, cfg: dict) -> dict[str, pl.DataFrame]:
    s = cfg["split"]["train"]["vintages"]
    done = completed_by_train_end(liq, cfg)
    vi = pl.col("vintage").map_elements(vintage_index, return_dtype=pl.Int32)
    train_v = done.filter(vi.is_between(vintage_index(s["from"]), vintage_index(s["to"])))
    return {
        "base": segment_lgd(done, cfg),
        "after_mi": segment_lgd(done, cfg, "lgd_net_mi"),
        "undiscounted": segment_lgd(done, cfg, "lgd_undisc"),
        "train_vintages": segment_lgd(train_v, cfg),
        "constant": _constant(done, "lgd"),
        "constant_after_mi": _constant(done, "lgd_net_mi"),
        "constant_undiscounted": _constant(done, "lgd_undisc"),
    }
