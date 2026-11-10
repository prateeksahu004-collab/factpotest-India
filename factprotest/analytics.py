"""Statistical analysis of opinions: polarization, uncertainty, association tests and a scorer-bias audit."""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import stats

ORDER = ["Oppose", "Neutral", "Support"]


def stance_counts(stances: Sequence[str]) -> dict[str, int]:
    counts = Counter(stances)
    return {s: counts.get(s, 0) for s in ORDER}


def leik_dispersion(stances: Sequence[str]) -> float:
    """Leik's ordinal dispersion D on Oppose < Neutral < Support.
    0 = everyone in one category (consensus); 1 = split evenly between the two extremes (maximum polarization)."""
    counts = stance_counts(stances)
    n = sum(counts.values())
    if n == 0:
        return float("nan")
    cumulative = np.cumsum([counts[s] / n for s in ORDER])[:-1]
    return round(float(2 * np.sum(np.minimum(cumulative, 1 - cumulative)) / (len(ORDER) - 1)), 4)


def bootstrap_ci(values: Sequence[Any], statistic: Callable[[Sequence[Any]], float], n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap: (point estimate, lower, upper)."""
    values = list(values)
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = np.array([statistic([values[i] for i in row]) for row in idx])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(statistic(values)), 4), round(float(lo), 4), round(float(hi), 4)


def support_share(stances: Sequence[str]) -> float:
    return sum(s == "Support" for s in stances) / len(stances) if stances else float("nan")


def stance_by_state_test(states: Sequence[str], stances: Sequence[str], min_count: int = 8) -> dict[str, Any]:
    """Chi-square test of independence between state and stance (states with < min_count opinions pooled as 'Other')."""
    totals = Counter(states)
    keep = [s if totals[s] >= min_count and s != "Not Specified" else "Other" for s in states]
    rows = sorted(set(keep))
    table = np.array([[sum(1 for st, sn in zip(keep, stances) if st == r and sn == c) for c in ORDER] for r in rows])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"testable": False}
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {"testable": True, "chi2": round(float(chi2), 3), "p_value": float(p), "dof": int(dof),
            "cramers_v": round(float(cramers_v), 4), "n": int(n), "states": rows,
            "low_expected_cells": int((expected < 5).sum()), "table": table.tolist()}


def score_bias_audit(scores: Sequence[float], stances: Sequence[str]) -> dict[str, Any]:
    """Does the credibility scorer systematically favour one side? Kruskal-Wallis across stances, then
    Mann-Whitney Support vs Oppose with rank-biserial effect size (|r| < 0.1 is negligible)."""
    groups = {s: [float(x) for x, st in zip(scores, stances) if st == s and x is not None] for s in ORDER}
    present = {s: g for s, g in groups.items() if len(g) >= 5}
    out: dict[str, Any] = {"medians": {s: float(np.median(g)) for s, g in present.items()},
                           "means": {s: round(float(np.mean(g)), 2) for s, g in present.items()},
                           "n": {s: len(g) for s, g in present.items()}}
    if len(present) >= 2:
        h, p = stats.kruskal(*present.values())
        out["kruskal_h"], out["kruskal_p"] = round(float(h), 3), float(p)
    if len(groups["Support"]) >= 5 and len(groups["Oppose"]) >= 5:
        u, p = stats.mannwhitneyu(groups["Support"], groups["Oppose"], alternative="two-sided")
        n1, n2 = len(groups["Support"]), len(groups["Oppose"])
        out["mannwhitney_u"], out["mannwhitney_p"] = float(u), float(p)
        out["rank_biserial"] = round(float(2 * u / (n1 * n2) - 1), 4)
    return out


def protest_summary(opinions: list[dict[str, Any]], score_key: str = "credibility_score") -> dict[str, Any]:
    stances = [o["stance"] for o in opinions]
    scores = [o[score_key] for o in opinions if o.get(score_key) is not None]
    share = bootstrap_ci(stances, support_share)
    polarization = bootstrap_ci(stances, leik_dispersion)
    mean_score = bootstrap_ci(scores, lambda xs: float(np.mean(xs))) if scores else (float("nan"),) * 3
    return {"n": len(opinions), "counts": stance_counts(stances), "support_share": share,
            "polarization": polarization, "mean_credibility": mean_score}
