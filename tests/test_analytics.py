import numpy as np

from factprotest import analytics


def test_leik_dispersion_extremes():
    assert analytics.leik_dispersion(["Support"] * 10) == 0
    assert analytics.leik_dispersion(["Support"] * 5 + ["Oppose"] * 5) == 1
    assert 0 < analytics.leik_dispersion(["Support", "Neutral", "Oppose"]) < 1


def test_bootstrap_ci_contains_point_estimate():
    stances = ["Support"] * 30 + ["Oppose"] * 10
    point, lo, hi = analytics.bootstrap_ci(stances, analytics.support_share, n_boot=500)
    assert point == 0.75
    assert lo < point < hi


def test_stance_by_state_detects_strong_association():
    states = ["Punjab"] * 40 + ["Gujarat"] * 40
    stances = ["Support"] * 38 + ["Oppose"] * 2 + ["Oppose"] * 38 + ["Support"] * 2
    result = analytics.stance_by_state_test(states, stances)
    assert result["testable"] and result["p_value"] < 1e-6 and result["cramers_v"] > 0.7


def test_score_bias_audit_finds_no_bias_on_identical_distributions():
    rng = np.random.default_rng(0)
    scores = list(rng.integers(30, 90, 200))
    stances = ["Support", "Oppose"] * 100
    audit = analytics.score_bias_audit(scores, stances)
    assert audit["mannwhitney_p"] > 0.05
    assert abs(audit["rank_biserial"]) < 0.2
