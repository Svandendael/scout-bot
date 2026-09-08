"""Smoke tests on synthetic data. Run: python -m pytest -q  (or python tests/test_pipeline.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scout import load_config, load_universe
from scout.data import make_synthetic
from scout.signals import compute_features
from scout.portfolio import select_targets, target_weights, propose_orders


def test_features_and_selection():
    prices = make_synthetic(seed=3)
    cfg, uni = load_config(), load_universe()
    feat = compute_features(prices, None, cfg, uni)
    assert "IWDA" in feat.index and feat.loc["IWDA", "score"] == feat.loc["IWDA", "score"]
    # eligibility never true for watchlist
    assert not feat[feat["role"] == "watch"]["eligible"].any()
    sel = select_targets(feat, [], cfg, uni)
    assert len(sel["slots"]) == cfg["portfolio"]["top_n"]
    w = target_weights(sel["slots"])
    assert abs(sum(w.values()) - 1) < 1e-9


def test_no_lookahead():
    """Features as-of a past date must not change when later data is appended."""
    prices = make_synthetic(seed=5)
    cfg, uni = load_config(), load_universe()
    d = prices.index[-300]
    a = compute_features(prices.loc[:d], None, cfg, uni)["score"]
    b = compute_features(prices, d, cfg, uni)["score"]
    pd.testing.assert_series_equal(a, b)


def test_orders_whole_units_within_cash():
    prices = make_synthetic(seed=9)
    cfg, uni = load_config(), load_universe()
    feat = compute_features(prices, None, cfg, uni)
    sel = select_targets(feat, [], cfg, uni)
    holdings = pd.DataFrame(columns=["units", "avg_cost_eur"]).rename_axis("id")
    res = propose_orders(feat, target_weights(sel["slots"]), holdings, 1000.0, cfg, uni)
    spent = sum(o["value"] for o in res["orders"] if o["side"] == "BUY")
    assert spent <= 1000.0 and all(isinstance(o["units"], int) for o in res["orders"])
    assert res["cash_after"] >= 0


if __name__ == "__main__":
    test_features_and_selection(); test_no_lookahead(); test_orders_whole_units_within_cash()
    print("all tests passed")
