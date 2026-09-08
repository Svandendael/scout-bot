"""Monthly backtest of the v1 rule against a global equity ETF and 60/40.

Simplifications (stated on the page): fractional units, month-end fills, costs =
half-spread + TOB per side + commission as a % of a typical order. The universe is
today's list, so there is mild survivorship bias in the *choice* of ETFs; the rule
itself only ever sees data up to each decision date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import load_config, load_universe
from .signals import compute_features, month_ends
from .portfolio import select_targets, target_weights


def _metrics(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return {}
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    r = eq.pct_change().dropna()
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(12)
    dd = (eq / eq.cummax() - 1)
    return {"cagr": cagr, "vol": vol, "sharpe": (r.mean() * 12) / vol if vol else np.nan,
            "maxdd": dd.min(), "years": yrs, "final": eq.iloc[-1]}


def run_backtest(prices: pd.DataFrame, cfg: dict | None = None, uni: dict | None = None,
                 start: str | None = None, overrides: dict | None = None) -> dict:
    cfg = cfg or load_config()
    uni = uni or load_universe()
    if overrides:  # shallow override of dials for sweeps
        cfg = {k: dict(v) for k, v in cfg.items()}
        for sect, kv in overrides.items():
            cfg[sect].update(kv)
    c = cfg["costs"]
    cost_side = c["spread_pct"] / 2 + c["commission_per_order"] / c["typical_order_eur"]
    me = month_ends(prices)
    start = pd.Timestamp(start or cfg["backtest"]["start"])
    dates = [d for d in me.index if d >= start]

    weights_prev: dict[str, float] = {}
    slots_prev: list[str] = []
    eq = [1.0]
    eq_idx = [dates[0]]
    turnover_total, n_trades, regimes, holdings_log = 0.0, 0, [], []
    for i in range(len(dates) - 1):
        d0, d1 = dates[i], dates[i + 1]
        feat = compute_features(prices, d0, cfg, uni, me_all=me)
        if feat["eligible"].sum() == 0 and feat["score"].notna().sum() < 3:
            eq.append(eq[-1]); eq_idx.append(d1); continue
        sel = select_targets(feat, list(dict.fromkeys(slots_prev)), cfg, uni)
        w = target_weights(sel["slots"])
        # turnover and costs
        ids = set(w) | set(weights_prev)
        turn = sum(abs(w.get(k, 0) - weights_prev.get(k, 0)) for k in ids) / 2
        tob = sum(abs(w.get(k, 0) - weights_prev.get(k, 0)) * uni["by_id"][k]["tob"] for k in ids)
        cost = turn * 2 * cost_side + tob
        n_trades += sum(1 for k in ids if abs(w.get(k, 0) - weights_prev.get(k, 0)) > 1e-9)
        # period return
        ret = 0.0
        for k, wk in w.items():
            p0, p1 = me.loc[d0, k], me.loc[d1, k]
            ret += wk * (p1 / p0 - 1) if p0 == p0 and p1 == p1 else 0.0
        eq.append(eq[-1] * (1 + ret) * (1 - cost))
        eq_idx.append(d1)
        turnover_total += turn
        weights_prev, slots_prev = w, sel["slots"]
        regimes.append(sel["regime"])
        holdings_log.append({"date": d1, "slots": sel["slots"], "regime": sel["regime"]})
    curve = pd.Series(eq, index=eq_idx, name="scout")

    # benchmarks
    eqb = uni["equity_benchmark"]
    bh = me[eqb].loc[dates[0]:dates[-1]]
    bh = (bh / bh.iloc[0]).rename("global_etf")
    w6040 = cfg["backtest"]["benchmark_weights_60_40"]
    rets = me[list(w6040)].loc[dates[0]:dates[-1]].pct_change().fillna(0)
    mix = (1 + (rets * pd.Series(w6040)).sum(axis=1)).cumprod().rename("60_40")

    years = (dates[-1] - dates[0]).days / 365.25
    out = {"curve": curve, "global_etf": bh, "60_40": mix,
           "metrics": {"scout": _metrics(curve), "global_etf": _metrics(bh), "60_40": _metrics(mix)},
           "turnover_per_year": turnover_total / years if years else np.nan,
           "trades_per_year": n_trades / years if years else np.nan,
           "months_risk_off": sum(r == "risk-off" for r in regimes) / max(len(regimes), 1),
           "holdings_log": holdings_log, "start": dates[0], "end": dates[-1]}
    # quarterly returns table
    q = curve.resample("QE").last()
    qb = bh.resample("QE").last()
    out["quarterly"] = pd.DataFrame({"scout": q.pct_change(), "global_etf": qb.pct_change()}).dropna()
    return out


def sweep(prices: pd.DataFrame, cfg: dict | None = None, uni: dict | None = None) -> pd.DataFrame:
    """Robustness: nearby dials should give similar results (a plateau, not a peak)."""
    cfg = cfg or load_config()
    rows = []
    for top_n in (2, 3, 4):
        for sma in (8, 10, 12):
            for lb in ([3, 6, 9, 12], [6, 12], [12]):
                r = run_backtest(prices, cfg, uni, overrides={
                    "portfolio": {"top_n": top_n}, "signals": {"trend_sma_months": sma, "lookbacks_months": lb}})
                m = r["metrics"]["scout"]
                rows.append({"top_n": top_n, "sma": sma, "lookbacks": "/".join(map(str, lb)),
                             "cagr": m.get("cagr"), "maxdd": m.get("maxdd"), "sharpe": m.get("sharpe"),
                             "trades_yr": r["trades_per_year"]})
    return pd.DataFrame(rows)
