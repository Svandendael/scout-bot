"""Signals computed as-of a date, from daily prices.

compute_features(prices, asof) returns one row per ETF:
  score       mean of 3/6/9/12-month returns (ensemble momentum)
  r3..r12     the individual returns
  above_sma   close > 10-month SMA at this month-end
  trend_ok    above_sma for the last `trend_confirm_months` month-ends
  absmom      12-month return beats the cash benchmark
  dist_sma    close / SMA - 1   (how far from the trend line)
  high_ratio  close / 252-day high
  dd12        drawdown from the 12-month peak
  adj_score   score minus the TOB penalty
  eligible    core AND trend_ok AND absmom
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import load_config, load_universe


def month_ends(prices: pd.DataFrame) -> pd.DataFrame:
    """Last available close of each calendar month (indexed by the actual last trading day)."""
    return prices.groupby(prices.index.to_period("M")).tail(1)


def compute_features(prices: pd.DataFrame, asof: pd.Timestamp | None = None,
                     cfg: dict | None = None, uni: dict | None = None,
                     me_all: pd.DataFrame | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    uni = uni or load_universe()
    s = cfg["signals"]
    p = cfg["portfolio"]

    px = prices.loc[: asof] if asof is not None else prices
    asof = px.index[-1]
    if me_all is None:
        me = month_ends(px)                # month-end closes up to and including asof's month
    else:
        me = me_all.loc[:asof]
        if me.index[-1] != asof:           # asof is mid-month: the current partial month counts as "now"
            me = pd.concat([me, px.iloc[[-1]]])
    lookbacks = s["lookbacks_months"]
    need = max(max(lookbacks), s["absmom_months"], s["trend_sma_months"] + s["trend_confirm_months"])

    rows = []
    cash_id = uni["cash_benchmark"]
    cash_r = np.nan
    if cash_id in me and me[cash_id].dropna().shape[0] > s["absmom_months"]:
        c = me[cash_id].dropna()
        cash_r = c.iloc[-1] / c.iloc[-1 - s["absmom_months"]] - 1

    for etf in uni["etfs"]:
        eid = etf["id"]
        if eid not in me:
            continue
        m = me[eid].dropna()
        row = {"id": eid, "name": etf["name"], "sleeve": etf["sleeve"], "role": etf["role"],
               "tob": etf["tob"], "verify": bool(etf.get("verify", False)),
               "defensive": bool(etf.get("defensive", False)), "price": float(px[eid].dropna().iloc[-1])
               if px[eid].notna().any() else np.nan, "months": int(len(m))}
        if len(m) <= need:
            row.update(score=np.nan, adj_score=np.nan, eligible=False, trend_ok=False,
                       above_sma=False, absmom=False, dist_sma=np.nan, high_ratio=np.nan, dd12=np.nan)
            for k in lookbacks:
                row[f"r{k}"] = np.nan
            rows.append(row)
            continue
        last = m.iloc[-1]
        for k in lookbacks:
            row[f"r{k}"] = last / m.iloc[-1 - k] - 1
        row["score"] = float(np.mean([row[f"r{k}"] for k in lookbacks]))
        sma = m.rolling(s["trend_sma_months"]).mean()
        above = (m > sma)
        row["above_sma"] = bool(above.iloc[-1])
        row["trend_ok"] = bool(above.iloc[-s["trend_confirm_months"]:].all())
        row["dist_sma"] = float(last / sma.iloc[-1] - 1)
        r_abs = last / m.iloc[-1 - s["absmom_months"]] - 1
        row["r_abs"] = r_abs
        row["absmom"] = bool(r_abs > cash_r) if not np.isnan(cash_r) else bool(r_abs > 0)
        daily = px[eid].dropna()
        w = daily.iloc[-s["high_window_days"]:]
        row["high_ratio"] = float(daily.iloc[-1] / w.max())
        row["dd12"] = float(daily.iloc[-1] / w.max() - 1)
        row["adj_score"] = row["score"] - p["tob_penalty"] * 2 * etf["tob"]
        row["eligible"] = bool(etf["role"] == "core" and row["trend_ok"] and row["absmom"])
        rows.append(row)

    df = pd.DataFrame(rows).set_index("id")
    df.attrs["asof"] = asof
    df.attrs["cash_r12"] = cash_r
    df["rank"] = df["adj_score"].rank(ascending=False, method="first")
    return df.sort_values("adj_score", ascending=False)
