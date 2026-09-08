"""Turn features into a target portfolio, then into whole-unit orders."""
from __future__ import annotations

import math

import pandas as pd

from . import ROOT, load_config, load_universe


def select_targets(feat: pd.DataFrame, current: list[str], cfg: dict | None = None,
                   uni: dict | None = None) -> dict:
    """Decide which ids fill the `top_n` slots.

    Rules: eligible core ETFs ranked by adj_score; an incumbent keeps its slot while
    eligible unless a challenger beats it by `switch_margin`; empty slots go to the
    better of the defensive ETFs by 12-month return.
    Returns {"slots": [...ids...], "reasons": {id: text}, "regime": "risk-on|mixed|risk-off"}
    """
    cfg = cfg or load_config()
    uni = uni or load_universe()
    p = cfg["portfolio"]
    n = p["top_n"]
    core = feat[feat["role"] == "core"]
    elig = core[core["eligible"]].sort_values("adj_score", ascending=False)
    reasons: dict[str, str] = {}

    # incumbents first
    kept = [i for i in current if i in elig.index]
    challengers = [i for i in elig.index if i not in kept]
    slots = list(kept)
    for c in challengers:
        if len(slots) < n:
            slots.append(c)
            reasons[c] = "enters: eligible and in the top group"
            continue
        # try to displace the weakest incumbent if the margin is met
        weakest = min(slots, key=lambda i: elig.loc[i, "adj_score"])
        if elig.loc[c, "adj_score"] - elig.loc[weakest, "adj_score"] > p["switch_margin"]:
            reasons[c] = f"replaces {weakest}: score higher by more than {p['switch_margin']*100:.0f} pts"
            reasons[weakest] = f"drops out: beaten by {c} by more than the switch margin"
            slots.remove(weakest)
            slots.append(c)
        else:
            reasons[c] = f"would rank ahead of {weakest} but within the switch margin — no change"
    for i in current:
        if i not in slots and i not in reasons:
            if i in core.index:
                f = core.loc[i]
                why = []
                if not f["trend_ok"]:
                    why.append("below its 10-month average")
                if not f["absmom"]:
                    why.append("12-month return below cash")
                reasons[i] = "exits: " + (" and ".join(why) if why else "no longer in the top group")
            else:
                reasons[i] = "defensive slot released" if feat.loc[i, "defensive"] else "not in universe"
    for i in slots:
        reasons.setdefault(i, "holds: still eligible and in the top group")

    # defensive fill
    empty = n - len(slots)
    if empty > 0:
        dfn = feat[feat["defensive"] & feat["r_abs"].notna()].sort_values("r_abs", ascending=False)
        if len(dfn):
            d = dfn.index[0]
            reasons[d] = (f"defensive: {empty} slot(s) without an eligible ETF; "
                          f"best of {', '.join(dfn.index)} by 12-month return")
            slots += [d] * empty
    n_risk = sum(1 for s in slots if feat.loc[s, "role"] == "core" and not feat.loc[s, "defensive"])
    regime = "risk-on" if n_risk == n else ("risk-off" if n_risk == 0 else "mixed")
    return {"slots": slots, "reasons": reasons, "regime": regime}


def target_weights(slots: list[str]) -> dict[str, float]:
    w: dict[str, float] = {}
    for s in slots:
        w[s] = w.get(s, 0) + 1 / len(slots)
    return w


# ------------------------------------------------------------------ holdings & orders
def load_holdings() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "holdings.csv")
    return df.set_index("id") if len(df) else pd.DataFrame(columns=["units", "avg_cost_eur"]).rename_axis("id")


def propose_orders(feat: pd.DataFrame, weights: dict[str, float], holdings: pd.DataFrame,
                   cash: float, cfg: dict | None = None, uni: dict | None = None) -> dict:
    """Whole-unit orders. Sells first (anything not in target), then buys with cash,
    most-underweight first. Small drifts inside the rebalance band are left alone."""
    cfg = cfg or load_config()
    uni = uni or load_universe()
    p, c = cfg["portfolio"], cfg["costs"]
    prices = feat["price"]
    value = {i: float(holdings.loc[i, "units"] * prices.get(i, float("nan"))) for i in holdings.index if i in prices}
    total = sum(v for v in value.values() if v == v) + cash
    orders, notes = [], []

    # sells
    for i, v in value.items():
        if i not in weights and holdings.loc[i, "units"] > 0:
            units = int(holdings.loc[i, "units"])
            orders.append({"side": "SELL", "id": i, "units": units, "price": prices[i],
                           "value": units * prices[i], "tob": units * prices[i] * uni["by_id"][i]["tob"],
                           "reason": "no longer a target"})
            cash += units * prices[i] * (1 - uni["by_id"][i]["tob"]) - c["commission_per_order"]
    # drift-based trims (only if outside the band)
    for i, w in weights.items():
        cur = value.get(i, 0.0) / total if total else 0
        if cur - w > p["rebalance_band"] and i in prices:
            excess = (cur - w) * total
            units = int(excess // prices[i])
            if units > 0:
                orders.append({"side": "SELL", "id": i, "units": units, "price": prices[i],
                               "value": units * prices[i], "tob": units * prices[i] * uni["by_id"][i]["tob"],
                               "reason": f"overweight by {100*(cur-w):.0f} pts (band {100*p['rebalance_band']:.0f})"})
                cash += units * prices[i] * (1 - uni["by_id"][i]["tob"]) - c["commission_per_order"]
    # buys: most underweight first, whole units, within cash
    def underweight(i):
        cur = value.get(i, 0.0) / total if total else 0
        return weights[i] - cur
    for i in sorted(weights, key=underweight, reverse=True):
        if i not in prices or prices[i] != prices[i]:
            continue
        px = prices[i]
        unit_cost = px * (1 + uni["by_id"][i]["tob"] + c["spread_pct"] / 2) + c["commission_per_order"]
        gap_eur = max(underweight(i), 0) * total
        want = int(gap_eur // unit_cost) if total > 0 else int(cash // unit_cost)
        if want == 0 and underweight(i) > 0 and cash >= unit_cost:
            want = 1  # at least one unit if we are underweight and can afford it
        units = min(want, int(cash // unit_cost))
        if units > 0:
            orders.append({"side": "BUY", "id": i, "units": units, "price": px, "value": units * px,
                           "tob": units * px * uni["by_id"][i]["tob"],
                           "reason": f"underweight by {100*underweight(i):.0f} pts" if total else "initial buy"})
            cash -= units * unit_cost
        elif underweight(i) > 0.02:
            notes.append(f"{i}: one unit costs €{unit_cost:.2f}, cash left €{cash:.2f} — carried to next month")
    return {"orders": orders, "cash_after": cash, "notes": notes, "portfolio_value": total}
