#!/usr/bin/env python3
"""Scout CLI.

  python run.py fetch                  download / refresh prices into data/prices.csv
  python run.py report                 compute signals, decision, orders → docs/index.html
  python run.py backtest [--sweep]     run the backtest (and the robustness sweep)
  python run.py daily                  fetch + backtest(cached) + report   (what GitHub Actions runs)
  python run.py selftest               everything on synthetic data, no network needed

Decisions are taken at month-end and frozen in data/decision_YYYY-MM.json; the daily
run only refreshes prices and shows a preview of what a decision today would be.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from scout import ROOT, load_config, load_universe
from scout import data as D
from scout.signals import compute_features, month_ends
from scout.portfolio import select_targets, target_weights, load_holdings, propose_orders
from scout.backtest import run_backtest, sweep
from scout.report import render, write_page

DATA = ROOT / "data"
BT_FILE = DATA / "backtest.pkl"


def _decision_path(month_end: pd.Timestamp) -> Path:
    return DATA / f"decision_{month_end.strftime('%Y-%m')}.json"


def standing_decision(prices: pd.DataFrame, cfg: dict, uni: dict, current: list[str], freeze: bool = True) -> dict:
    """Load the frozen month-end decision, or take it now if this is the first run after month-end."""
    me = month_ends(prices)
    last_me = me.index[-1]
    today = pd.Timestamp(date.today())
    # the last completed month-end: the last month-end date that is not in the current month
    completed = [d for d in me.index if d.to_period("M") < today.to_period("M")]
    if not completed:
        completed = [last_me]
    d = completed[-1]
    path = _decision_path(d)
    if freeze and path.exists():
        dec = json.loads(path.read_text())
        # allow holdings changes to be reflected without changing slots
        return dec
    feat = compute_features(prices, d, cfg, uni)
    sel = select_targets(feat, current, cfg, uni)
    dec = {"asof": str(d.date()), "slots": sel["slots"], "reasons": sel["reasons"], "regime": sel["regime"],
           "taken_on": str(today.date())}
    if freeze:
        path.write_text(json.dumps(dec, indent=2))
        print(f"  decision for {d.date()} taken and frozen → {path.name}: {sel['slots']} ({sel['regime']})")
    return dec


def cmd_report(prices: pd.DataFrame, synthetic=False, title="Scout") -> str:
    cfg, uni = load_config(), load_universe()
    holdings = load_holdings()
    current = [i for i in holdings.index if holdings.loc[i, "units"] > 0]
    dec = standing_decision(prices, cfg, uni, current, freeze=not synthetic)
    feat_now = compute_features(prices, None, cfg, uni)
    # preview: what would the decision be if today were month-end?
    prev = select_targets(feat_now, current, cfg, uni)
    tw = target_weights(dec["slots"])
    cash = float(cfg["portfolio"]["monthly_contribution"]) + float(_cash_file())
    orders = propose_orders(feat_now, tw, holdings, cash, cfg, uni)
    meta = D.load_meta() if not synthetic else {"updated": str(date.today()), "sources": {}}
    meta["last_price_date"] = str(prices.index[-1].date())
    bt = pickle.loads(BT_FILE.read_bytes()) if BT_FILE.exists() else None
    ctx = {"features_now": feat_now, "decision": dec, "orders": orders, "holdings": holdings,
           "target_weights": tw, "current_ids": current, "cfg": cfg, "meta": meta, "backtest": bt,
           "preview_slots": prev["slots"], "preview_differs": sorted(prev["slots"]) != sorted(dec["slots"]),
           "synthetic": synthetic, "title": title}
    out = write_page(render(ctx))
    # machine-readable state next to the page
    (ROOT / "docs" / "state.json").write_text(json.dumps({
        "updated": str(date.today()), "decision": dec, "orders": orders["orders"],
        "cash_after": orders["cash_after"], "preview_slots": prev["slots"],
        "ranking": feat_now[["role", "score", "adj_score", "trend_ok", "absmom", "dist_sma", "price"]]
                     .reset_index().to_dict(orient="records")}, indent=2, default=str))
    print(f"  decision {dec['asof']}: {dec['slots']} · preview today: {prev['slots']} · orders: "
          f"{[(o['side'], o['id'], o['units']) for o in orders['orders']]}")
    return out


def _cash_file() -> float:
    p = ROOT / "cash.txt"
    try:
        return float(p.read_text().strip()) if p.exists() else 0.0
    except ValueError:
        return 0.0


def cmd_backtest(prices: pd.DataFrame, do_sweep=False):
    cfg, uni = load_config(), load_universe()
    bt = run_backtest(prices, cfg, uni)
    m = bt["metrics"]
    print(f"  {'':12s} {'CAGR':>7s} {'Vol':>7s} {'Sharpe':>7s} {'MaxDD':>7s}")
    for k in ("scout", "global_etf", "60_40"):
        x = m[k]
        print(f"  {k:12s} {x['cagr']*100:6.1f}% {x['vol']*100:6.1f}% {x['sharpe']:7.2f} {x['maxdd']*100:6.1f}%")
    print(f"  trades/yr {bt['trades_per_year']:.1f} · risk-off {bt['months_risk_off']*100:.0f}% of months · {bt['start'].date()} → {bt['end'].date()}")
    BT_FILE.write_bytes(pickle.dumps(bt))
    if do_sweep:
        sw = sweep(prices, cfg, uni)
        sw.to_csv(DATA / "sweep.csv", index=False)
        print(sw.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return bt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "report", "backtest", "daily", "selftest"])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        print("synthetic prices…")
        prices = D.make_synthetic()
        print("backtest…"); cmd_backtest(prices, do_sweep=a.sweep)
        print("report…"); out = cmd_report(prices, synthetic=True, title="Scout (self-test, synthetic data)")
        BT_FILE.unlink(missing_ok=True)   # never let a synthetic backtest leak into a real page
        print(f"OK → {out}")
        return
    if a.cmd == "fetch":
        D.update_prices(force=a.force); return
    if a.cmd == "daily":
        print("fetch…"); prices = D.update_prices(force=a.force)
        if not BT_FILE.exists() or date.today().day <= 3:
            print("backtest…"); cmd_backtest(prices)
        print("report…"); print("OK →", cmd_report(prices)); return
    prices = D.load_prices()
    if a.cmd == "backtest":
        cmd_backtest(prices, do_sweep=a.sweep)
    elif a.cmd == "report":
        print("OK →", cmd_report(prices))


if __name__ == "__main__":
    main()
