"""Price data: Yahoo Finance via yfinance (with alternates), Stooq fallback, CSV cache.

Everything downstream works on one DataFrame: daily adjusted closes in EUR-listed
terms, one column per ETF id, a DatetimeIndex, missing values forward-filled
(never back-filled, so an ETF only "exists" from its first real print).
"""
from __future__ import annotations

import io
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import ROOT, load_config, load_universe

CACHE = ROOT / load_config()["data"]["cache_dir"]
CACHE.mkdir(exist_ok=True)
PRICES_FILE = CACHE / "prices.csv"
LONG_FILE = CACHE / "prices_long.csv"      # prices with proxy backfill, used by the backtest
META_FILE = CACHE / "sources.json"


# ---------------------------------------------------------------- fetchers
def _yahoo(ticker: str, start: str) -> pd.Series | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, auto_adjust=True, progress=False, threads=False)
            if df is None or len(df) == 0:
                return None
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            s = close.dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            return s.astype(float) if len(s) > 50 else None
        except Exception as e:  # rate limits come and go; back off and retry
            wait = 5 * (attempt + 1)
            print(f"  yahoo {ticker}: {type(e).__name__}: {e} — retry in {wait}s")
            time.sleep(wait)
    return None


def _stooq(ticker: str) -> pd.Series | None:
    """Stooq uses lowercase tickers with .nl / .de / .uk suffixes."""
    sym = ticker.lower().replace(".as", ".nl").replace(".l", ".uk")
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200 or "Date" not in r.text[:100]:
            return None
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")
        s = df["Close"].dropna().astype(float)
        return s if len(s) > 50 else None
    except Exception:
        return None


def fetch_series(etf: dict, start: str) -> tuple[pd.Series | None, str]:
    for t in etf["tickers"]:
        s = _yahoo(t, start)
        if s is not None:
            return s, f"yahoo:{t}"
    for t in etf["tickers"]:
        s = _stooq(t)
        if s is not None:
            return s, f"stooq:{t}"
    return None, "none"


# ---------------------------------------------------------------- public API
def update_prices(start: str = "2008-01-01", force: bool = False) -> pd.DataFrame:
    """Download (or refresh) every ETF in the universe and write the cache."""
    uni = load_universe()
    old = load_prices() if PRICES_FILE.exists() and not force else None
    cols, sources = {}, {}
    for etf in uni["etfs"]:
        eid = etf["id"]
        s, src = fetch_series(etf, start)
        if s is None and old is not None and eid in old:
            s, src = old[eid].dropna(), "cache"
        if s is None:
            print(f"  {eid}: NO DATA from any source — skipped")
            continue
        cols[eid] = s
        sources[eid] = {"source": src, "first": str(s.index.min().date()),
                        "last": str(s.index.max().date()), "rows": int(len(s))}
        print(f"  {eid:5s} {src:18s} {sources[eid]['first']} → {sources[eid]['last']} ({len(s)} rows)")
    prices = pd.DataFrame(cols).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.ffill()
    prices.index.name = "date"
    prices.to_csv(PRICES_FILE, float_format="%.6f")
    meta = {"updated": str(date.today()), "sources": sources}
    if load_config()["backtest"].get("backfill"):
        long, info = backfill(prices)
        long.to_csv(LONG_FILE, float_format="%.6f")
        meta["backfill"] = info
    META_FILE.write_text(json.dumps(meta, indent=2))
    return prices


def splice(real: pd.Series, proxy: pd.Series, min_days: int = 60) -> pd.Series | None:
    """Prepend proxy history (scaled to meet the real series at its first print). Real data wins."""
    first = real.index[0]
    before = proxy.loc[:first]
    if len(before) < min_days:
        return None
    anchor = proxy.reindex([first], method="nearest").iloc[0]
    scaled = before[before.index < first] * (real.iloc[0] / anchor)
    col = pd.concat([scaled, real])
    return col[~col.index.duplicated(keep="last")].sort_index()


def backfill(prices: pd.DataFrame, start: str = "2000-01-01") -> tuple[pd.DataFrame, dict]:
    """Extend each ETF backwards with a longer-history proxy (USD twin or index), converted
    to EUR and scaled so the two series meet at the ETF's first print. Only history *before*
    inception is taken from the proxy; real ETF prices are never altered."""
    uni = load_universe()
    fx = None
    fx_sym = uni.get("fx")
    if fx_sym:
        fx = _yahoo(fx_sym, start)  # EURUSD=X → USD per EUR
        if fx is None:
            fx = _stooq("eurusd")
    out = prices.copy()
    info = {}
    for etf in uni["etfs"]:
        eid, proxies = etf["id"], etf.get("proxies") or []
        if eid not in out or not proxies:
            continue
        real = out[eid].dropna()
        for t in proxies:
            px = _yahoo(t, start)
            if px is None:
                continue
            usd = not (t.endswith(".L") or t.endswith(".DE") or t.endswith(".AS") or t.endswith(".PA"))
            if usd:
                if fx is None:
                    continue
                px = (px / fx.reindex(px.index).ffill()).dropna()
            col = splice(real, px)
            if col is None:
                continue
            first = real.index[0]
            out = out.reindex(out.index.union(col.index))
            out[eid] = col.reindex(out.index)
            info[eid] = {"proxy": t, "from": str(col.index[0].date()), "to": str(first.date()), "usd": usd}
            print(f"  {eid:5s} backfilled with {t:22s} {info[eid]['from']} → {info[eid]['to']}")
            break
    out = out.sort_index().ffill()
    out.index.name = "date"
    return out, info


def load_prices(long: bool = False) -> pd.DataFrame:
    f = LONG_FILE if long and LONG_FILE.exists() else PRICES_FILE
    df = pd.read_csv(f, index_col="date", parse_dates=True)
    return df.sort_index().ffill()


def load_meta() -> dict:
    return json.loads(META_FILE.read_text()) if META_FILE.exists() else {"updated": None, "sources": {}}


# ---------------------------------------------------------------- offline test data
def make_synthetic(seed: int = 7, start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """Plausible daily series for every ETF so the pipeline can be tested without network.

    Each ETF gets its own drift/vol and a shared market factor; bonds are low-vol,
    gold uncorrelated. Not for research — only for exercising the code paths.
    """
    rng = np.random.default_rng(seed)
    uni = load_universe()
    end = end or str(date.today())
    idx = pd.bdate_range(start, end)
    n = len(idx)
    market = rng.normal(0.00025, 0.009, n)
    # a couple of bear episodes so the trend rule has something to do
    for a, b in [(0.20, 0.27), (0.55, 0.60), (0.80, 0.84)]:
        i0, i1 = int(a * n), int(b * n)
        market[i0:i1] -= 0.0025
    profiles = {
        "Global equity": (0.00035, 0.9, 0.004), "US equity": (0.00045, 1.0, 0.005),
        "Europe equity": (0.00025, 0.9, 0.005), "Emerging markets": (0.00025, 1.0, 0.008),
        "Small cap": (0.00035, 1.1, 0.006), "Gold": (0.00025, 0.0, 0.010),
    }
    cols = {}
    for etf in uni["etfs"]:
        sleeve = etf["sleeve"]
        if sleeve in profiles:
            mu, beta, eps = profiles[sleeve]
        elif "government" in sleeve.lower() or "Treasury" in sleeve or "inflation" in sleeve.lower():
            mu, beta, eps = (0.00008, -0.05, 0.003 if "1-3" in sleeve else 0.0045)
        elif "corporate" in sleeve.lower() or "high yield" in sleeve.lower():
            mu, beta, eps = (0.00012, 0.2, 0.003)
        else:
            mu, beta, eps = (0.0004, 1.0, 0.007)
        r = mu + beta * market + rng.normal(0, eps, n)
        # give each ETF a different inception so the "insufficient history" path is exercised
        first = rng.integers(0, int(0.35 * n)) if etf["role"] == "watch" else 0
        level = float(rng.choice([28, 45, 70, 110, 180, 320, 560]))
        px = level * np.exp(np.cumsum(r))
        px[:first] = np.nan
        cols[etf["id"]] = px
    df = pd.DataFrame(cols, index=idx)
    df.index.name = "date"
    return df
