"""Render docs/index.html — the daily page — from features, decision, orders and backtest."""
from __future__ import annotations

import html
import json
from datetime import date

import numpy as np
import pandas as pd

from . import ROOT

PALETTE = {"scout": ("#2a78d6", "#3987e5"), "global_etf": ("#eb6834", "#d95926"), "blend": ("#4a3aa7", "#9085e9")}
LABELS = {"scout": "Scout (satellite alone)", "global_etf": "Global equity ETF", "blend": "Core + satellite"}


def pct(x, d=1, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:+.{d}f}%" if sign else f"{x*100:.{d}f}%"


def eur(x):
    return "—" if x is None or x != x else f"€{x:,.2f}"


def _pill(text, kind):
    return f'<span class="pill {kind}">{html.escape(text)}</span>'


def _svg_chart(series: dict[str, pd.Series], width=860, height=300) -> str:
    """Three indexed equity curves, one scale, hover crosshair. Growth of 1."""
    pad_l, pad_r, pad_t, pad_b = 48, 16, 12, 28
    df = pd.DataFrame(series).dropna(how="all").ffill()
    if df.empty:
        return "<p class='small'>No backtest yet.</p>"
    x0, x1 = df.index[0], df.index[-1]
    ymin, ymax = float(df.min().min()), float(df.max().max())
    ymin, ymax = min(ymin, 0.9), ymax * 1.03
    def X(t): return pad_l + (t - x0) / (x1 - x0) * (width - pad_l - pad_r)
    def Y(v): return pad_t + (ymax - v) / (ymax - ymin) * (height - pad_t - pad_b)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="Growth of 1 euro: scout versus benchmarks">']
    # grid + y labels
    ticks = np.linspace(ymin, ymax, 5)
    for t in ticks:
        y = Y(t)
        parts.append(f'<line x1="{pad_l}" x2="{width-pad_r}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l-6}" y="{y+4:.1f}" class="tick" text-anchor="end">{t:.2f}</text>')
    # x labels: each year
    for yr in range(x0.year, x1.year + 1):
        t = pd.Timestamp(year=yr, month=1, day=1)
        if x0 <= t <= x1:
            parts.append(f'<text x="{X(t):.1f}" y="{height-8}" class="tick" text-anchor="middle">{yr}</text>')
    for k, s in df.items():
        pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in s.dropna().items())
        parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--s-{k})" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        last_t, last_v = s.dropna().index[-1], float(s.dropna().iloc[-1])
        parts.append(f'<circle cx="{X(last_t):.1f}" cy="{Y(last_v):.1f}" r="4" fill="var(--s-{k})" stroke="var(--surface)" stroke-width="2"/>')
    # hover layer
    data = [{"t": str(t.date()), **{k: (None if pd.isna(v) else round(float(v), 4)) for k, v in row.items()}} for t, row in df.iterrows()]
    xs = [round(X(t), 1) for t in df.index]
    parts.append(f'<line id="xh" x1="0" x2="0" y1="{pad_t}" y2="{height-pad_b}" class="xhair" style="opacity:0"/>')
    parts.append(f'<rect x="{pad_l}" y="{pad_t}" width="{width-pad_l-pad_r}" height="{height-pad_t-pad_b}" fill="transparent" id="hit"/>')
    parts.append("</svg>")
    parts.append(f'<div id="tip" class="tip" hidden></div>')
    parts.append(f'<script>window.__chart={{xs:{json.dumps(xs)},data:{json.dumps(data)},labels:{json.dumps(LABELS)}}};</script>')
    return "".join(parts)


def render(ctx: dict) -> str:
    feat: pd.DataFrame = ctx["features_now"]
    dec = ctx["decision"]
    orders = ctx["orders"]
    bt = ctx.get("backtest")
    meta = ctx["meta"]
    today = date.today().isoformat()

    # ------------------------------------------------------------- ranking rows
    def row(i, f, slot_ids):
        role = f["role"]
        if i in slot_ids and role != "watch":
            verdict = _pill("HOLD" if i in ctx["current_ids"] else "BUY", "bull")
        elif i in ctx["current_ids"] and i not in slot_ids:
            verdict = _pill("SELL", "bear")
        elif role == "core" and f["eligible"]:
            verdict = _pill("eligible", "neutral")
        elif role == "watch":
            verdict = _pill("watch", "accent")
        elif role == "benchmark":
            verdict = _pill("benchmark", "neutral")
        elif f["defensive"] and role == "defensive":
            verdict = _pill("defensive", "neutral")
        else:
            verdict = ""
        why = []
        if f["score"] != f["score"]:
            why.append(f"only {f['months']} months of history")
        else:
            if role in ("core",) and not f["trend_ok"]:
                why.append("below 10-month average" if not f["above_sma"] else "above average only this month")
            if role in ("core",) and not f["absmom"]:
                why.append("12-month return below cash")
        reason = dec["reasons"].get(i, "; ".join(why))
        tob = f"{f['tob']*100:.2f}%" + (" ?" if f["verify"] else "")
        return (f"<tr class='{'dim' if role=='watch' else ''}'><td class='num'>{int(f['rank']) if f['rank']==f['rank'] else '—'}</td>"
                f"<td><b>{i}</b><br><span class='small'>{html.escape(f['name'])}</span></td>"
                f"<td class='num'>{pct(f['score'],1,True)}</td><td class='num'>{pct(f.get('r3'),1,True)}</td>"
                f"<td class='num'>{pct(f.get('r12'),1,True)}</td>"
                f"<td class='num {'g' if f['trend_ok'] else 'r'}'>{pct(f['dist_sma'],1,True)}</td>"
                f"<td class='c'>{'✓' if f['absmom'] else '✗'}</td><td class='num'>{pct(f['high_ratio'],0)}</td>"
                f"<td class='num'>{tob}</td><td class='num'>{eur(f['price'])}</td><td>{verdict}</td>"
                f"<td class='small'>{html.escape(reason)}</td></tr>")
    slot_ids = set(dec["slots"])
    order = ["core", "defensive", "benchmark", "watch"]
    fs = feat.copy()
    fs["_o"] = fs["role"].map({r: i for i, r in enumerate(order)})
    fs = fs.sort_values(["_o", "adj_score"], ascending=[True, False])
    ranking = "".join(row(i, f, slot_ids) for i, f in fs.iterrows())

    # ------------------------------------------------------------- holdings / orders
    h = ctx["holdings"]
    tw = ctx["target_weights"]
    total = orders["portfolio_value"]
    hrows = []
    ch = ctx.get("core_holdings")
    if ch is not None and len(ch):
        for i in ch.index:
            units = int(ch.loc[i, "units"]); px = feat["price"].get(i, np.nan); val = units * px if px == px else 0
            hrows.append(f"<tr><td><b>{i}</b> <span class='pill accent'>core</span></td><td class='num'>{units}</td><td class='num'>{eur(val)}</td>"
                         f"<td class='num'>{pct(val/total if total else 0,0)}</td><td class='num'>—</td><td class='num'>—</td></tr>")
    sat_total = total - orders.get("core_value", 0)
    for i in sorted(set(h.index) | set(tw)):
        units = int(h.loc[i, "units"]) if i in h.index else 0
        px = feat["price"].get(i, np.nan)
        val = units * px if px == px else 0
        cur_w = val / sat_total if sat_total else 0
        hrows.append(f"<tr><td><b>{i}</b> <span class='pill neutral'>scout</span></td><td class='num'>{units}</td><td class='num'>{eur(val)}</td>"
                     f"<td class='num'>{pct(cur_w,0)}</td><td class='num'>{pct(tw.get(i,0),0)}</td>"
                     f"<td class='num {'r' if abs(cur_w-tw.get(i,0))>ctx['cfg']['portfolio']['rebalance_band'] else ''}'>{pct(cur_w-tw.get(i,0),0,True)}</td></tr>")
    orows = "".join(
        f"<tr><td>{_pill(o['side'], 'bull' if o['side']=='BUY' else 'bear')}</td><td><b>{o['id']}</b></td>"
        f"<td class='num'>{o['units']}</td><td class='num'>{eur(o['price'])}</td><td class='num'>{eur(o['value'])}</td>"
        f"<td class='num'>{eur(o['tob'])}</td><td class='small'>{html.escape(o['reason'])}</td></tr>"
        for o in orders["orders"]) or "<tr><td colspan='7' class='small'>Nothing to do this month.</td></tr>"
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in orders["notes"])

    # ------------------------------------------------------------- alerts
    alerts = []
    for i in ctx["current_ids"]:
        if i in feat.index and feat.loc[i, "dd12"] == feat.loc[i, "dd12"] and feat.loc[i, "dd12"] < -ctx["cfg"]["portfolio"]["midmonth_alert_drawdown"]:
            alerts.append(f"{i} is {pct(feat.loc[i,'dd12'],0)} below its 12-month peak — no action until month-end, but be aware.")
    stale = [k for k, v in meta.get("sources", {}).items() if (pd.Timestamp(today) - pd.Timestamp(v["last"])).days > 7]
    if stale:
        alerts.append("Price data older than a week for: " + ", ".join(stale))
    if ctx.get("preview_differs"):
        alerts.append("If today were month-end the slots would be " + ", ".join(ctx["preview_slots"]) +
                      " — different from the standing decision. Decisions are only taken at month-end.")
    alert_html = "".join(f"<div class='callout warn'><p>{html.escape(a)}</p></div>" for a in alerts)

    # ------------------------------------------------------------- backtest
    bt_html = "<p class='small'>Backtest not run yet (python run.py backtest).</p>"
    if bt:
        m = bt["metrics"]
        def mrow(k):
            x = m[k]
            return (f"<tr><td><span class='swatch' style='background:var(--s-{k})'></span>{LABELS[k]}</td>"
                    f"<td class='num'>{pct(x.get('cagr'))}</td><td class='num'>{pct(x.get('vol'))}</td>"
                    f"<td class='num'>{x.get('sharpe', float('nan')):.2f}</td><td class='num'>{pct(x.get('maxdd'))}</td>"
                    f"<td class='num'>{x.get('final', float('nan')):.2f}</td></tr>")
        q = bt["quarterly"].tail(12)
        qrows = "".join(f"<tr><td>{d.year} Q{(d.month-1)//3+1}</td><td class='num {'g' if r.scout>=0 else 'r'}'>{pct(r.scout,1,True)}</td>"
                        f"<td class='num {'g' if r.global_etf>=0 else 'r'}'>{pct(r.global_etf,1,True)}</td>"
                        f"<td class='num {'g' if r.blend>=0 else 'r'}'>{pct(r.blend,1,True)}</td></tr>" for d, r in q.iterrows())
        wy = bt.get("worst_years")
        wrows = "".join(f"<tr><td>{y}</td><td class='num r'>{pct(r.global_etf,1)}</td><td class='num r'>{pct(r.scout,1)}</td><td class='num r'>{pct(r.blend,1)}</td></tr>"
                        for y, r in wy.iterrows()) if wy is not None and len(wy) else "<tr><td colspan='4' class='small'>No year with a >10% drawdown in the benchmark.</td></tr>"
        mrow6040 = (f"<tr><td><span class='swatch' style='background:var(--muted)'></span>60/40</td><td class='num'>{pct(m['60_40'].get('cagr'))}</td>"
                    f"<td class='num'>{pct(m['60_40'].get('vol'))}</td><td class='num'>{m['60_40'].get('sharpe', float('nan')):.2f}</td>"
                    f"<td class='num'>{pct(m['60_40'].get('maxdd'))}</td><td class='num'>{m['60_40'].get('final', float('nan')):.2f}</td></tr>")
        bt_html = f"""
        <p class='small'>{bt['start'].date()} → {bt['end'].date()} · month-end rebalancing · costs: half-spread, TOB per side, commission as % of a €{ctx['cfg']['costs']['typical_order_eur']} order ·
        {bt['trades_per_year']:.1f} trades/yr · risk-off {pct(bt['months_risk_off'],0)} of months · {'synthetic data — for testing the code only' if ctx.get('synthetic') else 'real prices'}</p>
        <div class='chartwrap'>{_svg_chart({'scout': bt['curve'], 'global_etf': bt['global_etf'], 'blend': bt['blend']})}</div>
        <div class='legend'>{''.join(f"<span><i style='background:var(--s-{k})'></i>{v}</span>" for k, v in LABELS.items())}</div>
        <div class='tablewrap'><table><tr><th>Series</th><th class='num'>CAGR</th><th class='num'>Volatility</th><th class='num'>Sharpe</th><th class='num'>Max drawdown</th><th class='num'>Growth of 1</th></tr>
        {mrow('blend')}{mrow('scout')}{mrow('global_etf')}{mrow6040}</table></div>
        <p class='small'>Core + satellite = {bt['core_share']:.0%} permanent global ETF, {1-bt['core_share']:.0%} scout, no rebalancing between the two — the shape of a monthly contributor who splits each deposit.</p>
        <h3>Bad years: how far each fell from its peak</h3>
        <div class='tablewrap'><table><tr><th>Year</th><th class='num'>Global ETF</th><th class='num'>Scout</th><th class='num'>Core + satellite</th></tr>{wrows}</table></div>
        <h3>Last 12 quarters</h3>
        <div class='tablewrap'><table><tr><th>Quarter</th><th class='num'>Scout</th><th class='num'>Global ETF</th><th class='num'>Core + satellite</th></tr>{qrows}</table></div>"""

    verify = [i for i, f in feat.iterrows() if f["verify"]]
    cash_r = feat.attrs.get("cash_r12")
    dec_date = dec["asof"]
    next_dec = (pd.Timestamp(dec_date) + pd.offsets.MonthEnd(1)).date()

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scout — {html.escape(ctx['title'])}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--paper:#F2F4F6;--surface:#fff;--ink:#141C24;--ink-2:#3A4652;--muted:#6B7783;--line:#D6DCE2;--line-2:#E6EAEE;--accent:#2C5E8A;--accent-soft:#E3ECF4;--bull:#1E8A5A;--bull-soft:#E2F2EA;--bear:#C43D35;--bear-soft:#F8E4E2;--warn:#9A6A12;--warn-soft:#F7EDD6;--code:#EEF1F4;
--s-scout:{PALETTE['scout'][0]};--s-global_etf:{PALETTE['global_etf'][0]};--s-blend:{PALETTE['blend'][0]}}}
@media (prefers-color-scheme:dark){{:root{{--paper:#0F151B;--surface:#161E26;--ink:#E8ECF0;--ink-2:#C4CCD4;--muted:#8A96A2;--line:#2A343E;--line-2:#222B34;--accent:#7FB0DC;--accent-soft:#1A2A3A;--bull:#4CC08A;--bull-soft:#15302A;--bear:#E8776F;--bear-soft:#3A1F1E;--warn:#E0B25A;--warn-soft:#332A16;--code:#1D262F;
--s-scout:{PALETTE['scout'][1]};--s-global_etf:{PALETTE['global_etf'][1]};--s-blend:{PALETTE['blend'][1]}}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:"Source Serif 4",Georgia,serif;font-size:16px;line-height:1.5}}
h1,h2,h3{{font-family:Archivo,Arial,sans-serif;margin:0;line-height:1.15;letter-spacing:-.01em}}h1{{font-size:2rem}}h2{{font-size:1.4rem;margin-top:2.6rem;padding-top:1rem;border-top:2px solid var(--ink)}}h3{{font-size:1.05rem;margin-top:1.4rem}}
.wrap{{max-width:1120px;margin:0 auto;padding:2rem 1.2rem 4rem}}p{{max-width:70ch}}.small{{font-size:.82rem;color:var(--muted)}}
.eyebrow{{font-family:Archivo,sans-serif;font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);font-weight:600}}
.meta{{display:flex;flex-wrap:wrap;gap:.5rem 1.5rem;font-family:Archivo,sans-serif;font-size:.8rem;color:var(--muted);margin:.8rem 0 1.2rem}}.meta b{{color:var(--ink)}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin:1rem 0}}.tiles div{{background:var(--surface);padding:.9rem 1rem}}.tiles .k{{font-family:Archivo,sans-serif;font-weight:700;font-size:1.4rem;font-variant-numeric:tabular-nums}}.tiles .l{{font-family:Archivo,sans-serif;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}}
table{{border-collapse:collapse;width:100%;font-size:.88rem}}.tablewrap{{overflow-x:auto;margin:.8rem 0}}th{{font-family:Archivo,sans-serif;font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;text-align:left;color:var(--muted);padding:.45rem .55rem;border-bottom:2px solid var(--line)}}td{{padding:.5rem .55rem;border-bottom:1px solid var(--line-2);vertical-align:top}}
.num{{text-align:right;font-family:"IBM Plex Mono",monospace;font-size:.8rem;font-variant-numeric:tabular-nums;white-space:nowrap}}th.num{{text-align:right}}.c{{text-align:center}}tr.dim td{{color:var(--ink-2)}}.g{{color:var(--bull)}}.r{{color:var(--bear)}}
.pill{{display:inline-block;font-family:Archivo,sans-serif;font-size:.68rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;padding:.15em .5em;border-radius:2px;white-space:nowrap}}.pill.bull{{background:var(--bull-soft);color:var(--bull)}}.pill.bear{{background:var(--bear-soft);color:var(--bear)}}.pill.neutral{{background:var(--code);color:var(--ink-2)}}.pill.accent{{background:var(--accent-soft);color:var(--accent)}}.pill.warn{{background:var(--warn-soft);color:var(--warn)}}
.callout{{border-left:4px solid var(--accent);background:var(--surface);padding:.8rem 1rem;margin:1rem 0;max-width:72ch}}.callout.warn{{border-color:var(--warn)}}.callout p{{margin:.3rem 0}}
.chartwrap{{position:relative;background:var(--surface);border:1px solid var(--line);padding:.6rem;margin:1rem 0}}.chart{{width:100%;height:auto;display:block}}.grid{{stroke:var(--line-2);stroke-width:1}}.tick{{font-family:"IBM Plex Mono",monospace;font-size:11px;fill:var(--muted)}}.xhair{{stroke:var(--muted);stroke-dasharray:3 3}}
.tip{{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--line);padding:.4rem .6rem;font-family:"IBM Plex Mono",monospace;font-size:.75rem;box-shadow:0 2px 8px rgba(0,0,0,.12);white-space:nowrap}}
.legend{{display:flex;gap:1.2rem;font-family:Archivo,sans-serif;font-size:.8rem;color:var(--ink-2)}}.legend i,.swatch{{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:.4rem;vertical-align:-1px}}
ul{{max-width:70ch}}
</style></head><body><div class="wrap">
<span class="eyebrow">Scout · monthly ETF ranking · Belgium</span>
<h1>{html.escape(ctx['title'])}</h1>
<div class="meta"><span>Page updated <b>{today}</b></span><span>Prices to <b>{meta.get('last_price_date','—')}</b></span><span>Standing decision <b>{dec_date}</b></span><span>Next decision <b>{next_dec}</b></span><span>Regime <b>{dec['regime']}</b></span></div>
<div class="tiles">
<div><div class="k">{(ctx.get('core_id') + ' + ') if ctx.get('core_share') else ''}{', '.join(dict.fromkeys(dec['slots'])) or '—'}</div><div class="l">{'core + ' if ctx.get('core_share') else ''}scout slots</div></div>
<div><div class="k">{eur(total)}</div><div class="l">portfolio value incl. cash</div></div>
<div><div class="k">{len(orders['orders'])}</div><div class="l">proposed orders</div></div>
<div><div class="k">{pct(cash_r,1,True) if cash_r==cash_r else '—'}</div><div class="l">cash benchmark, 12 months</div></div>
</div>
{alert_html}
<h2>Proposed orders</h2>
<p class="small">Whole units only. {f"{ctx['core_share']:.0%} of each contribution goes to the core ETF {ctx['core_id']} (never sold); the rest is run by the scout. " if ctx.get('core_share') else ''}TOB is the Belgian transaction tax you pay per side. Cash after orders: {eur(orders['cash_after'])}.</p>
<div class="tablewrap"><table><tr><th>Side</th><th>ETF</th><th class="num">Units</th><th class="num">Price</th><th class="num">Value</th><th class="num">TOB</th><th>Reason</th></tr>{orows}</table></div>
{('<ul class="small">'+notes+'</ul>') if notes else ''}
<h3>Holdings vs target</h3>
<p class="small">Scout weights and targets are relative to the scout bucket only.</p>
<div class="tablewrap"><table><tr><th>ETF</th><th class="num">Units</th><th class="num">Value</th><th class="num">Weight</th><th class="num">Target</th><th class="num">Drift</th></tr>{''.join(hrows) or "<tr><td colspan='6' class='small'>No holdings yet — edit holdings.csv after your first purchase.</td></tr>"}</table></div>
<h2>Ranking</h2>
<p class="small">Score = average of 3/6/9/12-month returns minus a penalty for high-tax ETFs. Trend = distance from the 10-month average (green: above for two month-ends). Cash ✓ = 12-month return beats short-term government bonds. 52w = price as % of its 52-week high. TOB "?" = inferred, verify on your broker's order screen.</p>
<div class="tablewrap"><table><tr><th class="num">#</th><th>ETF</th><th class="num">Score</th><th class="num">3m</th><th class="num">12m</th><th class="num">Trend</th><th>Cash</th><th class="num">52w</th><th class="num">TOB</th><th class="num">Price</th><th>Verdict</th><th>Why</th></tr>{ranking}</table></div>
<h2>How the rule has done</h2>
{bt_html}
<h2>Notes</h2>
<ul class="small">
<li>Decisions are taken once a month on the last trading day; this page refreshes daily so you can see them coming. Mid-month alerts never trigger orders.</li>
<li>Expected edge: about the return of a global equity ETF with roughly half the worst drawdown. Expect to lag buy-and-hold in most bull years.</li>
<li>Tax classes marked "?" ({', '.join(verify) or 'none'}) were inferred from the fund sponsor; confirm each once on your broker's order ticket and edit universe.yaml.</li>
<li>Data: {', '.join(sorted(set(v['source'].split(':')[0] for v in meta.get('sources',{}).values()))) or 'n/a'}. Nothing here is investment advice; it is a rules engine you configured.</li>
</ul>
</div>
<script>
(function(){{const c=window.__chart;if(!c)return;const svg=document.querySelector('svg.chart'),hit=document.getElementById('hit'),xh=document.getElementById('xh'),tip=document.getElementById('tip'),wrap=document.querySelector('.chartwrap');
function near(px){{let lo=0,hi=c.xs.length-1;while(hi-lo>1){{const m=(lo+hi)>>1;(c.xs[m]<px?lo=m:hi=m)}}return (px-c.xs[lo]<c.xs[hi]-px)?lo:hi}}
hit.addEventListener('mousemove',e=>{{const pt=svg.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;const p=pt.matrixTransform(svg.getScreenCTM().inverse());const i=near(p.x);const d=c.data[i];xh.setAttribute('x1',c.xs[i]);xh.setAttribute('x2',c.xs[i]);xh.style.opacity=1;
tip.innerHTML='<b>'+d.t+'</b><br>'+Object.keys(c.labels).map(k=>d[k]==null?'':c.labels[k]+': '+d[k].toFixed(2)).filter(Boolean).join('<br>');tip.hidden=false;const r=wrap.getBoundingClientRect();tip.style.left=Math.min(e.clientX-r.left+12,r.width-170)+'px';tip.style.top=(e.clientY-r.top+12)+'px'}});
hit.addEventListener('mouseleave',()=>{{xh.style.opacity=0;tip.hidden=true}});}})();
</script></body></html>"""


def write_page(html_text: str) -> str:
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    return str(out)
