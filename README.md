# Scout

A monthly ETF ranking bot for a Belgian retail investor. It downloads prices for a
fixed list of UCITS ETFs, ranks them on trend and momentum, decides once a month which
three to hold (or when to sit in short-term government bonds), proposes whole-unit
orders for your monthly contribution, and publishes a page you can check any day.
You place the trades at your broker.

Background and the reasoning behind every dial: the *Scout Bot Blueprint* and the
*Candle Bot Field Guide* (your Claude artifacts).

## Quick start (no network needed)

```bash
pip install -r requirements.txt
python run.py selftest          # synthetic data → docs/index.html, proves the pipeline works
```

## Real data, on your machine

```bash
python run.py fetch             # downloads ~20 ETFs from Yahoo (Stooq fallback) into data/prices.csv
python run.py backtest --sweep  # the rule vs a global ETF and 60/40, plus a robustness sweep
python run.py report            # today's ranking, the standing decision, proposed orders → docs/index.html
```

`python run.py daily` does fetch + backtest (first days of the month) + report — this is
what the GitHub job runs every weekday morning.

## Put it on GitHub so it runs by itself

1. Create a new **public** repository on GitHub (public = free unlimited Actions minutes)
   and push this folder to it.
2. In the repo: **Settings → Pages → Source: GitHub Actions**.
3. **Actions → scout daily → Run workflow** once. After ~2 minutes the page is live at
   `https://<your-user>.github.io/<repo>/`.
4. From then on it runs every weekday morning. Bookmark the page.

The job commits `data/prices.csv`, the frozen monthly decisions and `docs/` back into the
repo, so the history of what the bot decided is in git.

## Your side of the loop

- **After you buy or sell**, edit `holdings.csv` (units per ETF) and `cash.txt`
  (uninvested cash at the broker, before this month's contribution). The page then
  shows drift against target and proposes orders relative to what you really own.
- **Monthly contribution** is `portfolio.monthly_contribution` in `config.yaml`.
- **Tax classes**: entries with `verify: true` in `universe.yaml` were inferred; check the
  TOB rate on your broker's order ticket once and set `verify: false`.

## The dials (`config.yaml`)

| Dial | Default | What it does |
|---|---|---|
| `lookbacks_months` | 3/6/9/12 | ensemble momentum score = mean of these returns |
| `trend_sma_months` | 10 | above the 10-month average = uptrend |
| `trend_confirm_months` | 2 | must hold for two month-ends (whipsaw filter) |
| `absmom_months` | 12 | 12-month return must beat short-term government bonds |
| `top_n` | 3 | number of slots |
| `switch_margin` | 0.02 | a challenger must beat an incumbent by 2 points to replace it |
| `rebalance_band` | 0.05 | ignore weight drift under 5 points |
| `tob_penalty` | 0.5 | score penalty × round-trip TOB (keeps 1.32%-TOB ETFs out unless they earn it) |
| `midmonth_alert_drawdown` | 0.15 | show an alert (never an order) if a holding is >15% below its 12-month peak |
| `costs.*` | MeDirect | commission per order, spread, typical order size — used in the backtest |

## How a decision is made

1. On the last trading day of each month, for every ETF: 3/6/9/12-month returns,
   position vs the 10-month average, 12-month return vs the cash benchmark, distance
   from the 52-week high.
2. *Eligible* = core ETF, above its 10-month average for two month-ends, beating cash.
3. Rank eligible ETFs by score minus the tax penalty. Incumbents keep their slot unless
   beaten by the switch margin. Empty slots go to the better of the two defensive
   bond ETFs by 12-month return.
4. Orders: sell what is no longer a target, then buy whole units of the most underweight
   target with the cash available. Leftover cash carries to next month.
5. The decision is frozen in `data/decision_YYYY-MM.json`. Daily runs only refresh the
   picture and show what a decision *today* would look like.

## Layout

```
run.py              CLI (fetch / report / backtest / daily / selftest)
config.yaml         the dials
universe.yaml       the ETF list with tickers, ISINs, TOB classes and roles
holdings.csv        what you own (you edit this)
cash.txt            uninvested cash (you edit this)
scout/data.py       price download, cache, synthetic data
scout/signals.py    features per ETF as-of a date
scout/portfolio.py  slot selection, target weights, whole-unit orders
scout/backtest.py   monthly simulation vs benchmarks + parameter sweep
scout/report.py     the HTML page
docs/               the published page (index.html, state.json)
data/               prices.csv, sources.json, decision_*.json
.github/workflows/  the daily job
```

## Later: the "go" button

The order-proposal step returns a plain list of `{side, id, units}`. An execution adapter
for a broker with an API (Saxo OpenAPI or Interactive Brokers) plugs in there; the rest
does not change. Not built yet, on purpose — the scout should earn it first.

Nothing here is investment advice. It is a rules engine you configured.
