#!/usr/bin/env python3
"""
EGX Buyer/Seller Pressure Map
-----------------------------
Daily screen of Egyptian Exchange stocks:
  * Buyer-driven rise  : price up, closes near the high, volume well above normal
  * Seller-driven fall : price down, closes near the low, volume well above normal

Outputs (in ./output):
  egx_map.html         interactive treemap + tables (open in any browser)
  egx_signals.csv      full table for the latest session
  history/YYYY-MM-DD.csv   daily archive

Usage:
  python egx_map.py            # live data from Yahoo Finance
  python egx_map.py --demo     # synthetic data, to test the pipeline offline
"""
import argparse
import datetime as dt
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------------- settings
CFG = dict(
    avg_days=20,          # window for average volume
    min_change=0.5,       # % move required to call a rise/fall
    min_clv=0.3,          # close location value threshold (closes in top/bottom ~35% of range)
    min_rvol=1.5,         # relative volume threshold
    min_value=500_000,    # min traded value (EGP) to ignore illiquid noise
    max_stale_days=5,     # drop tickers whose last bar is older than this
    top_n=10,
)
BASE = Path(__file__).parent
OUT = BASE / "output"


# ---------------------------------------------------------------- data
def load_universe() -> pd.DataFrame:
    return pd.read_csv(BASE / "egx_tickers.csv")


def fetch_live(tickers: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(
        tickers, period="4mo", interval="1d", group_by="ticker",
        auto_adjust=False, progress=False, threads=True,
    )
    data = {}
    for t in tickers:
        try:
            df = raw[t].dropna(subset=["Close"]) if len(tickers) > 1 else raw.dropna(subset=["Close"])
            if len(df) > CFG["avg_days"] + 1:
                data[t] = df
        except Exception:
            pass  # ticker missing on Yahoo
    return data


def fetch_demo(tickers: list[str]) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=80)
    data = {}
    for t in tickers:
        price = rng.uniform(5, 80)
        base_vol = rng.uniform(2e5, 5e6)
        rows = []
        for i, _ in enumerate(idx):
            r = rng.normal(0, 0.015)
            vol = base_vol * rng.lognormal(0, 0.3)
            if i == len(idx) - 1:                      # make the last day interesting
                r = rng.normal(0, 0.03)
                vol *= rng.choice([0.8, 1.0, 2.0, 3.0])
            o = price
            c = price * (1 + r)
            hi = max(o, c) * (1 + abs(rng.normal(0, 0.006)))
            lo = min(o, c) * (1 - abs(rng.normal(0, 0.006)))
            rows.append((o, hi, lo, c, vol))
            price = c
        data[t] = pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])
    return data


# ---------------------------------------------------------------- signals
def compute(data: dict[str, pd.DataFrame], universe: pd.DataFrame) -> pd.DataFrame:
    meta = universe.set_index("ticker")
    rows = []
    for t, d in data.items():
        last, prev = d.iloc[-1], d.iloc[-2]
        age = (pd.Timestamp.today().normalize() - pd.Timestamp(d.index[-1]).normalize()).days
        if age > CFG["max_stale_days"]:
            continue
        rng_ = last.High - last.Low
        chg = (last.Close / prev.Close - 1) * 100
        # Close Location Value: +1 closed at the high, -1 closed at the low.
        # Zero range (locked at limit-up/down) -> use direction of the move.
        if rng_ > 1e-9:
            clv = ((last.Close - last.Low) - (last.High - last.Close)) / rng_
        else:
            clv = float(np.sign(chg))
        avg_vol = d.Volume.iloc[-CFG["avg_days"] - 1:-1].mean()
        rvol = last.Volume / avg_vol if avg_vol > 0 else np.nan
        value = last.Close * last.Volume
        score = clv * min(rvol, 3) if np.isfinite(rvol) else 0.0

        liquid = value >= CFG["min_value"]
        if liquid and chg >= CFG["min_change"] and clv >= CFG["min_clv"] and rvol >= CFG["min_rvol"]:
            state = "Buyer-driven rise"
        elif liquid and chg <= -CFG["min_change"] and clv <= -CFG["min_clv"] and rvol >= CFG["min_rvol"]:
            state = "Seller-driven fall"
        elif chg > 0:
            state = "Up, weak conviction"
        elif chg < 0:
            state = "Down, weak conviction"
        else:
            state = "Flat"

        rows.append(dict(
            ticker=t,
            name=meta.loc[t, "name"] if t in meta.index else t,
            sector=meta.loc[t, "sector"] if t in meta.index else "Other",
            date=pd.Timestamp(d.index[-1]).date().isoformat(),
            close=round(float(last.Close), 3),
            change_pct=round(float(chg), 2),
            clv=round(float(clv), 2),
            rel_volume=round(float(rvol), 2) if np.isfinite(rvol) else np.nan,
            value_traded=float(value),
            pressure=round(float(score), 2),
            state=state,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- output
def table_html(df: pd.DataFrame, title: str, color: str) -> str:
    if df.empty:
        return f"<h3 style='color:{color}'>{title}</h3><p>None today.</p>"
    body = "".join(
        f"<tr><td>{html.escape(r.ticker.replace('.CA',''))}</td><td>{html.escape(r['name'])}</td>"
        f"<td>{r.close:.2f}</td><td style='color:{color}'>{r.change_pct:+.2f}%</td>"
        f"<td>{r.rel_volume:.1f}x</td><td>{r.clv:+.2f}</td><td>{r.value_traded/1e6:,.1f}M</td></tr>"
        for _, r in df.iterrows()
    )
    return (f"<h3 style='color:{color}'>{title}</h3><table><tr><th>Ticker</th><th>Company</th>"
            f"<th>Close</th><th>Change</th><th>Rel. vol</th><th>CLV</th><th>Value (EGP)</th></tr>{body}</table>")


def build_html(df: pd.DataFrame, demo: bool) -> str:
    plot = df[df.value_traded > 0].copy()
    plot["label"] = plot.ticker.str.replace(".CA", "", regex=False)
    plot["value_m"] = (plot.value_traded / 1e6).round(2)
    fig = px.treemap(
        plot, path=[px.Constant("EGX"), "sector", "label"], values="value_traded",
        color="pressure", color_continuous_scale="RdYlGn",
        range_color=(-3, 3), color_continuous_midpoint=0,
        hover_data={"name": True, "change_pct": ":.2f", "rel_volume": ":.2f",
                    "clv": ":.2f", "value_m": True, "state": True, "pressure": ":.2f"},
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{customdata[1]:+.2f}%", textposition="middle center",
        root_color="rgba(0,0,0,0)",
    )
    fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=640,
                      coloraxis_colorbar=dict(title="Pressure<br>sell ← → buy"))

    buyers = df[df.state == "Buyer-driven rise"].sort_values("pressure", ascending=False).head(CFG["top_n"])
    sellers = df[df.state == "Seller-driven fall"].sort_values("pressure").head(CFG["top_n"])
    session = df.date.max()
    n_up = int((df.change_pct > 0).sum()); n_dn = int((df.change_pct < 0).sum())
    banner = "<p style='background:#fde68a;padding:8px'>DEMO MODE: synthetic data, not real prices.</p>" if demo else ""

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>EGX Pressure Map</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0 auto;max-width:1200px;padding:16px;color:#111}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin-bottom:20px}}
th,td{{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left}} th{{background:#f4f4f4}}
.small{{color:#666;font-size:13px}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h2>Egyptian Exchange: Buyer vs Seller Pressure</h2>
{banner}
<p class="small">Session: <b>{session}</b> &middot; {n_up} advancing / {n_dn} declining &middot;
Box size = value traded &middot; Color = pressure score (green = buyers in control, red = sellers).</p>
{fig.to_html(full_html=False, include_plotlyjs="cdn")}
<div class="grid"><div>{table_html(buyers, "Buyer-driven rises", "#15803d")}</div>
<div>{table_html(sellers, "Seller-driven falls", "#b91c1c")}</div></div>
<p class="small"><b>How it is measured.</b> Daily data cannot show who is "really" buying or selling, so this uses a proxy.
CLV = ((Close-Low)-(High-Close))/(High-Low): +1 means the stock closed at its high (buyers won the session), -1 at its low.
Pressure = CLV x min(relative volume, 3). A stock is flagged only if the move is at least {CFG['min_change']}%,
|CLV| &ge; {CFG['min_clv']}, relative volume &ge; {CFG['min_rvol']}x its {CFG['avg_days']}-day average and traded value &ge; EGP {CFG['min_value']:,}.
Stocks locked at the daily price limit have no range, so direction is used instead. For screening only, not investment advice.
Data: Yahoo Finance, possibly delayed or incomplete.</p>
</body></html>"""


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic data")
    args = ap.parse_args()

    uni = load_universe()
    tickers = uni.ticker.tolist()
    data = fetch_demo(tickers) if args.demo else fetch_live(tickers)
    if not data:
        print("No data downloaded. Check your internet connection or Yahoo Finance availability.", file=sys.stderr)
        return 1

    missing = sorted(set(tickers) - set(data))
    df = compute(data, uni)
    if df.empty:
        print("All tickers were stale or empty (market holiday or data outage?).", file=sys.stderr)
        return 1

    OUT.mkdir(exist_ok=True)
    (OUT / "history").mkdir(exist_ok=True)
    df.sort_values("pressure", ascending=False).to_csv(OUT / "egx_signals.csv", index=False)
    df.to_csv(OUT / "history" / f"{df.date.max()}.csv", index=False)
    (OUT / "egx_map.html").write_text(build_html(df, args.demo), encoding="utf-8")

    b = df[df.state == "Buyer-driven rise"].sort_values("pressure", ascending=False)
    s = df[df.state == "Seller-driven fall"].sort_values("pressure")
    print(f"Session {df.date.max()}: {len(df)} stocks, {len(b)} buyer-driven rises, {len(s)} seller-driven falls")
    for label, part in (("BUYER-DRIVEN", b), ("SELLER-DRIVEN", s)):
        for _, r in part.head(CFG["top_n"]).iterrows():
            print(f"  {label:14s} {r.ticker:9s} {r.change_pct:+6.2f}%  rvol {r.rel_volume:4.1f}x  clv {r.clv:+.2f}")
    if missing:
        print(f"Not found / too little history on Yahoo ({len(missing)}): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
