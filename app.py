import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from signals import add_smas, generate_signals, load_candles

st.set_page_config(page_title="Trading BUY/SELL Signal Generator", page_icon="📈", layout="wide")
st.title("📈 TRADING BUY/SELL SIGNAL GENERATOR")
st.caption("Multi-factor decision-support tool · Daily SMA (21/50/200) built from 4-hour closes · "
           "Fibonacci · ATH drawdown · Psychological levels")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("1. Data")
    up = st.file_uploader("Upload 4H OHLCV CSV (TradingView export)", type="csv")
    stock = st.text_input("Stock name", "ITC")
    st.header("2. Parameters")
    days = st.slider("Days shown on chart / table", 30, 500, 120, 10)
    k = st.slider("Swing sensitivity (bars each side)", 3, 15, 8)
    sma_tol = st.slider("Near-SMA tolerance (%)", 0.5, 3.0, 1.5, 0.1) / 100
    fib_tol = st.slider("Fib level tolerance (± ratio)", 0.01, 0.06, 0.03, 0.005)
    psy_step = st.selectbox("Psychological level step (₹)", [25, 50, 100], index=1)
    psy_tol = st.slider("Psychological zone (%)", 0.5, 4.0, 2.0, 0.1) / 100
    horizon = st.slider("Back-test horizon (4H bars)", 5, 40, 15)

default = Path(__file__).parent / "ITC_4hour_clean.csv"
if up is None and not default.exists():
    st.info("Upload a 4-hour CSV to begin."); st.stop()


@st.cache_data(show_spinner="Computing signals…")
def run(file_bytes, params):
    raw = pd.read_csv(file_bytes if file_bytes is not None else default)
    d = add_smas(load_candles(raw))
    return d, generate_signals(d, **dict(params))


params = (("k", k), ("sma_tol", sma_tol), ("fib_tol", fib_tol), ("psy_step", psy_step),
          ("psy_tol", psy_tol), ("horizon", horizon))
d, sig = run(up, params)

# ---------------- SQL layer (SQLite) ----------------
con = sqlite3.connect(":memory:")
d.assign(Datetime=d["Datetime"].astype(str)).drop(columns="Day").to_sql("candles", con, index=False)
sig.assign(Datetime=sig["Datetime"].astype(str)).to_sql("signals", con, index=False)
cutoff = str(d["Datetime"].max() - pd.Timedelta(days=days))
win_c = pd.read_sql("SELECT * FROM candles WHERE Datetime >= ? ORDER BY Datetime", con, params=(cutoff,))
win_s = pd.read_sql("SELECT Datetime, Price, Signal, Strength, Reason, Confluence, Trend "
                    "FROM signals WHERE Datetime >= ? ORDER BY Datetime DESC", con, params=(cutoff,))

# ---------------- Headline metrics ----------------
last = d.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{stock} last close", f"₹{last.Close:,.2f}")
c2.metric("Trend (latest signal context)", sig["Trend"].iat[-1] if len(sig) else "n/a")
c3.metric("SMA 50 / 200", f"{last.SMA50:,.1f} / {last.SMA200:,.1f}" if pd.notna(last.SMA200) else "warming up")
c4.metric("Signals in window", len(win_s))

# ---------------- Chart (full history; initial view = last N days, zoom/pan freely) ----------------
xd = d["Datetime"]
fig = go.Figure(go.Candlestick(x=xd, open=d.Open, high=d.High, low=d.Low, close=d.Close,
                               name="4H price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
for col, colour in (("SMA21", "#1e88e5"), ("SMA50", "#fb8c00"), ("SMA200", "#8e24aa")):
    fig.add_trace(go.Scatter(x=xd, y=d[col], mode="lines", name=col.replace("SMA", "SMA "), line=dict(color=colour, width=1.8)))
key = d.set_index(d["Datetime"].dt.strftime("%Y-%m-%d %H:%M"))
for side, colour, sym, ycol, mult, pos in (("BUY", "#00a651", "triangle-up", "Low", 0.994, "bottom center"),
                                          ("SELL", "#e53935", "triangle-down", "High", 1.006, "top center")):
    s = sig[sig.Signal == side]
    if s.empty:
        continue
    y = key[ycol].reindex(s["Datetime"].dt.strftime("%Y-%m-%d %H:%M")).values * mult
    fig.add_trace(go.Scatter(x=s["Datetime"], y=y, mode="markers+text", name=f"{side} signals",
                             marker=dict(symbol=sym, size=11, color=colour),
                             text=s["Strength"] + " " + side, textposition=pos, textfont=dict(size=9, color=colour),
                             hovertext=s["Reason"], hoverinfo="text"))

# hide weekends, holidays and non-trading hours so candles sit side by side (no gaps)
all_days = pd.date_range(d["Datetime"].min().normalize(), d["Datetime"].max().normalize())
missing = all_days.difference(d["Datetime"].dt.normalize().unique()).strftime("%Y-%m-%d").tolist()
v = d[d["Datetime"] >= d["Datetime"].max() - pd.Timedelta(days=days)]
pad = (v.High.max() - v.Low.min()) * 0.06
fig.update_layout(
    height=680, dragmode="pan", hovermode="x", legend=dict(orientation="h"),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(rangeslider_visible=False, tickangle=0,
               tickformatstops=[
                   dict(dtickrange=[None, 86400000 * 1.5], value="%d %b\n%H:%M"),
                   dict(dtickrange=[86400000 * 1.5, 86400000 * 20], value="%a %d %b"),
                   dict(dtickrange=[86400000 * 20, "M6"], value="%d %b '%y"),
                   dict(dtickrange=["M6", None], value="%b %Y")], showspikes=True, spikemode="across", spikethickness=1,
               range=[v["Datetime"].min(), d["Datetime"].max() + pd.Timedelta(days=2)],
               rangebreaks=[dict(values=missing), dict(bounds=[14, 5], pattern="hour")],
               fixedrange=False),
    yaxis=dict(side="right", title="Price (₹)", fixedrange=False, showspikes=True, spikethickness=1,
               range=[v.Low.min() - pad, v.High.max() + pad], showgrid=True,
               minor=dict(showgrid=True, ticks="outside", gridcolor="rgba(128,128,128,0.15)")),
)
st.plotly_chart(fig, use_container_width=True, config={
    "scrollZoom": True, "displaylogo": False, "doubleClick": "reset+autosize",
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"]})
st.caption("🖱️ Scroll wheel = zoom · drag = pan · drag on the price axis (right) or date axis (bottom) to stretch/squeeze "
           "that axis · toolbar: Zoom box, Autoscale, Reset · double-click = reset view")

# ---------------- Button + table ----------------
if st.button("GENERATE BUY/SELL SIGNAL", type="primary"):
    st.session_state["show"] = True

if st.session_state.get("show"):
    st.subheader("Signal table")
    if win_s.empty:
        st.warning("No signals in this window — try widening tolerances in the sidebar.")
    else:
        tbl = pd.DataFrame({"Date / time (IST)": win_s.Datetime.str[:16], "Price": win_s.Price,
                            "Possible signal": win_s.Strength + " " + win_s.Signal,
                            "Reason for signal (confluence factors)": win_s.Reason,
                            "# factors": win_s.Confluence})
        paint = lambda v: ("background-color:#c8f7c5;color:#0b6b0b;font-weight:bold" if str(v).endswith("BUY")
                           else "background-color:#f9c6c6;color:#a11212;font-weight:bold" if str(v).endswith("SELL") else "")
        st.dataframe(tbl.style.map(paint, subset=["Possible signal"]).format({"Price": "₹{:.2f}"}),
                     use_container_width=True, hide_index=True)
        st.download_button("Download table (CSV)", tbl.to_csv(index=False), f"{stock}_signals.csv", "text/csv")

    st.subheader(f"How reliable were these signals? (SQL back-test, {horizon} bars ≈ {horizon/3:.0f} trading days ahead)")
    bt = pd.read_sql(
        """SELECT Signal, Strength, COUNT(*) AS Signals,
                  ROUND(AVG(Fwd_Return_Pct), 2) AS "Avg return %",
                  ROUND(100.0 * AVG(CASE WHEN Fwd_Return_Pct > 0 THEN 1.0 ELSE 0.0 END), 1) AS "Hit rate %"
           FROM signals WHERE Fwd_Return_Pct IS NOT NULL
           GROUP BY Signal, Strength ORDER BY Signal, "Hit rate %" DESC""", con)
    st.dataframe(bt, use_container_width=True, hide_index=True)
    st.caption("Return is signed: positive = the signal was right (price rose after BUY / fell after SELL). "
               "Past hit-rates do not guarantee future results; this is decision support, not investment advice.")
