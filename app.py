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

# ---------------- Chart ----------------
x = win_c["Datetime"].str[:16]
fig = go.Figure(go.Candlestick(x=x, open=win_c.Open, high=win_c.High, low=win_c.Low, close=win_c.Close,
                               name="4H price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
for col, colour in (("SMA21", "#1e88e5"), ("SMA50", "#fb8c00"), ("SMA200", "#8e24aa")):
    fig.add_trace(go.Scatter(x=x, y=win_c[col], mode="lines", name=col.replace("SMA", "SMA "), line=dict(color=colour, width=1.8)))
for side, colour, sym, ycol, mult, pos in (("BUY", "#00a651", "triangle-up", "Low", 0.994, "bottom center"),
                                          ("SELL", "#e53935", "triangle-down", "High", 1.006, "top center")):
    s = win_s[win_s.Signal == side].copy()
    if s.empty:
        continue
    s["x"] = s["Datetime"].str[:16]
    y = win_c.set_index(x)[ycol].reindex(s["x"]).values * mult
    fig.add_trace(go.Scatter(x=s["x"], y=y, mode="markers+text", name=f"{side} signals",
                             marker=dict(symbol=sym, size=11, color=colour),
                             text=s["Strength"] + " " + side, textposition=pos, textfont=dict(size=9, color=colour),
                             hovertext=s["Reason"], hoverinfo="text"))
fig.update_layout(height=650, xaxis=dict(type="category", nticks=12, rangeslider_visible=False),
                  yaxis_title="Price (₹)", margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

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
