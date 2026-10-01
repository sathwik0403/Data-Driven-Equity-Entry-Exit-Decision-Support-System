"""Streamlit panels for the live / intraday features. Kept separate from the original historical-CSV flow."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import intraday as I
import sources as S
from context import DEFAULT_BANDS, build_context

DISCLAIMER = ("Decision-support only: signals describe rule conditions that were met, they do not predict prices and are not "
              "an instruction to trade.")


@st.cache_data(ttl=60, show_spinner="Fetching intraday candles…")
def get_intraday(source: str, symbol: str):
    df, info = S.yf_intraday(symbol) if source.startswith("yfinance") else S.nse_intraday(symbol)
    return df, info


@st.cache_data(ttl=30, show_spinner=False)
def get_nse_quote(symbol: str):
    return S.nse_quote(symbol)


@st.cache_data(ttl=900, show_spinner="Fetching context indicators…")
def get_context(sector, nifty, bands):
    return build_context(sector, nifty, dict(bands))


def badge(status: str) -> str:
    return {"available": "🟢", "stale": "🟠", "market closed": "⚪", "unverified time": "🟠", "unavailable": "🔴"}.get(status, "⚪")


def render_live(cfg: dict, hist: pd.DataFrame):
    sym, src = cfg["symbol"], cfg["source"]
    st.subheader(f"Live / intraday · {sym}")
    st.warning("Live intraday data comes from an **unofficial, best-effort** public source and may be delayed, incomplete or "
               "blocked. It is shown separately from the historical CSV (4H) analysis.", icon="⚠️")

    # ---- current price: snapshot source if NSEPython, else last completed 1m bar ----
    raw, info = get_intraday(src, sym)
    q = S.with_status(get_nse_quote(sym).copy()) if src.startswith("NSEPython") else None
    d1 = I.prep_1m(raw) if raw is not None else None
    if q is None:
        if d1 is not None and len(d1):
            q = S.with_status(dict(source=info["source"], price=float(d1.Close.iat[-1]),
                                   asof=d1.index[-1] + pd.Timedelta(minutes=1), tick=None, interval="last completed 1m bar"))
        else:
            q = dict(source=info["source"], price=None, status="unavailable", reason=info.get("reason", "no data"))

    c1, c2, c3 = st.columns(3)
    if q["price"] is None:
        c1.metric("Current price", "unavailable")
        c2.metric("Status", f"{badge('unavailable')} unavailable")
        st.error(f"{q.get('reason', 'No data')}")
    else:
        c1.metric("Current price", f"₹{q['price']:,.2f}")
        c2.metric("Status", f"{badge(q['status'])} {q['status']}")
        c3.metric("Latest update", q["asof"].strftime("%d %b %H:%M:%S IST") if q.get("asof") is not None else "not provided")
        st.caption(f"Source: {q['source']} · basis: {q.get('interval')} · refresh every {cfg['refresh']}s. "
                   "Market-closed status ignores NSE holidays (no holiday feed wired in).")
    if src.startswith("NSEPython"):
        st.info("NSEPython gives **snapshots only**: no verified 1-minute OHLCV, so candles, ORB, key-level setups and ATR are "
                "**unavailable** with this source. Pick yfinance in the sidebar for candles, or supply a broker/vendor feed.")
    if d1 is None or d1.empty:
        st.error("**ORB unavailable: intraday data missing.** " + info.get("reason", ""))
        return

    # ---- timeframe: only intervals the source can supply (genuine 1m base -> 5m, 15m) ----
    tf = st.radio("Candle timeframe", ["1m", "5m", "15m"], horizontal=True, index=2, key="tf")
    m = int(tf[:-1])
    c = I.resample(d1, m)
    cc = c[c.Complete]
    px = q["price"] if q["price"] is not None else float(d1.Close.iat[-1])
    a = I.atr(cc, 14)
    st.caption(f"{len(d1):,} one-minute bars, {d1.index[0].strftime('%d %b')} → {d1.index[-1].strftime('%d %b %H:%M')} IST · candles aligned to 09:15 IST · "
               f"last candle {'complete' if c.Complete.iat[-1] else 'still forming (excluded from confirmations)'}. "
               f"Missing 1m bars inside a candle: {int((m - c.Bars).clip(lower=0).sum())}.")

    # ---- ORB ----
    orb = I.opening_range(d1, c, cfg["orb_close"], cfg["orb_vol"], 20)
    st.markdown("#### 15-minute Opening Range Breakout (09:15–09:30 IST)")
    if orb["status"] != "ready":
        st.info(orb["status"])
    else:
        st.write(f"Session {orb['session']} · range high **{orb['high']:.2f}**, low **{orb['low']:.2f}** · now: {orb['live_bias']}")
        if orb["rows"]:
            r = pd.DataFrame(orb["rows"])
            st.dataframe(r, hide_index=True, use_container_width=True)
        else:
            st.caption("No completed candle has broken the range yet (not confirmed).")

    # ---- levels + proximity ----
    levels = I.key_levels(d1, c, hist, cfg["swing_k"], cfg["levels"])
    if orb["status"] == "ready":
        levels += [("ORB high", orb["high"], "opening range"), ("ORB low", orb["low"], "opening range")]
    tick = q.get("tick") or cfg["tick"]
    band, why = I.proximity_band(cfg["prox_mode"], px, a, tick, cfg["fixed"], cfg["pct"], cfg["atr_mult"])
    st.markdown("#### Key levels and proximity")
    st.caption(f"ATR(14) on **{tf}** completed candles = {a:.2f} (average recent price movement, not direction). "
               f"'Near' band = {band:.2f} ({why}). Tick size {tick:g} "
               f"({'read from source' if q.get('tick') else 'assumed: not provided by this source, editable in sidebar'}). "
               "Defaults are a heuristic starting value that needs backtesting.")
    watch, setups = I.evaluate_levels(levels, c, px, band, tf, cfg["buffer"], cfg["lvl_vol"])
    events = I.store_and_invalidate(sym, tf, setups, cc)
    if levels:
        lv = pd.DataFrame(levels, columns=["Level", "Price", "Note"])
        lv["Distance"] = px - lv["Price"]
        lv["Near?"] = lv["Distance"].abs() <= band
        st.dataframe(lv.sort_values("Price", ascending=False), hide_index=True, use_container_width=True)
    if watch:
        st.dataframe(pd.DataFrame(watch), hide_index=True, use_container_width=True)
        st.caption("WATCH / NEAR LEVEL is an alert only. Proximity alone does not create a BUY or SELL.")
    st.markdown("#### Level setups (completed candles only)")
    if setups:
        for s in setups:
            icon = "🟢" if s["side"] == "BUY" else "🔴"
            st.write(f"{icon} **{s['side']} · {s['setup']}** at {s['level']} ({s['level_price']:.2f}) · price {s['price']:.2f} · "
                     f"{s['tf']} candle {s['candle_ts'].strftime('%d %b %H:%M')} IST")
            st.caption("Conditions met: " + s["rules"])
    else:
        st.caption("WAIT / NO SIGNAL: no level setup satisfied its rules on the latest completed candle.")
    if len(events):
        with st.expander("Stored setup events (de-duplicated per level, candle and setup; ACTIVE → INVALIDATED if a close goes through the level)"):
            st.dataframe(events, hide_index=True, use_container_width=True)
    st.caption(DISCLAIMER)

    # ---- chart ----
    days = sorted(c.index.normalize().unique())[-cfg["sessions"]:]
    v = c[c.index.normalize().isin(days)]
    fig = go.Figure(go.Candlestick(x=v.index, open=v.Open, high=v.High, low=v.Low, close=v.Close, name=f"{tf} live",
                                   increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
    if orb["status"] == "ready":
        t0 = pd.Timestamp(f"{orb['session']} 09:15", tz=S.IST)
        fig.add_shape(type="rect", x0=t0, x1=v.index[-1], y0=orb["low"], y1=orb["high"], fillcolor="rgba(30,136,229,0.12)", line_width=0)
        if orb["rows"]:
            fig.add_trace(go.Scatter(x=[orb["ts"]], y=[v.loc[orb["ts"], "Close"] if orb["ts"] in v.index else px], mode="markers+text",
                                     marker=dict(size=13, symbol="star", color="#fb8c00"), text=[f"ORB {orb['rows'][0]['Direction']} "
                                     f"({'confirmed' if orb['confirmed'] else 'unconfirmed'})"], textposition="top center", name="ORB breakout"))
    for nm, lvl, _ in levels:
        fig.add_hline(y=lvl, line_dash="dot", line_width=1, line_color="rgba(128,128,128,0.7)", annotation_text=nm, annotation_font_size=9)
    fig.update_layout(height=620, dragmode="pan", margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"),
                      xaxis=dict(rangeslider_visible=False, rangebreaks=[dict(bounds=[15.5, 9.25], pattern="hour"), dict(bounds=["sat", "mon"])]),
                      yaxis=dict(side="right", title="Price (₹)"))
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})


def render_context(cfg: dict):
    st.subheader("Pre-open / global context")
    st.caption("Context only, kept apart from technical, ORB and key-level signals. Nothing here produces BUY/SELL. "
               "Thresholds are prototype heuristics, not proven predictors; edit them below.")
    with st.expander("Interpretation thresholds (editable)"):
        b = {}
        cols = st.columns(4)
        for i, (k, v) in enumerate(DEFAULT_BANDS.items()):
            b[k] = cols[i % 4].number_input(k, value=float(v), step=0.05, key=f"band_{k}")
    df, us, raw = get_context(cfg["sector"], cfg["nifty"], tuple(sorted(b.items())))
    df = df.copy()
    df["Value"] = df["Value"].map(lambda x: "" if x is None else str(x))
    df["Status"] = df["Status"].map(lambda s: f"{badge(s)} {s}")
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.write(f"**Previous U.S. session (Dow/Nasdaq/S&P, completed sessions only):** {us}. This is a cue, not a prediction of India's open.")
    if raw is not None:
        st.markdown("**FII/DII (as returned by NSEPython, backward-looking)**")
        st.dataframe(raw, hide_index=True, use_container_width=True)
    st.caption("Values use daily closes from Yahoo Finance (unofficial). Unavailable rows are excluded from the summary; "
               "no score is computed.")
