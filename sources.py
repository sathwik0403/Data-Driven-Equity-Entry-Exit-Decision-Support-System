"""Data-source adapters. Every call returns data plus an explicit status; nothing is invented or silently substituted.

NSEPython and yfinance both scrape public endpoints: UNOFFICIAL, best-effort, may be delayed, rate-limited or break.
"""
import pandas as pd

IST = "Asia/Kolkata"
NSEPY_LABEL = "NSEPython (unofficial NSE public API, snapshot quote)"
YF_LABEL = "yfinance / Yahoo Finance (unofficial, may be delayed)"


def now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=IST)


def market_open(now=None) -> bool:
    """Weekday 09:15-15:30 IST. NSE holidays are NOT detected (no verified holiday feed is wired in)."""
    now = now or now_ist()
    return now.dayofweek < 5 and 555 <= now.hour * 60 + now.minute < 930


def _find(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                return v
            r = _find(v, key)
            if r is not None:
                return r
    return None


def with_status(q: dict, stale_after_s: int = 180) -> dict:
    """Adds status: available / stale / market closed / unverified time / unavailable."""
    if q.get("price") is None:
        q["status"] = "unavailable"
    elif q.get("asof") is None:
        q["status"] = "unverified time"          # price returned but source gave no usable timestamp
    elif not market_open():
        q["status"] = "market closed"            # last known value, not live
    elif (now_ist() - q["asof"]).total_seconds() > stale_after_s:
        q["status"] = "stale"
    else:
        q["status"] = "available"
    return q


def nse_quote(symbol: str) -> dict:
    """Current price snapshot via NSEPython nse_eq(). A snapshot, NOT a stream, and NOT candles."""
    try:
        from nsepython import nse_eq
    except Exception as e:
        return dict(source=NSEPY_LABEL, price=None, reason=f"nsepython not importable: {e}")
    try:
        q = nse_eq(symbol)
        price = float(q["priceInfo"]["lastPrice"])
    except Exception as e:
        return dict(source=NSEPY_LABEL, price=None,
                    reason=f"NSE request failed or response layout differed ({e!r}). NSE often blocks cloud/server IPs.")
    try:
        asof = pd.to_datetime(_find(q, "lastUpdateTime"), format="%d-%b-%Y %H:%M:%S").tz_localize(IST)
    except Exception:
        asof = None
    tick = _find(q, "tickSize")
    return dict(source=NSEPY_LABEL, price=price, asof=asof, tick=float(tick) if tick else None,
                interval="snapshot")


def yf_intraday(symbol: str, period: str = "5d"):
    """Genuine 1-minute OHLCV from Yahoo (about 7 days of history max). Returns (df|None, info)."""
    info = dict(source=YF_LABEL, interval="1m")
    try:
        import yfinance as yf
        df = yf.download(f"{symbol}.NS", period=period, interval="1m", progress=False, auto_adjust=False)
    except Exception as e:
        return None, {**info, "reason": f"yfinance error: {e!r}"}
    if df is None or df.empty:
        return None, {**info, "reason": "Yahoo returned no 1-minute rows (symbol wrong, market data blocked, or rate limit)."}
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]], info


def nse_intraday(symbol: str):
    """NSEPython: no verified 1-minute OHLCV function. Reported honestly instead of guessing an endpoint."""
    return None, dict(source=NSEPY_LABEL, interval="n/a",
                      reason="NSEPython offers quote snapshots; no 1-minute OHLCV function was verified. "
                             "Choose yfinance for candles, or supply a broker/vendor feed (e.g. Kite Connect, Upstox, "
                             "Dhan: needs API key + access token in st.secrets).")
