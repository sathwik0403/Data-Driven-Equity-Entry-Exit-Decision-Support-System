"""Intraday logic on genuine 1-minute OHLCV. All times Asia/Kolkata; NSE session 09:15-15:30."""
import sqlite3
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

from signals import find_pivots
from sources import IST, now_ist

DB_PATH = Path(__file__).parent / "live_events.db"


def prep_1m(df: pd.DataFrame, now=None) -> pd.DataFrame:
    """tz-aware IST, de-duplicated, session hours only, still-forming minute dropped."""
    d = df.copy()
    idx = pd.DatetimeIndex(d.index)
    d.index = (idx.tz_localize("UTC") if idx.tz is None else idx).tz_convert(IST)
    d = d[~d.index.duplicated(keep="last")].sort_index().dropna(subset=["Open", "High", "Low", "Close"])
    mins = d.index.hour * 60 + d.index.minute
    d = d[(mins >= 555) & (mins < 930) & (d.index.dayofweek < 5)]
    now = now if now is not None else now_ist()
    return d[d.index + pd.Timedelta(minutes=1) <= now]


def resample(d1: pd.DataFrame, m: int) -> pd.DataFrame:
    """1m -> m-minute candles anchored at 09:15 per session. O=first, H=max, L=min, C=last, V=sum.
    Complete = the candle's window has fully elapsed within the data. Bars = 1m bars present (gaps visible)."""
    parts = []
    for _, day in d1.groupby(d1.index.normalize()):
        origin = day.index[0].normalize() + pd.Timedelta(hours=9, minutes=15)
        r = day.resample(f"{m}min", origin=origin).agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        r["Bars"] = day["Close"].resample(f"{m}min", origin=origin).count()
        r = r[r["Bars"] > 0]
        r["Complete"] = (r.index + pd.Timedelta(minutes=m)) <= day.index[-1] + pd.Timedelta(minutes=1)
        parts.append(r)
    return pd.concat(parts) if parts else pd.DataFrame()


def atr(c: pd.DataFrame, n: int = 14) -> float:
    """Wilder ATR on the given (completed) candles: average recent range, NOT direction."""
    if len(c) <= n:
        return float("nan")
    pc = c["Close"].shift()
    tr = pd.concat([c.High - c.Low, (c.High - pc).abs(), (c.Low - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().iat[-1])


def _vol_ok(c, ts, mult, lookback):
    """Volume of candle ts >= mult x mean volume of the previous `lookback` completed candles. mult 0 = off."""
    if mult <= 0:
        return True, "volume check off"
    prior = c[c.index < ts]["Volume"].tail(lookback)
    if len(prior) < 3 or prior.mean() <= 0:
        return False, "volume baseline unavailable"
    v = c.loc[ts, "Volume"]
    return bool(v >= mult * prior.mean()), f"volume {v:,.0f} vs {mult:g} x avg {prior.mean():,.0f}"


def opening_range(d1, c, require_close=True, vol_mult=1.0, vol_lookback=20, now=None):
    """15-minute ORB for the latest session. Range from 1m candles 09:15-09:29 only; never from hourly data."""
    now = now if now is not None else now_ist()
    out = dict(status="ORB unavailable: intraday data missing")
    if d1 is None or d1.empty:
        return out
    day = d1.index[-1].normalize()
    s = d1[d1.index.normalize() == day]
    if now.normalize() == day and now.time() < dtime(9, 30):
        return dict(status="Opening range forming (09:15-09:30): breakouts not evaluated yet")
    rng = s.between_time("09:15", "09:29")
    if len(rng) < 15:
        return dict(status=f"ORB unavailable: opening range has {len(rng)}/15 one-minute candles")
    hi, lo = float(rng.High.max()), float(rng.Low.min())
    out = dict(status="ready", session=day.date(), high=hi, low=lo, direction="none", rows=[])
    cc = c[(c.index.normalize() == day) & (c.index.time >= dtime(9, 30)) & c["Complete"]]
    for ts, r in cc.iterrows():
        up = (r.Close > hi) if require_close else (r.High > hi)
        dn = (r.Close < lo) if require_close else (r.Low < lo)
        if not (up or dn):
            continue
        vok, vtxt = _vol_ok(c, ts, vol_mult, vol_lookback)
        side = "LONG" if up else "SHORT"
        out["direction"] = side
        out["rows"].append({"Breakout time": ts.strftime("%H:%M"), "Direction": side, "Range high": hi, "Range low": lo,
                            "Confirmed": "YES" if vok else "NO (volume)",
                            "Rules": f"{'close' if require_close else 'wick'} beyond range: pass; {vtxt}: {'pass' if vok else 'fail'}"})
        out["ts"] = ts
        out["confirmed"] = vok
        break                                   # first breakout of the session
    last = float(s.Close.iat[-1])
    out["live_bias"] = ("LONG bias (price above range high, unconfirmed)" if last > hi else
                        "SHORT bias (price below range low, unconfirmed)" if last < lo else "Inside range")
    return out


def vwap(d1):
    s = d1[d1.index.normalize() == d1.index[-1].normalize()]
    if s.empty or s.Volume.sum() <= 0:
        return None
    tp = (s.High + s.Low + s.Close) / 3
    return float((tp * s.Volume).sum() / s.Volume.sum())


def key_levels(d1, c, hist=None, swing_k=5, cfg=None):
    """Returns list of (name, price, note). cfg toggles groups."""
    cfg = cfg or {}
    L = []
    days = sorted(d1.index.normalize().unique()) if d1 is not None and not d1.empty else []
    if len(days) >= 2 and cfg.get("prev", True):
        p = d1[d1.index.normalize() == days[-2]]
        t = days[-2].strftime("%d %b")
        L += [(f"Prev-day high ({t})", float(p.High.max()), ""), (f"Prev-day low ({t})", float(p.Low.min()), ""),
              (f"Prev-day close ({t})", float(p.Close.iat[-1]), "reference/pivot, not guaranteed S/R")]
    elif hist is not None and cfg.get("prev", True):    # fallback: historical CSV, labelled as such
        h = hist.assign(D=hist.Datetime.dt.normalize()).groupby("D").agg(H=("High", "max"), L=("Low", "min"), C=("Close", "last"))
        if len(h):
            t = h.index[-1].strftime("%d %b %Y")
            L += [(f"Prev high (CSV, {t})", float(h.H.iat[-1]), "from CSV: may be stale"),
                  (f"Prev low (CSV, {t})", float(h.L.iat[-1]), "from CSV: may be stale"),
                  (f"Prev close (CSV, {t})", float(h.C.iat[-1]), "reference only; from CSV: may be stale")]
    if cfg.get("swings", True) and c is not None and len(c[c.Complete]) > 2 * swing_k + 2:
        cd = c[c.Complete].tail(400).reset_index()
        pv = find_pivots(cd, swing_k)
        for typ, nm in (("H", "Swing high"), ("L", "Swing low")):
            for _, i, _, px in [x for x in pv if x[2] == typ][-2:]:
                L.append((f"{nm} @ {cd.iloc[i, 0].strftime('%d %b %H:%M')}", float(px), "support/resistance candidate"))
    if cfg.get("vwap", True) and d1 is not None and not d1.empty:
        v = vwap(d1)
        if v:
            L.append(("Session VWAP", v, "moves through the day"))
    return L


def proximity_band(mode, px, atr_v, tick, fixed=1.0, pct=0.10, atr_mult=0.10, ticks=2):
    """Returns (band, explanation). Default heuristic = max(ticks*tick, pct% of price, atr_mult*ATR): needs backtesting."""
    if mode == "Fixed rupees":
        return fixed, f"fixed ₹{fixed:g}"
    if mode == "% of price":
        return px * pct / 100, f"{pct:g}% of price"
    if mode == "ATR multiple":
        return (atr_mult * atr_v, f"{atr_mult:g} x ATR") if atr_v == atr_v else (float("nan"), "ATR unavailable")
    parts = {f"{ticks} ticks": ticks * tick, f"{pct:g}% price": px * pct / 100}
    if atr_v == atr_v:
        parts[f"{atr_mult:g} x ATR"] = atr_mult * atr_v
    k = max(parts, key=parts.get)
    return parts[k], f"max of ({', '.join(parts)}) -> {k}"


def evaluate_levels(levels, c, px, band, tf, buffer_pct=0.05, vol_mult=0.0, vol_lookback=20):
    """WATCH rows (live price) + setups from the latest completed candle. Proximity alone never makes BUY/SELL."""
    watch, setups = [], []
    cc = c[c.Complete]
    if len(cc) < 2 or band != band:
        return watch, setups
    k, prev = cc.iloc[-1], cc.iloc[-2]
    kts = cc.index[-1]
    for name, lvl, _ in levels:
        d = px - lvl
        if abs(d) <= band:
            watch.append({"Status": "WATCH / NEAR LEVEL", "Level": name, "Level price": lvl, "Current price": px,
                          "Distance": d, "Timeframe": tf, "As of": now_ist().strftime("%d %b %H:%M:%S IST")})
        buf = lvl * buffer_pct / 100
        vok, vtxt = _vol_ok(cc, kts, vol_mult, vol_lookback)
        checks = {
            "SUPPORT BOUNCE": ("BUY", [(f"low tested level (low {k.Low:.2f} <= {lvl + band:.2f})", k.Low <= lvl + band),
                                      (f"completed candle closed back above ({k.Close:.2f} > {lvl:.2f})", k.Close > lvl)]),
            "RESISTANCE REJECTION": ("SELL", [(f"high tested level (high {k.High:.2f} >= {lvl - band:.2f})", k.High >= lvl - band),
                                             (f"completed candle closed back below ({k.Close:.2f} < {lvl:.2f})", k.Close < lvl)]),
            "BREAKOUT UP": ("BUY", [(f"close above level+buffer ({k.Close:.2f} > {lvl + buf:.2f})", k.Close > lvl + buf),
                                    (f"previous close not already above ({prev.Close:.2f} <= {lvl:.2f})", prev.Close <= lvl),
                                    (vtxt, vok)]),
            "BREAKDOWN": ("SELL", [(f"close below level-buffer ({k.Close:.2f} < {lvl - buf:.2f})", k.Close < lvl - buf),
                                   (f"previous close not already below ({prev.Close:.2f} >= {lvl:.2f})", prev.Close >= lvl),
                                   (vtxt, vok)]),
        }
        for setup, (side, rules) in checks.items():
            if all(ok for _, ok in rules):
                setups.append(dict(setup=setup, side=side, level=name, level_price=lvl, price=float(k.Close), candle_ts=kts,
                                   tf=tf, rules="; ".join(t for t, _ in rules)))
    return watch, setups


# ---------- persistence: signal events only, de-duplicated by (symbol, tf, level, candle, setup) ----------
def _db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS live_events(
        ev_key TEXT PRIMARY KEY, symbol TEXT, tf TEXT, candle_ts TEXT, level TEXT, level_price REAL,
        setup TEXT, side TEXT, price REAL, rules TEXT, status TEXT DEFAULT 'ACTIVE')""")
    return con


def store_and_invalidate(symbol, tf, setups, cc):
    """Insert new setups (duplicates ignored); mark ACTIVE setups INVALIDATED if the latest completed close went against them."""
    con = _db()
    try:
        for s in setups:
            key = f"{symbol}|{tf}|{s['level']}|{s['candle_ts']}|{s['setup']}"
            con.execute("INSERT OR IGNORE INTO live_events(ev_key,symbol,tf,candle_ts,level,level_price,setup,side,price,rules) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (key, symbol, tf, str(s["candle_ts"]), s["level"], s["level_price"], s["setup"], s["side"], s["price"], s["rules"]))
        if len(cc):
            close = float(cc.Close.iat[-1])
            con.execute("UPDATE live_events SET status='INVALIDATED' WHERE symbol=? AND tf=? AND status='ACTIVE' AND "
                        "((side='BUY' AND ? < level_price) OR (side='SELL' AND ? > level_price)) AND candle_ts < ?",
                        (symbol, tf, close, close, str(cc.index[-1])))
        con.commit()
        return pd.read_sql("SELECT candle_ts,level,setup,side,price,status,rules FROM live_events "
                           "WHERE symbol=? AND tf=? ORDER BY candle_ts DESC LIMIT 50", con, params=(symbol, tf))
    finally:
        con.close()
