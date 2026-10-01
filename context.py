"""Market-context indicators (separate from technical / ORB / key-level signals; never produces BUY/SELL).

Source: Yahoo Finance via yfinance (UNOFFICIAL). Values are daily-close based: previous completed session vs the one before.
Anything without a verified source is returned as 'unavailable' with the reason, never estimated.
"""
import pandas as pd

from sources import IST, now_ist

DEFAULT_BANDS = dict(
    idx_small=0.5, idx_notable=1.0, idx_strong=2.0,            # abs % move, US/Asian indices
    gift_flat=0.15, gift_modest=0.5, gift_notable=1.0,         # abs % gap vs previous Nifty close
    vix_calm=12.0, vix_mod=18.0, vix_elev=25.0,
    oil_small=1.0, oil_notable=2.0,
    sector_rel=0.3,                                            # percentage points vs Nifty
    breadth_pos=60.0, breadth_neg=40.0,
)

US_INDICES = {"Dow Jones": "^DJI", "Nasdaq Composite": "^IXIC", "S&P 500": "^GSPC"}
ASIA_INDICES = {"Nikkei 225": "^N225", "Hang Seng": "^HSI"}
TZ = {"^DJI": ("America/New_York", 16), "^IXIC": ("America/New_York", 16), "^GSPC": ("America/New_York", 16),
      "^N225": ("Asia/Tokyo", 15), "^HSI": ("Asia/Hong_Kong", 16)}


def size_label(pct, b, keys=("idx_small", "idx_notable", "idx_strong"), names=("small", "notable", "strong", "unusually large")):
    a = abs(pct)
    return names[0] if a < b[keys[0]] else names[1] if a < b[keys[1]] else names[2] if a < b[keys[2]] else names[3]


def _row(name, instrument, source="Yahoo Finance via yfinance (unofficial)", **kw):
    return dict(Factor=name, Instrument=instrument, Value=None, **{"Change %": None, "As of": None, "Reading": "",
                "Status": "unavailable", "Source": source, "Note": ""}, **kw)


def daily_pair(ticker: str):
    """(last_value, prev_value, last_session_date) from completed daily bars, or (None, reason)."""
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period="15d", interval="1d", auto_adjust=False)["Close"].dropna()
    except Exception as e:
        return None, f"yfinance error: {e!r}"
    if len(h) < 2:
        return None, "fewer than 2 daily rows returned"
    idx = pd.DatetimeIndex(h.index)
    tzname, close_h = TZ.get(ticker, (IST, 16))
    local_now = pd.Timestamp.now(tz=tzname)
    # drop today's bar if that exchange's session may still be running (avoids using an incomplete session)
    if idx[-1].date() == local_now.date() and local_now.hour < close_h + 0:
        h = h.iloc[:-1]
    if len(h) < 2:
        return None, "only an in-progress session available"
    return (float(h.iat[-1]), float(h.iat[-2]), h.index[-1].date()), None


def _status(session_date):
    age = (now_ist().date() - session_date).days
    return "stale" if age > 5 else "available"


def build_context(sector_ticker="^CNXFMCG", nifty_ticker="^NSEI", bands=None):
    b = {**DEFAULT_BANDS, **(bands or {})}
    rows = []

    # --- global indices: previous completed session ---
    signs = []
    for nm, tk in {**US_INDICES, **ASIA_INDICES}.items():
        r = _row(nm, tk)
        res, err = daily_pair(tk)
        if res:
            v, p, d = res
            pct = (v / p - 1) * 100
            r.update(Value=round(v, 2), **{"Change %": round(pct, 2)}, **{"As of": f"session {d}"},
                     Reading=f"{size_label(pct, b)} {'up' if pct > 0 else 'down'} move", Status=_status(d))
            if r["Status"] == "available" and nm in US_INDICES:
                signs.append(1 if pct > 0 else -1 if pct < 0 else 0)
        else:
            r["Note"] = err
        rows.append(r)
    us = ("positive" if signs and all(s > 0 for s in signs) else "negative" if signs and all(s < 0 for s in signs)
          else "mixed" if signs else "unavailable")

    # --- GIFT Nifty: no verified source wired in ---
    g = _row("GIFT Nifty gap", "GIFT Nifty futures", source="none")
    g["Note"] = ("Needs a verified GIFT Nifty quote (e.g. NSE IX / broker / vendor feed). Gap = (GIFT - prev Nifty close) / prev close; "
                 f"bands: flat within ±{b['gift_flat']}%, modest <{b['gift_modest']}%, notable <{b['gift_notable']}%, else large.")
    rows.append(g)

    # --- India VIX ---
    r = _row("India VIX", "^INDIAVIX")
    res, err = daily_pair("^INDIAVIX")
    if res:
        v, p, d = res
        lab = ("relatively calm" if v < b["vix_calm"] else "moderate" if v < b["vix_mod"] else "elevated" if v < b["vix_elev"]
               else "high expected volatility")
        r.update(Value=round(v, 2), **{"Change %": round((v / p - 1) * 100, 2), "As of": f"session {d}"},
                 Reading=lab, Status=_status(d), Note="volatility context only, not direction")
    else:
        r["Note"] = err
    rows.append(r)

    # --- Brent ---
    r = _row("Brent crude", "BZ=F (Yahoo front-month continuous; contract/exchange not verified)")
    res, err = daily_pair("BZ=F")
    if res:
        v, p, d = res
        pct = (v / p - 1) * 100
        lab = size_label(pct, b, ("oil_small", "oil_notable", "oil_notable"), ("small", "notable", "sharp", "sharp"))
        r.update(Value=round(v, 2), **{"Change %": round(pct, 2), "As of": f"session {d}"},
                 Reading=f"{lab} {'rise' if pct > 0 else 'fall'}", Status=_status(d),
                 Note="vs previous daily close (not verified as exchange settlement). Rising oil can be a broad headwind for India; "
                      "effects differ by company/sector.")
    else:
        r["Note"] = err
    rows.append(r)

    # --- USD/INR ---
    r = _row("USD/INR", "INR=X (Yahoo spot quote; NOT an NDF)")
    res, err = daily_pair("INR=X")
    if res:
        v, p, d = res
        pct = (v / p - 1) * 100
        r.update(Value=round(v, 3), **{"Change %": round(pct, 2), "As of": f"session {d}"},
                 Reading="rupee weakened" if pct > 0 else "rupee strengthened" if pct < 0 else "unchanged", Status=_status(d),
                 Note="No overnight NDF feed wired in; no universal thresholds applied. Impact differs for importers/exporters.")
    else:
        r["Note"] = err
    rows.append(r)

    # --- Sector vs Nifty ---
    r = _row("Sector vs Nifty", f"{sector_ticker} vs {nifty_ticker}")
    s, e1 = daily_pair(sector_ticker)
    n, e2 = daily_pair(nifty_ticker)
    if s and n:
        sp, npct = (s[0] / s[1] - 1) * 100, (n[0] / n[1] - 1) * 100
        rel = sp - npct
        lab = "outperforming" if rel > b["sector_rel"] else "underperforming" if rel < -b["sector_rel"] else "in line"
        r.update(Value=f"sector {sp:+.2f}% / Nifty {npct:+.2f}%", **{"Change %": round(rel, 2), "As of": f"session {s[2]} / {n[2]}"},
                 Reading=f"{lab} (relative, pp)", Status="available" if s[2] == n[2] else "stale",
                 Note="Sessions differ: not comparable" if s[2] != n[2] else f"threshold ±{b['sector_rel']} pp")
    else:
        r["Note"] = e1 or e2
    rows.append(r)

    # --- no verified source wired in ---
    for nm, note in (("Company news / events",
                      "No verified announcements feed wired in. Would show separate flags (results, guidance, contracts, regulatory, "
                      "filings after prior close) with publication time + source. Headlines never become BUY/SELL."),
                     ("Market breadth", "Needs a defined universe (e.g. Nifty 500 constituents' prev close vs close). Not computed: "
                                        f"labels would be >{b['breadth_pos']}% broad positive, <{b['breadth_neg']}% broad negative, else mixed.")):
        x = _row(nm, "n/a", source="none")
        x["Note"] = note
        rows.append(x)

    # --- FII/DII (backward looking) ---
    f = _row("FII/DII net flows", "NSE via NSEPython nse_fiidii()", source="NSEPython (unofficial)")
    raw = None
    try:
        from nsepython import nse_fiidii
        raw = nse_fiidii()
        f.update(Status="available", Note="Backward-looking, previous reported day: not live pre-open data. See table below.")
        f["As of"] = "see 'date' column"
    except Exception as e:
        f["Note"] = f"nsepython unavailable or request failed: {e!r}"
    rows.append(f)
    return pd.DataFrame(rows), us, raw
