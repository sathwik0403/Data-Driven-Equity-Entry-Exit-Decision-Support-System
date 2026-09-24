"""Signal engine: SMA bounce, Fibonacci retracement, ATH drawdown, psychological levels."""
import numpy as np
import pandas as pd

RANK = {"WEAK": 1, "SUFFICIENT": 2, "NORMAL": 3, "GOOD": 4, "MODERATE": 5, "STRONG": 6}

# Fibonacci level -> (direction, strength) for each trend
FIB_UP = {0.236: ("BUY", "WEAK"), 0.382: ("BUY", "NORMAL"), 0.5: ("BUY", "MODERATE"),
          0.618: ("BUY", "STRONG"), 0.786: ("SELL", "MODERATE")}
FIB_DOWN = {0.236: ("SELL", "WEAK"), 0.382: ("SELL", "NORMAL"), 0.5: ("SELL", "MODERATE"),
            0.618: ("SELL", "STRONG"), 0.786: ("BUY", "MODERATE")}
ATH_LEVELS = {0.382: "SUFFICIENT", 0.5: "MODERATE", 0.618: "STRONG"}


def load_candles(raw: pd.DataFrame) -> pd.DataFrame:
    """Accepts a TradingView / yfinance style export and standardises it."""
    d = raw.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    tcol = next(c for c in d.columns if c in ("datetime", "time", "date", "timestamp"))
    if pd.api.types.is_numeric_dtype(d[tcol]):
        t = pd.to_datetime(d[tcol], unit="s", utc=True)
    else:
        t = pd.to_datetime(d[tcol], utc=True)
    d["Datetime"] = t.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    d = d.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    if "Volume" not in d:
        d["Volume"] = 0
    d = d[["Datetime", "Open", "High", "Low", "Close", "Volume"]].dropna()
    return d.drop_duplicates("Datetime").sort_values("Datetime").reset_index(drop=True)


def add_smas(d: pd.DataFrame, periods=(21, 50, 200)) -> pd.DataFrame:
    """Daily SMA built from the last 4H close of each day (previous completed day, so no look-ahead)."""
    d = d.copy()
    d["Day"] = d["Datetime"].dt.normalize()
    daily = d.groupby("Day")["Close"].last()
    sma = pd.DataFrame({f"SMA{p}": daily.rolling(p).mean() for p in periods}).shift(1)
    return d.join(sma, on="Day")


def find_pivots(d: pd.DataFrame, k: int):
    hi, lo, out = d["High"].values, d["Low"].values, []
    for i in range(k, len(d) - k):
        if hi[i - k:i + k + 1].argmax() == k:
            out.append((i + k, i, "H", hi[i]))       # (confirmed_at, index, type, price)
        if lo[i - k:i + k + 1].argmin() == k:
            out.append((i + k, i, "L", lo[i]))
    return sorted(out)


def generate_signals(d, k=8, sma_tol=0.015, fib_tol=0.03, ath_tol=0.02,
                     psy_step=50, psy_tol=0.02, cooldown=6, horizon=15):
    piv, p = find_pivots(d, k), 0
    highs, lows, rows = [], [], []
    last = {"BUY": (-999, 0), "SELL": (-999, 0)}
    close, high = d["Close"].values, d["High"].values
    ath = np.maximum.accumulate(high)

    for t in range(len(d)):
        while p < len(piv) and piv[p][0] <= t:      # only pivots already confirmed at bar t
            (highs if piv[p][2] == "H" else lows).append(piv[p][1:])
            p += 1
        if len(highs) < 2 or len(lows) < 2:
            continue
        px = close[t]
        up = highs[-1][2] > highs[-2][2] and lows[-1][2] > lows[-2][2]
        down = highs[-1][2] < highs[-2][2] and lows[-1][2] < lows[-2][2]
        trend = "UPTREND" if up else "DOWNTREND" if down else "SIDEWAYS"
        hits = []  # (signal, strength, reason)

        if down:  # Rules 1 & 2: bounce off SMAs
            for col, s, lab in (("SMA200", "STRONG", "200-day SMA"), ("SMA50", "GOOD", "50-day SMA"),
                                ("SMA21", "NORMAL", "21-day SMA")):
                v = d[col].iat[t]
                if pd.notna(v) and abs(px / v - 1) <= sma_tol:
                    hits.append(("BUY", s, f"Downtrend + price near {lab} ({v:.1f})"))
            dd = (ath[t] - px) / ath[t]            # Rule 5: drawdown from ATH
            for lvl, s in ATH_LEVELS.items():
                if abs(dd - lvl) <= ath_tol:
                    hits.append(("BUY", s, f"Downtrend + {lvl*100:.1f}% down from ATH ({ath[t]:.1f})"))

        if up or down:  # Rules 3 & 4: Fibonacci retracement of the latest swing leg
            if up:
                H = highs[-1]; L = next((x for x in reversed(lows) if x[0] < H[0]), None)
                rng = H[2] - L[2] if L else 0
                retr = (H[2] - px) / rng if rng > 0 else None
                table = FIB_UP
            else:
                L = lows[-1]; H = next((x for x in reversed(highs) if x[0] < L[0]), None)
                rng = H[2] - L[2] if H else 0
                retr = (px - L[2]) / rng if rng > 0 else None
                table = FIB_DOWN
            if retr is not None:
                for lvl, (sig, s) in table.items():
                    if abs(retr - lvl) <= fib_tol:
                        hits.append((sig, s, f"{trend.title()} + {lvl*100:.1f}% Fib retracement "
                                             f"(swing {L[2]:.1f} to {H[2]:.1f})"))

        lvl_up = np.ceil(px / psy_step) * psy_step   # Rule 6: psychological level just above
        gap = (lvl_up - px) / lvl_up
        if gap <= 0.4 * psy_tol:
            hits.append(("SELL", "STRONG", f"At psychological level {lvl_up:.0f} (resistance)"))
        elif gap <= psy_tol:
            hits.append(("BUY", "STRONG", f"Approaching psychological level {lvl_up:.0f} (magnet)"))

        for sig in ("BUY", "SELL"):
            h = [x for x in hits if x[0] == sig]
            if not h:
                continue
            rank = max(RANK[x[1]] for x in h)
            lt, lr = last[sig]
            if t - lt < cooldown and rank <= lr:
                continue
            last[sig] = (t, rank)
            best = max(h, key=lambda x: RANK[x[1]])[1]
            fwd = close[t + horizon] / px - 1 if t + horizon < len(d) else np.nan
            signed = fwd if sig == "BUY" else -fwd
            rows.append(dict(Datetime=d["Datetime"].iat[t], Price=round(px, 2), Signal=sig, Strength=best,
                             Reason=" + ".join(x[2] for x in h), Confluence=len(h), Trend=trend,
                             Fwd_Return_Pct=round(signed * 100, 2) if not np.isnan(fwd) else np.nan))
    return pd.DataFrame(rows)
