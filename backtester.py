"""
backtester.py — Simulasi strategi MA+RSI+MACD pada data harian 90 hari ke belakang.

Logika:
  - Ambil data harian 1 tahun terakhir
  - Rolling-window: untuk setiap hari, hitung indikator menggunakan data sebelumnya
  - Jika sinyal BUY/STRONG BUY → catat harga entry, ukur return 5 hari ke depan
  - Hitung: total sinyal, win rate, rata-rata return, max drawdown, best trade
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import ccxt
import streamlit as st

# Konstanta sama dengan scanner.py
MA_FAST   = 20
MA_SLOW   = 50
RSI_LEN   = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIG  = 9
BB_LEN    = 20
BB_STD    = 2.0
VOL_LEN   = 20

FORWARD_BARS  = 5    # cek return 5 hari ke depan
MIN_BARS      = MA_SLOW + 15   # minimal bar untuk hitung semua indikator


# ─── FETCH DATA HARIAN ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)   # cache 1 jam (data harian tidak berubah cepat)
def _fetch_daily_stock(ticker: str) -> pd.DataFrame | None:
    try:
        raw = yf.download(ticker, period="1y", interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return raw.reset_index(drop=True)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_daily_crypto(symbol: str) -> pd.DataFrame | None:
    try:
        ex = ccxt.binance({"enableRateLimit": True})
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="1d", limit=365)
        if not ohlcv:
            return None
        df = pd.DataFrame(ohlcv, columns=["Datetime","Open","High","Low","Close","Volume"])
        df["Close"]  = df["Close"].astype(float)
        df["Volume"] = df["Volume"].astype(float)
        return df[["Open","High","Low","Close","Volume"]].reset_index(drop=True)
    except Exception:
        return None


# ─── SINYAL PADA SATU WINDOW ──────────────────────────────────────────────────
def _signal_on_window(df: pd.DataFrame) -> str:
    """Hitung sinyal strategi pada window df (tanpa caching, dipakai loop)."""
    if len(df) < MIN_BARS:
        return "WAIT"

    close  = df["Close"].astype(float)
    volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(dtype=float)

    ma20 = ta.sma(close, length=MA_FAST)
    ma50 = ta.sma(close, length=MA_SLOW)
    rsi  = ta.rsi(close, length=RSI_LEN)
    macd_df = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIG)
    bb_df   = ta.bbands(close, length=BB_LEN, std=BB_STD)

    # Ambil nilai terakhir
    def last_val(series, default=0.0):
        if series is None: return default
        v = series.dropna()
        return float(v.iloc[-1]) if not v.empty else default

    lp       = last_val(close)
    _ma20    = last_val(ma20)
    _ma50    = last_val(ma50)
    _rsi     = last_val(rsi)
    _macd    = last_val(macd_df.iloc[:, 0] if macd_df is not None else None)
    _mhist   = last_val(macd_df.iloc[:, 1] if macd_df is not None else None)
    _msig    = last_val(macd_df.iloc[:, 2] if macd_df is not None else None)
    _bb_u    = last_val(bb_df.iloc[:, 0]   if bb_df   is not None else None, lp * 1.02)
    _bb_l    = last_val(bb_df.iloc[:, 2]   if bb_df   is not None else None, lp * 0.98)
    _bb_m    = (_bb_u + _bb_l) / 2

    # Slope MA20
    ma20_arr = ma20.dropna()
    slope = float(ma20_arr.iloc[-1] - ma20_arr.iloc[-6]) if len(ma20_arr) >= 6 else 0.0

    # Volume
    vol_ratio = 0.0
    if not volume.empty:
        vol_arr = volume.iloc[-VOL_LEN:]
        avg = float(vol_arr.mean())
        if avg > 0:
            vol_ratio = float(volume.iloc[-1]) / avg

    # Skor (logika sama persis dengan scanner.py)
    score = 0
    if lp > _ma20:                                              score += 15
    if slope > 0:                                               score += 10
    if lp > _ma50:                                              score += 5
    if   50 <= _rsi <= 65:                                      score += 25
    elif 45 <= _rsi < 50 or 65 < _rsi <= 70:                   score += 15
    elif 40 <= _rsi < 45 or 70 < _rsi <= 75:                   score += 8
    if _macd > _msig:                                           score += 10
    if _mhist > 0:                                              score += 8
    bb_range = _bb_u - _bb_l
    if bb_range > 0:
        bb_pos = (lp - _bb_l) / bb_range
        if   0.30 <= bb_pos <= 0.70:                            score += 10
        elif 0.70 <  bb_pos <  0.90:                            score += 5
    if   vol_ratio >= 1.5:                                      score += 10
    elif vol_ratio >= 1.0:                                      score += 5

    score = min(100, max(0, score))

    if   _rsi > 75:    return "OVERBOUGHT"
    elif score >= 75:  return "STRONG BUY"
    elif score >= 55:  return "BUY"
    else:              return "WAIT"


# ─── BACKTEST UTAMA ───────────────────────────────────────────────────────────
def run_backtest(symbol: str, category: str) -> dict:
    """
    Jalankan backtest rolling 90 hari pada data harian.
    Kembalikan statistik performa strategi.
    """
    df = (_fetch_daily_crypto(symbol)
          if category == "Kripto"
          else _fetch_daily_stock(symbol))

    if df is None or len(df) < MIN_BARS + FORWARD_BARS + 5:
        return {"error": "Data tidak cukup untuk backtest."}

    trades: list[dict] = []

    # Rolling window: mulai dari bar ke-MIN_BARS, berhenti FORWARD_BARS sebelum akhir
    for i in range(MIN_BARS, len(df) - FORWARD_BARS):
        window  = df.iloc[:i]
        signal  = _signal_on_window(window)
        if signal not in ("BUY", "STRONG BUY"):
            continue

        entry_price = float(df["Close"].iloc[i])
        exit_price  = float(df["Close"].iloc[i + FORWARD_BARS])

        if entry_price <= 0:
            continue

        ret_pct = (exit_price - entry_price) / entry_price * 100
        trades.append({
            "bar":          i,
            "signal":       signal,
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "return_pct":   round(ret_pct, 2),
            "is_win":       ret_pct > 0,
        })

    if not trades:
        return {
            "total_signals": 0,
            "win_rate":      0.0,
            "avg_return":    0.0,
            "best_trade":    0.0,
            "worst_trade":   0.0,
            "max_drawdown":  0.0,
            "trades":        [],
            "error":         None,
        }

    returns      = [t["return_pct"] for t in trades]
    wins         = [r for r in returns if r > 0]
    win_rate     = len(wins) / len(trades) * 100
    avg_return   = float(np.mean(returns))
    best_trade   = float(max(returns))
    worst_trade  = float(min(returns))

    # Max drawdown: berdasarkan kurva ekuitas kumulatif
    cumulative = np.cumprod([1 + r / 100 for r in returns])
    running_max = np.maximum.accumulate(cumulative)
    drawdowns   = (cumulative - running_max) / running_max * 100
    max_dd      = float(min(drawdowns)) if len(drawdowns) else 0.0

    # Profit factor
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss   = abs(sum(r for r in returns if r < 0))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    # Hanya simpan 90 hari terakhir worth of trades untuk ditampilkan
    recent_trades = trades[-90:]

    return {
        "total_signals":  len(trades),
        "win_rate":       round(win_rate, 1),
        "avg_return":     round(avg_return, 2),
        "best_trade":     round(best_trade, 2),
        "worst_trade":    round(worst_trade, 2),
        "max_drawdown":   round(max_dd, 2),
        "profit_factor":  profit_factor,
        "trades":         recent_trades,
        "error":          None,
    }
