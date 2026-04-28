"""
scanner.py — Engine utama Market Scanner
Mengambil data real-time dari yfinance (saham) & Binance/ccxt (kripto)
dengan indikator MA, RSI, MACD, Bollinger Bands, dan Volume.
"""

import warnings
warnings.filterwarnings("ignore")

import io
import base64
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import ccxt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ─── KONSTANTA INDIKATOR ──────────────────────────────────────────────────────
MA_FAST    = 20
MA_SLOW    = 50
RSI_LEN    = 14
BB_LEN     = 20
BB_STD     = 2.0
MACD_FAST  = 12
MACD_SLOW  = 26
MACD_SIG   = 9
VOL_LEN    = 20

# ─── KLASIFIKASI SYARIAH ──────────────────────────────────────────────────────
# Referensi: Daftar Efek Syariah (DES) OJK periode II 2024 — saham ID
# Referensi: DJIM / AAOIFI screening — saham US (non-bank, non-alkohol, non-senjata)
# Kripto: mayoritas ulama masih debatable → default Konvensional
SYARIAH_MAP: dict[str, str] = {
    # ── Saham Indonesia (DES OJK) ──────────────────────────────────────────
    "TLKM.JK": "Syariah",   "ASII.JK": "Syariah",   "UNVR.JK": "Syariah",
    "GOTO.JK": "Syariah",   "TOWR.JK": "Syariah",   "EXCL.JK":  "Syariah",
    "ICBP.JK": "Syariah",   "INDF.JK": "Syariah",   "KLBF.JK":  "Syariah",
    "SIDO.JK": "Syariah",   "CPIN.JK": "Syariah",   "JPFA.JK":  "Syariah",
    "MYOR.JK": "Syariah",   "HRUM.JK": "Syariah",   "PTBA.JK":  "Syariah",
    "ADRO.JK": "Syariah",   "INCO.JK": "Syariah",   "ANTM.JK":  "Syariah",
    "MDKA.JK": "Syariah",   "SMGR.JK": "Syariah",   "WIKA.JK":  "Syariah",
    "WSKT.JK": "Syariah",   "JSMR.JK": "Syariah",   "PGAS.JK":  "Syariah",
    "AKRA.JK": "Syariah",   "SCMA.JK": "Syariah",   "MNCN.JK":  "Syariah",
    # Bank konvensional — tidak masuk DES OJK
    "BBCA.JK": "Konvensional", "BMRI.JK": "Konvensional",
    "BBRI.JK": "Konvensional", "BBNI.JK": "Konvensional",
    "BNGA.JK": "Konvensional", "BTPS.JK": "Konvensional",
    # ── Saham US (DJIM/AAOIFI screening) ──────────────────────────────────
    "AAPL":  "Syariah", "MSFT":  "Syariah", "GOOGL": "Syariah",
    "NVDA":  "Syariah", "TSLA":  "Syariah", "META":  "Syariah",
    "AMZN":  "Syariah", "AMD":   "Syariah", "INTC":  "Syariah",
    "NFLX":  "Syariah", "ORCL":  "Syariah", "CRM":   "Syariah",
    "UBER":  "Syariah", "LYFT":  "Syariah", "SPOT":  "Syariah",
    # Bank & keuangan konvensional
    "JPM":   "Konvensional", "BAC": "Konvensional", "GS": "Konvensional",
    "V":     "Konvensional", "MA":  "Konvensional",
    # ── Kripto — default Konvensional ──────────────────────────────────────
    "BTC/USDT": "Konvensional", "ETH/USDT": "Konvensional",
    "SOL/USDT": "Konvensional", "BNB/USDT": "Konvensional",
    "XRP/USDT": "Konvensional", "ADA/USDT": "Konvensional",
    "DOGE/USDT":"Konvensional", "AVAX/USDT":"Konvensional",
    "DOT/USDT": "Konvensional", "MATIC/USDT":"Konvensional",
}

def get_syariah(symbol: str) -> str:
    return SYARIAH_MAP.get(symbol, "Konvensional")


# ─── FETCH DATA ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)   # cache 15 menit
def fetch_stock(ticker: str, interval: str = "15m", period: str = "60d") -> pd.DataFrame | None:
    """Ambil OHLCV saham dari yfinance. Interval 15m didukung hingga 60 hari."""
    try:
        raw = yf.download(ticker, period=period, interval=interval,
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None
        # Flatten MultiIndex columns (yfinance >= 0.2.x)
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        raw.index = pd.to_datetime(raw.index)
        return raw
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)   # cache 15 menit
def fetch_crypto(symbol: str, timeframe: str = "15m", limit: int = 600) -> pd.DataFrame | None:
    """Ambil OHLCV kripto dari Binance via ccxt."""
    try:
        ex = ccxt.binance({"enableRateLimit": True})
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            return None
        df = pd.DataFrame(ohlcv, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        df["Close"]  = df["Close"].astype(float)
        df["Volume"] = df["Volume"].astype(float)
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit="ms")
        df.set_index("Datetime", inplace=True)
        return df
    except Exception:
        return None


# ─── HITUNG INDIKATOR ─────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> dict | None:
    """
    Hitung MA20, MA50, RSI-14, MACD(12,26,9), Bollinger Bands(20,2), Volume.
    Kembalikan dict berisi semua nilai indikator + skor + sinyal.
    """
    if df is None or len(df) < MA_SLOW + 10:
        return None

    df = df.copy().reset_index(drop=True)
    close  = df["Close"].astype(float)
    volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(dtype=float)

    # ── Moving Averages ───────────────────────────────────────────────────
    df["MA20"] = ta.sma(close, length=MA_FAST)
    df["MA50"] = ta.sma(close, length=MA_SLOW)

    # ── RSI ───────────────────────────────────────────────────────────────
    df["RSI"] = ta.rsi(close, length=RSI_LEN)

    # ── MACD ─────────────────────────────────────────────────────────────
    macd_df = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIG)
    if macd_df is not None and not macd_df.empty:
        df["MACD"]      = macd_df.iloc[:, 0]   # MACD line
        df["MACD_HIST"] = macd_df.iloc[:, 1]   # Histogram
        df["MACD_SIG"]  = macd_df.iloc[:, 2]   # Signal line

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb_df = ta.bbands(close, length=BB_LEN, std=BB_STD)
    if bb_df is not None and not bb_df.empty:
        df["BB_UPPER"] = bb_df.iloc[:, 0]
        df["BB_MID"]   = bb_df.iloc[:, 1]
        df["BB_LOWER"] = bb_df.iloc[:, 2]

    df.dropna(inplace=True)
    if len(df) < 3:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    def safe(col, default=0.0):
        v = last.get(col, default)
        return float(v) if pd.notna(v) else default

    lp          = safe("Close")
    ma20        = safe("MA20")
    ma50        = safe("MA50")
    rsi         = safe("RSI")
    macd        = safe("MACD")
    macd_sig    = safe("MACD_SIG")
    macd_hist   = safe("MACD_HIST")
    prev_hist   = float(prev.get("MACD_HIST", macd_hist)) if pd.notna(prev.get("MACD_HIST", None)) else macd_hist
    bb_upper    = safe("BB_UPPER", lp * 1.02)
    bb_mid      = safe("BB_MID",   ma20)
    bb_lower    = safe("BB_LOWER", lp * 0.98)

    # MA20 slope: sekarang vs 5 bar lalu
    ma20_slope  = ma20 - float(df["MA20"].iloc[-6]) if len(df) >= 6 else 0.0

    # Volume ratio terhadap rata-rata 20 bar
    vol_ratio   = 0.0
    if not volume.empty and "Volume" in df.columns:
        vol_avg = float(df["Volume"].iloc[-VOL_LEN:].mean())
        if vol_avg > 0:
            vol_ratio = float(last.get("Volume", 0)) / vol_avg

    # ── SCORING (0–100) ───────────────────────────────────────────────────
    score = 0

    # Tren — MA (30 poin)
    if lp > ma20:                score += 15
    if ma20_slope > 0:           score += 10
    if lp > ma50:                score += 5

    # Momentum — RSI (25 poin)
    if   50 <= rsi <= 65:        score += 25
    elif 45 <= rsi < 50 or 65 < rsi <= 70:   score += 15
    elif 40 <= rsi < 45 or 70 < rsi <= 75:   score += 8
    # RSI < 40 atau > 75 → tidak ada poin RSI

    # Momentum — MACD (25 poin)
    if macd > macd_sig:          score += 10
    if macd_hist > 0:            score += 8
    if macd_hist > prev_hist:    score += 7

    # Posisi harga di Bollinger Bands (10 poin)
    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        bb_pos = (lp - bb_lower) / bb_range   # 0=lower, 1=upper
        if   0.30 <= bb_pos <= 0.70:  score += 10  # zona ideal tengah
        elif 0.70 <  bb_pos <  0.90:  score += 5   # masih oke, tapi mulai extended

    # Volume konfirmasi (10 poin)
    if   vol_ratio >= 1.5:       score += 10
    elif vol_ratio >= 1.0:       score += 5

    score = min(100, max(0, score))

    # ── SINYAL ────────────────────────────────────────────────────────────
    if   rsi > 75:               signal = "⚠️ OVERBOUGHT"
    elif score >= 75:            signal = "🚀 STRONG BUY"
    elif score >= 55:            signal = "⬆️ BUY"
    else:                        signal = "😴 WAIT"

    # ── Perubahan 24 jam ──────────────────────────────────────────────────
    bars_24h = 96   # 96 × 15 menit = 24 jam
    chg_pct  = None
    if len(df) >= bars_24h:
        p24 = float(df["Close"].iloc[-bars_24h])
        if p24 > 0:
            chg_pct = round((lp - p24) / p24 * 100, 2)

    return {
        "price":      lp,
        "ma20":       ma20,
        "ma50":       ma50,
        "rsi":        rsi,
        "macd":       macd,
        "macd_hist":  macd_hist,
        "bb_upper":   bb_upper,
        "bb_mid":     bb_mid,
        "bb_lower":   bb_lower,
        "vol_ratio":  vol_ratio,
        "score":      score,
        "signal":     signal,
        "chg_pct":    chg_pct,
        "sparkline":  close.iloc[-60:].tolist(),   # 60 titik terakhir
    }


# ─── SPARKLINE ────────────────────────────────────────────────────────────────
def make_sparkline(prices: list[float], signal: str) -> str:
    """Buat sparkline sebagai base64 PNG transparan (150×40px)."""
    if not prices or len(prices) < 2:
        return ""
    color_map = {
        "STRONG BUY": "#10b981",
        "BUY":        "#34d399",
        "OVERBOUGHT": "#f59e0b",
        "WAIT":       "#94a3b8",
        "ERROR":      "#ef4444",
    }
    color = next((v for k, v in color_map.items() if k in signal), "#94a3b8")

    fig, ax = plt.subplots(figsize=(1.8, 0.45))
    ax.plot(prices, color=color, linewidth=1.3, solid_capstyle="round")
    ax.fill_between(range(len(prices)), prices, min(prices),
                    alpha=0.18, color=color)
    ax.set_xlim(0, len(prices) - 1)
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight",
                pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ─── SCAN SATU ASET ───────────────────────────────────────────────────────────
def scan_single(symbol: str, category: str, interval: str = "15m") -> dict:
    """Fetch + hitung indikator untuk satu aset."""
    df = (fetch_crypto(symbol, timeframe=interval, limit=600)
          if category == "Kripto"
          else fetch_stock(symbol, interval=interval))

    tech = compute_indicators(df)

    if tech:
        spark = make_sparkline(tech["sparkline"], tech["signal"])
        return {
            "Simbol":    symbol,
            "Kategori":  category,
            "Syariah":   get_syariah(symbol),
            "Harga":     round(tech["price"],      6),
            "MA20":      round(tech["ma20"],        6),
            "MA50":      round(tech["ma50"],        6),
            "RSI":       round(tech["rsi"],         2),
            "MACD Hist": round(tech["macd_hist"],   6),
            "BB Mid":    round(tech["bb_mid"],      6),
            "Vol Ratio": round(tech["vol_ratio"],   2),
            "Δ24h (%)":  tech["chg_pct"],
            "Skor":      tech["score"],
            "Sinyal":    tech["signal"],
            "_spark":    spark,
            "_bb_upper": round(tech["bb_upper"],    6),
            "_bb_lower": round(tech["bb_lower"],    6),
        }
    else:
        return {
            "Simbol":    symbol,
            "Kategori":  category,
            "Syariah":   get_syariah(symbol),
            "Harga":     None, "MA20": None, "MA50": None,
            "RSI":       None, "MACD Hist": None, "BB Mid": None,
            "Vol Ratio": None, "Δ24h (%)": None,
            "Skor":      -1,
            "Sinyal":    "❌ ERROR",
            "_spark":    "",
            "_bb_upper": None, "_bb_lower": None,
        }


# ─── SCAN PARALEL ─────────────────────────────────────────────────────────────
def scan_all_parallel(
    assets: dict[str, list[str]],
    interval: str,
    progress_cb=None,
    status_cb=None,
    max_workers: int = 8,
) -> list[dict]:
    """Scan semua aset secara paralel menggunakan ThreadPoolExecutor."""
    all_items = [(sym, cat)
                 for cat, syms in assets.items()
                 for sym in syms]
    total = len(all_items)
    results: list[dict] = []
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(scan_single, sym, cat, interval): (sym, cat)
            for sym, cat in all_items
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done += 1
            if progress_cb:
                progress_cb(done / total)
            if status_cb:
                status_cb(result["Simbol"], done, total)

    results.sort(key=lambda x: x["Skor"], reverse=True)
    return results
