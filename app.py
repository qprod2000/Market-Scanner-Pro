"""
app.py — Market Scanner Streamlit App
Analisis investasi real-time: MA20, MA50, RSI, MACD, Bollinger Bands, Volume
Data: yfinance (saham) + Binance/ccxt (kripto) · Interval 15 menit
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

from scanner import scan_all_parallel, fetch_stock, fetch_crypto, compute_indicators
from backtester import run_backtest
from telegram_notif import (
    is_telegram_configured, send_scan_summary, send_alert, format_signal_message
)

# ─── KONFIGURASI HALAMAN ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── STYLING ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px;
}
[data-testid="stMetricLabel"] { font-size: 0.72rem !important; color: #64748b !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.4px; }
[data-testid="stMetricValue"] { font-size: 1.55rem !important; font-weight: 700 !important; }

/* Signal badges */
.badge { display:inline-block; padding:3px 9px; border-radius:5px; font-size:0.73rem; font-weight:700; white-space:nowrap; line-height:1.6; }
.b-sb   { background:#dcfce7; color:#15803d; border:1px solid #86efac; }
.b-buy  { background:#d1fae5; color:#065f46; border:1px solid #6ee7b7; }
.b-ob   { background:#fef3c7; color:#92400e; border:1px solid #fcd34d; }
.b-wait { background:#f1f5f9; color:#475569; border:1px solid #cbd5e1; }
.b-err  { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; }
.b-sy   { background:#ecfdf5; color:#065f46; border:1px solid #6ee7b7; font-size:0.68rem; padding:2px 7px; }
.b-kv   { background:#fff7ed; color:#9a3412; border:1px solid #fdba74; font-size:0.68rem; padding:2px 7px; }

/* Top cards */
.top-card { background:#fff; border:1.5px solid #e2e8f0; border-radius:14px; padding:16px 18px; }
.top-card.gold   { border-color:#f59e0b; background:#fffbeb; }
.top-card.silver { border-color:#94a3b8; background:#f8fafc; }
.top-card.bronze { border-color:#cd7f32; background:#fdf6ec; }
.top-card-sym   { font-size:1.05rem; font-weight:700; color:#0f172a; }
.top-card-meta  { font-size:0.73rem; color:#64748b; margin:2px 0 8px; }
.kv-row { display:flex; justify-content:space-between; font-size:0.78rem; margin-top:5px; color:#374151; }
.score-bg   { height:4px; background:#e2e8f0; border-radius:2px; margin-top:9px; }
.score-fill { height:4px; border-radius:2px; }

/* Table */
.scan-table { width:100%; border-collapse:collapse; font-size:0.8rem; }
.scan-table thead tr { border-bottom:2px solid #e2e8f0; background:#f8fafc; }
.scan-table th { padding:9px 11px; text-align:left; color:#475569; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; white-space:nowrap; }
.scan-table td { padding:8px 11px; border-bottom:1px solid #f1f5f9; vertical-align:middle; }
.scan-table tr:hover td { background:#f0f9ff; }
.num { font-family:monospace; font-size:0.78rem; }
.up   { color:#15803d; font-weight:600; }
.down { color:#b91c1c; font-weight:600; }
.neutral { color:#64748b; }
.dot-live { display:inline-block; width:7px; height:7px; background:#10b981; border-radius:50%; margin-right:5px; animation:blink 1.3s infinite; }
.dot-idle { display:inline-block; width:7px; height:7px; background:#94a3b8; border-radius:50%; margin-right:5px; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }

/* Sidebar */
section[data-testid="stSidebar"] { background:#f8fafc; }
.sidebar-hd { font-size:0.7rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.6px; margin:1rem 0 0.3rem; }

/* Info box */
.info-box { background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; padding:10px 14px; font-size:0.8rem; color:#1e40af; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

# ─── DEFAULT ASET ─────────────────────────────────────────────────────────────
DEFAULT_ASSETS = {
    "Saham ID": ["BBCA.JK","TLKM.JK","ASII.JK","GOTO.JK","BMRI.JK","BBRI.JK","UNVR.JK"],
    "Saham US": ["AAPL","TSLA","NVDA","MSFT","GOOGL","META","AMZN"],
    "Kripto":   ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT"],
}

INTERVAL_OPTIONS = {
    "15 Menit (default)": "15m",
    "1 Jam":              "1h",
    "4 Jam":              "4h",
    "1 Hari":             "1d",
}

# ─── SESSION STATE ────────────────────────────────────────────────────────────
for key, val in {
    "results": [], "last_update": None, "scanning": False,
    "bt_cache": {}, "notif_sent": set(),
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─── HELPER HTML ──────────────────────────────────────────────────────────────
def badge_signal(sig: str) -> str:
    if "STRONG BUY" in sig: return f'<span class="badge b-sb">🚀 STRONG BUY</span>'
    if "BUY"        in sig: return f'<span class="badge b-buy">⬆️ BUY</span>'
    if "OVERBOUGHT" in sig: return f'<span class="badge b-ob">⚠️ OVERBOUGHT</span>'
    if "ERROR"      in sig: return f'<span class="badge b-err">❌ ERROR</span>'
    return f'<span class="badge b-wait">😴 WAIT</span>'

def badge_syariah(s: str) -> str:
    return (f'<span class="badge b-sy">✓ Syariah</span>'
            if s == "Syariah" else f'<span class="badge b-kv">Konvensional</span>')

def fmt_num(v, dec=4) -> str:
    if v is None: return "—"
    return f"{v:,.{dec}f}"

def fmt_chg(v) -> str:
    if v is None: return "—"
    sign = "▲" if v >= 0 else "▼"
    cls  = "up" if v >= 0 else "down"
    return f'<span class="{cls}">{sign} {abs(v):.2f}%</span>'

def fmt_rsi(v) -> str:
    if v is None: return "—"
    cls = "up" if 50 <= v <= 65 else "down" if v > 70 else "neutral"
    return f'<span class="{cls}">{v:.1f}</span>'

def score_color(s: int) -> str:
    if s >= 75: return "#10b981"
    if s >= 55: return "#f59e0b"
    if s >= 0:  return "#94a3b8"
    return "#ef4444"


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Market Scanner")
    st.caption("MA20 · MA50 · RSI · MACD · BB · Volume")
    st.divider()

    st.markdown('<div class="sidebar-hd">⏱ Interval & Strategi</div>', unsafe_allow_html=True)
    interval_label = st.selectbox("Timeframe", list(INTERVAL_OPTIONS.keys()), index=0)
    interval       = INTERVAL_OPTIONS[interval_label]

    with st.expander("Parameter Indikator", expanded=False):
        ma_fast   = st.slider("MA Fast",       5,  50, 20)
        ma_slow   = st.slider("MA Slow",       20, 100, 50)
        rsi_len   = st.slider("RSI Period",    5,  21,  14)
        score_buy = st.slider("Threshold BUY", 40, 70,  55)
        score_sb  = st.slider("Threshold STRONG BUY", 60, 95, 75)

    st.divider()
    st.markdown('<div class="sidebar-hd">📋 Daftar Aset</div>', unsafe_allow_html=True)
    saham_id = st.text_area("Saham Indonesia (.JK)", "\n".join(DEFAULT_ASSETS["Saham ID"]), height=130)
    saham_us = st.text_area("Saham US",              "\n".join(DEFAULT_ASSETS["Saham US"]), height=130)
    kripto   = st.text_area("Kripto (Binance)",      "\n".join(DEFAULT_ASSETS["Kripto"]),   height=110)

    user_assets = {
        "Saham ID": [s.strip().upper() for s in saham_id.splitlines() if s.strip()],
        "Saham US": [s.strip().upper() for s in saham_us.splitlines() if s.strip()],
        "Kripto":   [s.strip().upper() for s in kripto.splitlines()   if s.strip()],
    }

    st.divider()
    st.markdown('<div class="sidebar-hd">🔎 Filter</div>', unsafe_allow_html=True)
    f_cat  = st.multiselect("Kategori", ["Saham ID","Saham US","Kripto"],
                             default=["Saham ID","Saham US","Kripto"])
    f_sy   = st.multiselect("Status", ["Syariah","Konvensional"],
                             default=["Syariah","Konvensional"])
    f_sig  = st.multiselect("Sinyal", ["🚀 STRONG BUY","⬆️ BUY","⚠️ OVERBOUGHT","😴 WAIT"],
                             default=["🚀 STRONG BUY","⬆️ BUY","⚠️ OVERBOUGHT","😴 WAIT"])
    f_min_score = st.slider("Skor Minimum", 0, 100, 0)

    st.divider()
    st.markdown('<div class="sidebar-hd">📣 Telegram Alert</div>', unsafe_allow_html=True)
    if is_telegram_configured():
        st.success("✅ Telegram terhubung", icon="✅")
        tg_threshold = st.slider("Alert jika Skor ≥", 50, 100, 75)
    else:
        st.info("Tambahkan TELEGRAM_BOT_TOKEN dan TELEGRAM_CHAT_ID di Streamlit Secrets untuk aktifkan alert.", icon="ℹ️")
        tg_threshold = 75

    st.divider()
    st.markdown('<div class="sidebar-hd">🔄 Auto Refresh</div>', unsafe_allow_html=True)
    auto_refresh    = st.toggle("Auto refresh", value=False)
    refresh_minutes = st.slider("Interval (menit)", 5, 60, 15, disabled=not auto_refresh)

# ─── HEADER ───────────────────────────────────────────────────────────────────
col_h, col_btn = st.columns([3, 1])
with col_h:
    st.markdown("## 📊 Market Scanner")
    st.caption(f"Data real-time · Interval {interval_label} · 5 Indikator Teknikal")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    scan_clicked = st.button("🔍  Scan Sekarang", type="primary", use_container_width=True)

# ─── AUTO REFRESH ─────────────────────────────────────────────────────────────
if auto_refresh and st.session_state.last_update:
    elapsed = (datetime.now() - datetime.strptime(
        st.session_state.last_update, "%H:%M:%S, %d %b %Y")).seconds / 60
    if elapsed >= refresh_minutes:
        scan_clicked = True


# ─── PROSES SCAN ──────────────────────────────────────────────────────────────
if scan_clicked:
    prog    = st.progress(0.0, text="Memulai scan paralel…")
    status  = st.empty()
    t_start = time.time()

    def on_progress(pct):
        prog.progress(pct, text=f"Memproses… {int(pct*100)}%")

    def on_status(sym, done, total):
        status.markdown(
            f'<span class="dot-live"></span> <small>Mengambil <b>{sym}</b> '
            f'({done}/{total})</small>', unsafe_allow_html=True)

    st.session_state.results = scan_all_parallel(
        user_assets, interval,
        progress_cb=on_progress, status_cb=on_status, max_workers=8
    )
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S, %d %b %Y")
    elapsed = round(time.time() - t_start, 1)

    prog.empty(); status.empty()
    st.toast(f"✅ Scan selesai dalam {elapsed} detik!", icon="✅")

    # Kirim alert Telegram jika dikonfigurasi
    if is_telegram_configured():
        sent, failed = send_scan_summary(
            st.session_state.results, min_score=tg_threshold)
        if sent > 0:
            st.toast(f"📣 {sent} alert Telegram terkirim!", icon="📣")


results: list[dict] = st.session_state.results

# ─── STATUS ───────────────────────────────────────────────────────────────────
if st.session_state.last_update:
    total_items = sum(len(v) for v in user_assets.values())
    st.markdown(
        f'<span class="dot-idle"></span> <small>Terakhir diperbarui: <b>'
        f'{st.session_state.last_update}</b> · {len(results)}/{total_items} aset · '
        f'Interval: {interval_label}</small>',
        unsafe_allow_html=True)
else:
    st.markdown(
        '<span class="dot-idle"></span> <small>Belum ada data. '
        'Klik <b>Scan Sekarang</b>.</small>', unsafe_allow_html=True)

st.divider()

if not results:
    st.info("Klik **Scan Sekarang** untuk memulai analisis pasar.", icon="📡")
    st.stop()


# ─── FILTER ───────────────────────────────────────────────────────────────────
def passes_filter(r: dict) -> bool:
    if r["Kategori"] not in f_cat:         return False
    if r["Syariah"] not in f_sy:            return False
    if r["Skor"] < f_min_score:             return False
    sig = r["Sinyal"]
    return any(
        (k in sig) for k in
        [s.replace("🚀 ","").replace("⬆️ ","").replace("⚠️ ","").replace("😴 ","")
         for s in f_sig]
    ) if f_sig else False

filtered = [r for r in results if passes_filter(r)]


# ─── METRIK RINGKASAN ─────────────────────────────────────────────────────────
total    = len(filtered)
n_sb     = sum(1 for r in filtered if "STRONG BUY" in r["Sinyal"])
n_buy    = sum(1 for r in filtered if r["Sinyal"] == "⬆️ BUY")
n_ob     = sum(1 for r in filtered if "OVERBOUGHT" in r["Sinyal"])
n_wait   = sum(1 for r in filtered if "WAIT" in r["Sinyal"])
n_sy     = sum(1 for r in filtered if r["Syariah"] == "Syariah")
avg_sc   = round(sum(r["Skor"] for r in filtered if r["Skor"] >= 0) / max(1, total), 1)

c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
c1.metric("Total Aset",    total)
c2.metric("🚀 Strong Buy", n_sb)
c3.metric("⬆️ Buy",         n_buy)
c4.metric("⚠️ Overbought",  n_ob)
c5.metric("😴 Wait",        n_wait)
c6.metric("✅ Syariah",     n_sy)
c7.metric("⌀ Skor",        avg_sc)

st.markdown("<br>", unsafe_allow_html=True)


# ─── TOP 3 KARTU ──────────────────────────────────────────────────────────────
top3 = [r for r in filtered if r["Skor"] >= 55][:3]
if top3:
    st.markdown("### 🌟 Top 3 Aset Paling Prospektif")
    medals      = ["🥇", "🥈", "🥉"]
    card_styles = ["gold", "silver", "bronze"]
    cols = st.columns(len(top3))
    for i, (col, a) in enumerate(zip(cols, top3)):
        with col:
            fill_color = score_color(a["Skor"])
            st.markdown(f"""
            <div class="top-card {card_styles[i]}">
              <div style="font-size:1.3rem">{medals[i]}</div>
              <div class="top-card-sym">{a["Simbol"]}</div>
              <div class="top-card-meta">{a["Kategori"]} &nbsp;·&nbsp; {badge_syariah(a["Syariah"])}</div>
              {badge_signal(a["Sinyal"])}
              <div class="kv-row"><span>Harga</span><b>{fmt_num(a["Harga"])}</b></div>
              <div class="kv-row"><span>RSI</span>{fmt_rsi(a["RSI"])}</div>
              <div class="kv-row"><span>MACD Hist</span>
                <span class="{'up' if (a['MACD Hist'] or 0) > 0 else 'down'}">{fmt_num(a["MACD Hist"],6)}</span>
              </div>
              <div class="kv-row"><span>Δ24h</span>{fmt_chg(a["Δ24h (%)"])}</div>
              <div class="kv-row"><span>Vol Ratio</span>
                <b class="{'up' if (a['Vol Ratio'] or 0) >= 1 else 'neutral'}">{a["Vol Ratio"] or 0:.2f}×</b>
              </div>
              <div class="score-bg">
                <div class="score-fill" style="width:{a['Skor']}%;background:{fill_color}"></div>
              </div>
              <div style="font-size:0.68rem;color:#94a3b8;text-align:right;margin-top:3px">
                Skor {a["Skor"]}/100
              </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ─── TAB UTAMA ────────────────────────────────────────────────────────────────
tab_scan, tab_bt, tab_chart, tab_download = st.tabs([
    "📊 Hasil Scan", "🧪 Backtest", "📈 Chart Detail", "⬇️ Export"
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — TABEL SCAN
# ══════════════════════════════════════════════════════════════════════════════
with tab_scan:
    sub_all, sub_sy, sub_kv = st.tabs(["Semua", "✅ Syariah", "🔸 Konvensional"])

    def render_scan_table(data: list[dict]):
        if not data:
            st.info("Tidak ada aset yang sesuai filter.", icon="🔍")
            return

        rows = ""
        for r in data:
            sc       = r["Skor"]
            clr      = score_color(sc)
            spark    = r.get("_spark", "")
            spark_html = (f'<img src="{spark}" style="height:30px;width:110px;vertical-align:middle">'
                          if spark else "—")
            rows += f"""
            <tr>
              <td style="font-weight:700">{r["Simbol"]}</td>
              <td><small style="color:#64748b">{r["Kategori"]}</small></td>
              <td>{badge_syariah(r["Syariah"])}</td>
              <td class="num">{fmt_num(r["Harga"])}</td>
              <td class="num {'up' if r['Harga'] and r['MA20'] and r['Harga'] > r['MA20'] else 'down'}">{fmt_num(r["MA20"])}</td>
              <td class="num">{fmt_num(r["MA50"])}</td>
              <td>{fmt_rsi(r["RSI"])}</td>
              <td class="num {'up' if (r['MACD Hist'] or 0) > 0 else 'down'}">{fmt_num(r["MACD Hist"],6)}</td>
              <td class="num {'up' if (r['Vol Ratio'] or 0) >= 1.5 else ('neutral' if (r['Vol Ratio'] or 0) >= 1 else 'down')}">{r["Vol Ratio"]:.2f}×</td>
              <td>{fmt_chg(r["Δ24h (%)"])}</td>
              <td><b style="color:{clr}">{sc if sc >= 0 else '—'}</b></td>
              <td>{badge_signal(r["Sinyal"])}</td>
              <td>{spark_html}</td>
            </tr>"""

        st.markdown(f"""
        <div style="overflow-x:auto">
        <table class="scan-table">
          <thead><tr>
            <th>Simbol</th><th>Jenis</th><th>Status</th>
            <th>Harga</th><th>MA20</th><th>MA50</th>
            <th>RSI</th><th>MACD Hist</th><th>Vol Ratio</th>
            <th>Δ24h</th><th>Skor</th><th>Sinyal</th><th>Sparkline</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with sub_all:
        render_scan_table(filtered)
    with sub_sy:
        render_scan_table([r for r in filtered if r["Syariah"] == "Syariah"])
    with sub_kv:
        render_scan_table([r for r in filtered if r["Syariah"] == "Konvensional"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.markdown("#### 🧪 Backtest Strategi — Data Harian 1 Tahun")
    st.markdown("""
    <div class="info-box">
    Simulasi menggunakan <b>data harian 1 tahun</b> ke belakang. Setiap kali strategi 
    menghasilkan sinyal BUY/STRONG BUY, dicatat apakah harga naik dalam <b>5 hari ke depan</b>.
    Metrik yang ditampilkan: Win Rate, Rata-rata Return, Profit Factor, Max Drawdown.
    </div>
    """, unsafe_allow_html=True)

    valid_assets = [r for r in filtered if r["Skor"] >= 0]
    bt_sym = st.selectbox(
        "Pilih aset untuk dibacktest:",
        options=[r["Simbol"] for r in valid_assets],
        format_func=lambda s: next(
            (f"{r['Simbol']} ({r['Kategori']})" for r in valid_assets if r["Simbol"] == s), s)
    ) if valid_assets else None

    if bt_sym and st.button("▶ Jalankan Backtest", type="secondary"):
        cat = next(r["Kategori"] for r in valid_assets if r["Simbol"] == bt_sym)
        with st.spinner(f"Menjalankan backtest {bt_sym}…"):
            bt = run_backtest(bt_sym, cat)
            st.session_state.bt_cache[bt_sym] = bt

    # Tampilkan hasil backtest
    if bt_sym and bt_sym in st.session_state.bt_cache:
        bt = st.session_state.bt_cache[bt_sym]
        if bt.get("error") and bt["total_signals"] == 0:
            st.warning(bt["error"])
        else:
            wr  = bt["win_rate"]
            wr_color = "green" if wr >= 55 else "orange" if wr >= 45 else "red"
            bc1,bc2,bc3,bc4,bc5 = st.columns(5)
            bc1.metric("Total Sinyal",   bt["total_signals"])
            bc2.metric("Win Rate",       f'{bt["win_rate"]}%',
                       delta="Baik" if wr >= 55 else "Perhatikan",
                       delta_color="normal" if wr >= 55 else "inverse")
            bc3.metric("Avg Return",     f'{bt["avg_return"]:+.2f}%')
            bc4.metric("Profit Factor",  f'{bt["profit_factor"]:.2f}×')
            bc5.metric("Max Drawdown",   f'{bt["max_drawdown"]:.2f}%')

            st.metric("Best Trade",  f'+{bt["best_trade"]:.2f}%')

            # Grafik distribusi return
            trades_df = pd.DataFrame(bt["trades"])
            if not trades_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=trades_df["return_pct"],
                    nbinsx=20,
                    marker_color=["#10b981" if v >= 0 else "#ef4444"
                                  for v in trades_df["return_pct"]],
                    name="Return (%)",
                ))
                fig.update_layout(
                    title=f"Distribusi Return per Trade — {bt_sym}",
                    xaxis_title="Return (%)",
                    yaxis_title="Frekuensi",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)

                # Equity curve
                cumulative = (1 + trades_df["return_pct"] / 100).cumprod() * 100
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    y=cumulative, mode="lines",
                    line=dict(color="#3b82f6", width=2),
                    fill="tonexty", fillcolor="rgba(59,130,246,0.08)",
                    name="Equity Curve",
                ))
                fig2.update_layout(
                    title=f"Equity Curve — {bt_sym} (mulai 100)",
                    yaxis_title="Nilai (mulai 100)",
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor="white", paper_bgcolor="white",
                )
                st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — CHART DETAIL
# ══════════════════════════════════════════════════════════════════════════════
with tab_chart:
    st.markdown("#### 📈 Chart Teknikal Detail")
    valid_ch = [r for r in filtered if r["Skor"] >= 0]
    ch_sym = st.selectbox(
        "Pilih aset:",
        options=[r["Simbol"] for r in valid_ch],
        format_func=lambda s: next(
            (f"{r['Simbol']} ({r['Kategori']})" for r in valid_ch if r["Simbol"] == s), s),
        key="chart_select"
    ) if valid_ch else None

    if ch_sym:
        ch_cat = next(r["Kategori"] for r in valid_ch if r["Simbol"] == ch_sym)
        with st.spinner(f"Memuat chart {ch_sym}…"):
            ch_df = (fetch_crypto(ch_sym, timeframe=interval, limit=200)
                     if ch_cat == "Kripto"
                     else fetch_stock(ch_sym, interval=interval))

        if ch_df is not None and not ch_df.empty:
            import pandas_ta as _ta
            cdf = ch_df.copy()
            cl  = cdf["Close"].astype(float)
            cdf["MA20"]  = _ta.sma(cl, length=20)
            cdf["MA50"]  = _ta.sma(cl, length=50)
            cdf["RSI"]   = _ta.rsi(cl, length=14)
            macd_r = _ta.macd(cl)
            if macd_r is not None:
                cdf["MACD"]      = macd_r.iloc[:, 0]
                cdf["MACD_HIST"] = macd_r.iloc[:, 1]
                cdf["MACD_SIG"]  = macd_r.iloc[:, 2]
            bb_r = _ta.bbands(cl)
            if bb_r is not None:
                cdf["BB_U"] = bb_r.iloc[:, 0]
                cdf["BB_M"] = bb_r.iloc[:, 1]
                cdf["BB_L"] = bb_r.iloc[:, 2]
            cdf.dropna(inplace=True)
            cdf = cdf.tail(150)

            idx = list(range(len(cdf)))

            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                row_heights=[0.55, 0.22, 0.23],
                vertical_spacing=0.04,
                subplot_titles=[f"{ch_sym} — Harga & Bollinger Bands", "RSI-14", "MACD(12,26,9)"],
            )

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=idx,
                open=cdf["Open"], high=cdf["High"],
                low=cdf["Low"],   close=cdf["Close"],
                name="OHLC",
                increasing_line_color="#10b981",
                decreasing_line_color="#ef4444",
            ), row=1, col=1)

            # MA
            fig.add_trace(go.Scatter(x=idx, y=cdf["MA20"], name="MA20",
                line=dict(color="#3b82f6", width=1.5, dash="solid")), row=1, col=1)
            fig.add_trace(go.Scatter(x=idx, y=cdf["MA50"], name="MA50",
                line=dict(color="#f59e0b", width=1.5, dash="dash")), row=1, col=1)

            # Bollinger Bands
            if "BB_U" in cdf:
                fig.add_trace(go.Scatter(x=idx, y=cdf["BB_U"], name="BB Upper",
                    line=dict(color="#94a3b8", width=1, dash="dot"),
                    showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=idx, y=cdf["BB_L"], name="BB Lower",
                    fill="tonexty", fillcolor="rgba(148,163,184,0.06)",
                    line=dict(color="#94a3b8", width=1, dash="dot"),
                    showlegend=False), row=1, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=idx, y=cdf["RSI"], name="RSI",
                line=dict(color="#8b5cf6", width=2)), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="#ef4444", line_width=1, row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="#94a3b8", line_width=1, row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="#10b981",  line_width=1, row=2, col=1)

            # MACD
            if "MACD" in cdf:
                colors = ["#10b981" if v >= 0 else "#ef4444"
                          for v in cdf["MACD_HIST"]]
                fig.add_trace(go.Bar(x=idx, y=cdf["MACD_HIST"], name="MACD Hist",
                    marker_color=colors, opacity=0.7), row=3, col=1)
                fig.add_trace(go.Scatter(x=idx, y=cdf["MACD"], name="MACD",
                    line=dict(color="#3b82f6", width=1.5)), row=3, col=1)
                fig.add_trace(go.Scatter(x=idx, y=cdf["MACD_SIG"], name="Signal",
                    line=dict(color="#f59e0b", width=1.5, dash="dash")), row=3, col=1)

            fig.update_layout(
                height=640, margin=dict(l=10, r=10, t=40, b=10),
                xaxis_rangeslider_visible=False,
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
                font=dict(size=11),
            )
            fig.update_yaxes(gridcolor="#f1f5f9")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Gagal memuat data chart untuk {ch_sym}.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_download:
    st.markdown("#### ⬇️ Export Data")
    if filtered:
        export_cols = ["Simbol","Kategori","Syariah","Harga","MA20","MA50",
                       "RSI","MACD Hist","BB Mid","Vol Ratio","Δ24h (%)","Skor","Sinyal"]
        df_exp = pd.DataFrame([{k: r.get(k) for k in export_cols} for r in filtered])
        ts     = datetime.now().strftime("%Y%m%d_%H%M")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Unduh CSV",
                data=df_exp.to_csv(index=False).encode("utf-8"),
                file_name=f"market_scan_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "⬇️ Unduh JSON",
                data=df_exp.to_json(orient="records", indent=2).encode("utf-8"),
                file_name=f"market_scan_{ts}.json",
                mime="application/json",
                use_container_width=True,
            )
        st.dataframe(df_exp, use_container_width=True, hide_index=True)
    else:
        st.info("Lakukan scan terlebih dahulu.", icon="📡")


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer:** Aplikasi ini hanya untuk edukasi dan informasi. "
    "Bukan merupakan rekomendasi investasi. Selalu lakukan riset mandiri. "
    "Klasifikasi syariah bersifat indikatif — referensi DES OJK & DJIM. "
    "Konsultasikan dengan ahli syariah sebelum keputusan investasi."
)
