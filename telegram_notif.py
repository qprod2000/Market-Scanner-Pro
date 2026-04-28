"""
telegram_notif.py — Kirim alert Telegram saat sinyal STRONG BUY terdeteksi.

Setup:
  1. Buat bot via @BotFather di Telegram → dapatkan BOT_TOKEN
  2. Kirim pesan apa saja ke bot Anda → dapatkan CHAT_ID via:
     https://api.telegram.org/bot<TOKEN>/getUpdates
  3. Simpan di Streamlit secrets (Settings → Secrets):
       TELEGRAM_BOT_TOKEN = "123456:ABC-..."
       TELEGRAM_CHAT_ID   = "987654321"
"""

import requests
import streamlit as st
from datetime import datetime


TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _get_credentials() -> tuple[str | None, str | None]:
    """Baca token dan chat_id dari st.secrets. Return None jika belum dikonfigurasi."""
    try:
        token   = st.secrets.get("TELEGRAM_BOT_TOKEN")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
        if token and chat_id and token != "YOUR_BOT_TOKEN_HERE":
            return str(token), str(chat_id)
    except Exception:
        pass
    return None, None


def is_telegram_configured() -> bool:
    token, chat_id = _get_credentials()
    return bool(token and chat_id)


def send_alert(message: str) -> bool:
    """Kirim pesan teks ke Telegram. Kembalikan True jika berhasil."""
    token, chat_id = _get_credentials()
    if not token or not chat_id:
        return False
    try:
        resp = requests.post(
            TELEGRAM_API.format(token=token),
            json={
                "chat_id":    chat_id,
                "text":       message,
                "parse_mode": "HTML",
            },
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def format_signal_message(asset: dict) -> str:
    """Format pesan alert untuk satu aset."""
    now   = datetime.now().strftime("%d %b %Y, %H:%M")
    sym   = asset.get("Simbol", "?")
    cat   = asset.get("Kategori", "")
    sy    = asset.get("Syariah", "")
    harga = asset.get("Harga", 0)
    rsi   = asset.get("RSI", 0)
    skor  = asset.get("Skor", 0)
    chg   = asset.get("Δ24h (%)", None)
    sig   = asset.get("Sinyal", "")

    chg_str = f"{chg:+.2f}%" if chg is not None else "—"
    sy_icon = "✅" if sy == "Syariah" else "🔸"

    return (
        f"<b>📈 Market Scanner Alert</b>\n"
        f"─────────────────────\n"
        f"<b>Aset :</b> {sym} ({cat})\n"
        f"<b>Status :</b> {sy_icon} {sy}\n"
        f"<b>Sinyal :</b> {sig}\n"
        f"<b>Harga :</b> {harga:,.4f}\n"
        f"<b>RSI :</b> {rsi:.1f}\n"
        f"<b>Δ24h :</b> {chg_str}\n"
        f"<b>Skor :</b> {skor}/100\n"
        f"─────────────────────\n"
        f"<i>{now}</i>\n"
        f"<i>⚠️ Bukan rekomendasi investasi.</i>"
    )


def send_scan_summary(results: list[dict], min_score: int = 75) -> tuple[int, int]:
    """
    Kirim alert untuk semua aset dengan skor >= min_score.
    Kembalikan (jumlah_dikirim, jumlah_gagal).
    """
    targets = [r for r in results if r.get("Skor", 0) >= min_score]
    sent, failed = 0, 0
    for asset in targets:
        msg = format_signal_message(asset)
        if send_alert(msg):
            sent += 1
        else:
            failed += 1
    return sent, failed
