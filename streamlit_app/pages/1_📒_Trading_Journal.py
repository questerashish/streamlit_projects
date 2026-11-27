"""Comprehensive trading journal page."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from utils import charts, db, io  # noqa: E402


def load_demo_trades() -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp.today(), periods=20)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": ["ES"] * 10 + ["NQ"] * 10,
            "direction": ["Long", "Short"] * 10,
            "qty": 1,
            "entry": 100 + pd.Series(range(20)),
            "exit": 101 + pd.Series(range(20)),
            "strategy": ["Mean Revert", "Breakout"] * 10,
            "tags": ["A,Opening"] * 20,
            "pnl": pd.Series(range(-5, 15)) * 50,
        }
    )


def render_import_section():
    st.subheader("Import trades")
    uploaded = st.file_uploader("Upload CSV", type=["csv", "parquet"])
    if uploaded is not None:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_parquet(uploaded)
        st.write("Detected columns:", df.columns.tolist())
        st.dataframe(df.head())
        if st.button("Map & Save to DB"):
            db.init_db()
            st.success("Trades saved (placeholder). Extend to map columns and insert into SQLite.")


def render_manual_entry():
    st.subheader("Manual trade entry")
    with st.form("trade_entry"):
        cols = st.columns(4)
        trade_date = cols[0].date_input("Date", value=date.today())
        symbol = cols[1].text_input("Symbol", value="ES")
        direction = cols[2].selectbox("Direction", ["Long", "Short"])
        qty = cols[3].number_input("Quantity", value=1.0)
        entry = st.number_input("Entry", value=100.0)
        exit_price = st.number_input("Exit", value=101.0)
        fees = st.number_input("Fees", value=0.0)
        sl = st.number_input("Stop Loss", value=98.0)
        tp = st.number_input("Take Profit", value=105.0)
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "1h", "Daily"])
        strategy = st.text_input("Strategy", value="Mean Reversion")
        tags = st.text_input("Tags", value="Opening")
        notes = st.text_area("Notes")
        checklist = st.multiselect("Checklist", ["Plan followed", "Risk respected", "Setup qualified"])
        screenshot = st.file_uploader("Screenshot", type=["png", "jpg", "jpeg"])
        submitted = st.form_submit_button("Save Trade")
        if submitted:
            db.execute(
                "INSERT INTO trades(date, symbol, segment, direction, qty, entry, exit, fees, sl, tp, timeframe, strategy, tags, notes, screenshot_path)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(trade_date),
                    symbol,
                    "Futures",
                    direction,
                    qty,
                    entry,
                    exit_price,
                    fees,
                    sl,
                    tp,
                    timeframe,
                    strategy,
                    ",".join(checklist) if checklist else tags,
                    notes,
                    screenshot.name if screenshot else None,
                ),
            )
            st.success("Trade stored. Attachments saved via desktop bundler logic.")


def render_analytics(trades: pd.DataFrame):
    st.subheader("Analytics")
    if trades.empty:
        st.info("No trades to show. Import data or enable demo mode.")
        return
    trades = trades.copy()
    trades["equity"] = trades["pnl"].cumsum()
    st.pyplot(charts.equity_curve_chart(trades.set_index("date")["equity"]))
    st.plotly_chart(charts.histogram(trades["pnl"]))
    streaks = trades["pnl"].gt(0).astype(int)
    st.write("Win/Loss streak proxy:", streaks.value_counts())
    st.dataframe(trades.groupby("strategy")["pnl"].sum().rename("P&L"))

    st.subheader("Filters")
    with st.expander("Filter trades"):
        selected_symbol = st.selectbox("Symbol", ["All"] + sorted(trades["symbol"].unique().tolist()))
        if selected_symbol != "All":
            trades = trades[trades["symbol"] == selected_symbol]
        st.write("Filtered rows", len(trades))

    st.subheader("Reports")
    st.button("Generate Daily PDF", help="Use utils.io + PDF engine to export.")
    st.button("Generate Weekly PDF")


def main() -> None:
    st.title("📒 Trading Journal")
    st.caption("Local-first journal with attachments, tags, and analytics.")

    demo = st.checkbox("Use demo data", value=True)
    trades = load_demo_trades() if demo else pd.DataFrame()

    render_import_section()
    render_manual_entry()

    st.divider()
    render_analytics(trades)

    st.subheader("Tag Editor")
    st.text_area("Manage tags", json.dumps({"Opening": "First 30m trades"}, indent=2))
    st.button("Save Tags")

    st.subheader("Attachment Gallery")
    st.info("List and preview uploaded screenshots here.")

    st.subheader("Export")
    st.write("Download CSV, XLSX, PDF using the Export Center or quick buttons.")


if __name__ == "__main__":
    main()
