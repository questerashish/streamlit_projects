"""Landing page for the trading desktop suite."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils import db


def ensure_environment() -> None:
    db.init_db()
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)


def main() -> None:
    st.set_page_config(page_title="Trader Control Center", layout="wide")
    ensure_environment()

    st.title("📊 Trader Control Center")
    st.write(
        "Welcome to the all-in-one desktop suite for journaling, analytics, risk and strategy tooling."
    )
    st.success(
        "Use the sidebar navigation to access the 20 dedicated workspaces. "
        "All data is stored locally in SQLite (`app.db`) and the `data/` folder for maximum privacy."
    )

    st.header("Quick Start")
    st.markdown(
        """
        1. Upload or log your first trades via **Trading Journal**.
        2. Explore **Performance Analytics** to review stats.
        3. Configure your **Risk Dashboard** profile in the settings area.
        4. Import or build strategies in the **Strategy Vault**.
        5. Assemble baskets, run backtests, and export polished reports.
        """
    )

    st.header("Demo Data Toggle")
    st.info(
        "Turn on demo data on each page to explore the workflows before importing your own trades."
    )

    st.caption("Desktop build ready — use PyInstaller or Streamlit's native packaging to ship.")


if __name__ == "__main__":
    main()
