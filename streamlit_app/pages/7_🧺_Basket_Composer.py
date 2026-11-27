"""Strategy basket composer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from utils import charts, db  # noqa: E402


def demo_components() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": ["Mean Revert", "Breakout", "Options"],
            "weight": [0.4, 0.4, 0.2],
            "equity": [np.linspace(10000, 12000, 50), np.linspace(10000, 15000, 50), np.linspace(10000, 13000, 50)],
        }
    )


def main() -> None:
    st.title("🧺 Strategy Basket Composer")
    st.caption("Blend strategies into deployable portfolios.")

    strategies = [row["name"] for row in db.fetch_all("SELECT name FROM strategies")] or ["Mean Revert", "Breakout"]
    selected = st.multiselect("Select strategies", strategies, default=strategies)
    weights = {name: st.slider(f"Weight for {name}", 0.0, 1.0, 1 / max(len(selected), 1)) for name in selected}

    st.subheader("Constraints")
    st.number_input("Max concurrent positions", value=5, step=1)
    st.number_input("Segment cap %", value=30, step=5)

    st.subheader("Backfill & Simulation")
    st.write("Use journal trades or uploaded return series to compute portfolio equity.")

    comp = demo_components()
    equity_df = pd.DataFrame({row.strategy: pd.Series(row.equity) for row in comp.itertuples()})
    st.plotly_chart(charts.multi_equity_plot(equity_df.to_dict(orient="series")))

    st.subheader("Attribution")
    st.bar_chart(pd.Series(weights))

    st.subheader("Rebalancing")
    st.selectbox("Rule", ["Monthly", "Quarterly", "Drift > 5%"])


if __name__ == "__main__":
    main()
