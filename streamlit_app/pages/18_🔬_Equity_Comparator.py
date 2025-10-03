"""Equity curve comparator."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from utils import charts  # noqa: E402


def main() -> None:
    st.title("🔬 Equity Curve Comparator")
    st.caption("Overlay multiple strategies or baskets to inspect correlation and alpha.")

    series_count = st.slider("Number of series", 2, 5, 3)
    data = {
        f"Series {i+1}": pd.Series(
            np.cumprod(1 + np.random.normal(0.0005 * (i + 1), 0.01, 200)) * 10000
        )
        for i in range(series_count)
    }

    st.plotly_chart(charts.multi_equity_plot(data))
    frame = pd.DataFrame(data)
    st.subheader("Correlation matrix")
    st.dataframe(frame.pct_change().corr())

    st.subheader("Drawdown overlap")
    dd = frame.apply(lambda col: col / col.cummax() - 1)
    st.area_chart(dd)


if __name__ == "__main__":
    main()
