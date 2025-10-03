"""Monte Carlo simulator page."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))


def monte_carlo_paths(returns: pd.Series, paths: int = 100) -> pd.DataFrame:
    sims = []
    for _ in range(paths):
        resampled = returns.sample(len(returns), replace=True).reset_index(drop=True)
        sims.append((1 + resampled).cumprod())
    return pd.DataFrame(sims).T


def main() -> None:
    st.title("🎲 Monte Carlo Simulator")
    st.caption("Bootstrap equity paths to stress test the edge.")

    paths = st.slider("Paths", min_value=10, max_value=500, value=100)
    returns = pd.Series(np.random.normal(0.002, 0.01, 200))
    paths_df = monte_carlo_paths(returns, paths)
    st.line_chart(paths_df, height=250)

    percentiles = paths_df.iloc[-1].quantile([0.05, 0.5, 0.95])
    st.write("Terminal equity percentiles", percentiles)

    st.subheader("Drawdown distribution")
    dd = paths_df.apply(lambda col: col / col.cummax() - 1)
    st.area_chart(dd, height=200)


if __name__ == "__main__":
    main()
