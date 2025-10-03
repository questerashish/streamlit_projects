"""Walk-forward optimisation page."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))


def main() -> None:
    st.title("🧪 Walk-Forward & OOS Validator")
    st.caption("Split data into training/testing folds to fight overfitting.")

    data_points = st.slider("Data points", min_value=100, max_value=1000, value=300)
    folds = st.slider("Folds", min_value=2, max_value=10, value=3)
    param_grid = st.text_area("Parameter grid", "sma=10,20,50\nrsi=20,30")

    st.subheader("Results")
    metrics = pd.DataFrame(
        {
            "fold": range(1, folds + 1),
            "in_sample": np.random.uniform(0.5, 1.5, folds),
            "out_sample": np.random.uniform(0.3, 1.2, folds),
        }
    )
    metrics["stability"] = metrics["out_sample"] / metrics["in_sample"]
    st.dataframe(metrics)
    st.line_chart(metrics.set_index("fold"))

    st.subheader("Best-on-median heuristic")
    st.write("Select params that perform closest to the median across folds.")


if __name__ == "__main__":
    main()
