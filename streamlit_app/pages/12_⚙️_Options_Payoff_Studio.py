"""Options strategy payoff visualiser."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def payoff(strikes, premiums, lots, price_grid):
    payoff_matrix = np.zeros_like(price_grid, dtype=float)
    for strike, premium, lot in zip(strikes, premiums, lots):
        intrinsic = np.maximum(price_grid - strike, 0)
        payoff_matrix += (intrinsic - premium) * lot
    return payoff_matrix


def main() -> None:
    st.title("⚙️ Options Strategy Studio")
    st.caption("Build custom spreads, visualise payoff, and review Greeks approximations.")

    cols = st.columns(3)
    strikes = st.text_input("Strikes", value="95,100,105")
    premiums = st.text_input("Premiums", value="3.5,2.0,1.0")
    lots = st.text_input("Lots", value="1,-2,1")
    expiry = cols[0].date_input("Expiry")
    implied_vol = cols[1].number_input("Implied vol", value=0.25)
    margin = cols[2].number_input("Margin estimate", value=1000.0)

    strikes_list = [float(x) for x in strikes.split(",")]
    premiums_list = [float(x) for x in premiums.split(",")]
    lots_list = [float(x) for x in lots.split(",")]
    price_grid = np.linspace(min(strikes_list) * 0.8, max(strikes_list) * 1.2, 100)
    payoff_values = payoff(strikes_list, premiums_list, lots_list, price_grid)

    chart_df = pd.DataFrame({"Price": price_grid, "Payoff": payoff_values})
    st.line_chart(chart_df.set_index("Price"))

    st.subheader("Break-even points")
    breakevens = price_grid[np.isclose(payoff_values, 0, atol=0.5)]
    st.write(np.unique(np.round(breakevens, 2)))

    st.subheader("Templates")
    st.selectbox("Template", ["Custom", "Iron Condor", "Vertical Spread", "Butterfly", "Calendar"])


if __name__ == "__main__":
    main()
