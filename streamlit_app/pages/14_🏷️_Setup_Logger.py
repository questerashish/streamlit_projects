"""Setup and pattern logger."""
from __future__ import annotations

import streamlit as st


def main() -> None:
    st.title("🏷️ Setup / Pattern Logger")
    st.caption("Build a gallery of your favourite trade setups with outcomes.")

    st.subheader("Upload annotated chart")
    image = st.file_uploader("Chart image", type=["png", "jpg", "jpeg"])
    tags = st.text_input("Tags", value="ORBO,TrendDay")
    outcome = st.selectbox("Outcome", ["Winner", "Loser", "Scratch"]) 
    notes = st.text_area("Notes")
    link_trade = st.text_input("Link to trade ID")

    if st.button("Save setup"):
        st.success("Image stored to data folder and metadata linked to trades table (extend logic).")

    st.subheader("Gallery")
    st.info("Display stored images with filters by tag and outcome.")


if __name__ == "__main__":
    main()
