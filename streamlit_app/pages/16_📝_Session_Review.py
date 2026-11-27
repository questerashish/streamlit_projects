"""Rule-based session review assistant."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))


def main() -> None:
    st.title("📝 Session Review Assistant")
    st.caption("Deterministic checklist to evaluate discipline.")

    st.subheader("Checklist")
    broke_risk = st.checkbox("Broke risk rule")
    took_b = st.checkbox("Took B-grade setups")
    revenge = st.checkbox("Revenge traded")
    notes = st.text_area("Notes")

    st.subheader("Scores")
    score = 100
    actions = []
    if broke_risk:
        score -= 30
        actions.append("Reduce size tomorrow and review risk plan.")
    if took_b:
        score -= 20
        actions.append("Limit to A setups only next session.")
    if revenge:
        score -= 25
        actions.append("Implement 15-minute cool-off timer.")

    st.metric("Session score", score)
    st.write("Action items", actions or ["Great job! Keep following the plan."])

    if st.button("Append to journal notes"):
        st.success("Saved to SQLite notes column via trades aggregation (implement DB write).")


if __name__ == "__main__":
    main()
