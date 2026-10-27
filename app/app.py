"""Streamlit prediction app.

Run from the repo root:
    streamlit run app/app.py

Week 1 version: uses the Poisson model only (fitted on the last 2 seasons).
TODO (week 4, Chhoungkorng): load the best ML model from results/models/ and show its
probabilities; keep Poisson for the scoreline table.
Owner: Chhoungkorng
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.load_data import read_matches                       # noqa: E402
from src.poisson import (expected_goals, fit_poisson, outcome_probs_from_grid,  # noqa: E402
                         scoreline_grid, top_scorelines)


@st.cache_resource
def load():
    m = read_matches()
    last_season = m["season"].max()
    recent = m[m.date >= m.date.max() - pd.Timedelta(days=730)]
    model = fit_poisson(recent, xi=0.0019)
    teams = sorted(set(m.loc[m.season == last_season, "home"]))
    return model, teams, last_season


st.title("Premier League match predictor")
model, teams, season = load()
st.caption(f"Teams from season {season[:2]}/{season[2:]}. Poisson model fitted on the last 2 seasons.")

col1, col2 = st.columns(2)
home = col1.selectbox("Home team", teams)
away = col2.selectbox("Away team", [t for t in teams if t != home])

if st.button("Predict"):
    lh, la = expected_goals(model, [home], [away])
    grid = scoreline_grid(lh[0], la[0])
    ph, pd_, pa = outcome_probs_from_grid(grid)
    st.subheader("Result probabilities")
    st.bar_chart(pd.DataFrame({"probability": [ph, pd_, pa]},
                              index=["Home win", "Draw", "Away win"]))
    st.write(f"Expected goals: **{home} {lh[0]:.2f} – {la[0]:.2f} {away}**")
    st.subheader("Most likely scorelines")
    top = pd.DataFrame(top_scorelines(grid), columns=["probability", "score"])
    top["probability"] = (top["probability"] * 100).round(1).astype(str) + "%"
    st.table(top[["score", "probability"]])

st.caption("For learning and fan use only. Not betting advice.")
