"""Poisson goal model + walk-forward probabilities (for the Poisson baseline and Setup D).

Expected goals = exp(intercept + home + attack[team] + defence[opponent]), fitted as a
Poisson regression. Optional time-decay weights make recent matches count more.

TODO (week 2, Chhoungkorng): Dixon-Coles low-score correction (rho) - see gameplan 6.6.
Owner: Chhoungkorng
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson

OUT_FILE = Path("data/processed/poisson_probs.csv")
MAX_GOALS = 10


def _long_goals(df):
    return pd.concat([
        pd.DataFrame({"goals": df.hg.values, "team": df.home.values, "opp": df.away.values,
                      "home": 1, "date": df.date.values}),
        pd.DataFrame({"goals": df.ag.values, "team": df.away.values, "opp": df.home.values,
                      "home": 0, "date": df.date.values})], ignore_index=True)


def fit_poisson(train, xi=0.0, ref_date=None):
    """Fit on `train` matches. xi > 0 adds time decay: weight = exp(-xi * days_ago)
    (try xi between 0.0018 and 0.0065 per day)."""
    gl = _long_goals(train)
    weights = None
    if xi > 0:
        ref_date = ref_date or gl["date"].max()
        days = (pd.Timestamp(ref_date) - pd.to_datetime(gl["date"])).dt.days
        weights = np.exp(-xi * days.to_numpy())
    return smf.glm("goals ~ home + C(team) + C(opp)", data=gl,
                   family=sm.families.Poisson(), freq_weights=weights).fit()


def expected_goals(model, homes, aways):
    """Vectorised: expected home goals and away goals for lists of fixtures."""
    lam_h = model.predict(pd.DataFrame({"team": homes, "opp": aways, "home": 1}))
    lam_a = model.predict(pd.DataFrame({"team": aways, "opp": homes, "home": 0}))
    return np.asarray(lam_h), np.asarray(lam_a)


def scoreline_grid(lam_h, lam_a, max_goals=MAX_GOALS):
    """grid[i, j] = P(home scores i, away scores j)."""
    g = np.arange(max_goals + 1)
    return np.outer(poisson.pmf(g, lam_h), poisson.pmf(g, lam_a))


def outcome_probs_from_grid(grid):
    p_home = np.tril(grid, -1).sum()
    p_draw = np.trace(grid)
    p_away = np.triu(grid, 1).sum()
    total = p_home + p_draw + p_away                    # tiny mass lost above MAX_GOALS
    return p_home / total, p_draw / total, p_away / total


def top_scorelines(grid, n=5):
    cells = [(grid[i, j], f"{i}-{j}") for i in range(grid.shape[0]) for j in range(grid.shape[1])]
    return sorted(cells, reverse=True)[:n]


def walk_forward_probs(m, window_days=730, xi=0.0, min_train=300):
    """For every month: fit ONLY on matches before that month (last `window_days`),
    then predict that month. No leakage, so these probabilities can be features (Setup D).

    Teams the model has never seen (e.g. promoted team in August) cannot be predicted;
    those matches get NaN and `pois_missing = 1` (fill them later, see fill_missing_probs).
    """
    m = m.sort_values("date")
    out = []
    for mon in m["date"].dt.to_period("M").unique():
        start = mon.to_timestamp()
        train = m[(m.date < start) & (m.date >= start - pd.Timedelta(days=window_days))]
        target = m[m.date.dt.to_period("M") == mon]
        if len(train) < min_train:
            continue
        model = fit_poisson(train, xi=xi, ref_date=start)
        known = set(train.home) | set(train.away)
        ok = target[target.home.isin(known) & target.away.isin(known)]
        if len(ok):
            lh, la = expected_goals(model, ok.home.values, ok.away.values)
            for mid, a, b in zip(ok.match_id, lh, la):
                out.append((mid, *outcome_probs_from_grid(scoreline_grid(a, b)), a, b, 0))
        for mid in target.match_id[~target.match_id.isin(ok.match_id)]:
            out.append((mid, np.nan, np.nan, np.nan, np.nan, np.nan, 1))
    return pd.DataFrame(out, columns=["match_id", "pois_pH", "pois_pD", "pois_pA",
                                      "pois_xg_h", "pois_xg_a", "pois_missing"])


def fill_missing_probs(probs, fallback=(0.45, 0.25, 0.30)):
    """Simple documented rule: unseen-team matches get historical H/D/A frequencies."""
    p = probs.copy()
    for col, val in zip(["pois_pH", "pois_pD", "pois_pA"], fallback):
        p[col] = p[col].fillna(val)
    return p


def main():
    from src.load_data import read_matches
    probs = walk_forward_probs(read_matches())
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    probs.to_csv(OUT_FILE, index=False)
    print(f"saved {len(probs)} rows -> {OUT_FILE}; unseen-team matches: {int(probs.pois_missing.sum())}")


if __name__ == "__main__":
    main()
