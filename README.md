# Is Head-to-Head History Overrated?

**Predicting Premier League match results with engineered features and a hybrid Poisson–machine learning model.**
ITM-390 Machine Learning group project · Advisor: Kuntha Pin

| Member | Main role | Owns |
|---|---|---|
| Lyzing Seang (leader) | Data, rest days + shots features, report lead | `src/load_data.py` |
| Darareach Nhoeun | Elo, form, home/away form, head-to-head | `src/elo.py`, `src/h2h.py`, `src/features.py` |
| Cheypol Khin | Evaluation, ML models, setups A–D, tuning, SHAP | `src/evaluate.py`, `src/models.py` |
| Chhoungkorng Tang | Poisson / Dixon–Coles, hybrid, Streamlit app | `src/poisson.py`, `app/app.py` |

---

## Quick start

```bash
git clone https://github.com/<your-username>/football-prediction.git
cd football-prediction
pip install -r requirements.txt

python -m src.load_data     # 1. download + clean -> data/processed/matches.csv
python -m src.features      # 2. features         -> data/processed/features.csv
python -m src.poisson       # 3. walk-forward Poisson probabilities -> data/processed/poisson_probs.csv
python -m pytest -q         # 4. tests (includes the leakage tests)
streamlit run app/app.py    # 5. the app
```

Google Colab: open `notebooks/01_week1_quickstart.ipynb` and run the first cell. It clones the repo and installs everything.

**Always run commands from the repo root** (the folder with this README). Otherwise `from src...` imports fail.

---

## Agreed column names (don't rename without telling the group)

| Column | Meaning |
|---|---|
| `match_id` | unique id, in date order |
| `season` | text code, e.g. `"1011"` = 2010/11 |
| `date` | match date |
| `home`, `away` | team names (football-data.co.uk spelling, e.g. `Man United`) |
| `hg`, `ag` | full-time home / away goals |
| `result` | `H`, `D`, `A` |
| `label` | **0 = H, 1 = D, 2 = A** (order matters for RPS) |
| `hs`, `as_`, `hst`, `ast` | shots and shots on target, home / away (**this match: never a feature**) |
| `odds_h`, `odds_d`, `odds_a` | Bet365 odds (**benchmark only, never a feature**) |

Feature columns: `home_*`, `away_*`, `diff_*` for each team feature, plus `elo_home`, `elo_away`, `elo_diff`, `h2h_score`, `h2h_n`, and `pois_pH/pD/pA` (Setup D). The exact lists are `SETUPS` in `src/features.py`.

## Data splits (by season, never random)

| Split | Seasons | Use |
|---|---|---|
| Warm-up | 2005/06 – 2009/10 | Build Elo/form history only |
| Train | 2010/11 – 2021/22 | Fit models |
| Validation | 2022/23 | Choose settings (use freely) |
| Test | 2023/24 – 2024/25 | **Run ONCE**, end of week 3 |

## Feature setups

| Setup | Adds | Research question |
|---|---|---|
| A | Season-to-date points/goals per game | Baseline |
| B | Elo, form, home/away form, rest days, shots-on-target ratio | RQ1: does feature engineering help? |
| C | Smoothed head-to-head | RQ2: is head-to-head overrated? |
| D | Walk-forward Poisson probabilities (hybrid) | RQ3: does the hybrid help? |

## Leakage rules

1. Sort by date before any rolling or loop feature.
2. Every rolling feature uses `shift(1)`. Elo and head-to-head record the value **before** updating.
3. Never fill missing values with an average over the whole dataset (it includes future matches). Use fixed values chosen in advance.
4. Poisson probabilities used as features come from `walk_forward_probs` only.
5. `tests/test_pipeline.py` changes future results and checks that past features don't move. **Run `pytest` after every feature change.**

## Repo layout

```
data/raw/            downloaded season files (never edit by hand)
data/processed/      matches.csv, features.csv, poisson_probs.csv
src/                 shared code (one owner per file)
notebooks/           personal experiments; import from src/
app/app.py           Streamlit app
results/             results_table.csv, figures/, models/
tests/               fake-data tests incl. leakage tests
report/, slides/     final write-up
```

## Team workflow

- Pull before you start, push when you stop (at least once per work session).
- Shared code goes in `src/`. Experiments go in your own notebook.
- Log every run in the Results Log sheet: `date | person | model | setup | settings | split | accuracy | macro_F1 | brier | rps | notes`.
- Check-ins Monday + Thursday: done / next / blocked.

## Status

- [x] Repo structure, data loader, Elo, form/rest/shots features, head-to-head, Poisson + walk-forward, metrics + benchmarks, model training/tuning helpers, basic app, leakage tests
- [ ] Run on real data + sanity checks (Lyzing, week 1)
- [ ] Dixon–Coles upgrade (Chhoungkorng, week 2)
- [ ] Setups A–D validation + tuning (Cheypol, weeks 2–3)
- [ ] Final test run (week 3), SHAP + draws (week 4), app with best ML model (week 4)

*Starter code was drafted with AI assistance (Claude). Read and understand each file before building on it, and mention it in your course's AI-use disclosure if one is required.*
