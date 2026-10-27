"""ML models (Logistic Regression, Random Forest, XGBoost) for any setup.

A setup is just a list of column names (see src/features.py SETUPS).
Owner: Cheypol
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import evaluate, split

# small grids for week 3 tuning (choose by validation RPS)
GRIDS = {
    "LR": [{"C": c} for c in (0.01, 0.1, 1, 10)],
    "RF": [{"max_depth": d, "min_samples_leaf": s} for d in (4, 6, 8, None) for s in (5, 20, 50)],
    "XGB": [{"max_depth": d, "learning_rate": lr, "n_estimators": n, "subsample": 0.8}
            for d in (2, 3, 4) for lr in (0.03, 0.1) for n in (200, 400)],
}


def make_model(name, **params):
    if name == "LR":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, **params))
    if name == "RF":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, **params))
    if name == "XGB":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                             random_state=42, n_jobs=-1, **params)
    raise ValueError(name)


def fit_predict(name, train, test, cols, **params):
    model = make_model(name, **params).fit(train[cols], train["label"])
    return model, model.predict_proba(test[cols])          # columns: H, D, A


def run_validation(df, setups, models=("LR", "RF", "XGB")):
    """Default settings, train -> validation. One row per (model, setup)."""
    train, val, _ = split(df)
    rows = []
    for s, cols in setups.items():
        for name in models:
            _, P = fit_predict(name, train, val, cols)
            rows.append({"model": name, "setup": s, "split": "val", **evaluate(val.label, P)})
    return pd.DataFrame(rows)


def tune(df, name, cols, grid=None):
    """Try each setting on validation; return (best_params, table sorted by RPS)."""
    train, val, _ = split(df)
    rows = []
    for params in grid or GRIDS[name]:
        _, P = fit_predict(name, train, val, cols, **params)
        rows.append({"params": params, **evaluate(val.label, P)})
    table = pd.DataFrame(rows).sort_values("rps")
    return table.iloc[0]["params"], table


def final_test(df, name, cols, **params):
    """RUN ONCE (end of week 3): retrain on train + val, score on test."""
    train, val, test = split(df)
    model, P = fit_predict(name, pd.concat([train, val]), test, cols, **params)
    return model, P, evaluate(test.label, P)
