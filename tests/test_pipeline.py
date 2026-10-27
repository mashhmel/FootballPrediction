"""Run from the repo root:  python -m pytest -q

These tests use FAKE data (tests/fake_data.py) so they run offline.
The leakage tests are the important ones: if they fail, a feature is peeking at the future.
"""
import numpy as np
import pytest

from src.evaluate import brier, evaluate, rps, split
from src.features import SETUPS, build_features
from src.poisson import outcome_probs_from_grid, scoreline_grid, walk_forward_probs
from tests.fake_data import make_fake_matches


@pytest.fixture(scope="module")
def matches():
    return make_fake_matches()


@pytest.fixture(scope="module")
def features(matches):
    return build_features(matches)


# ---- metrics -----------------------------------------------------------------------------
def test_rps_perfect_is_zero():
    assert rps([0, 1, 2], np.eye(3)) == 0


def test_rps_values():
    # home win happened; predicting a draw is "less wrong" than predicting an away win
    assert rps([0], [[0, 1, 0]]) < rps([0], [[0, 0, 1]])
    assert rps([0], [[0, 0, 1]]) == pytest.approx(1.0)


def test_brier_range():
    assert brier([0], [[1, 0, 0]]) == 0
    assert brier([0], [[0, 0, 1]]) == pytest.approx(2.0)


# ---- data shape ------------------------------------------------------------------------
def test_shape(matches, features):
    assert len(features) == len(matches) == 20 * 380
    assert features["match_id"].is_unique


def test_no_missing_in_setup_c(features):
    assert features[SETUPS["C"]].isna().sum().sum() == 0


def test_splits_are_by_season(features):
    train, val, test = split(features)
    assert train.date.max() < val.date.min() < test.date.min()
    assert len(test) == 760 and len(val) == 380


# ---- LEAKAGE -----------------------------------------------------------------------------
def _flip(result):
    return {"H": "A", "A": "H", "D": "H"}[result]


def test_no_leakage_from_own_result(matches, features):
    """Changing a match's own result/score must NOT change that match's features."""
    target = len(matches) // 2
    m2 = matches.copy()
    m2.loc[target, ["hg", "ag"]] = m2.loc[target, ["ag", "hg"]].values + [3, 0]
    m2.loc[target, "result"] = _flip(m2.loc[target, "result"])
    m2.loc[target, ["hst", "ast"]] = [20, 0]
    f2 = build_features(m2)
    cols = SETUPS["C"]
    row1 = features.loc[features.match_id == target, cols].to_numpy()
    row2 = f2.loc[f2.match_id == target, cols].to_numpy()
    assert np.allclose(row1, row2, equal_nan=True)


def test_no_leakage_from_future(matches, features):
    """Changing every result in the LAST season must not change any earlier match's features."""
    m2 = matches.copy()
    last = m2.season == "2425"
    m2.loc[last, ["hg", "ag"]] = m2.loc[last, ["ag", "hg"]].values
    m2.loc[last, "result"] = m2.loc[last, "result"].map(_flip)
    f2 = build_features(m2)
    early = features.season != "2425"
    cols = SETUPS["C"]
    a = features.loc[early].sort_values("match_id")[cols].to_numpy()
    b = f2.loc[f2.season != "2425"].sort_values("match_id")[cols].to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_poisson_walk_forward_no_leakage(matches):
    """Poisson probs for a month must not change if a LATER month's results change."""
    sub = matches[matches.season.isin(["2223", "2324"])].copy()
    p1 = walk_forward_probs(sub, min_train=100)
    sub2 = sub.copy()
    late = sub2.date >= sub2.date.max() - np.timedelta64(30, "D")
    sub2.loc[late, ["hg", "ag"]] = sub2.loc[late, ["ag", "hg"]].values
    p2 = walk_forward_probs(sub2, min_train=100)
    early_ids = sub.loc[sub.date < sub.date.max() - np.timedelta64(60, "D"), "match_id"]
    a = p1[p1.match_id.isin(early_ids)].sort_values("match_id")["pois_pH"].to_numpy()
    b = p2[p2.match_id.isin(early_ids)].sort_values("match_id")["pois_pH"].to_numpy()
    assert len(a) > 0 and np.allclose(a, b, equal_nan=True)


# ---- Poisson ---------------------------------------------------------------------------
def test_poisson_probs_sum_to_one():
    ph, pd_, pa = outcome_probs_from_grid(scoreline_grid(1.6, 1.1))
    assert ph + pd_ + pa == pytest.approx(1.0)
    assert ph > pa                                        # stronger expected attack wins more


# ---- models run end to end -------------------------------------------------------------
def test_models_run(features):
    from src.models import fit_predict
    train, val, _ = split(features)
    for name in ("LR", "RF"):
        _, P = fit_predict(name, train, val, SETUPS["B"])
        assert P.shape == (len(val), 3)
        assert np.allclose(P.sum(axis=1), 1)
        assert evaluate(val.label, P)["accuracy"] > 0.33   # better than random on fake data
