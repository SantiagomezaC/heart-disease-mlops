"""Pruebas de calidad del modelo serializado (umbral mínimo de desempeño)."""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]
MIN_TEST_AUC = 0.90
MIN_TEST_ACCURACY = 0.85


@pytest.fixture(scope="module")
def model():
    return joblib.load(ROOT / "app" / "model.joblib")


@pytest.fixture(scope="module")
def test_split():
    df = pd.read_csv(ROOT / "data" / "heart.csv")
    _, X_test, _, y_test = train_test_split(
        df[FEATURES], df["HeartDisease"], test_size=0.2,
        stratify=df["HeartDisease"], random_state=42)
    return X_test, y_test


def test_model_is_full_pipeline(model):
    assert list(model.named_steps) == ["prep", "reduce", "clf"]
    assert list(model.feature_names_in_) == FEATURES


def test_model_meets_auc_threshold(model, test_split):
    X_test, y_test = test_split
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    assert auc >= MIN_TEST_AUC, f"AUC de prueba {auc:.3f} < {MIN_TEST_AUC}"


def test_model_meets_accuracy_threshold(model, test_split):
    X_test, y_test = test_split
    acc = accuracy_score(y_test, model.predict(X_test))
    assert acc >= MIN_TEST_ACCURACY


def test_metadata_matches_model(model, test_split):
    meta = json.loads((ROOT / "app" / "model_metadata.json").read_text(
        encoding="utf-8"))
    X_test, y_test = test_split
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    assert meta["features"] == FEATURES
    assert auc == pytest.approx(meta["metrics"]["test_roc_auc"], abs=1e-3)


def test_root_copy_is_identical():
    a = (ROOT / "app" / "model.joblib").read_bytes()
    b = (ROOT / "model.joblib").read_bytes()
    assert a == b
