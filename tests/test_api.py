"""Pruebas funcionales de la API de inferencia (Etapa 3)."""

import pytest
from fastapi.testclient import TestClient

from app.api import FEATURES, app

client = TestClient(app)

HIGH_RISK = [65, "M", "ASY", 150, 0, 1, "ST", 100, "Y", 2.5, "Flat"]
LOW_RISK = [35, "F", "ATA", 120, 200, 0, "Normal", 180, "N", 0.0, "Up"]


def as_patient(values):
    return dict(zip(FEATURES, values))


def test_root_lists_feature_order():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["features_order"] == FEATURES


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info_has_metrics():
    body = client.get("/model-info").json()
    assert body["decision_threshold"] == 0.5
    assert body["metrics"]["test_roc_auc"] > 0.5


@pytest.mark.parametrize("values", [HIGH_RISK, LOW_RISK])
def test_predict_contract(values):
    r = client.post("/predict", json={"features": values})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"heart_disease_probability", "prediction"}
    assert 0.0 <= body["heart_disease_probability"] <= 1.0
    assert body["prediction"] == int(body["heart_disease_probability"] > 0.5)


def test_predict_separates_clinical_profiles():
    high = client.post("/predict", json={"features": HIGH_RISK}).json()
    low = client.post("/predict", json={"features": LOW_RISK}).json()
    assert high["prediction"] == 1
    assert low["prediction"] == 0
    assert high["heart_disease_probability"] > low[
        "heart_disease_probability"]


def test_both_endpoints_agree():
    a = client.post("/predict", json={"features": HIGH_RISK}).json()
    b = client.post("/predict/patient", json=as_patient(HIGH_RISK)).json()
    assert a == pytest.approx(b)


def test_predict_rejects_wrong_length():
    r = client.post("/predict", json={"features": HIGH_RISK[:-1]})
    assert r.status_code == 422


def test_predict_rejects_non_numeric_value():
    bad = list(HIGH_RISK)
    bad[0] = "sesenta"
    r = client.post("/predict", json={"features": bad})
    assert r.status_code == 422


def test_patient_rejects_invalid_category():
    patient = as_patient(HIGH_RISK)
    patient["ChestPainType"] = "XYZ"
    r = client.post("/predict/patient", json=patient)
    assert r.status_code == 422


def test_predict_tolerates_unseen_category():
    # /predict no restringe categorías: el OneHotEncoder las ignora.
    values = list(LOW_RISK)
    values[6] = "Desconocido"
    r = client.post("/predict", json={"features": values})
    assert r.status_code == 200
