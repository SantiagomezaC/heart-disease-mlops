"""Etapa 6 — Monitoreo de deriva de datos con Evidently.

Genera dos reportes:

1. ``drift_report.html`` (raíz del repositorio): compara los datos de
   entrenamiento (referencia) con los de prueba (actuales), tal como
   propone el enunciado. Ambos provienen de la misma población, por lo
   que se espera ausencia de deriva.
2. ``reports/drift_report_simulated.html``: compara la referencia con un
   lote de "producción" simulado con cambios plausibles en la población
   atendida, para verificar que el monitoreo detecta la deriva.

Además escribe ``reports/drift_summary.json`` con el resultado por variable.

Uso (desde la raíz del repositorio):
    python monitoring/drift_report.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]
NUMERICAL = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CATEGORICAL = ["Sex", "ChestPainType", "FastingBS", "RestingECG",
               "ExerciseAngina", "ST_Slope"]
SEED = 42

COLUMN_MAPPING = ColumnMapping(
    target=None,
    prediction="prediction",
    numerical_features=NUMERICAL,
    categorical_features=CATEGORICAL,
)


def load_sets():
    """Conjuntos exportados por notebooks/2_model_pipeline_cv.ipynb."""
    reference = pd.read_csv(ROOT / "data" / "reference.csv")
    current = pd.read_csv(ROOT / "data" / "current.csv")
    cols = FEATURES + ["prediction"]
    return reference[cols], current[cols]


def simulate_production_batch(base: pd.DataFrame, model, n: int = 400,
                              seed: int = SEED) -> pd.DataFrame:
    """Lote de producción con deriva controlada.

    Escenario: el servicio comienza a recibir pacientes de una unidad de
    cardiología geriátrica. Se remuestrea la base y se introducen cambios
    coherentes con esa población: mayor edad (+8 años en promedio), menor
    frecuencia cardíaca máxima, más dolor torácico asintomático y un
    laboratorio que deja de reportar colesterol en el 35 % de los casos.
    """
    rng = np.random.default_rng(seed)
    batch = base.sample(n=n, replace=True, random_state=seed)
    batch = batch[FEATURES].reset_index(drop=True)
    batch["Age"] = np.clip(batch["Age"] + rng.normal(8, 3, n), 28, 95)
    batch["Age"] = batch["Age"].round().astype(int)
    batch["MaxHR"] = np.clip(batch["MaxHR"] - rng.normal(12, 5, n), 60, 202)
    batch["MaxHR"] = batch["MaxHR"].round().astype(int)
    to_asy = rng.random(n) < 0.30
    batch.loc[to_asy, "ChestPainType"] = "ASY"
    no_chol = rng.random(n) < 0.35
    batch.loc[no_chol, "Cholesterol"] = 0
    batch["prediction"] = model.predict(batch[FEATURES])
    return batch


def run_report(reference, current, path: Path) -> dict:
    """Ejecuta DataDriftPreset, guarda el HTML y devuelve un resumen."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current,
               column_mapping=COLUMN_MAPPING)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(path))
    return summarize(report.as_dict())


def summarize(result: dict) -> dict:
    dataset, table = None, None
    for metric in result["metrics"]:
        if metric["metric"] == "DatasetDriftMetric":
            dataset = metric["result"]
        elif metric["metric"] == "DataDriftTable":
            table = metric["result"]
    columns = {
        name: {
            "stattest": info["stattest_name"],
            "drift_score": round(float(info["drift_score"]), 6),
            "threshold": info["stattest_threshold"],
            "drift_detected": bool(info["drift_detected"]),
        }
        for name, info in table["drift_by_columns"].items()
    }
    return {
        "dataset_drift": bool(dataset["dataset_drift"]),
        "drift_share_threshold": dataset["drift_share"],
        "number_of_columns": dataset["number_of_columns"],
        "number_of_drifted_columns": dataset["number_of_drifted_columns"],
        "share_of_drifted_columns": round(
            float(dataset["share_of_drifted_columns"]), 4),
        "columns": columns,
    }


def main():
    reference, current = load_sets()
    model = joblib.load(ROOT / "app" / "model.joblib")
    simulated = simulate_production_batch(current, model)

    summary = {
        "train_vs_test": run_report(
            reference, current, ROOT / "drift_report.html"),
        "train_vs_simulated_production": run_report(
            reference, simulated,
            ROOT / "reports" / "drift_report_simulated.html"),
        "positive_rate": {
            "reference": round(float(reference["prediction"].mean()), 4),
            "current": round(float(current["prediction"].mean()), 4),
            "simulated_production": round(
                float(simulated["prediction"].mean()), 4),
        },
    }
    out = ROOT / "reports" / "drift_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for name in ("train_vs_test", "train_vs_simulated_production"):
        s = summary[name]
        print(f"{name}: deriva del conjunto = {s['dataset_drift']} "
              f"({s['number_of_drifted_columns']}/{s['number_of_columns']} "
              f"columnas con deriva)")
    print("Tasa de predicciones positivas:", summary["positive_rate"])


if __name__ == "__main__":
    main()
