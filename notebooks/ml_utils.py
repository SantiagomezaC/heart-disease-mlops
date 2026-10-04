"""Funciones reutilizables para el proyecto de predicción de falla cardíaca.

El módulo concentra la lógica de carga de datos, construcción del
preprocesamiento, entrenamiento con ``GridSearchCV`` y evaluación, de modo
que ambos cuadernos (``1_model_leakage_demo`` y ``2_model_pipeline_cv``)
compartan exactamente las mismas definiciones.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "HeartDisease"

# Variables según su naturaleza clínica y su tipo de dato.
ZERO_AS_MISSING = ["RestingBP", "Cholesterol"]  # un cero es fisiológicamente imposible
NUMERIC = ["Age", "MaxHR", "Oldpeak"]
BINARY_NUMERIC = ["FastingBS"]
CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "heart.csv"


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Lee el conjunto Heart Failure Prediction (Kaggle, fedesoriano 2021)."""
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Columnas ausentes en el archivo: {sorted(missing)}")
    return df[FEATURES + [TARGET]]


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE,
               random_state: int = RANDOM_STATE):
    """Partición estratificada entrenamiento/prueba, previa a toda transformación."""
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=random_state)


def build_preprocessor(scaler=None) -> ColumnTransformer:
    """Construye el ``ColumnTransformer`` de preprocesamiento.

    * ``RestingBP`` y ``Cholesterol``: el valor 0 se trata como faltante, se
      imputa con la mediana del entrenamiento y se añade un indicador binario
      de ausencia, porque el patrón de ausencia es informativo.
    * Numéricas continuas: escalado (``MinMaxScaler`` por defecto, como en el
      enunciado; puede sustituirse con el argumento ``scaler``).
    * Categóricas nominales: codificación one-hot; ``drop='if_binary'`` evita
      la trampa de la variable ficticia en las binarias y
      ``handle_unknown='ignore'`` protege la inferencia ante categorías nuevas.
    """
    scaler = scaler if scaler is not None else MinMaxScaler()
    zero_missing = Pipeline([
        ("imputer", SimpleImputer(missing_values=0, strategy="median",
                                  add_indicator=True)),
        ("scaler", scaler),
    ])
    numeric = Pipeline([("scaler", scaler)])
    categorical = OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("zero_missing", zero_missing, ZERO_AS_MISSING),
            ("num", numeric, NUMERIC),
            ("bin", "passthrough", BINARY_NUMERIC),
            ("cat", categorical, CATEGORICAL),
        ],
        verbose_feature_names_out=True,
    )


def make_cv(n_splits: int = 5, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """Validación cruzada estratificada y reproducible."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def train_pipeline(X_train, y_train, model, param_grid, preprocessor=None,
                   cv=None, scoring="roc_auc", extra_steps=None) -> GridSearchCV:
    """Entrena ``preprocesamiento -> [pasos extra] -> clasificador`` con GridSearchCV.

    Toda transformación se ajusta dentro de cada pliegue, de modo que la
    información de validación nunca participa en el ajuste del preprocesador.
    """
    steps = [("prep", preprocessor if preprocessor is not None else build_preprocessor())]
    steps += list(extra_steps or [])
    steps.append(("clf", model))
    grid = GridSearchCV(
        Pipeline(steps), param_grid=param_grid, cv=cv or make_cv(),
        scoring={"roc_auc": "roc_auc", "accuracy": "accuracy"}, refit=scoring,
        n_jobs=-1, return_train_score=True,
    )
    grid.fit(X_train, y_train)
    return grid


def evaluate_model(estimator, X_test, y_test, threshold: float = 0.5) -> dict:
    """AUC y exactitud en el conjunto de prueba."""
    proba = estimator.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {"test_auc": roc_auc_score(y_test, proba),
            "test_accuracy": accuracy_score(y_test, pred)}


def bootstrap_auc_ci(y_true, proba, n_boot: int = 2000, alpha: float = 0.05,
                     random_state: int = RANDOM_STATE) -> tuple[float, float]:
    """Intervalo de confianza percentil del AUC por remuestreo bootstrap."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    stats = []
    n = len(y_true)
    while len(stats) < n_boot:
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        stats.append(roc_auc_score(y_true[idx], proba[idx]))
    return tuple(np.quantile(stats, [alpha / 2, 1 - alpha / 2]))


def run_benchmark(models: dict, X_train, y_train, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    """Ajusta cada modelo candidato y devuelve la tabla comparativa y los objetos ajustados."""
    rows, fitted = [], {}
    for name, spec in models.items():
        start = time.perf_counter()
        grid = train_pipeline(X_train, y_train, spec["model"], spec["grid"],
                              extra_steps=spec.get("extra_steps"))
        elapsed = time.perf_counter() - start
        best = grid.best_index_
        res = grid.cv_results_
        test = evaluate_model(grid.best_estimator_, X_test, y_test)
        proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
        lo, hi = bootstrap_auc_ci(y_test, proba)
        rows.append({
            "Modelo": name,
            "AUC CV (media)": res["mean_test_roc_auc"][best],
            "AUC CV (d.e.)": res["std_test_roc_auc"][best],
            "Accuracy CV": res["mean_test_accuracy"][best],
            "AUC entrenamiento": res["mean_train_roc_auc"][best],
            "AUC prueba": test["test_auc"],
            "IC95% AUC prueba": f"[{lo:.3f}, {hi:.3f}]",
            "Accuracy prueba": test["test_accuracy"],
            "Combinaciones": len(res["params"]),
            "Tiempo (s)": elapsed,
        })
        fitted[name] = grid
    table = (pd.DataFrame(rows)
             .sort_values(["AUC CV (media)", "AUC prueba"], ascending=False)
             .reset_index(drop=True))
    table.index = table.index + 1
    table.index.name = "Ranking"
    return table, fitted


def univariate_auc_audit(X: pd.DataFrame, y, threshold: float = 0.95) -> pd.DataFrame:
    """Auditoría de fuga de objetivo: AUC univariado de cada predictor.

    Un predictor que por sí solo separa casi perfectamente las clases
    (AUC >= ``threshold`` o <= 1 - ``threshold``) es sospechoso de contener
    información posterior al desenlace y debe revisarse antes de modelar.
    Las variables categóricas se codifican por la tasa de positivos de su
    categoría, calculada sobre los mismos datos auditados.
    """
    y = pd.Series(np.asarray(y), index=X.index)
    rows = []
    for col in X.columns:
        x = X[col]
        if x.dtype == object:
            x = x.map(y.groupby(x).mean())
        auc = roc_auc_score(y, x)
        rows.append({"Variable": col, "AUC univariado": auc,
                     "Poder discriminante |AUC-0.5|": abs(auc - 0.5)})
    out = (pd.DataFrame(rows)
           .sort_values("Poder discriminante |AUC-0.5|", ascending=False)
           .reset_index(drop=True))
    out["Sospecha de fuga"] = np.where(
        (out["AUC univariado"] >= threshold) | (out["AUC univariado"] <= 1 - threshold),
        "Sí", "No")
    return out


# ---------------------------------------------------------------------------
# Estilo gráfico común
# ---------------------------------------------------------------------------
COLORS = {
    "neg": "#2a78d6",      # HeartDisease = 0
    "pos": "#eb6834",      # HeartDisease = 1
    "aux": "#1baf7a",
    "muted": "#8a8984",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e6e5e0",
}
CLASS_LABELS = {0: "Sin enfermedad (0)", 1: "Con enfermedad (1)"}


def set_plot_style() -> None:
    """Estilo sobrio tipo artículo para todas las figuras."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.labelcolor": COLORS["ink2"],
        "axes.edgecolor": "#b9b8b2",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.7,
        "xtick.color": COLORS["ink2"],
        "ytick.color": COLORS["ink2"],
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


def save_figure(fig, name: str) -> Path:
    """Guarda la figura en ``reports/figures`` con resolución de publicación."""
    out = PROJECT_ROOT / "reports" / "figures" / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    return out


# ---------------------------------------------------------------------------
# Modelos candidatos y espacios de búsqueda
# ---------------------------------------------------------------------------
def candidate_models(random_state: int = RANDOM_STATE) -> dict:
    """Clasificadores vistos en el curso con sus rejillas de hiperparámetros.

    Las rejillas se ampliaron hasta que ningún óptimo quedara en el borde del
    espacio explorado (salvo cuando el borde es el valor natural, p. ej.
    ``max_depth=None``).
    """
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    rs = random_state
    return {
        "SVC (RBF)": {
            "model": SVC(probability=True, random_state=rs),
            "grid": {"clf__C": [0.01, 0.1, 1, 10, 100],
                     "clf__gamma": ["scale", 0.001, 0.01, 0.1, 1]}},
        "Regresión logística": {
            "model": LogisticRegression(solver="liblinear", max_iter=5000),
            "grid": {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
                     "clf__penalty": ["l1", "l2"]}},
        "k-NN": {
            "model": KNeighborsClassifier(),
            "grid": {"clf__n_neighbors": list(range(3, 82, 2)),
                     "clf__weights": ["uniform", "distance"],
                     "clf__p": [1, 2]}},
        "Naive Bayes gaussiano": {
            "model": GaussianNB(),
            "grid": {"clf__var_smoothing": np.logspace(-9, 1, 11)}},
        "Árbol de decisión": {
            "model": DecisionTreeClassifier(random_state=rs),
            "grid": {"clf__max_depth": [2, 3, 4, 5, 6, 8, None],
                     "clf__min_samples_leaf": [1, 5, 10, 20, 40],
                     "clf__criterion": ["gini", "entropy"]}},
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=300, random_state=rs),
            "grid": {"clf__max_depth": [None, 4, 6, 8],
                     "clf__max_features": ["sqrt", 0.5],
                     "clf__min_samples_leaf": [1, 3, 5, 10]}},
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=rs),
            "grid": {"clf__n_estimators": [50, 100, 200, 400],
                     "clf__learning_rate": [0.01, 0.05, 0.1],
                     "clf__max_depth": [1, 2, 3],
                     "clf__subsample": [0.8, 1.0]}},
        "XGBoost": {
            "model": XGBClassifier(random_state=rs, eval_metric="logloss", n_jobs=1),
            "grid": {"clf__n_estimators": [100, 300, 600],
                     "clf__learning_rate": [0.005, 0.01, 0.05, 0.1],
                     "clf__max_depth": [2, 3, 4],
                     "clf__subsample": [0.8, 1.0],
                     "clf__colsample_bytree": [0.5, 0.7, 1.0]}},
        "PCA + SVC (RBF)": {
            "model": SVC(probability=True, random_state=rs),
            "extra_steps": [("pca", PCA(random_state=rs))],
            "grid": {"pca__n_components": [2, 4, 6, 8, 10, 12],
                     "clf__C": [0.1, 1, 10],
                     "clf__gamma": ["scale", 0.01, 0.1]}},
    }
