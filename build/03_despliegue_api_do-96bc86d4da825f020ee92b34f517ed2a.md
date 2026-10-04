---
title: Parte III.1. Despliegue local con FastAPI y Docker
short_title: III.1 FastAPI y Docker
---

**Predicción de falla cardíaca con *pipelines* de aprendizaje automático** · Etapa 3 del proyecto integrador

Manuel Meza · Kevin Clemente — Machine Learning, Maestría en Ciencias de la Tierra, Prof. Dr. Lihki Rubio — Octubre de 2026

---

**Resumen.** El pipeline exportado en la Parte II se sirve mediante una API REST construida con FastAPI y se empaqueta en una imagen Docker basada en `python:3.10-slim`. La API conserva el contrato del enunciado (`POST /predict` con una lista de variables y respuesta `heart_disease_probability`, `prediction`) y lo complementa con un endpoint de campos nombrados y validados, una sonda de salud y un endpoint de metadatos. La imagen se construyó y ejecutó en local; el contenedor alcanzó el estado `healthy` y reprodujo exactamente las probabilidades calculadas en el cuaderno.

## 1. Diseño de la API

El archivo `app/api.py` carga una única vez el objeto `model.joblib`, que contiene el `Pipeline` completo: imputación, escalado, codificación y clasificador. En consecuencia, la API recibe las variables clínicas en crudo y no replica ninguna lógica de preprocesamiento, lo que elimina la principal fuente de desajuste entre entrenamiento y producción (*training-serving skew*).

El enunciado propone transformar la entrada con `np.array(data.features).reshape(1, -1)`. Esa conversión no es aplicable a este modelo, porque el `ColumnTransformer` selecciona las columnas por nombre y mezcla variables de texto y numéricas, que un arreglo de NumPy convertiría a un único tipo. La API conserva el mismo esquema de entrada, una lista con las once variables, pero la convierte en un `DataFrame` con los nombres de columna en el orden de `FEATURES`, valida la longitud y el tipo de cada valor y responde con un error 422 ante entradas mal formadas.

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Información del servicio y orden esperado de las variables |
| GET | `/health` | Sonda de vida y disponibilidad, usada por Docker y Kubernetes |
| GET | `/model-info` | Estimador, métricas de validación y prueba y fecha de entrenamiento |
| POST | `/predict` | Contrato del enunciado: lista ordenada de 11 valores |
| POST | `/predict/patient` | Objeto con campos nombrados; las categorías y los rangos se validan con Pydantic |

El orden de `features` es `Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope`. La clase predicha usa el umbral de 0.5 del enunciado (`prediction = int(proba > 0.5)`).

```{literalinclude} ../app/api.py
:language: python
:caption: app/api.py
```

## 2. Exportación del modelo

La exportación se realiza al final de la Parte II con `joblib.dump(grid.best_estimator_, "app/model.joblib")`, como indica el enunciado. Se deja una copia en la raíz del repositorio (`model.joblib`), conforme a la estructura de la Etapa 0, y se guarda `app/model_metadata.json` con las métricas, los hiperparámetros, el orden de las variables y las versiones de las librerías. Una prueba automática verifica que ambas copias sean idénticas byte a byte y que las métricas declaradas coincidan con las que el modelo reproduce sobre el conjunto de prueba.

## 3. Imagen Docker

El `Dockerfile` sigue el del enunciado y añade cuatro prácticas de producción: variables de entorno que evitan archivos `.pyc` y el búfer de salida, un usuario sin privilegios, la declaración del puerto expuesto y un `HEALTHCHECK` que consulta `/health` cada 30 segundos. Las dependencias están fijadas a las versiones exactas con que se entrenó el modelo, porque un objeto serializado con `joblib` solo es portable entre versiones compatibles de scikit-learn y XGBoost. Se usa `xgboost-cpu`, que contiene la misma clase `XGBClassifier` pero excluye las bibliotecas de GPU, innecesarias para inferencia y responsables de varios cientos de megabytes. El archivo `.dockerignore` excluye datos, cuadernos y reportes del contexto de construcción.

```{literalinclude} ../docker/Dockerfile
:language: docker
:caption: docker/Dockerfile
```

```{literalinclude} ../docker/requirements.txt
:language: text
:caption: docker/requirements.txt
```

Construcción y ejecución local, desde la raíz del repositorio:

```bash
docker build -t heart-api -f docker/Dockerfile .
docker run -p 8000:8000 heart-api
```

## 4. Evidencia de ejecución

La API se ejecutó primero con `uvicorn` y luego dentro del contenedor, en Windows 11 con Docker Desktop 29.8. La imagen resultante ocupa 159 MB comprimida y el contenedor alcanza el estado `healthy` tras el periodo de arranque del `HEALTHCHECK`.

```{literalinclude} ../reports/evidence/03_docker_images.txt
:language: text
:caption: Imagen construida y estado de salud del contenedor
```

```{literalinclude} ../reports/evidence/03_docker_run.json
:language: json
:caption: Respuesta de /health y /predict desde el contenedor
```

```{literalinclude} ../reports/evidence/03_api_local.txt
:language: text
:caption: API servida con uvicorn, incluido el rechazo de una entrada de longitud incorrecta
```

La probabilidad devuelta para el paciente de ejemplo, 0.7452, es idéntica en la ejecución con `uvicorn`, en el contenedor y en el clúster de Kubernetes (Parte III.2), lo que confirma que el artefacto desplegado es exactamente el modelo evaluado. El registro completo de la construcción sin caché se encuentra en `reports/evidence/03_docker_build.log`.
