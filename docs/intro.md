---
title: Predicción de falla cardíaca con pipelines de aprendizaje automático
subtitle: Flujo MLOps local — Pipeline y GridSearchCV, FastAPI, Docker, Kubernetes, GitHub Actions y Evidently
short_title: Presentación
---

Proyecto integrador de aprendizaje automático, capítulo 10 (*Pipelines*) · Machine Learning, Maestría en Ciencias de la Tierra · Prof. Dr. Lihki Rubio

## Resumen

Las enfermedades cardiovasculares son la primera causa de muerte en el mundo y su detección temprana condiciona el pronóstico de los pacientes. Este trabajo desarrolla, de extremo a extremo y en un entorno local, el ciclo de vida de un clasificador binario que estima el riesgo de enfermedad cardíaca (`HeartDisease = 1`) a partir de once variables clínicas de rutina del conjunto *Heart Failure Prediction* (918 pacientes).

El análisis exploratorio revela que 172 registros de colesterol están codificados como cero y que esa ausencia se asocia con una prevalencia de enfermedad del 88.4 %, lo que motiva un preprocesamiento que imputa con la mediana de entrenamiento y conserva un indicador de ausencia. Todo el preprocesamiento se encapsula junto con el clasificador en un `Pipeline` optimizado con `GridSearchCV` y validación cruzada estratificada. Una serie de experimentos controlados separa tres mecanismos de fuga de datos: la variable con fuga de objetivo del enunciado lleva el AUC a 1.000 incluso dentro del `Pipeline`; el preprocesamiento no supervisado ajustado con todos los datos introduce un sesgo inferior a 10⁻⁴; y la selección de variables previa a la validación cruzada fabrica un AUC de 0.707 sobre etiquetas sin señal.

De nueve clasificadores vistos en el curso, el modelo seleccionado es un XGBoost que alcanza en el conjunto de prueba un **AUC de 0.935** (IC 95 %: 0.895–0.969), una **exactitud de 0.897**, una sensibilidad de 0.912 y una especificidad de 0.878. El modelo se sirve mediante una API REST con FastAPI, se empaqueta en una imagen Docker, se despliega en Kubernetes local con Minikube, se verifica en cada cambio con GitHub Actions y se vigila con reportes de deriva de datos generados con Evidently.

## Datos y objetivo

| Aspecto | Descripción |
|---|---|
| Fuente | [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (fedesoriano, 2021), que integra las bases de Cleveland, Hungría, Suiza, Long Beach VA y Statlog del repositorio UCI |
| Universo | 918 pacientes, sin duplicados ni valores `NaN` explícitos |
| Variable objetivo | `HeartDisease` (1 = enfermedad cardíaca, 55.3 %; 0 = sin enfermedad, 44.7 %) |
| Predictores | 6 numéricos (`Age`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak`) y 5 categóricos (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) |
| Partición | 80/20 estratificada, semilla 42 (734 / 184 pacientes), realizada antes de cualquier transformación |

## Organización del libro

| Capítulo | Etapa del enunciado | Contenido |
|---|---|---|
| Parte I | 1. Análisis exploratorio y preprocesamiento | EDA, estrategia de preprocesamiento, demostración y diagnóstico de la fuga de datos y comparación de nueve clasificadores |
| Parte II | 2. Entrenamiento seguro | Búsqueda conjunta de modelo e hiperparámetros, matriz de confusión, curva ROC, AUC, sobreajuste, interpretación y exportación |
| Parte III.1 | 3. Despliegue local (FastAPI + Docker) | API REST, imagen de contenedor y evidencia de ejecución |
| Parte III.2 | 4. Orquestación (Kubernetes) | Manifiestos `Deployment` y `Service`, despliegue en Minikube y escalado |
| Parte III.3 | 5. Integración continua (GitHub Actions) | Lint, pruebas automáticas, construcción de la imagen y despliegue en un clúster efímero |
| Parte IV | 6. Monitoreo (Evidently) | Deriva de datos entre entrenamiento, prueba y un lote de producción simulado |
| Síntesis | — | Conclusiones generales y limitaciones |

La estructura de carpetas de la Etapa 0 está en el [repositorio del proyecto](https://github.com/SantiagomezaC/heart-disease-mlops), donde también se encuentra el cuaderno compilado `proyecto_heart_disease_mlops.ipynb`, que reúne todos los capítulos con sus salidas.

## Resultados principales

| Modelo | AUC validación cruzada | AUC prueba | Exactitud prueba |
|---|---|---|---|
| **XGBoost (modelo desplegado)** | **0.939 ± 0.024** | **0.935** | **0.897** |
| Random Forest | 0.934 ± 0.026 | 0.930 | 0.880 |
| Gradient Boosting | 0.934 ± 0.022 | 0.932 | 0.908 |
| k-NN | 0.930 ± 0.032 | 0.939 | 0.908 |
| PCA + SVC (RBF) | 0.929 ± 0.035 | 0.934 | 0.875 |
| Regresión logística | 0.929 ± 0.037 | 0.927 | 0.880 |
| Naive Bayes gaussiano | 0.928 ± 0.035 | 0.927 | 0.886 |
| SVC (RBF) | 0.928 ± 0.036 | 0.935 | 0.880 |
| Árbol de decisión | 0.910 ± 0.028 | 0.887 | 0.777 |

```{figure} ../reports/figures/fig07_curva_roc.png
:width: 55%
:align: center

Curva ROC del modelo desplegado: pliegues de validación cruzada, su promedio y el conjunto de prueba.
```

## Criterios metodológicos

La partición estratificada se realiza sobre los datos crudos antes de cualquier transformación, y todas las transformaciones aprendidas (imputación, escalado, codificación y, cuando corresponde, PCA) se ajustan dentro de cada pliegue mediante `Pipeline`. La selección del modelo se basa exclusivamente en el AUC de validación cruzada; el conjunto de prueba se usa una sola vez, al final, y su incertidumbre se cuantifica con intervalos bootstrap. El umbral de decisión de la API es 0.5, como en el enunciado, y su efecto sobre la sensibilidad se analiza con predicciones fuera de pliegue del entrenamiento. Todo el código es determinista (semilla 42) y las dependencias del servicio están fijadas a las mismas versiones con que se entrenó el modelo.
