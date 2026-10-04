---
title: Síntesis y conclusiones
short_title: Conclusiones
---

**Predicción de falla cardíaca con *pipelines* de aprendizaje automático**

Manuel Meza · Kevin Clemente — Machine Learning, Maestría en Ciencias de la Tierra, Prof. Dr. Lihki Rubio — Octubre de 2026

---

## 1. Conclusiones

El proyecto confirma, con evidencia propia, la tesis que articula el capítulo de *pipelines*: el desempeño de un modelo depende de la representación de los datos y de la honestidad del procedimiento con que se evalúa, tanto o más que del algoritmo elegido. Una vez resuelto el preprocesamiento, ocho de los nueve clasificadores del curso se concentraron en una franja de AUC de validación cruzada de apenas 0.011 (de 0.928 a 0.939), con intervalos de confianza en prueba que se solapan casi por completo. En cambio, decisiones de representación como tratar los ceros de colesterol como ausentes y conservar un indicador de ausencia aportaron información que el modelo final aprovecha: el colesterol es la cuarta variable en importancia, y su contribución proviene sobre todo del patrón de ausencia.

Los experimentos de fuga de datos precisan el alcance del `Pipeline`. La encapsulación es indispensable cuando un paso de preprocesamiento usa la variable respuesta (la selección de variables previa a la validación cruzada inventó un AUC de 0.707 sobre etiquetas aleatorias), pero no detecta una variable que ya contiene el desenlace, que exige auditoría explícita. En cambio, el sesgo del preprocesamiento no supervisado ajustado con todos los datos fue despreciable en este conjunto (inferior a 10⁻⁴ en AUC). Reconocer esta jerarquía de riesgos es más útil que tratar toda fuga como igualmente grave.

El modelo desplegado, un XGBoost seleccionado por AUC de validación cruzada (0.939 ± 0.024), obtuvo en su única evaluación sobre prueba un AUC de 0.935 (IC 95 %: 0.895–0.969), con sensibilidad de 0.912 y especificidad de 0.878. La coincidencia entre la estimación interna y la externa, junto con la curva de aprendizaje estabilizada, respalda que el procedimiento de validación no está sesgado y que la flexibilidad del ensamble no se tradujo en pérdida de generalización.

En la dimensión operativa, el mismo artefacto produjo la misma probabilidad (0.7452 para el paciente de referencia) en el cuaderno, en la API local, en el contenedor Docker y en el clúster de Kubernetes, lo que valida la cadena de exportación y empaquetado. El flujo de integración continua convierte esa verificación en una compuerta automática que se repite en cada cambio, y el monitoreo con Evidently detectó la deriva introducida en un lote de producción simulado. Este último ejercicio mostró además que la deriva de las entradas no implica deriva de las predicciones cuando las variables desplazadas tienen poco peso en el modelo, y que ninguna de las dos garantiza que el desempeño se conserve.

## 2. Limitaciones y trabajo futuro

El conjunto de datos es pequeño (918 pacientes) y proviene de cohortes históricas de las décadas de 1980 y 1990 con protocolos distintos, por lo que el desempeño reportado no es transferible sin más a una población actual. Con 184 pacientes de prueba, el intervalo de confianza del AUC tiene una amplitud cercana a 0.07, lo que impide discriminar entre los mejores modelos. Una validación externa con datos de otra institución sería el paso natural antes de cualquier uso clínico.

El umbral de 0.5 de la API, fijado por fidelidad al enunciado, no es necesariamente el más adecuado para tamizaje: el análisis de la Parte II sugiere que un umbral de 0.4 reduciría un tercio de los falsos negativos con una pérdida moderada de especificidad, decisión que corresponde a criterio clínico. Por último, el monitoreo implementado vigila la deriva de covariables; detectar la deriva de concepto exige incorporar los desenlaces reales a medida que estén disponibles y recalcular el desempeño por ventanas, según los lineamientos propuestos en la Parte IV.

## 3. Referencias

- Ambroise, C. y McLachlan, G. J. (2002). Selection bias in gene extraction on the basis of microarray gene-expression data. *PNAS*, 99(10), 6562–6566.
- Chen, T. y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794.
- Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304–310.
- fedesoriano (2021). *Heart Failure Prediction Dataset*. Kaggle. https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- Kaufman, S., Rosset, S., Perlich, C. y Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 1–21.
- Müller, A. C. y Guido, S. (2016). *Introduction to Machine Learning with Python*. O'Reilly Media.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Rabanser, S., Günnemann, S. y Lipton, Z. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. *Advances in Neural Information Processing Systems*, 32.
- Rubio, L. (2023). *Machine Learning*, capítulo 10: Pipelines. https://lihkir.github.io/MachineLearning/chains_pipelines.html
