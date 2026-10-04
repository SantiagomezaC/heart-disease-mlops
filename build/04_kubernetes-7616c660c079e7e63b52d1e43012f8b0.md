---
title: Parte III.2. Orquestación con Kubernetes local
short_title: III.2 Kubernetes
---

**Predicción de falla cardíaca con *pipelines* de aprendizaje automático** · Etapa 4 del proyecto integrador

Manuel Meza · Kevin Clemente — Machine Learning (maestría), Prof. Dr. Lihki Rubio — Octubre de 2026

---

**Resumen.** El contenedor de la Parte III.1 se despliega en un clúster local de Kubernetes creado con Minikube (Kubernetes 1.37). Un `Deployment` mantiene dos réplicas del servicio con sondas de disponibilidad y de vida, y un `Service` de tipo `LoadBalancer` las expone en el puerto 80. El despliegue completó su *rollout*, respondió predicciones a través del `Service` y escaló a tres réplicas sin interrupción.

## 1. Manifiestos

El `Deployment` parte del manifiesto del enunciado y lo completa en tres aspectos. La imagen, indicada en el enunciado como `<TU_USUARIO_DOCKER>/heart-api`, se publica en GitHub Container Registry desde el flujo de integración continua (`ghcr.io/santiagomezac/heart-api:latest`), lo que evita gestionar credenciales de Docker Hub; con `imagePullPolicy: IfNotPresent`, Minikube usa la imagen construida localmente si ya está cargada y, en otro caso, la descarga del registro público. El número de réplicas se eleva de una a dos para que el `Service` tenga más de un destino y el servicio sobreviva a la caída de un pod. Por último, se declaran solicitudes y límites de CPU y memoria y dos sondas HTTP sobre `/health`: la de disponibilidad (*readiness*) impide que el `Service` envíe tráfico a un pod que aún está cargando el modelo, y la de vida (*liveness*) reinicia el contenedor si deja de responder.

```{literalinclude} ../k8s/deployment.yaml
:language: yaml
:caption: k8s/deployment.yaml
```

```{literalinclude} ../k8s/service.yaml
:language: yaml
:caption: k8s/service.yaml
```

## 2. Despliegue en Minikube

```bash
minikube start --driver=docker
minikube image load ghcr.io/santiagomezac/heart-api:latest
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get svc
```

En Minikube un `Service` de tipo `LoadBalancer` permanece con `EXTERNAL-IP` en estado `<pending>` porque no existe un balanceador de nube que le asigne una dirección. Para obtenerla se ejecuta `minikube tunnel` en una terminal aparte; alternativamente, `minikube service heart-service --url` devuelve una URL accesible a través del `NodePort` que Kubernetes asigna automáticamente, y `kubectl port-forward svc/heart-service 8080:80` redirige un puerto local al `Service`. La secuencia completa, con la prueba del `Service` y el escalado, está automatizada en `scripts/deploy_local.ps1`.

## 3. Evidencia de ejecución

```{literalinclude} ../reports/evidence/04_kubectl_get.txt
:language: text
:caption: Estado del Deployment, los pods y el Service tras el rollout
```

```{literalinclude} ../reports/evidence/04_k8s_predict.json
:language: json
:caption: Predicción servida a través del Service
```

```{literalinclude} ../reports/evidence/04_kubectl_scale.txt
:language: text
:caption: Escalado horizontal a tres réplicas
```

Las dos réplicas alcanzaron el estado `Running` y quedaron disponibles (2/2) en menos de 15 segundos. El `Service` recibió la IP de clúster 10.111.143.225 y el `NodePort` 32181, y respondió la misma probabilidad (0.7452) que el contenedor aislado. El escalado a tres réplicas se completó sin interrumpir el servicio, porque las sondas de disponibilidad retienen el tráfico hacia el pod nuevo hasta que carga el modelo. El mismo despliegue se repite en cada ejecución del flujo de integración continua sobre un clúster efímero creado con kind (Parte III.3).
