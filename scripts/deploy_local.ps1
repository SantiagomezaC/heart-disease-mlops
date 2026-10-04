<#
.SYNOPSIS
  Etapas 3 y 4 en local: construye la imagen Docker, la prueba y la despliega
  en Minikube. Guarda la evidencia de cada paso en reports/evidence/.

.NOTES
  Requisitos: Docker Desktop en ejecución, minikube y kubectl.
  Uso (desde la raíz del repositorio):
      powershell -ExecutionPolicy Bypass -File scripts/deploy_local.ps1
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$evidence = Join-Path $root "reports/evidence"
New-Item -ItemType Directory -Force $evidence | Out-Null
$image = "ghcr.io/santiagomezac/heart-api:latest"
$body = '{"features":[54,"M","ASY",140,239,0,"Normal",160,"N",1.2,"Flat"]}'

function Wait-Http($url) {
    for ($i = 0; $i -lt 60; $i++) {
        try { return Invoke-RestMethod $url -TimeoutSec 2 } catch { Start-Sleep 1 }
    }
    throw "El servicio no respondió en $url"
}

Write-Host "== Etapa 3: construcción de la imagen Docker" -ForegroundColor Cyan
docker build -t heart-api -f docker/Dockerfile . 2>&1 | Tee-Object "$evidence/03_docker_build.log"
docker tag heart-api $image
docker images heart-api | Tee-Object "$evidence/03_docker_images.txt"

Write-Host "== Etapa 3: ejecución del contenedor" -ForegroundColor Cyan
docker rm -f heart-api-local 2>$null | Out-Null
docker run -d --name heart-api-local -p 8000:8000 heart-api | Out-Null
$health = Wait-Http "http://localhost:8000/health"
$pred = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/predict" -ContentType "application/json" -Body $body
@{ health = $health; predict_request = ($body | ConvertFrom-Json); predict_response = $pred } |
    ConvertTo-Json -Depth 5 | Tee-Object "$evidence/03_docker_run.json"
docker ps --filter name=heart-api-local | Tee-Object -Append "$evidence/03_docker_images.txt"
docker rm -f heart-api-local | Out-Null

Write-Host "== Etapa 4: clúster local con Minikube" -ForegroundColor Cyan
minikube start --driver=docker
minikube image load $image
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/heart-model --timeout=180s
kubectl get deployments,pods,svc -o wide | Tee-Object "$evidence/04_kubectl_get.txt"

Write-Host "== Etapa 4: prueba del Service" -ForegroundColor Cyan
$pf = Start-Process kubectl -ArgumentList "port-forward svc/heart-service 8080:80" -PassThru -WindowStyle Hidden
try {
    $health = Wait-Http "http://localhost:8080/health"
    $pred = Invoke-RestMethod -Method Post -Uri "http://localhost:8080/predict" -ContentType "application/json" -Body $body
    @{ health = $health; predict_response = $pred } | ConvertTo-Json -Depth 5 |
        Tee-Object "$evidence/04_k8s_predict.json"
} finally {
    Stop-Process -Id $pf.Id -Force
}

Write-Host "== Etapa 4: escalabilidad (3 réplicas)" -ForegroundColor Cyan
kubectl scale deployment/heart-model --replicas=3
kubectl rollout status deployment/heart-model --timeout=180s
kubectl get pods -l app=heart-model -o wide | Tee-Object "$evidence/04_kubectl_scale.txt"
kubectl scale deployment/heart-model --replicas=2 | Out-Null

Write-Host "Listo. Para acceder por la IP del LoadBalancer ejecute 'minikube tunnel' en otra terminal," -ForegroundColor Green
Write-Host "o bien 'minikube service heart-service --url'." -ForegroundColor Green
