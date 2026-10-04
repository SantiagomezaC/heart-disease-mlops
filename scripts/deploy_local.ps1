<#
.SYNOPSIS
  Etapas 3 y 4 en local: construye la imagen Docker, la prueba y la despliega
  en Minikube. Guarda la evidencia de cada paso en reports/evidence/.

.NOTES
  Requisitos: Docker Desktop en ejecución, minikube y kubectl.
  Uso (desde la raíz del repositorio):
      powershell -ExecutionPolicy Bypass -File scripts/deploy_local.ps1
#>
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$evidence = Join-Path $root "reports/evidence"
New-Item -ItemType Directory -Force $evidence | Out-Null
$image = "ghcr.io/santiagomezac/heart-api:latest"
$body = '{"features":[54,"M","ASY",140,239,0,"Normal",160,"N",1.2,"Flat"]}'

function Run([string]$cmd, [string]$log = $null) {
    # Ejecuta un comando nativo combinando stdout y stderr; falla si el código de salida no es 0.
    Write-Host "> $cmd" -ForegroundColor DarkGray
    $out = cmd /c "$cmd 2>&1"
    $code = $LASTEXITCODE
    $out | ForEach-Object { Write-Host $_ }
    if ($log) { "> $cmd" | Out-File -Append -Encoding utf8 $log; $out | Out-File -Append -Encoding utf8 $log }
    if ($code -ne 0) { throw "Falló ($code): $cmd" }
}

function Wait-Http($url) {
    for ($i = 0; $i -lt 90; $i++) {
        try { return Invoke-RestMethod $url -TimeoutSec 2 } catch { Start-Sleep 1 }
    }
    throw "El servicio no respondió en $url"
}

Get-ChildItem $evidence -Filter "0[34]_*" -Exclude "03_api_local.txt" | Remove-Item -Force

Write-Host "== Etapa 3: construcción de la imagen Docker" -ForegroundColor Cyan
Run "docker build --progress=plain -t heart-api -f docker/Dockerfile ." "$evidence/03_docker_build.log"
Run "docker tag heart-api $image"
Run "docker images heart-api" "$evidence/03_docker_images.txt"

Write-Host "== Etapa 3: ejecución del contenedor" -ForegroundColor Cyan
cmd /c "docker rm -f heart-api-local >nul 2>&1"
Run "docker run -d --name heart-api-local -p 8000:8000 heart-api"
$health = Wait-Http "http://localhost:8000/health"
$pred = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/predict" -ContentType "application/json" -Body $body
[ordered]@{ comando = "docker run -p 8000:8000 heart-api"; health = $health
            predict_request = ($body | ConvertFrom-Json); predict_response = $pred } |
    ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 "$evidence/03_docker_run.json"
Run "docker ps --filter name=heart-api-local" "$evidence/03_docker_images.txt"
Run "docker inspect --format ""Estado de salud del contenedor: {{.State.Health.Status}}"" heart-api-local" "$evidence/03_docker_images.txt"
cmd /c "docker rm -f heart-api-local >nul 2>&1"

Write-Host "== Etapa 4: clúster local con Minikube" -ForegroundColor Cyan
Run "minikube start --driver=docker" "$evidence/04_minikube_start.log"
Run "minikube image load $image"
Run "kubectl apply -f k8s/deployment.yaml" "$evidence/04_kubectl_apply.txt"
Run "kubectl apply -f k8s/service.yaml" "$evidence/04_kubectl_apply.txt"
Run "kubectl rollout status deployment/heart-model --timeout=180s" "$evidence/04_kubectl_apply.txt"
Run "kubectl get deployments,pods,svc -o wide" "$evidence/04_kubectl_get.txt"

Write-Host "== Etapa 4: prueba del Service" -ForegroundColor Cyan
$pf = Start-Process kubectl -ArgumentList "port-forward svc/heart-service 8080:80" -PassThru -WindowStyle Hidden
try {
    $health = Wait-Http "http://localhost:8080/health"
    $pred = Invoke-RestMethod -Method Post -Uri "http://localhost:8080/predict" -ContentType "application/json" -Body $body
    [ordered]@{ comando = "kubectl port-forward svc/heart-service 8080:80"; health = $health
                predict_request = ($body | ConvertFrom-Json); predict_response = $pred } |
        ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 "$evidence/04_k8s_predict.json"
} finally {
    Stop-Process -Id $pf.Id -Force
}

Write-Host "== Etapa 4: escalabilidad (3 réplicas)" -ForegroundColor Cyan
Run "kubectl scale deployment/heart-model --replicas=3" "$evidence/04_kubectl_scale.txt"
Run "kubectl rollout status deployment/heart-model --timeout=180s" "$evidence/04_kubectl_scale.txt"
Run "kubectl get pods -l app=heart-model -o wide" "$evidence/04_kubectl_scale.txt"
Run "kubectl scale deployment/heart-model --replicas=2"

Write-Host "Listo. Para acceder por la IP del LoadBalancer ejecute 'minikube tunnel' en otra terminal," -ForegroundColor Green
Write-Host "o bien 'minikube service heart-service --url'." -ForegroundColor Green
