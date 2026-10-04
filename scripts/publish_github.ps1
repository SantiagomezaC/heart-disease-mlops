<#
.SYNOPSIS
  Crea el repositorio público en GitHub y publica el proyecto (dispara el CI).

.NOTES
  Requiere GitHub CLI autenticado una única vez:  gh auth login
  Uso (desde la raíz del repositorio):
      powershell -ExecutionPolicy Bypass -File scripts/publish_github.ps1
#>
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
gh auth status
gh repo create SantiagomezaC/heart-disease-mlops --public `
    --description "MLOps local para predicción de falla cardíaca: Pipeline + GridSearchCV, FastAPI, Docker, Kubernetes, GitHub Actions y Evidently" `
    --source . --remote origin --push
gh run watch --exit-status
