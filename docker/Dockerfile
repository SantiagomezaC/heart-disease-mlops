# Imagen de inferencia del modelo de falla cardíaca (Etapa 3).
# Construcción desde la raíz del repositorio:
#   docker build -t heart-api -f docker/Dockerfile .
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Ejecución con un usuario sin privilegios
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
