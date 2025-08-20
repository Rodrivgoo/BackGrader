#!/bin/bash
set -e

# Iniciar la API
echo "Iniciando InsightGrader API..."
# Asegúrate de que la variable GOOGLE_API_KEY se pasa al contenedor Docker en el `docker run` o `docker build`
# El Dockerfile ya está configurado para tomarla de --build-arg
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
