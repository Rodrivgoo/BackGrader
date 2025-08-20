FROM python:3.10-slim

# Instalar dependencias del sistema: poppler para pdf2image
# Tesseract OCR ya no es necesario
RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivo de credenciales de Google Cloud y establecer la variable de entorno
# Asegúrate de que el nombre del archivo local (primero) coincida con tu archivo JSON
COPY rag-n8n-454003-b80e9a6d04a8.json /app/gcp_credentials.json
ENV GOOGLE_APPLICATION_CREDENTIALS /app/gcp_credentials.json

# Copiar archivos de requisitos
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Crear directorio para archivos temporales
RUN mkdir -p temp

# Exponer puerto para la API
EXPOSE 8000

# Script de inicio
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Comando para ejecutar la aplicación
CMD ["/start.sh"] 