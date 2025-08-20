# InsightGrader Backend API

Backend de calificación automática con IA para exámenes académicos. Utiliza Google Cloud Vision para OCR y Google Generative AI para evaluación automatizada.

## 🚀 Características

- **OCR Avanzado**: Extracción de texto de imágenes usando Google Cloud Vision
- **Evaluación con IA**: Calificación automatizada usando Google Generative AI  
- **URLs Firmadas**: Descarga segura de archivos desde Cloudflare R2
- **Procesamiento por Lotes**: Evaluación de múltiples pruebas simultáneamente
- **API RESTful**: Endpoints stateless para integración con frontend

## 📋 Requisitos

- Python 3.8+
- Cuenta de Google Cloud con Vision API y Generative AI habilitados
- Credenciales de Cloudflare R2 (las mismas que el frontend)

## ⚙️ Configuración

### 1. Variables de Entorno

Crear archivo `.env` en la raíz del proyecto:

```bash
# Google API Configuration
GOOGLE_API_KEY=tu_clave_google_api_aqui

# Cloudflare R2 Configuration (mismas credenciales que el frontend)
URL_R2=https://tu_dominio.r2.cloudflarestorage.com
ACCESS_KEY_ID=tu_access_key_id
SECRET_ACCESS_KEY=tu_secret_access_key
BUCKET_NAME=nombre_de_tu_bucket

# Configuración opcional
GOOGLE_MODEL_NAME=gemma-3-27b-it
```

### 2. Obtener Credenciales

#### **Google API Key:**
1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Crea un nuevo proyecto o selecciona uno existente
3. Habilita las APIs necesarias:
   - **Cloud Vision API** (para OCR)
   - **Generative Language API** (para evaluación con IA)
4. Ve a **Credenciales → Crear Credenciales → API Key**
5. Copia la clave generada para `GOOGLE_API_KEY`

#### **Cloudflare R2:**
Las credenciales de R2 son **las mismas que se configuran en el frontend**:
- `URL_R2`: URL base de tu dominio R2
- `ACCESS_KEY_ID`: Clave de acceso de Cloudflare
- `SECRET_ACCESS_KEY`: Clave secreta de Cloudflare  
- `BUCKET_NAME`: Nombre del bucket donde se almacenan las pruebas

### 3. Instalación

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 4. Ejecutar

```bash
# Activar entorno virtual
source venv/bin/activate

# Ejecutar servidor
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

El servidor estará disponible en `http://localhost:8000`

## 🔄 Flujo de Funcionamiento

### 1. **Recepción de Request**
- El frontend envía una petición POST a `/evaluar-lote`
- Incluye: datos de la prueba, rúbrica, y keys de archivos R2

### 2. **Generación de URLs Firmadas**
- Para cada `test_key` recibido, se genera una URL firmada temporal (5 min)
- Utiliza credenciales R2 y `boto3` para crear URLs seguras
- Permite descarga directa sin exponer credenciales

### 3. **Descarga de Archivos**
- Descarga cada archivo de prueba usando las URLs firmadas
- Almacena temporalmente en el sistema de archivos
- Maneja errores de conectividad y timeouts

### 4. **Extracción de Texto (OCR)**
- Procesa imágenes con **Google Cloud Vision API**
- Extrae texto completo de cada página de la prueba
- Normaliza y limpia el texto extraído

### 5. **Evaluación con IA**
- Utiliza **Google Generative AI** (Gemini) para calificar
- Aplica la rúbrica proporcionada (simple o avanzada)
- Genera feedback específico por pregunta/criterio
- Calcula calificaciones en escala 1-7

### 6. **Procesamiento de Resultados**
- Convierte scores internos (0-10) a escala final (1-7)
- Estructura response con metadata detallada
- Incluye tiempos de procesamiento y nivel de confianza

### 7. **Limpieza**
- Elimina archivos temporales descargados
- Libera recursos utilizados
- Retorna resultados estructurados al frontend

## 📚 API Endpoints

### POST `/evaluar-lote`

Evalúa múltiples pruebas en lote usando IA.

**Request Body:**
```json
{
  "gradedBy": "AI",
  "submissions": [
    {
      "submissionId": 1021,
      "test_key": "archivo-prueba-1.jpeg"
    }
  ],
  "rubric_data": {
    "id": 6,
    "mode": "simple",
    "questions": [...]
  },
  "test_data": {
    "id": 76,
    "name": "Examen Final",
    "max_grade": "7.00"
  }
}
```

**Response:**
```json
{
  "message": "Evaluación por lotes completada para 2 submissions.",
  "gradedBy": "AI", 
  "results": [
    {
      "status": "completed",
      "overall_score": 7,
      "confidence": 1.0,
      "detailed_scores": {...},
      "processing_time_seconds": 15.3
    }
  ]
}
```

### POST `/evaluar-directo`

Evaluación directa de una prueba individual.

### GET `/health`

Verificación de salud del servicio.

### GET `/`

Información general de la API.

## 🔧 Funciones Principales

### `get_signed_url(key: str) -> str`
**Propósito**: Genera URL firmada temporal para acceso seguro a archivos R2
- **Parámetros**: `key` - Clave del archivo en R2
- **Retorna**: URL firmada válida por 5 minutos
- **Excepciones**: `HTTPException` si falla la generación

### `download_file_from_url(url: str) -> str`
**Propósito**: Descarga archivo desde URL firmada y lo guarda temporalmente
- **Parámetros**: `url` - URL firmada del archivo
- **Retorna**: Ruta del archivo temporal
- **Validaciones**: Esquema HTTP/HTTPS, dominio válido

### `evaluate_test_with_rubric(text: str, rubric: dict) -> dict`
**Propósito**: Evalúa texto de prueba usando rúbrica con Google AI
- **Parámetros**: 
  - `text` - Texto extraído por OCR
  - `rubric` - Datos de rúbrica (simple/avanzada)
- **Retorna**: Diccionario con calificaciones y feedback
- **Procesamiento**: Convierte escala 0-10 a 1-7

### `extract_text_google_vision(file_path: str) -> str`
**Propósito**: Extrae texto de imagen usando Google Cloud Vision
- **Parámetros**: `file_path` - Ruta del archivo de imagen
- **Retorna**: Texto extraído y normalizado
- **Formatos**: JPEG, PNG, PDF

### `normalize_text(text: str) -> str`
**Propósito**: Normaliza y filtra texto extraído por OCR
- **Parámetros**: `text` - Texto bruto de OCR
- **Retorna**: Texto limpio y estructurado
- **Procesamiento**: Corrección ortográfica, filtrado de contenido

### `cleanup_temp_file(filepath: str)`
**Propósito**: Elimina archivos temporales después del procesamiento
- **Parámetros**: `filepath` - Ruta del archivo a eliminar
- **Función**: Gestión de memoria y almacenamiento

## 🛡️ Manejo de Errores

- **URLs Inválidas**: Validación de esquema y dominio
- **Fallos de Descarga**: Reintentos y timeouts
- **Errores de OCR**: Fallback a procesamiento básico  
- **Fallas de IA**: Scores por defecto y logging
- **Archivos Corruptos**: Detección y omisión

## 📊 Escalas de Calificación

- **Interna AI**: 0-10 puntos (procesamiento)
- **Final Usuario**: 1-7 puntos (escala chilena)
- **Conversión**: `score_final = max(1.0, (score_ai / 10) * 6 + 1)`

## 🔍 Logging

- **INFO**: Flujo normal de procesamiento
- **WARNING**: Situaciones recoverable
- **ERROR**: Fallos de procesamiento por submission
- **CRITICAL**: Fallos de configuración del sistema

## 🚨 Solución de Problemas

### Error: "Cliente S3/R2 no configurado"
- Verificar credenciales R2 en `.env`
- Confirmar que `URL_R2` no tiene `/` final

### Error: "GOOGLE_API_KEY no configurada"
- Verificar clave en `.env`
- Confirmar que las APIs están habilitadas en Google Cloud

### Error: "No se pudo extraer texto"
- Verificar calidad de imagen
- Confirmar formato compatible (JPEG, PNG)
- Revisar conectividad con Google Vision API

### Error: "Error al descargar desde Cloudflare R2"
- Verificar que las URLs firmadas no han expirado (5 min)
- Confirmar credenciales R2 válidas
- Revisar conectividad de red

## 📈 Rendimiento

- **Procesamiento OCR**: ~2-5 segundos por imagen
- **Evaluación IA**: ~3-8 segundos por prueba
- **Descarga R2**: ~1-2 segundos por archivo
- **Total por submission**: ~6-15 segundos

## 🔒 Seguridad

- URLs firmadas con expiración de 5 minutos
- Validación de esquemas de URL
- Limpieza automática de archivos temporales
- Sin almacenamiento persistente de datos sensibles # BackGrader
# BackGrader
