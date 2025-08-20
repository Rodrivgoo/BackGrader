import os
import time
import base64
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import aiohttp
import tempfile
import uuid
from urllib.parse import urlparse
import boto3
from botocore.config import Config

from app.services.ocr_services import extract_text_google_vision
from app.utils.normalizer import normalize_text
from app.utils.evaluator import evaluate_test_with_rubric, GOOGLE_API_KEY, GOOGLE_MODEL_NAME
from app.schemas import DirectEvaluationRequest, DirectEvaluationResponse, BatchEvaluationRequest

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de S3/R2 para URLs firmadas
s3_client = None
try:
    s3_client = boto3.client(
        's3',
        region_name='auto',
        endpoint_url=os.getenv('URL_R2'),
        aws_access_key_id=os.getenv('ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('SECRET_ACCESS_KEY'),
        config=Config(signature_version='s3v4')
    )
    logger.info("Cliente S3/R2 configurado correctamente para URLs firmadas")
except Exception as e:
    logger.warning(f"No se pudo configurar cliente S3/R2: {e}")

async def get_signed_url(key: str) -> str:
    """
    Genera una URL firmada temporal para acceder a un archivo en Cloudflare R2.
    """
    try:
        if not s3_client:
            raise ValueError("Cliente S3/R2 no configurado")
        
        bucket_name = os.getenv('BUCKET_NAME', 'insightgradertests')
        if not bucket_name:
            raise ValueError("BUCKET_NAME no está configurada")
        
        # Generar URL firmada válida por 5 minutos (300 segundos)
        signed_url = s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': bucket_name,
                'Key': key
            },
            ExpiresIn=300  # 5 minutos
        )
        
        logger.info(f"URL firmada generada para key: {key}")
        return signed_url
        
    except Exception as e:
        logger.error(f"Error generando URL firmada para {key}: {e}")
        raise HTTPException(status_code=500, detail=f"Error generando URL firmada: {str(e)}")

# Verificar configuración crítica
if not GOOGLE_API_KEY:
    logger.critical("CRITICAL ERROR: GOOGLE_API_KEY no está configurada.")
logger.info(f"Evaluator API Key Check: OK. Using model: {GOOGLE_MODEL_NAME}")

app = FastAPI(
    title="InsightGrader API",
    description="API stateless para evaluación de exámenes con IA usando URLs y rúbricas JSON.",
    version="3.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def download_file_from_url(url: str) -> str:
    """
    Descarga un archivo desde una URL de Cloudflare R2.
    Solo acepta URLs HTTP/HTTPS válidas.
    Retorna la ruta del archivo temporal.
    """
    try:
        # Validar que sea una URL HTTP/HTTPS válida
        parsed_url = urlparse(url)
        if not parsed_url.scheme in ['http', 'https']:
            raise HTTPException(
                status_code=400, 
                detail=f"URL inválida. Solo se aceptan URLs HTTP/HTTPS. Recibido: {parsed_url.scheme}://"
            )
        
        if not parsed_url.netloc:
            raise HTTPException(
                status_code=400,
                detail="URL inválida. Falta el dominio."
            )
        
        # Obtener nombre del archivo desde la URL
        filename = os.path.basename(parsed_url.path) or f"temp_{uuid.uuid4().hex[:8]}.pdf"
        
        # Crear archivo temporal con la extensión correcta
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
        temp_path = temp_file.name
        temp_file.close()
        
        logger.info(f"Descargando archivo desde Cloudflare R2: {url}")
        
        # Descargar el archivo
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(temp_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    logger.info(f"✅ Archivo descargado exitosamente: {temp_path}")
                    return temp_path
                else:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Error al descargar desde Cloudflare R2: HTTP {response.status}"
                    )
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error descargando desde URL {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Error descargando archivo: {str(e)}")

def cleanup_temp_file(filepath: str):
    """Elimina un archivo temporal."""
    try:
        if os.path.exists(filepath):
            os.unlink(filepath)
            logger.info(f"Archivo temporal eliminado: {filepath}")
    except Exception as e:
        logger.warning(f"No se pudo eliminar archivo temporal {filepath}: {e}")

@app.get("/")
async def read_root():
    return JSONResponse(content={
        "message": "InsightGrader API v3.0 - Evaluación directa stateless",
        "description": "Envía datos de prueba + rúbrica → Recibe evaluación inmediata",
        "docs_url": "/docs",
        "main_endpoint": "/evaluar-directo"
    })

@app.post("/evaluar-directo", response_model=DirectEvaluationResponse)
async def evaluar_directo(request: DirectEvaluationRequest):
    """
    Endpoint principal para evaluación directa stateless.
    
    Recibe:
    - test_data: Metadatos de la prueba desde tu BD
    - rubric_data: Datos completos de la rúbrica desde tu BD  
    - test_url: URL de Cloudflare R2 con el archivo de la prueba (legacy)
    - test_key: Key del archivo en R2 para generar URL firmada (recomendado)
    
    Procesa:
    1. Genera URL firmada o usa URL directa
    2. Descarga el archivo desde Cloudflare R2
    3. Extrae texto usando Google Vision OCR
    4. Evalúa usando la rúbrica con IA
    5. Retorna resultado inmediatamente
    6. Limpia archivos temporales
    """
    start_time = time.time()
    temp_file_path = None
    
    try:
        if not GOOGLE_API_KEY:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY no configurada")
        
        if not (request.test_url or request.test_key) or not request.rubric_data:
            raise HTTPException(status_code=400, detail="(test_url o test_key) y rubric_data son requeridos")
        
        logger.info(f"Iniciando evaluación directa para prueba: {request.test_data.get('name', 'N/A')}")
        
        # 1. Obtener URL de descarga (firmada o directa)
        download_url = request.test_url
        if request.test_key:
            logger.info(f"Generando URL firmada para key: {request.test_key}")
            download_url = await get_signed_url(request.test_key)
        
        # 2. Procesar archivo desde URL
        logger.info("Procesando archivo de prueba...")
        temp_file_path = await download_file_from_url(download_url)
        
        # 2. Extraer texto usando OCR
        logger.info("Extrayendo texto con Google Vision OCR...")
        raw_text = extract_text_google_vision(temp_file_path)
        
        if not raw_text or raw_text.strip() == "":
            raise HTTPException(status_code=422, detail="No se pudo extraer texto de la prueba")
        
        # 3. Normalizar texto
        logger.info("Normalizando texto extraído...")
        normalized_text = normalize_text(raw_text)
        
        # 4. Evaluar con rúbrica
        logger.info("Evaluando con IA...")
        evaluation_result = evaluate_test_with_rubric(normalized_text, request.rubric_data)
        
        if "error" in evaluation_result:
            raise HTTPException(status_code=500, detail=f"Error en evaluación: {evaluation_result['error']}")
        
        # 5. Preparar respuesta
        processing_time = time.time() - start_time
        
        response = DirectEvaluationResponse(
            status="completed",
            general_feedback=evaluation_result.get("general_feedback", "Evaluación completada"),
            overall_score=evaluation_result.get("overall_score", 1.0),
            confidence=evaluation_result.get("confidence", 0.8),
            detailed_scores=evaluation_result.get("detailed_scores", {}),
            test_metadata={
                "test_name": request.test_data.get("name"),
                "test_id": request.test_data.get("id"),
                "rubric_name": request.rubric_data.get("name"),
                "text_length": len(normalized_text),
                "original_url": request.test_url,
                "original_key": request.test_key,
                "download_url": download_url
            },
            processing_time_seconds=round(processing_time, 2)
        )
        
        logger.info(f"Evaluación completada en {processing_time:.2f}s - Nota: {response.overall_score}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error crítico en evaluación directa: {str(e)}")
        processing_time = time.time() - start_time
        
        return DirectEvaluationResponse(
            status="error",
            general_feedback=f"Error durante la evaluación: {str(e)}",
            overall_score=1.0,
            confidence=0.0,
            detailed_scores={},
            processing_time_seconds=round(processing_time, 2),
            error=str(e)
        )
    
    finally:
        # Limpiar archivo temporal
        if temp_file_path:
            cleanup_temp_file(temp_file_path)

@app.post("/evaluar-lote")
async def evaluar_lote(request: BatchEvaluationRequest):
    """
    Endpoint para evaluación por lotes stateless.
    
    Recibe:
    - submissions: Una lista de objetos, cada uno con "submissionId" y "test_url".
    - rubric_data: Datos completos de la rúbrica.
    - test_data: Metadatos de la prueba.
    
    Procesa cada submission de forma secuencial y retorna los resultados.
    """
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY no configurada")

    logger.info(f"Iniciando evaluación por lotes para {len(request.submissions)} submissions, calificado por: {request.gradedBy}")
    
    batch_results = []
    
    for submission in request.submissions:
        start_time_submission = time.time()
        temp_file_path = None
        
        try:
            # 1. Obtener URL de descarga (firmada o directa)
            download_url = submission.test_url
            if submission.test_key:
                logger.info(f"Generando URL firmada para submissionId {submission.submissionId}, key: {submission.test_key}")
                download_url = await get_signed_url(submission.test_key)
                logger.info(f"Procesando submissionId: {submission.submissionId} desde URL firmada")
            else:
                logger.info(f"Procesando submissionId: {submission.submissionId} desde {submission.test_url}")
            
            # 2. Descargar archivo
            temp_file_path = await download_file_from_url(download_url)
            
            # 2. Extraer texto
            raw_text = extract_text_google_vision(temp_file_path)
            if not raw_text or raw_text.strip() == "":
                raise ValueError("No se pudo extraer texto de la prueba")
            
            # 3. Normalizar texto
            normalized_text = normalize_text(raw_text)
            
            # 4. Evaluar con rúbrica
            evaluation_result = evaluate_test_with_rubric(normalized_text, request.rubric_data)
            if "error" in evaluation_result:
                raise ValueError(f"Error en evaluación IA: {evaluation_result['error']}")
            
            # 5. Preparar resultado exitoso para esta submission
            processing_time = time.time() - start_time_submission
            
            result = DirectEvaluationResponse(
                status="completed",
                general_feedback=evaluation_result.get("general_feedback", "Evaluación completada"),
                overall_score=evaluation_result.get("overall_score", 1.0),
                confidence=evaluation_result.get("confidence", 0.8),
                detailed_scores=evaluation_result.get("detailed_scores", {}),
                test_metadata={
                    "submission_id": submission.submissionId,
                    "original_url": submission.test_url,
                    "original_key": submission.test_key,
                    "download_url": download_url,
                    "text_length": len(normalized_text)
                },
                processing_time_seconds=round(processing_time, 2)
            )
            batch_results.append(result.dict())
            
        except Exception as e:
            logger.error(f"Error procesando submissionId {submission.submissionId}: {e}")
            processing_time = time.time() - start_time_submission
            
            error_response = DirectEvaluationResponse(
                status="error",
                general_feedback=f"Error procesando submission: {str(e)}",
                overall_score=1.0,
                confidence=0.0,
                detailed_scores={},
                test_metadata={
                    "submission_id": submission.submissionId, 
                    "original_url": submission.test_url,
                    "original_key": submission.test_key
                },
                processing_time_seconds=round(processing_time, 2),
                error=str(e)
            )
            batch_results.append(error_response.dict())
            
        finally:
            if temp_file_path:
                cleanup_temp_file(temp_file_path)

    return JSONResponse(content={
        "message": f"Evaluación por lotes completada para {len(request.submissions)} submissions.",
        "gradedBy": request.gradedBy,
        "results": batch_results
    })


@app.get("/health")
async def health_check():
    """Verificación de salud del servicio."""
    return {
        "status": "ok",
        "version": "3.0.0",
        "google_api_configured": bool(GOOGLE_API_KEY),
        "endpoints": ["/evaluar-directo", "/health"]
    } 