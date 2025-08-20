from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

# Schema principal para evaluación directa (stateless)
class DirectEvaluationRequest(BaseModel):
    test_data: Dict[str, Any]  # Datos de la prueba desde la BD del frontend
    rubric_data: Dict[str, Any]  # Datos completos de la rúbrica desde la BD
    test_url: Optional[str] = None  # URL de Cloudflare R2 donde está la prueba (legacy)
    test_key: Optional[str] = None  # Key del archivo en R2 para generar URL firmada

# Schemas legacy mantenidos para compatibilidad
class EvaluationRequest(BaseModel):
    evaluation_id: str

class AnswerKeyUpload(BaseModel):
    evaluation_id: str

# Nuevos schemas para URLs y rúbricas JSON
class PruebaUrlUpload(BaseModel):
    url: str
    description: Optional[str] = None

class RubricaJsonUpload(BaseModel):
    rubrica_data: Dict[str, Any]
    name: str
    description: Optional[str] = None

class ErrorDetail(BaseModel):
    loc: Optional[List[str]] = None
    msg: str
    type: Optional[str] = None

class QuestionDetail(BaseModel):
    id: str
    text: Optional[str] = None
    answer: Optional[str] = None
    # Otros campos que puedas necesitar para una pregunta

class StructureAnalysisResult(BaseModel):
    total_questions: int = 0
    numbering_format: Optional[str] = None
    questions: List[QuestionDetail] = []
    error_api: Optional[str] = None
    error_parsing: Optional[str] = None
    error_config: Optional[str] = None
    error_critical: Optional[str] = None

class AnswerExtractionResult(BaseModel):
    answers: Dict[str, str] = {} # Clave es ID de pregunta, valor es respuesta del estudiante
    error_api: Optional[str] = None
    error_parsing: Optional[str] = None
    error_config: Optional[str] = None
    error_critical: Optional[str] = None
    error_dependency: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class OCRResult(BaseModel):
    raw_text: str
    normalized_text: Optional[str] = None
    error: Optional[str] = None

class DetailedScoreItem(BaseModel):
    student_answer: Optional[str] = "No provista"
    correct_answer: Optional[str] = "No provista"
    evaluation: Optional[str] = "No evaluado"
    feedback: Optional[str] = "Sin feedback específico."
    score: float = Field(default=1.0, description="Puntaje en escala 1-7") # Escala 1-7
    original_score_0_10: Optional[float] = Field(default=0.0, description="Puntaje original en escala 0-10")


class EvaluationResult(BaseModel):
    status: str
    test_extracted_text: Optional[str] = None
    answer_key_extracted_text: Optional[str] = None
    # El feedback general ahora es un string, como lo espera evaluate_test
    general_feedback: Optional[str] = "Evaluación en proceso o no disponible."
    error: Optional[str] = None # Para errores generales del proceso
    confidence: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    overall_score: Optional[float] = Field(default=1.0, description="Puntaje global en escala 1-7") # Escala 1-7
    detailed_scores: Optional[Dict[str, DetailedScoreItem]] = {}
    # Campos adicionales de las funciones de evaluator que podrían ser útiles
    original_overall_score_percentage: Optional[float] = None
    error_api: Optional[str] = None # Errores específicos de la API
    error_parsing: Optional[str] = None # Errores de parseo JSON
    error_config: Optional[str] = None # Errores de configuración (ej. API Key)
    error_critical: Optional[str] = None # Otros errores críticos
    structure_analysis_warning: Optional[str] = None # Advertencias del análisis de estructura
    error_detail_extraction: Optional[str] = None # Errores en extracción de respuestas
    error_prerequisite: Optional[str] = None # Errores de prerrequisitos (pasos previos fallidos)

# Respuesta simplificada para evaluación directa
class DirectEvaluationResponse(BaseModel):
    status: str
    general_feedback: str
    overall_score: float = Field(description="Puntaje global en escala 1-7")
    confidence: float = Field(ge=0.0, le=1.0)
    detailed_scores: Dict[str, DetailedScoreItem]
    test_metadata: Optional[Dict[str, Any]] = None  # Metadata de la prueba procesada
    processing_time_seconds: Optional[float] = None
    error: Optional[str] = None

# Nuevos esquemas para evaluación por lotes
class Submission(BaseModel):
    submissionId: int
    test_url: Optional[str] = None  # URL directa (legacy)
    test_key: Optional[str] = None  # Key del archivo en R2 (nuevo)

class BatchEvaluationRequest(BaseModel):
    gradedBy: str
    submissions: List[Submission]
    rubric_data: Dict[str, Any]
    test_data: Dict[str, Any]

# Modelos para los endpoints, si se quiere ser más específico que usar EvaluationResult para todo
class UploadResponse(BaseModel):
    id: str
    status: str

class EvaluationStatusResponse(EvaluationResult): # Hereda de EvaluationResult para consistencia
    pass 