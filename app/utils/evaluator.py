import os
import json
import logging
import math # Para redondear
import google.generativeai as genai
# langchain_community.llms.Ollama ya no es necesario
from langchain.prompts import PromptTemplate # Se mantiene para la construcción de prompts
from typing import Dict, List, Any, Tuple

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ya no se necesitan configuraciones de Ollama
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct-q5_K_M")
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Configurar API Key de Google
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Clave API de Google configurada.")
else:
    # Este es un punto crítico. Si no hay clave, la aplicación no puede funcionar.
    # Podríamos lanzar un error aquí para detener la inicialización si es preferible.
    logger.error("¡¡¡ALERTA CRÍTICA!!! GOOGLE_API_KEY no encontrada en el entorno. La aplicación no podrá evaluar.")
    # raise ValueError("GOOGLE_API_KEY es requerida y no fue encontrada en el entorno.")

# Nombre del modelo de Google a usar, configurable por variable de entorno
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini-2.0-flash-lite")
logger.info(f"Usando el modelo de Google: {GOOGLE_MODEL_NAME}")

# --- Nueva función de conversión de escala ---
def convert_to_1_to_7_scale(score: float, scale_max: int = 10, precision: int = 1) -> float:
    """
    Convierte un puntaje dado (ej. 0-10 o 0-100) a una escala de 1 a 7.
    Args:
        score (float): El puntaje original.
        scale_max (int): El máximo del puntaje original (10 para preguntas, 100 para porcentaje general).
        precision (int): Número de decimales para la nota final.
    Returns:
        float: El puntaje convertido a la escala 1-7.
    """
    if scale_max <= 0:
        return 1.0 # Evitar división por cero, devolver nota mínima
    
    # Asegurarse que el score no exceda los límites esperados para la conversión
    score = max(0, min(score, scale_max))
    
    nota_1_7 = 1 + (score / scale_max) * 6
    return round(nota_1_7, precision)

# --- Fin nueva función ---

def call_google_api(prompt_text: str, model_name: str = None) -> str:
    """
    Llama a la API de Google Generative AI con el prompt dado.
    
    Args:
        prompt_text (str): El prompt para enviar al modelo.
        model_name (str): El nombre del modelo a usar (ej: "gemini-1.5-flash-latest").
        
    Returns:
        str: La respuesta del modelo.
    """
    if not GOOGLE_API_KEY:
        logger.error("No se puede llamar a la API de Google: GOOGLE_API_KEY no configurada.")
        # Este error se propagará a las funciones que llaman
        raise ValueError("API Key de Google no configurada. La evaluación no puede continuar.")
        
    try:
        # Usar GOOGLE_MODEL_NAME leído del entorno
        logger.info(f"Llamando a la API de Google con el modelo: {model_name if model_name else GOOGLE_MODEL_NAME} para el prompt (primeros 300 chars): {prompt_text[:300]}...")
        actual_model_name = model_name if model_name else GOOGLE_MODEL_NAME
        model = genai.GenerativeModel(actual_model_name)
        response = model.generate_content(prompt_text)
        
        # Loguear la respuesta completa para depuración, antes de intentar acceder a sus partes
        logger.info(f"Respuesta completa de Google API: {response}")

        if response.parts:
            # Asegurarse de que parts[0] tiene 'text' y no es None
            if hasattr(response.parts[0], 'text') and response.parts[0].text is not None:
                logger.info("Respuesta de Google API extraída de response.parts[0].text")
                return response.parts[0].text
            else:
                logger.warning("response.parts[0] no tiene atributo 'text' o es None.")
        
        # Verificar candidatos, que es la estructura más común para Gemini
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts and len(candidate.content.parts) > 0:
                if hasattr(candidate.content.parts[0], 'text') and candidate.content.parts[0].text is not None:
                    logger.info("Respuesta de Google API extraída de response.candidates[0].content.parts[0].text")
                    return candidate.content.parts[0].text
                else:
                    logger.warning("response.candidates[0].content.parts[0] no tiene atributo 'text' o es None.")
            # Manejar el caso de finalización por seguridad u otros motivos directamente desde el candidato
            if candidate.finish_reason and candidate.finish_reason.name not in ["STOP", "MAX_TOKENS"]:
                 # OTHER, SAFETY, RECITATION, UNKNOWN, UNSPECIFIED
                reason = candidate.finish_reason.name
                logger.error(f"La generación de contenido de Google API finalizó por: {reason}")
                return f'{{"error": "Generación de contenido detenida por la API de Google", "reason": "{reason}"}}'
        
        # Fallback si la estructura no es ninguna de las anteriores o si text es None
        logger.warning(f"Respuesta de Google API no contenía texto en las ubicaciones esperadas (parts o candidates). Verifique la respuesta completa logueada.")
        # Intentar obtener el prompt_feedback si existe, puede indicar un bloqueo general
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
            reason = response.prompt_feedback.block_reason.name
            logger.error(f"La solicitud a la API de Google fue bloqueada (prompt_feedback). Razón: {reason}")
            return f'{{"error": "Solicitud bloqueada por la API de Google (prompt_feedback)", "reason": "{reason}"}}'
        
        # Si después de todas las verificaciones no hay texto útil, devolver error genérico.
        return '{"error": "Respuesta vacía o con formato inesperado de la API de Google tras verificar todas las estructuras conocidas."}'

    except Exception as e:
        logger.error(f"Error crítico al llamar a la API de Google: {str(e)}")
        logger.exception("Detalles de la excepción en call_google_api:")
        return f'{{"error": "Excepción crítica en call_google_api: {str(e)}"}}'

# La función get_ollama_model() ya no es necesaria y se elimina.

def _parse_json_from_response(response_text: str, logger_func_name: str) -> Any:
    """Función helper para parsear JSON, reintentando con limpieza."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        logger.warning(f"JSONDecodeError inicial en {logger_func_name}. Intentando limpiar respuesta: {response_text[:200]}...")
        import re
        json_str_to_parse = None
        match_md = re.search(r"```json\\s*(.*?)\\s*```", response_text, re.DOTALL)
        
        if match_md:
            json_str_to_parse = match_md.group(1).strip()
            logger.info(f"JSON extraído de bloque markdown ```json``` en {logger_func_name}.")
        else:
            start_brace = response_text.find('{')
            end_brace = response_text.rfind('}')
            start_bracket = response_text.find('[')
            end_bracket = response_text.rfind(']')

            is_object_primary = start_brace != -1 and end_brace != -1 and start_brace < end_brace
            is_array_primary = start_bracket != -1 and end_bracket != -1 and start_bracket < end_bracket
            
            if is_object_primary and is_array_primary:
                if start_bracket < start_brace and end_bracket > end_brace :
                     json_str_to_parse = response_text[start_bracket : end_bracket + 1].strip()
                else:
                     json_str_to_parse = response_text[start_brace : end_brace + 1].strip()
            elif is_object_primary:
                json_str_to_parse = response_text[start_brace : end_brace + 1].strip()
            elif is_array_primary:
                json_str_to_parse = response_text[start_bracket : end_bracket + 1].strip()

            if json_str_to_parse:
                 logger.info(f"JSON extraído por búsqueda heurística en {logger_func_name}.")
            else:
                logger.error(f"No se pudo extraer una subcadena JSON candidata en {logger_func_name} con heurísticas.")


        if json_str_to_parse:
            try:
                return json.loads(json_str_to_parse)
            except json.JSONDecodeError as e_retry:
                logger.error(f"Error al decodificar JSON (con limpieza) en {logger_func_name}: {e_retry}. Contenido: {json_str_to_parse[:500]}...")
                raise e_retry # Re-lanzar para que la función llamante maneje
        else: # Si json_str_to_parse sigue siendo None
            logger.error(f"Fallo en la extracción de JSON con limpieza en {logger_func_name}. Respuesta original: {response_text[:500]}")
            raise json.JSONDecodeError("No se pudo extraer JSON con limpieza", response_text, 0)

def analyze_structure(answer_key_text: str) -> Dict[str, Any]:
    """
    Analiza la estructura de la pauta para determinar el número de preguntas,
    su formato y características para usar como referencia al evaluar.
    
    Args:
        answer_key_text (str): Texto normalizado de la pauta
        
    Returns:
        Dict: Estructura identificada con información sobre las preguntas
    """
    try:
        logger.info("Analizando estructura de la pauta usando Google API...")
        
        prompt_template_str = ("""
Eres un experto en análisis de exámenes académicos. Tu tarea es analizar la estructura de una pauta de respuestas para determinar su organización.

PAUTA DE RESPUESTAS:
{answer_key_text}

Analiza y extrae la siguiente información:
1. Número total de preguntas
2. Formato de numeración (ejemplo: "1.-", "Pregunta 1:", etc.)
3. Para cada pregunta, identifica:
   - Número o identificador de la pregunta
   - Texto o enunciado de la pregunta
   - Respuesta esperada

Responde en formato JSON con la siguiente estructura:
{{  // Estas llaves dobles escapan las llaves para que sean literales en f-string/format
    "total_questions": 5,
    "numbering_format": "1.-",
    "questions": [
        {{
            "id": "1",
            "text": "Dirección de red y máscara de la sucursal LC.",
            "answer": "40.41/24"
        }},
        ... (para cada pregunta)
    ]
}}

Asegúrate de identificar correctamente cada pregunta numerada y su respuesta correspondiente, incluso si hay texto adicional o formato irregular.
IMPORTANTE: Tu respuesta DEBE SER EXCLUSIVAMENTE un objeto JSON válido que siga la estructura especificada. No incluyas ```json```, explicaciones, comentarios o cualquier otro texto fuera del propio objeto JSON.
""")
        prompt_template_obj = PromptTemplate.from_template(prompt_template_str)
        prompt = prompt_template_obj.format(answer_key_text=answer_key_text)
        
        response_text = call_google_api(prompt)
            
        try:
            # parsed_response = json.loads(response_text) # Original
            # Lógica de limpieza movida a _parse_json_from_response
            parsed_response = _parse_json_from_response(response_text, "analyze_structure")

            if isinstance(parsed_response, dict) and "error" in parsed_response:
                error_detail = parsed_response.get('reason', parsed_response['error'])
                logger.error(f"Error de la API de Google al analizar estructura: {error_detail}")
                return {"total_questions": 0, "numbering_format": "", "questions": [], "error_api": error_detail}
            
            # Si no es un diccionario de error, entonces debería ser la estructura esperada
            logger.info(f"Estructura analizada: {len(parsed_response.get('questions', []))} preguntas identificadas")
            return parsed_response # Esta es la 'structure'
        except json.JSONDecodeError: # Captura re-lanzado de _parse_json_from_response
            logger.error(f"Fallo final al parsear JSON de analyze_structure. Respuesta original de call_google_api: {response_text[:500]}")
            return {"total_questions": 0, "numbering_format": "", "questions": [], "error_parsing": "Fallo al parsear JSON de Google API (analyze_structure)"}
    
    except ValueError as ve: 
        logger.error(f"Error de configuración impidió el análisis de estructura: {ve}")
        return {"total_questions": 0, "numbering_format": "", "questions": [], "error_config": str(ve)}
    except Exception as e:
        logger.error(f"Error crítico al analizar estructura (Google API): {str(e)}")
        logger.exception("Detalles de la excepción en analyze_structure:")
        return {"total_questions": 0, "numbering_format": "", "questions": [], "error_critical": str(e)}

def extract_student_answers(student_text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrae las respuestas del estudiante basándose en la estructura identificada en la pauta.
    """
    try:
        logger.info("Extrayendo respuestas del estudiante según estructura (Google API)...")
        
        if not structure or "questions" not in structure or len(structure["questions"]) == 0:
            logger.warning("No se pudo extraer respuestas: estructura de preguntas vacía o con error previo.")
            if any(err_key in structure for err_key in ["error_api", "error_parsing", "error_config", "error_critical"]):
                 return {"error_dependency": "La estructura previa falló.", "details": structure}
            return {}
            
        prompt_template_str = ("""
Eres un experto en procesamiento de exámenes académicos. Tu tarea es extraer las respuestas de un estudiante basándote en la estructura de preguntas identificada.

PRUEBA DEL ESTUDIANTE:
{student_text}

ESTRUCTURA DE PREGUNTAS (DE LA PAUTA):
{structure_json}

Instrucciones:
1. Para cada pregunta en la estructura, busca la respuesta correspondiente en la prueba del estudiante.
2. Si la respuesta no está clara o hay múltiples respuestas posibles, selecciona la que parezca ser la respuesta final o definitiva.
3. Si no encuentras una respuesta para alguna pregunta, indica "No encontrada".

IMPORTANTE: 
- Busca respuestas que correspondan al mismo número/identificador de pregunta.
- La respuesta podría estar en un formato diferente al esperado.
- Si hay texto adicional, extrae solo la parte que constituye la respuesta.
- Para respuestas que incluyen cálculos o múltiples elementos, captura la respuesta completa.

Responde en formato JSON con la siguiente estructura:
{{ // Estas llaves dobles escapan las llaves para que sean literales en f-string/format
    "1": "Respuesta del estudiante a la pregunta 1",
    "2": "Respuesta del estudiante a la pregunta 2",
    ... (para cada pregunta en la estructura)
}}

Usa los mismos identificadores de pregunta que aparecen en la estructura.
IMPORTANTE: Tu respuesta DEBE SER EXCLUSIVAMENTE un objeto JSON válido que siga la estructura especificada. No incluyas ```json```, explicaciones, comentarios o cualquier otro texto fuera del propio objeto JSON.
""")
        prompt_template_obj = PromptTemplate.from_template(prompt_template_str)

        structure_json = json.dumps(structure, ensure_ascii=False)
        prompt = prompt_template_obj.format(
            student_text=student_text,
            structure_json=structure_json
        )
        
        response_text = call_google_api(prompt)
            
        try:
            # parsed_response = json.loads(response_text) # Original
            # Lógica de limpieza movida a _parse_json_from_response
            parsed_response = _parse_json_from_response(response_text, "extract_student_answers")

            if isinstance(parsed_response, dict) and "error" in parsed_response:
                error_detail = parsed_response.get('reason', parsed_response['error'])
                logger.error(f"Error de la API de Google al extraer respuestas: {error_detail}")
                return {"error_api": error_detail}
            
            logger.info(f"Respuestas extraídas (Google API): {len(parsed_response)} preguntas")
            return parsed_response # Este es el diccionario de 'answers'
        except json.JSONDecodeError: # Captura re-lanzado de _parse_json_from_response
            logger.error(f"Fallo final al parsear JSON de extract_student_answers. Respuesta original de call_google_api: {response_text[:500]}")
            return {"error_parsing": "Fallo al parsear JSON de Google API (extract_student_answers)"}
            
    except ValueError as ve:
        logger.error(f"Error de configuración impidió la extracción de respuestas: {ve}")
        return {"error_config": str(ve)}
    except Exception as e:
        logger.error(f"Error crítico al extraer respuestas (Google API): {str(e)}")
        logger.exception("Detalles de la excepción en extract_student_answers:")
        return {"error_critical": str(e)}

def evaluate_test_with_rubric(student_text: str, rubrica_data: dict) -> Dict[str, Any]:
    """
    Evalúa un examen usando una rúbrica estructurada en formato JSON.
    
    Args:
        student_text (str): Texto extraído de la prueba del estudiante
        rubrica_data (dict): Datos de la rúbrica en formato JSON
        
    Returns:
        Dict[str, Any]: Resultado de la evaluación con scores detallados
    """
    try:
        if not GOOGLE_API_KEY:
            logger.error("Evaluación abortada: GOOGLE_API_KEY no está configurada.")
            return {
                "detailed_scores": {}, "overall_score": 1.0,
                "general_feedback": "Error de configuración: GOOGLE_API_KEY no encontrada.",
                "confidence": 0, "error": "Configuración API Google incompleta."
            }

        if not student_text or not rubrica_data:
            logger.error("Texto del estudiante o datos de rúbrica están vacíos.")
            return {
                "detailed_scores": {}, "overall_score": 1.0,
                "general_feedback": "Se requiere texto de la prueba y datos de la rúbrica.",
                "confidence": 0, "error": "Input de texto o rúbrica vacío."
            }
        
        logger.info(f"Iniciando evaluación con rúbrica JSON usando Google API.")

        # Crear un prompt especializado para evaluación con rúbrica
        rubrica_json_str = json.dumps(rubrica_data, ensure_ascii=False, indent=2)
        
        prompt_template_str = ("""
Eres un profesor experto en evaluación académica. Tu tarea es evaluar la prueba de un estudiante usando una rúbrica específica proporcionada.

TEXTO DE LA PRUEBA DEL ESTUDIANTE:
{student_text}

RÚBRICA DE EVALUACIÓN (JSON):
{rubrica_json}

Instrucciones para la evaluación:
1. Usa ESTRICTAMENTE la rúbrica proporcionada para evaluar la prueba.
2. Si la rúbrica contiene criterios específicos, evalúa según esos criterios.
3. Si la rúbrica contiene preguntas y respuestas esperadas, compara las respuestas del estudiante.
4. Para cada elemento evaluable (pregunta, criterio, etc.), asigna un puntaje de 0 a 10.
5. Proporciona feedback específico basado en los criterios de la rúbrica.
6. Calcula un puntaje general de 0 a 100 basado en los pesos de la rúbrica (si los hay).
7. Asigna un nivel de confianza (0 a 1) en tu evaluación.

Formato de Respuesta JSON Esperado:
{{
    "detailed_scores": {{
        "Elemento1 (pregunta/criterio)": {{
            "student_answer": "Respuesta o evidencia del estudiante...",
            "correct_answer": "Respuesta esperada según rúbrica...",
            "evaluation": "Evaluación según criterios de rúbrica...",
            "feedback": "Feedback específico basado en rúbrica...",
            "score": 0-10 // Puntaje basado en criterios de rúbrica
        }},
        // ... más elementos según la rúbrica
    }},
    "overall_score": 85, // Puntaje general (0-100) calculado según rúbrica
    "general_feedback": "Feedback general basado en la rúbrica...",
    "confidence": 0.9
}}

IMPORTANTE: 
- Tu respuesta DEBE SER EXCLUSIVAMENTE un objeto JSON válido.
- No incluyas ```json```, explicaciones o texto adicional.
- Basa tu evaluación ÚNICAMENTE en los criterios definidos en la rúbrica.
- Si la rúbrica tiene pesos específicos, úsalos para calcular el puntaje general.
""")

        prompt_template_obj = PromptTemplate.from_template(prompt_template_str)
        prompt = prompt_template_obj.format(
            student_text=student_text, 
            rubrica_json=rubrica_json_str
        )
        
        response_text = call_google_api(prompt)

        try:
            result = _parse_json_from_response(response_text, "evaluate_test_with_rubric")
            
            if isinstance(result, dict) and "error" in result:
                error_detail = result.get('reason', result['error'])
                logger.error(f"Error de la API de Google en evaluación con rúbrica: {error_detail}")
                return {
                    "detailed_scores": {}, "overall_score": 1.0, 
                    "general_feedback": f"Error API Google: {error_detail}", 
                    "confidence": 0, "error_api": error_detail
                }

            # Conversión de escalas de 0-100 a 1-7
            if "overall_score" in result and isinstance(result["overall_score"], (int, float)):
                result["original_overall_score_percentage"] = result["overall_score"]
                result["overall_score"] = convert_to_1_to_7_scale(result["overall_score"], scale_max=100)
            else:
                result["overall_score"] = 1.0

            # Conversión de detailed_scores de 0-10 a 1-7
            if "detailed_scores" in result and isinstance(result["detailed_scores"], dict):
                for q_id, q_eval in result["detailed_scores"].items():
                    if isinstance(q_eval, dict) and "score" in q_eval and isinstance(q_eval["score"], (int, float)):
                        q_eval["original_score_0_10"] = q_eval["score"]
                        q_eval["score"] = convert_to_1_to_7_scale(q_eval["score"], scale_max=10)
            else:
                result["detailed_scores"] = {}
            
            # Asegurar campos requeridos
            if "general_feedback" not in result: 
                result["general_feedback"] = "Evaluación completada según rúbrica proporcionada."
            if "confidence" not in result: 
                result["confidence"] = 0.8
            
            logger.info("Evaluación con rúbrica completada y convertida a escala 1-7.")
            return result
            
        except json.JSONDecodeError:
            logger.error(f"Fallo al parsear JSON de evaluate_test_with_rubric. Respuesta: {response_text[:500]}")
            return {
                "detailed_scores": {}, "overall_score": 1.0, 
                "general_feedback": "Error: La respuesta del modelo no fue un JSON válido.", 
                "confidence": 0, "error_parsing": "Fallo al parsear JSON de Google API"
            }
    
    except ValueError as ve:
        logger.error(f"Error de configuración en evaluación con rúbrica: {ve}")
        return {
            "detailed_scores": {}, "overall_score": 1.0, 
            "general_feedback": f"Error de configuración: {ve}", 
            "confidence": 0, "error_config": str(ve)
        }
    except Exception as e:
        logger.error(f"Error crítico en evaluación con rúbrica: {str(e)}")
        logger.exception("Detalles de la excepción en evaluate_test_with_rubric:")
        return {
            "detailed_scores": {}, "overall_score": 1.0, 
            "general_feedback": f"Error crítico durante la evaluación: {str(e)}", 
            "confidence": 0, "error_critical": str(e)
        }

# Mantener la función original para compatibilidad hacia atrás
def evaluate_test(student_text: str, rubrica_data_or_answer_key: any) -> Dict[str, Any]:
    """
    Función de compatibilidad que determina si usar evaluación con rúbrica o método legacy.
    
    Args:
        student_text (str): Texto del estudiante
        rubrica_data_or_answer_key: Puede ser dict (rúbrica) o str (pauta legacy)
    """
    if isinstance(rubrica_data_or_answer_key, dict):
        # Es una rúbrica JSON, usar nuevo método
        logger.info("Detectada rúbrica JSON, usando evaluate_test_with_rubric")
        return evaluate_test_with_rubric(student_text, rubrica_data_or_answer_key)
    else:
        # Es texto de pauta legacy, usar método original
        logger.info("Detectado texto de pauta, usando método legacy")
        return evaluate_test_legacy(student_text, str(rubrica_data_or_answer_key))

def evaluate_test_legacy(student_text: str, answer_key_text: str) -> Dict[str, Any]:
    """
    Función legacy para evaluación con pautas de texto (método anterior).
    """
    # base_error_response ya no se usa para simplificar, se define el error en cada return
    try:
        if not GOOGLE_API_KEY:
            logger.error("Evaluación abortada: GOOGLE_API_KEY no está configurada.")
            return {
                "detailed_scores": {}, "overall_score": 1.0,
                "general_feedback": "Error de configuración: GOOGLE_API_KEY no encontrada.",
                "confidence": 0, "error": "Configuración API Google incompleta."
            }

        if not student_text or not answer_key_text:
            logger.error("Texto del estudiante o de la pauta está vacío.")
            return {
                "detailed_scores": {}, "overall_score": 1.0,
                "general_feedback": "Se requiere texto de la prueba y de la pauta.",
                "confidence": 0, "error": "Input de texto vacío."
            }
        
        logger.info(f"Iniciando evaluación LEGACY con Google API.")

        structure = analyze_structure(answer_key_text)
        
        structure_error = None
        if isinstance(structure, dict) and any(err_key in structure for err_key in ["error_api", "error_parsing", "error_config", "error_critical"]):
            structure_error = structure.get("error_api") or structure.get("error_parsing") or structure.get("error_config") or structure.get("error_critical", "Error desconocido en análisis de estructura")
            logger.error(f"Fallo en analyze_structure. Error: {structure_error}. Intentando evaluación directa.")
        elif not structure or "questions" not in structure or not structure["questions"]:
            structure_error = "Estructura no determinada o vacía tras analyze_structure."
            logger.warning(structure_error + " Usando evaluación directa.")

        if structure_error:
            direct_eval_result = evaluate_direct(student_text, answer_key_text)
            if "error" in direct_eval_result or "error_api" in direct_eval_result or "error_parsing" in direct_eval_result:
                 direct_eval_result["structure_analysis_warning"] = structure_error
                 return direct_eval_result
            else:
                direct_eval_result["general_feedback"] = f"Advertencia: {structure_error} {direct_eval_result.get('general_feedback', '')}".strip()
                if "error" in direct_eval_result and direct_eval_result["error"] == "Ver campo 'error' para detalles.":
                    direct_eval_result.pop("error", None)
                if "feedback" in direct_eval_result and direct_eval_result["feedback"] == "Ver campo 'error' para detalles.":
                     direct_eval_result.pop("feedback", None)
                return direct_eval_result
        
        student_answers = extract_student_answers(student_text, structure)
        if isinstance(student_answers, dict) and any(err_key in student_answers for err_key in ["error_api", "error_parsing", "error_config", "error_critical", "error_dependency"]):
            error_detail = student_answers.get("error_api") or student_answers.get("error_parsing") or student_answers.get("error_config") or student_answers.get("error_critical") or student_answers.get("error_dependency", "Error desconocido en extracción de respuestas")
            logger.error(f"Fallo en extract_student_answers. Error: {error_detail}. No se puede realizar evaluación estructurada.")
            return {
                "detailed_scores": {}, "overall_score": 1.0,
                "general_feedback": f"Error al procesar respuestas del estudiante ({error_detail}). No se pudo completar la evaluación estructurada.",
                "confidence": 0, "error": "Fallo en extracción de respuestas", "error_detail_extraction": error_detail
            }
        
        if not isinstance(student_answers, dict):
            logger.error(f"Tipo inesperado para student_answers: {type(student_answers)}. Contenido: {str(student_answers)[:200]}")
            return {
                "detailed_scores": {}, "overall_score": 1.0,
                "general_feedback": f"Error interno: formato inesperado de respuestas del estudiante.",
                "confidence": 0, "error": "Error de formato interno en respuestas."
            }

        structured_eval_result = evaluate_structured(student_answers, structure, student_text, answer_key_text)
        if not ("error" in structured_eval_result or "error_api" in structured_eval_result or "error_parsing" in structured_eval_result):
            structured_eval_result.pop("error", None)
            structured_eval_result.pop("feedback", None)
        return structured_eval_result
        
    except ValueError as ve:
        logger.error(f"Error de Valor (ej. API Key o input) al evaluar la prueba: {str(ve)}")
        return {
            "detailed_scores": {}, "overall_score": 1.0,
            "general_feedback": f"Error durante la evaluación: {str(ve)}",
            "confidence": 0, "error": str(ve)
        }
    except Exception as e:
        logger.error(f"Error crítico al evaluar la prueba (Google API): {str(e)}")
        logger.exception("Detalles de la excepción en evaluate_test_legacy:")
        return {
            "detailed_scores": {}, "overall_score": 1.0,
            "general_feedback": f"Error crítico durante la evaluación: {str(e)}",
            "confidence": 0, "error": str(e)
        }

def evaluate_direct(student_text: str, answer_key_text: str) -> Dict[str, Any]:
    """
    Evaluación directa. Ahora solo usa Google API.
    """
    try:
        logger.info("Realizando evaluación directa usando Google API...")
        prompt_template_str = ("""
Eres un profesor experto en evaluar exámenes académicos. Tu tarea es comparar la prueba de un estudiante con la pauta de respuestas y proporcionar una evaluación detallada.

PRUEBA DEL ESTUDIANTE:
{student_text}

PAUTA DE RESPUESTAS:
{answer_key_text}

Instrucciones para la evaluación:
1.  Compara cuidadosamente las respuestas del estudiante con las respuestas esperadas en la pauta.
2.  Para cada pregunta o sección principal que puedas identificar, determina si la respuesta del estudiante es Correcta, Parcialmente Correcta, Incorrecta o No Respondida. Asigna un puntaje de 0 a 10 para cada una (0 para incorrecta/no respondida, 10 para totalmente correcta).
3.  Proporciona un feedback específico para cada pregunta o sección, explicando por qué la respuesta es correcta, incorrecta o parcial. Incluye sugerencias de mejora si es posible.
4.  Asigna un puntaje numérico general (de 0 a 100) que refleje el desempeño global del estudiante.
5.  Proporciona un nivel de confianza (de 0 a 1) en tu evaluación.
6.  Resume el feedback general en un párrafo conciso.

Formato de Respuesta JSON Esperado (asegúrate de que el JSON sea válido y completo):
{{
    "detailed_scores": {{
        "Pregunta 1 (o Tema Principal 1)": {{
            "student_answer": "Respuesta del estudiante...",
            "correct_answer": "Respuesta esperada...",
            "evaluation": "Correcta/Parcialmente Correcta/Incorrecta",
            "feedback": "Feedback específico...",
            "score": 0-10 // Puntaje original 0-10
        }}
        // ... más preguntas o secciones
    }},
    "overall_score": 85, // Puntaje general original (0-100)
    "general_feedback": "Feedback general conciso...",
    "confidence": 0.9 
}}
Si no puedes identificar preguntas específicas, evalúa el texto completo y refleja esto en `detailed_scores` con una entrada general como "Evaluación General".
Intenta ser lo más detallado posible en `detailed_scores`.
IMPORTANTE: Tu respuesta DEBE SER EXCLUSIVAMENTE un objeto JSON válido que siga la estructura especificada. No incluyas ```json```, explicaciones, comentarios o cualquier otro texto fuera del propio objeto JSON.
""")
        prompt_template_obj = PromptTemplate.from_template(prompt_template_str)
        prompt = prompt_template_obj.format(student_text=student_text, answer_key_text=answer_key_text)
        response_text = call_google_api(prompt)

        try:
            result = _parse_json_from_response(response_text, "evaluate_direct")
            if isinstance(result, dict) and "error" in result:
                error_detail = result.get('reason', result['error'])
                logger.error(f"Error de la API de Google en evaluación directa: {error_detail}")
                return {"detailed_scores": {}, "overall_score": 1.0, "general_feedback": f"Error API Google: {error_detail}", "confidence": 0, "error_api": error_detail}

            # Conversión de escalas
            if "overall_score" in result and isinstance(result["overall_score"], (int, float)):
                result["original_overall_score_percentage"] = result["overall_score"] # Guardar original si se desea
                result["overall_score"] = convert_to_1_to_7_scale(result["overall_score"], scale_max=100)
            else: # Si no hay overall_score o no es numérico, poner nota mínima
                 result["overall_score"] = 1.0

            if "detailed_scores" in result and isinstance(result["detailed_scores"], dict):
                for q_id, q_eval in result["detailed_scores"].items():
                    if isinstance(q_eval, dict) and "score" in q_eval and isinstance(q_eval["score"], (int, float)):
                        q_eval["original_score_0_10"] = q_eval["score"] # Guardar original
                        q_eval["score"] = convert_to_1_to_7_scale(q_eval["score"], scale_max=10)
            else:
                result["detailed_scores"] = {}
            
            if "general_feedback" not in result: result["general_feedback"] = "No se pudo generar feedback."
            if "confidence" not in result: result["confidence"] = 0
            
            logger.info("Evaluación directa completada y convertida a escala 1-7 (Google API).")
            return result
        except json.JSONDecodeError:
            logger.error(f"Fallo final al parsear JSON de evaluate_direct. Respuesta: {response_text[:500]}")
            return {"detailed_scores": {}, "overall_score": 1.0, "general_feedback": "Error: La respuesta del modelo no fue un JSON válido (Google API).", "confidence": 0, "error_parsing": "Fallo al parsear JSON de Google API (evaluate_direct)"}
    
    except ValueError as ve:
        logger.error(f"Error de configuración impidió la evaluación directa: {ve}")
        return {"detailed_scores": {}, "overall_score": 1.0, "general_feedback": f"Error de configuración: {ve}", "confidence": 0, "error_config": str(ve)}
    except Exception as e:
        logger.error(f"Error crítico en la evaluación directa (Google API): {str(e)}")
        logger.exception("Detalles de la excepción en evaluate_direct:")
        return {"detailed_scores": {}, "overall_score": 1.0, "general_feedback": f"Error crítico durante la evaluación directa: {str(e)}", "confidence": 0, "error_critical": str(e)}

def evaluate_structured(student_answers: Dict[str, Any], structure: Dict[str, Any], 
                         student_text: str, answer_key_text: str) -> Dict[str, Any]:
    try:
        logger.info("Realizando evaluación estructurada usando Google API...")

        if isinstance(student_answers, dict) and any(err_key in student_answers for err_key in ["error_api", "error_parsing", "error_config", "error_critical", "error_dependency"]):
            logger.error(f"Evaluación estructurada no puede continuar debido a error previo en student_answers: {student_answers}")
            error_detail = student_answers.get("error_api") or student_answers.get("error_parsing") or student_answers.get("error_config") or student_answers.get("error_critical") or student_answers.get("error_dependency", "Error desconocido en paso anterior")
            return {"detailed_scores": {}, "overall_score": 1.0, "general_feedback": f"Error previo impidió evaluación estructurada: {error_detail}", "confidence": 0, "error_prerequisite": error_detail}

        questions_for_prompt = []
        for i, q_struct in enumerate(structure.get("questions", [])):
            q_id = q_struct.get("id", str(i+1))
            q_text = q_struct.get("text", "Pregunta sin texto")
            q_answer_key = q_struct.get("answer", "Respuesta no especificada en pauta")
            current_student_ans = student_answers.get(str(q_id), "No encontrada") if isinstance(student_answers, dict) else "Error en respuestas previas"
            questions_for_prompt.append({"id": q_id, "question_text": q_text, "expected_answer": q_answer_key, "student_answer": current_student_ans})

        questions_json_for_prompt = json.dumps(questions_for_prompt, ensure_ascii=False, indent=2)

        prompt_template_str = ("""
Eres un profesor experto evaluando exámenes. Dada una lista de preguntas, sus respuestas esperadas (pauta) y las respuestas de un estudiante, evalúa cada pregunta.

DATOS DE LAS PREGUNTAS Y RESPUESTAS:
{questions_data_json}

PAUTA GENERAL ADICIONAL (Contexto si es necesario):
{answer_key_text_full}

TEXTO COMPLETO DEL ESTUDIANTE (Contexto si es necesario):
{student_text_full}

Instrucciones para la evaluación de CADA PREGUNTA:
1.  Compara la "student_answer" con la "expected_answer".
2.  Determina si la respuesta es: "Correcta", "Parcialmente Correcta", "Incorrecta", o "No Respondida".
3.  Proporciona un "feedback" conciso y específico para cada pregunta, explicando tu evaluación.
4.  Asigna un "score" numérico de 0 a 10 para cada pregunta (0 para incorrecta/no respondida, 10 para totalmente correcta, valores intermedios para parcial).

Formato de Respuesta JSON Esperado (un diccionario donde cada clave es el 'id' de la pregunta):
{{
    "ID_PREGUNTA_1": {{
        "evaluation": "Correcta", 
        "feedback": "La respuesta es clara y cumple todos los criterios.",
        "score": 10 // Puntaje original 0-10
    }},
    "ID_PREGUNTA_2": {{
        "evaluation": "Parcialmente Correcta",
        "feedback": "Menciona X pero falta Y.",
        "score": 6 // Puntaje original 0-10
    }},
    // ... para cada pregunta
}}
Asegúrate de que el JSON de salida sea válido y siga estrictamente este formato, usando los 'id' de las preguntas como claves principales.
IMPORTANTE: Tu respuesta DEBE SER EXCLUSIVAMENTE un objeto JSON válido que siga la estructura especificada. No incluyas ```json```, explicaciones, comentarios o cualquier otro texto fuera del propio objeto JSON.
""")
        prompt_template_obj = PromptTemplate.from_template(prompt_template_str)
        prompt = prompt_template_obj.format(questions_data_json=questions_json_for_prompt, answer_key_text_full=answer_key_text, student_text_full=student_text)
        response_text = call_google_api(prompt)

        try:
            evaluation_results_per_question = _parse_json_from_response(response_text, "evaluate_structured")
            if isinstance(evaluation_results_per_question, dict) and "error" in evaluation_results_per_question:
                error_detail = evaluation_results_per_question.get('reason', evaluation_results_per_question['error'])
                logger.error(f"Error de la API de Google en evaluación estructurada: {error_detail}")
                return {"detailed_scores": {}, "overall_score": 1.0, "general_feedback": f"Error API Google: {error_detail}", "confidence": 0, "error_api": error_detail}

            detailed_scores_converted = {}
            sum_original_scores = 0
            num_questions_evaluated = 0
            max_possible_score_per_question = 10 # Asumiendo que la IA da puntaje 0-10

            for q_struct in structure.get("questions", []):
                q_id_str = str(q_struct.get("id")) 
                question_eval_original = evaluation_results_per_question.get(q_id_str)
                student_ans_for_q = student_answers.get(q_id_str, "No encontrada") if isinstance(student_answers, dict) else "Error en respuestas previas"

                if question_eval_original and isinstance(question_eval_original, dict) and "score" in question_eval_original and isinstance(question_eval_original["score"], (int, float)):
                    original_score = question_eval_original["score"]
                    converted_score = convert_to_1_to_7_scale(original_score, scale_max=max_possible_score_per_question)
                    
                    detailed_scores_converted[q_id_str] = {
                        "student_answer": student_ans_for_q,
                        "correct_answer": q_struct.get("answer", ""),
                        "evaluation": question_eval_original.get("evaluation", "Error en evaluación"),
                        "feedback": question_eval_original.get("feedback", "Sin feedback del LLM."),
                        "score": converted_score, # Nota 1-7
                        "original_score_0_10": original_score # Guardar original
                    }
                    sum_original_scores += original_score
                    num_questions_evaluated +=1
                else:
                    detailed_scores_converted[q_id_str] = {
                        "student_answer": student_ans_for_q,
                        "correct_answer": q_struct.get("answer", ""),
                        "evaluation": "Error en Formato de Respuesta del LLM",
                        "feedback": f"El LLM no devolvió una evaluación válida para la pregunta {q_id_str}. Respuesta: {question_eval_original}",
                        "score": 1.0, # Nota mínima
                        "original_score_0_10": 0
                    }
                    if not (question_eval_original and isinstance(question_eval_original, dict) and "score" in question_eval_original and isinstance(question_eval_original["score"], (int, float))):
                         num_questions_evaluated +=1

            overall_score_percentage = 0
            if num_questions_evaluated > 0:
                max_total_score_possible = num_questions_evaluated * max_possible_score_per_question
                if max_total_score_possible > 0 :
                     overall_score_percentage = (sum_original_scores / max_total_score_possible) * 100
            
            overall_score_1_to_7 = convert_to_1_to_7_scale(overall_score_percentage, scale_max=100)
            
            general_feedback_summary = f"El estudiante obtuvo una nota final de {overall_score_1_to_7:.1f} (equivalente a {overall_score_percentage:.1f}% de logro)."
            
            logger.info("Evaluación estructurada completada y convertida a escala 1-7 (Google API).")
            return {
                "detailed_scores": detailed_scores_converted,
                "overall_score": overall_score_1_to_7,
                "general_feedback": general_feedback_summary,
                "confidence": 0.85, 
                "original_overall_score_percentage": round(overall_score_percentage, 1)
            }

        except json.JSONDecodeError:
            logger.error(f"Fallo final al parsear JSON de evaluate_structured. Respuesta: {response_text[:500]}")
            return {"detailed_scores": {}, "overall_score": 1.0, "general_feedback": "Error: La respuesta del modelo para la evaluación estructurada no fue un JSON válido (Google API).", "confidence": 0, "error_parsing": "Fallo al parsear JSON de Google API (evaluate_structured)"}
    
    except ValueError as ve: 
        logger.error(f"Error de configuración impidió la evaluación estructurada: {ve}")
        return {"detailed_scores": {}, "overall_score": 1.0, "general_feedback": f"Error de configuración: {ve}", "confidence": 0, "error_config": str(ve)}
    except Exception as e:
        logger.error(f"Error crítico en la evaluación estructurada (Google API): {str(e)}")
        logger.exception("Detalles de la excepción en evaluate_structured:")
        return {"detailed_scores": {}, "overall_score": 1.0, "general_feedback": f"Error crítico durante la evaluación estructurada: {str(e)}", "confidence": 0, "error_critical": str(e)} 