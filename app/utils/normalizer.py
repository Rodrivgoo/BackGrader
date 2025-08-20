import os
import logging
from langchain.prompts import PromptTemplate
from app.utils.evaluator import call_google_api, GOOGLE_API_KEY # Importar desde evaluator

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def filter_relevant_content(normalized_text: str, context: str = "exam") -> str:
    """
    Filtra el contenido normalizado para mantener solo lo relevante para la evaluación:
    elimina títulos, nombres, información personal y otros elementos no evaluables.
    Utiliza la API de Google.
    
    Args:
        normalized_text (str): Texto ya normalizado
        context (str): Contexto del texto ("exam" o "answer_key")
    
    Returns:
        str: Texto filtrado con solo el contenido relevante para evaluación
    """
    if not GOOGLE_API_KEY:
        logger.error("No se puede filtrar contenido: GOOGLE_API_KEY no configurada.")
        return normalized_text # Devolver sin filtrar si no hay API key

    try:
        # Verificar que hay texto para filtrar
        if not normalized_text or len(normalized_text.strip()) < 10:
            logger.warning("Texto demasiado corto para filtrar, devolviendo original.")
            return normalized_text
        
        # Prompt para filtrar contenido no relevante
        prompt_template_str = (
            "Eres un asistente especializado en procesar exámenes académicos.\n\n"
            "TEXTO NORMALIZADO:\n"
            "{normalized_text}\n\n"
            "Tu tarea es EXTRAER ÚNICAMENTE las preguntas y respuestas relevantes para la evaluación.\n\n"
            "ELIMINA completamente los siguientes elementos:\n"
            "1. Títulos del documento o encabezados institucionales\n"
            "2. Nombres de estudiantes o profesores\n"
            "3. RUT, números de identificación o matrícula\n"
            "4. Fechas, horas o duraciones de la prueba\n"
            "5. Instrucciones generales\n"
            "6. Cualquier otro contenido no relacionado directamente con las preguntas y respuestas evaluables\n\n"
            "MANTÉN SOLAMENTE:\n"
            "1. Números y enunciados de las preguntas\n"
            "2. Las respuestas proporcionadas para cada pregunta\n"
            "3. Cualquier contenido que sea esencial para evaluar el conocimiento\n\n"
            "El resultado debe contener ÚNICAMENTE el contenido evaluable, manteniendo el formato de numeración original.\n\n"
            "NO añadas prefacios, introducciones ni explicaciones. SOLO devuelve el contenido relevante sin texto adicional.\n"
            "Si el texto original ya parece ser solo contenido evaluable (por ejemplo, solo una lista de preguntas y respuestas), devuélvelo tal cual.\n\n"
            "CONTENIDO RELEVANTE:"
        )
        prompt_template_obj = PromptTemplate.from_template(prompt_template_str)
        
        # Preparar prompt con el texto normalizado
        prompt = prompt_template_obj.format(normalized_text=normalized_text)
        
        logger.info("Filtrando contenido relevante usando Google API...")
        # Realizar la inferencia con Google API
        filtered_text = call_google_api(prompt, model_name="gemini-2.0-flash-lite") # Usar el mismo modelo que en evaluator
        
        # Limpiar el resultado
        filtered_text = filtered_text.strip()
        
        # Eliminar prefacios comunes
        prefixes_to_remove = [
            "¡Claro!",
            "A continuación",
            "Aquí está",
            "Contenido relevante:",
            "El contenido relevante es:",
            "TEXTO RELEVANTE:", # Añadido por si acaso
            "Este es el contenido relevante:"
        ]
        
        for prefix in prefixes_to_remove:
            # Usar lower() para comparación insensible a mayúsculas/minúsculas al inicio
            if filtered_text.lower().startswith(prefix.lower()):
                filtered_text = filtered_text[len(prefix):].strip()
        
        logger.info(f"Contenido filtrado correctamente usando Google API.")
        return filtered_text
    
    except Exception as e:
        logger.error(f"Error al filtrar contenido relevante con Google API: {str(e)}")
        logger.exception("Detalles de la excepción en filter_relevant_content:")
        # En caso de error, devolver el texto sin filtrar
        return normalized_text

def normalize_text(raw_text: str, context: str ="exam", filter_content: bool = True) -> str:
    """
    Normaliza el texto extraído por OCR utilizando la API de Google.
    
    Args:
        raw_text (str): Texto extraído por OCR
        context (str): Contexto del texto ("exam" o "answer_key")
        filter_content (bool): Si se debe filtrar contenido no relevante después de normalizar
    
    Returns:
        str: Texto normalizado y opcionalmente filtrado
    """
    if not GOOGLE_API_KEY:
        logger.error("No se puede normalizar texto: GOOGLE_API_KEY no configurada.")
        return raw_text # Devolver original si no hay API key

    try:
        # Verificar que hay texto para normalizar
        if not raw_text or len(raw_text.strip()) < 10:
            logger.warning("Texto demasiado corto para normalizar, devolviendo original.")
            return raw_text
        
        prompt_template_str: str
        # Seleccionar prompt según el contexto
        if context == "exam":
            prompt_template_str = (
                "Eres un asistente especializado en corregir y normalizar texto extraído por OCR de exámenes académicos.\n\n"
                "TEXTO OCR:\n"
                "{raw_text}\n\n"
                "Tu tarea es corregir los errores de OCR, teniendo en cuenta que estás procesando un examen o prueba académica. \n"
                "Presta especial atención a:\n"
                "1. Símbolos matemáticos (corrige \"x\" por \"×\", \"z\" por \"2\", etc.)\n"
                "2. Números y letras mal interpretados\n"
                "3. Formato de preguntas y respuestas\n"
                "4. Ecuaciones y fórmulas\n\n"
                "Mantén la estructura original del documento y no agregues información nueva.\n"
                "Importante: No añadas prefacios, introducciones ni explicaciones. Solo devuelve el texto normalizado.\n"
                "Si el texto parece ya estar bien formateado y sin errores obvios de OCR, devuélvelo tal cual.\n\n"
                "TEXTO NORMALIZADO:"
            )
        elif context == "answer_key":
            prompt_template_str = (
                "Eres un asistente especializado en corregir y normalizar texto extraído por OCR de pautas de respuestas (answer keys) para exámenes académicos.\n\n"
                "TEXTO OCR:\n"
                "{raw_text}\n\n"
                "Tu tarea es corregir los errores de OCR, teniendo en cuenta que estás procesando una pauta de respuestas. \n"
                "Presta especial atención a:\n"
                "1. Respuestas correctas y su numeración\n"
                "2. Símbolos matemáticos (corrige \"x\" por \"×\", \"z\" por \"2\", etc.)\n"
                "3. Números y letras mal interpretados\n"
                "4. Ecuaciones y fórmulas\n\n"
                "Mantén la estructura original del documento y asegúrate de que las respuestas sean claras.\n"
                "Importante: No añadas prefacios, introducciones ni explicaciones. Solo devuelve el texto normalizado sin texto adicional.\n"
                "No inicies con frases como \"¡Claro!\" o \"A continuación\". No expliques lo que has corregido al final.\n"
                "Si el texto parece ya estar bien formateado y sin errores obvios de OCR, devuélvelo tal cual.\n\n"
                "TEXTO NORMALIZADO:"
            )
        else: # Contexto genérico
            prompt_template_str = (
                "Corrige y normaliza el siguiente texto extraído por OCR:\n\n"
                "TEXTO OCR:\n"
                "{raw_text}\n\n"
                "No añadas prefacios, introducciones ni explicaciones. Solo devuelve el texto normalizado.\n"
                "Si el texto parece ya estar bien formateado y sin errores obvios de OCR, devuélvelo tal cual.\n\n"
                "TEXTO NORMALIZADO:"
            )
        
        prompt_template_obj = PromptTemplate.from_template(prompt_template_str)
        # Preparar prompt con el texto
        prompt = prompt_template_obj.format(raw_text=raw_text)
        
        logger.info(f"Normalizando texto (contexto: {context}) usando Google API...")
        # Realizar la inferencia con Google API
        normalized_text_from_api = call_google_api(prompt, model_name="gemini-2.0-flash-lite")
        
        # Eliminar posibles introducciones o explicaciones
        cleaned_text = normalized_text_from_api.strip()
        
        # Eliminar prefacios comunes
        prefixes_to_remove = [
            "¡Claro!",
            "A continuación",
            "Aquí está",
            "Texto normalizado:",
            "El texto normalizado es:",
            "TEXTO NORMALIZADO:" # Añadido por si acaso
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned_text.lower().startswith(prefix.lower()): # Usar lower() para comparación insensible
                cleaned_text = cleaned_text[len(prefix):].strip()
        
        # Eliminar explicaciones finales comunes (revisar si esto sigue siendo necesario con Gemini)
        # Puede ser menos propenso a añadir estos comentarios que Llama3
        end_markers = [
            "He corregido los errores",
            "He normalizado el texto",
            "La estructura original"
        ]
        
        for marker in end_markers:
            # Buscar de forma insensible a mayúsculas/minúsculas
            # y solo si el marcador está a más de la mitad del texto (para evitar cortar el inicio si es corto)
            marker_lower = marker.lower()
            cleaned_text_lower = cleaned_text.lower()
            find_pos = cleaned_text_lower.rfind(marker_lower) # Buscar desde el final
            if find_pos > len(cleaned_text_lower) / 2:
                 # Verificar que no estamos cortando algo esencial si el marcador es muy común
                 # Esta lógica es un poco arriesgada, considerar si es realmente necesaria
                 # Por ahora, la mantenemos pero con logging.
                logger.info(f"Posible marcador final '{marker}' encontrado y eliminado de la normalización.")
                cleaned_text = cleaned_text[:find_pos].strip()

        logger.info(f"Texto normalizado correctamente usando Google API (antes de filtro opcional).")
        
        # Aplicar filtrado de contenido si está habilitado
        if filter_content:
            logger.info("Aplicando filtro de contenido después de la normalización...")
            cleaned_text = filter_relevant_content(cleaned_text, context) # Ya usa Google API
            
        return cleaned_text
    
    except Exception as e:
        logger.error(f"Error al normalizar texto con Google API: {str(e)}")
        logger.exception("Detalles de la excepción en normalize_text:")
        # En caso de error, devolver el texto original
        return raw_text 