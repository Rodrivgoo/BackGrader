from PIL import Image
import io
import os
import logging
from typing import List

# Importaciones para Google Cloud Vision
from google.cloud import vision

# Importación para conversión de PDF a imagen
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

logger = logging.getLogger(__name__)

# PDF_TEMP_PAGES_DIR ya no se usa directamente, convert_from_path usa output_folder
# y los archivos se manejan por su nombre completo devuelto por pdf2image.

def extract_text_google_vision(file_path: str) -> str:
    """
    Detecta y extrae texto de un archivo local (imagen o PDF) usando Google Cloud Vision AI.
    Si es un PDF, lo convierte a imágenes página por página y procesa cada una.
    """
    client = vision.ImageAnnotatorClient() # Asume GOOGLE_APPLICATION_CREDENTIALS está configurada
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        logger.info(f"Archivo PDF detectado: {file_path}. Convirtiendo a imágenes...")
        all_text_parts = []
        # temp_page_files almacenará las rutas de los archivos generados por convert_from_path para limpieza
        temp_page_files_generated = [] 

        try:
            base_temp_dir = "temp" 
            pdf_pages_output_dir = os.path.join(base_temp_dir, "pdf_processing_pages")
            # Asegurarse que el directorio base 'temp' y el subdirectorio para páginas existan.
            # main.py crea 'temp'. Aquí creamos el subdirectorio.
            os.makedirs(pdf_pages_output_dir, exist_ok=True)
            
            logger.info(f"Convirtiendo PDF {file_path} a imágenes en {pdf_pages_output_dir}")
            # convert_from_path devuelve una lista de objetos PIL.Image
            # cada objeto tiene un atributo .filename con la ruta donde se guardó el archivo si output_folder se usa.
            images_pil_list = convert_from_path(file_path, dpi=300, output_folder=pdf_pages_output_dir, fmt='png', thread_count=2, paths_only=False)
            
            if not images_pil_list:
                logger.warning(f"pdf2image no devolvió imágenes para {file_path}")
                return "" 

            logger.info(f"PDF convertido a {len(images_pil_list)} imágenes. Procesando con Google Vision...")

            for i, image_pil_obj in enumerate(images_pil_list):
                # Guardar la ruta del archivo temporal para limpiarlo después
                if hasattr(image_pil_obj, 'filename') and image_pil_obj.filename:
                    temp_page_files_generated.append(image_pil_obj.filename)
                
                img_byte_arr = io.BytesIO()
                image_pil_obj.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                image_pil_obj.close() # Cerrar el objeto imagen PIL después de usarlo
                
                logger.info(f"Procesando página {i+1}/{len(images_pil_list)} de {file_path} (Bytes directos)")
                vision_image = vision.Image(content=img_bytes)
                response = client.document_text_detection(image=vision_image)

                if response.error.message:
                    logger.error(f"Error de Google Cloud Vision API para página {i+1} de {file_path}: {response.error.message}")
                    all_text_parts.append(f"[Error en OCR de página {i+1}: {response.error.message}]")
                    continue 
                
                if response.full_text_annotation:
                    all_text_parts.append(response.full_text_annotation.text)
                else:
                    logger.info(f"No se detectó texto en la página {i+1} de {file_path}.")
            
            final_text = "\n\n--- Nueva Página ---\n\n".join(all_text_parts)
            logger.info(f"Texto extraído de PDF {file_path} (todas las páginas):\n{final_text[:500]}...")
            return final_text

        except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e_pdf:
            logger.error(f"Error de pdf2image al procesar {file_path}: {e_pdf}")
            raise Exception(f"Fallo en la conversión de PDF a imágenes: {e_pdf}")
        except Exception as e:
            logger.error(f"Error crítico procesando PDF {file_path} con Google Vision: {e}")
            logger.exception(f"Detalles de la excepción en extract_text_google_vision (PDF path) para {file_path}:")
            raise
        finally:
            logger.info(f"Limpiando {len(temp_page_files_generated)} archivos temporales de páginas PDF...")
            for path_to_remove in temp_page_files_generated:
                try:
                    if os.path.exists(path_to_remove):
                        os.remove(path_to_remove)
                        logger.debug(f"Archivo temporal {path_to_remove} eliminado.")
                except Exception as e_clean:
                    logger.warning(f"No se pudo eliminar el archivo temporal {path_to_remove}: {e_clean}")

    else: # Es un archivo de imagen, no PDF
        logger.info(f"Archivo de imagen detectado: {file_path}. Procesando directamente con Google Vision.")
        try:
            with io.open(file_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            response = client.document_text_detection(image=image)
            
            if response.error.message:
                logger.error(f"Error de Google Cloud Vision API para imagen {file_path}: {response.error.message}")
                raise Exception(f"Google Cloud Vision API error: {response.error.message}")

            if response.full_text_annotation:
                text = response.full_text_annotation.text
                logger.info(f"Texto extraído con Google Vision de {file_path}:\n{text[:500]}...")
                return text
            else:
                logger.info(f"No se detectó texto con Google Vision en {file_path}.")
                return ""
        except Exception as e:
            logger.error(f"Error crítico al usar Google Cloud Vision AI para imagen {file_path}: {e}")
            logger.exception(f"Detalles de la excepción en extract_text_google_vision (Image path) para {file_path}:")
            raise

# Las funciones relacionadas con Tesseract (process_image_tesseract, ocr_pdf_tesseract) han sido eliminadas. 