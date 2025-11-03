import os
import re
import json
import time
import requests
import pandas as pd
from io import BytesIO
from PIL import Image

# === OCR y NLTK ===
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    TESSERACT_AVAILABLE = False
    print("pytesseract no disponible: OCR no se ejecutará (instala pytesseract + tesseract).")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    stop_words = set(stopwords.words('english')).union(set(stopwords.words('spanish')))
except Exception:
    stop_words = set()
    print("No se pudieron cargar las stopwords. Se continuará sin filtrarlas.")

PROCESSED_FILE = 'notebooks/data/processed/fakenewsnet_processed.csv'


# --- Funciones auxiliares ---

def clean_text(text):
    """Limpia texto: minúsculas, solo letras, sin stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text)  # Solo letras y espacios
    tokens = word_tokenize(text)
    cleaned_tokens = [w for w in tokens if w not in stop_words]
    return " ".join(cleaned_tokens)


def first_image_url_from_field(images_field):
    """Devuelve la primera URL válida desde el campo 'images'."""
    if pd.isna(images_field) or images_field is None:
        return None
    if isinstance(images_field, (list, tuple)) and len(images_field) > 0:
        return images_field[0]
    if isinstance(images_field, str):
        s = images_field.strip()
        # Si parece un JSON array
        if s.startswith('[') and s.endswith(']'):
            try:
                arr = json.loads(s)
                if isinstance(arr, list) and len(arr) > 0:
                    return arr[0]
            except Exception:
                pass
        # Buscar primera URL http(s)
        m = re.search(r'https?://[^\s,\]\)\'"]+', s)
        if m:
            return m.group(0)
    return None


def ocr_from_image_url(url, timeout=8):
    """Descarga imagen y aplica OCR."""
    if not TESSERACT_AVAILABLE or not url:
        return ""
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return ""
        img = Image.open(BytesIO(resp.content)).convert('RGB')
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception:
        return ""


# --- Proceso principal ---
if __name__ == "__main__":
    try:
        df = pd.read_csv(PROCESSED_FILE)
        print(f"Archivo cargado correctamente: {PROCESSED_FILE} ({len(df)} filas)")
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo '{PROCESSED_FILE}'.")
        exit()

    # Verificar columnas básicas
    for col in ['title', 'text', 'images', 'label']:
        if col not in df.columns:
            df[col] = ''

    # Limpiar texto base
    print("Limpieza de texto base (title + text)...")
    df['title_clean'] = df['title'].apply(clean_text)
    df['text_clean'] = df['text'].apply(clean_text)

    # Crear X_baseline
    df['X_baseline'] = (df['title_clean'].fillna('') + ' ' + df['text_clean'].fillna('')).str.strip()

    # Obtener la primera URL de imagen
    print("Extrayendo la primera URL de imagen...")
    df['image_url_first'] = df['images'].apply(first_image_url_from_field)

    # Aplicar OCR
    print("Aplicando OCR a las imágenes (esto puede tardar unos minutos)...")
    ocr_cache = {}
    ocr_texts = []
    for idx, url in enumerate(df['image_url_first'].tolist()):
        if not url:
            ocr_texts.append('')
            continue
        if url in ocr_cache:
            ocr_texts.append(ocr_cache[url])
            continue
        txt = ocr_from_image_url(url)
        txt_clean = clean_text(txt)
        ocr_cache[url] = txt_clean
        ocr_texts.append(txt_clean)
        time.sleep(0.3)  # pequeña pausa entre requests

    df['ocr_text'] = ocr_texts

    # Crear X_ocr_model
    df['X_ocr_model'] = (df['X_baseline'].fillna('') + ' ' + df['ocr_text'].fillna('')).str.strip()

    # Guardar CSV actualizado
    try:
        os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
        df.to_csv(PROCESSED_FILE, index=False, encoding='utf-8')
        print(f"\nArchivo actualizado con nuevas columnas guardado en:\n→ {PROCESSED_FILE}")
    except Exception as e:
        print(f"Error al guardar el archivo procesado: {e}")

    # Mostrar vista previa
    print("\nMuestra de columnas generadas:")
    print(df[['title', 'X_baseline', 'X_ocr_model', 'label']].head(2))
    print(f"\nTotal de artículos procesados: {len(df)}")
