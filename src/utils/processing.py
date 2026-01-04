import os
import re
import json
import time
import argparse
import requests
import pandas as pd
from io import BytesIO
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    TESSERACT_AVAILABLE = False
    print("pytesseract no disponible. OCR deshabilitado.")

# NLP
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    # Descargar recursos si no existen
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    STOP_WORDS = set(stopwords.words('english')).union(set(stopwords.words('spanish')))
    NLTK_AVAILABLE = True
except Exception as e:
    STOP_WORDS = set()
    NLTK_AVAILABLE = False
    print(f"NLTK no completamente disponible: {e}")


# Configuración
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'


class TextCleaner: # Limpieza y normalización de texto
    
    def __init__(self, remove_stopwords=True, language='both'):
        self.remove_stopwords = remove_stopwords
        self.language = language
        
        if not NLTK_AVAILABLE:
            self.remove_stopwords = False
    
    def clean(self, text):
        if not isinstance(text, str):
            return ""
        
        # Minúsculas
        text = text.lower()
        
        # Eliminar URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Eliminar menciones y hashtags (para contenido de redes sociales)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Mantener solo letras, números y espacios
        text = re.sub(r'[^a-záéíóúüñ0-9\s]', ' ', text)
        
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remover stopwords si está habilitado
        if self.remove_stopwords and NLTK_AVAILABLE:
            tokens = word_tokenize(text)
            tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
            text = ' '.join(tokens)
        
        return text


class OCRExtractor: # Extracción de texto desde imágenes usando OCR
    
    def __init__(self, timeout=10, cache=True):
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("Tesseract no está disponible. Instala pytesseract y tesseract-ocr")
        
        self.timeout = timeout
        self.cache = cache
        self._cache_dict = {}
    
    def extract_from_url(self, url): # Extrae texto de una imagen URL usando OCR
    
        if not url or not isinstance(url, str):
            return ""
        
        # Verificar cache
        if self.cache and url in self._cache_dict:
            return self._cache_dict[url]
        
        # Validar URL
        if not url.startswith('http'):
            return ""
        
        try:
            # Descargar imagen
            response = requests.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            # Abrir con PIL
            img = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Aplicar OCR
            text = pytesseract.image_to_string(img, lang='eng+spa')
            text = text.strip()
            
            # Guardar en cache
            if self.cache:
                self._cache_dict[url] = text
            
            return text
        
        except requests.RequestException:
            return ""
        except Exception:
            return ""
    
    def get_cache_stats(self): # Retorna estadísticas del cache
        return {
            'cached_urls': len(self._cache_dict),
            'cache_enabled': self.cache
        }


class DatasetProcessor: # Procesador principal de datasets
    
    def __init__(self, dataset_name, use_ocr=True, clean_text=True):
        self.dataset_name = dataset_name
        self.use_ocr = use_ocr and TESSERACT_AVAILABLE
        self.clean_text = clean_text
        
        # Rutas
        self.raw_dir = DATA_DIR / dataset_name / 'raw'
        self.processed_dir = DATA_DIR / dataset_name / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Componentes
        self.text_cleaner = TextCleaner() if clean_text else None
        self.ocr_extractor = OCRExtractor() if use_ocr else None
    
    def load_raw_data(self): # Carga datos sin procesar
        processed_file = self.raw_dir.parent / 'processed' / 'processed.csv'
        
        if not processed_file.exists():
            raise FileNotFoundError(
                f"No se encontró {processed_file}. "
                f"Primero ejecuta: python src/data_loader.py --dataset {self.dataset_name} --load"
            )
        
        df = pd.read_csv(processed_file)
        print(f"Cargado: {len(df)} filas desde {processed_file}")
        return df
    
    def extract_first_image_url(self, images_field): # Extrae la primera URL de imagen del campo 'images'
        if pd.isna(images_field) or not images_field:
            return None
        
        # Si es una lista/JSON
        if isinstance(images_field, str):
            images_field = images_field.strip()
            
            # Intentar parsear como JSON
            if images_field.startswith('['):
                try:
                    arr = json.loads(images_field)
                    if isinstance(arr, list) and len(arr) > 0:
                        return str(arr[0])
                except json.JSONDecodeError:
                    pass
            
            # Buscar primera URL
            match = re.search(r'https?://[^\s,\]\)\'"]+', images_field)
            if match:
                return match.group(0)
        
        return str(images_field) if images_field else None
    
    def process(self):
        print(f"Procesando: {self.dataset_name.upper()}")
        
        # Cargar datos
        df = self.load_raw_data()
        
        # Asegurar columnas
        for col in ['title', 'text', 'images', 'label']:
            if col not in df.columns:
                df[col] = ''
        
        # LIMPIEZA DE TEXTO 
        if self.clean_text:
            print("Limpiando texto...")
            df['title_clean'] = df['title'].apply(
                lambda x: self.text_cleaner.clean(x) if self.text_cleaner else x
            )
            df['text_clean'] = df['text'].apply(
                lambda x: self.text_cleaner.clean(x) if self.text_cleaner else x
            )
            print("Texto limpiado")
        else:
            df['title_clean'] = df['title']
            df['text_clean'] = df['text']
        
        # Crear baseline textual (sin OCR)
        df['X_baseline'] = (
            df['title_clean'].fillna('') + ' ' + df['text_clean'].fillna('')
        ).str.strip()
        
        # EXTRACCIÓN DE URLs DE IMÁGENES 
        print("Extrayendo URLs de imágenes...")
        df['image_url'] = df['images'].apply(self.extract_first_image_url)
        valid_images = df['image_url'].notna().sum()
        print(f"{valid_images}/{len(df)} URLs válidas")
        
        # OCR 
        if self.use_ocr and self.ocr_extractor:
            print(f"Aplicando OCR ({valid_images} imágenes)...")
            
            ocr_texts = []
            for url in tqdm(df['image_url'], desc="OCR Progress"):
                if pd.isna(url):
                    ocr_texts.append('')
                else:
                    text = self.ocr_extractor.extract_from_url(url)
                    clean_text = self.text_cleaner.clean(text) if self.text_cleaner else text
                    ocr_texts.append(clean_text)
                
                # Pequeña pausa para no sobrecargar
                time.sleep(0.2)
            
            df['ocr_text'] = ocr_texts
            
            # Estadísticas de OCR
            non_empty_ocr = sum(1 for t in ocr_texts if t.strip())
            print(f"OCR completado: {non_empty_ocr}/{valid_images} con texto")
            
            # Cache stats
            if hasattr(self.ocr_extractor, 'get_cache_stats'):
                stats = self.ocr_extractor.get_cache_stats()
                print(f"Cache: {stats['cached_urls']} URLs")
        else:
            df['ocr_text'] = ''
            print("OCR deshabilitado o no disponible")
        
        # Crear X_multimodal (baseline + OCR)
        df['X_multimodal'] = (
            df['X_baseline'].fillna('') + ' ' + df['ocr_text'].fillna('')
        ).str.strip()
        
        # GUARDAR 
        output_file = self.processed_dir / f'{self.dataset_name}_with_ocr.csv'
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\nProcesamiento completado")
        print(f"Guardado en: {output_file}")
        
        # Resumen
        self._print_summary(df)
        
        return df
    
    def _print_summary(self, df): # Imprime resumen del procesamiento
        print(f"Total de muestras: {len(df)}")
        print(f"\nDistribución de clases:")
        print(df['label'].value_counts())
        
        print(f"\nEstadísticas de texto:")
        print(f"Baseline promedio: {df['X_baseline'].str.len().mean():.1f} caracteres")
        print(f"Multimodal promedio: {df['X_multimodal'].str.len().mean():.1f} caracteres")
        
        if 'ocr_text' in df.columns:
            non_empty = (df['ocr_text'].str.len() > 0).sum()
            print(f"  - Muestras con OCR: {non_empty}/{len(df)} ({non_empty/len(df)*100:.1f}%)")
        
        print(f"{'='*60}\n")


def main(): # CLI para procesamiento
    parser = argparse.ArgumentParser(
        description='Procesamiento de datasets con OCR y limpieza de texto'
    )
    
    parser.add_argument(
        '--dataset',
        choices=['FakeNewsNet', 'Fakeddit', 'all'],
        default='all',
        help='Dataset a procesar'
    )
    
    parser.add_argument(
        '--no-ocr',
        action='store_true',
        help='Desactivar OCR (solo limpieza de texto)'
    )
    
    parser.add_argument(
        '--no-clean',
        action='store_true',
        help='Desactivar limpieza de texto'
    )
    
    args = parser.parse_args()
    
    datasets = ['FakeNewsNet', 'Fakeddit'] if args.dataset == 'all' else [args.dataset]
    
    for dataset_name in datasets:
        try:
            processor = DatasetProcessor(
                dataset_name,
                use_ocr=not args.no_ocr,
                clean_text=not args.no_clean
            )
            processor.process()
        except Exception as e:
            print(f"Error procesando {dataset_name}: {e}\n")
            continue

if __name__ == '__main__':
    main()