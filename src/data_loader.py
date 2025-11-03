import os
import pandas as pd
import requests
import json
from io import BytesIO
from PIL import Image
from bs4 import BeautifulSoup
import time # Para agregar pausas entre requests (buena práctica de scraping)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# Rutas y configuración
CSV_PATH = os.path.join('data', 'FakeNewsNet', 'dataset')
REQUIRED_COLUMNS = ['title', 'news_url', 'image_url', 'text', 'label']

try:
    import pytesseract
except ImportError:
    pytesseract = None
    print("La librería pytesseract no se pudo importar")

def scrape_article_content(url): # Descarga la página web de la noticia y extrae el texto del artículo y la URL de la imagen principal.
    
    text = ""
    image_url = ""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return text, image_url
        try:
            soup = BeautifulSoup(response.content, 'lxml')
        except Exception:
             # Si falla el parsing simplemente devolvemos vacío y saltamos este artículo
            return text, image_url
            
        # 1. Extraer URL de imagen 
        og_image = soup.find('meta', property='og:image')
        if og_image and 'content' in og_image.attrs:
            image_url = og_image['content']

        # 2. Extraer texto del artículo: Método heurístico
        article_body = soup.find('article') or soup.find('div', class_=lambda c: c and ('article-body' in c or 'content' in c))
        
        if article_body:
            paragraphs = article_body.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
        
        if not text:
             paragraphs = soup.find_all('p')
             text = ' '.join([p.get_text() for p in paragraphs])
             
        text = text[:10000] # Limitar a 10k caracteres
        
    except requests.exceptions.RequestException:
        # Captura errores de red/timeout
        pass
    except Exception:
        # Captura cualquier otro error durante el proceso de scraping
        pass
    return text, image_url

def load_fakenewsnet_data_from_csv(): # Carga los datos de PolitiFact
    
    full_path = os.path.abspath(CSV_PATH)
    
    # 1. Cargar metadatos
    data_frames = []
    for ftype, label in [('politifact_real.csv', 0), ('politifact_fake.csv', 1)]:
        file_path = os.path.join(full_path, ftype)
        try:
            df = pd.read_csv(file_path)
            df['label'] = label
            data_frames.append(df)
        except FileNotFoundError:
            pass

    if not data_frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.concat(data_frames, ignore_index=True)
    
    # 2. Verificar si necesitamos scraping
    available_cols = df.columns.tolist()
    critical_missing = [col for col in ['image_url', 'text'] if col not in available_cols]
    
    if critical_missing and 'news_url' in available_cols:
        print(f"\nINICIANDO WEB SCRAPING: Faltan columnas: {critical_missing}")
        
        df['text'] = ''
        df['image_url'] = ''
        
        # df_to_scrape = df
        df_to_scrape = df.head(10)
        
        # Aplicar la función de scraping
        scraped_results = []
        
        iterator = tqdm(df_to_scrape.iterrows(), total=len(df_to_scrape), desc="Scraping Artículos")
            
        for index, row in iterator:
            text, image_url = scrape_article_content(row['news_url'])
            scraped_results.append({'index': index, 'text': text, 'image_url': image_url})
            time.sleep(0.5) # Pausa de 0.5 segundos 
        
        # Merge de los resultados scrapeados
        scraped_df = pd.DataFrame(scraped_results).set_index('index')
        df.update(scraped_df) # Actualizar el DataFrame principal con los resultados
        
        # Eliminar filas donde el scraping no pudo obtener texto ni imagen
        df = df.dropna(subset=['text', 'image_url']) 
        df = df[df['text'] != '']
        df = df[df['image_url'] != '']

    elif critical_missing:
        print(f"\n¡ERROR! Las columnas críticas faltan: {critical_missing}. No se puede continuar.")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    # 3. Limpieza
    df = df[REQUIRED_COLUMNS] # Seleccionar las columnas finales
    df = df.reset_index(drop=True)
    
    print(f"\nSe completó la fase de carga. Total de artículos listos para procesamiento: {len(df)}")
    return df

def extract_text_from_url(image_url): # Descarga una imagen desde una URL y aplica OCR (usando Tesseract).
    
    if pytesseract is None:
        return ""
        
    if not isinstance(image_url, str) or not image_url.startswith('http'):
        return ""
        
    try:
        response = requests.get(image_url, timeout=5)
        if response.status_code != 200:
            return ""

        img = Image.open(BytesIO(response.content))
        text = pytesseract.image_to_string(img)
        
        return text
    except Exception:
        return ""

if __name__ == '__main__':
    print("Iniciando prueba de carga")
    df = load_fakenewsnet_data_from_csv()
    
    print("\nMuestra de datos después del scraping")
    print(df[['title', 'news_url', 'image_url', 'text', 'label']].head())
    print(f"\nTotal de noticias procesadas: {len(df)}")
    
    if not df.empty:
        print(f"Distribución de etiquetas:\n{df['label'].value_counts()}")