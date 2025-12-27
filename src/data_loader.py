import os
import pandas as pd
import requests
import json
from io import BytesIO
from PIL import Image
from bs4 import BeautifulSoup
import time
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# Intentar importar kagglehub 
try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    KAGGLEHUB_AVAILABLE = True
except Exception:
    KAGGLEHUB_AVAILABLE = False

# Rutas y configuración
CSV_PATH = os.path.join('data', 'FakeNewsNet', 'dataset')
ISOT_CSV_PATH = os.path.join('data', 'ISOT', 'ISOT_fake_news.csv')
os.makedirs(CSV_PATH, exist_ok=True)
REQUIRED_COLUMNS = ['title', 'canonical_link', 'images', 'text', 'id']

try:
    import pytesseract
except ImportError:
    pytesseract = None
    print("La librería pytesseract no se pudo importar")

def scrape_article_content(url):
    text = ""
    image_url = ""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return text, image_url

        soup = BeautifulSoup(response.content, 'lxml')

        # Intentar obtener og:image
        og_image = soup.find('meta', property='og:image')
        if og_image and 'content' in og_image.attrs:
            image_url = og_image['content']

        # Extraer texto del artículo
        article_body = soup.find('article') or soup.find(
            'div', class_=lambda c: c and ('article-body' in c or 'content' in c))
        if article_body:
            paragraphs = article_body.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])

        if not text:
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])

        text = text[:10000]
    except Exception:
        pass
    return text, image_url

def try_download_from_kagglehub(target_files=('PolitiFact_real_news_content.csv', 'PolitiFact_fake_news_content.csv')):
    downloaded = []
    if not KAGGLEHUB_AVAILABLE:
        print("kagglehub no está disponible en este entorno. Para descarga automática instala 'kagglehub'.")
        return downloaded

    print("Intentando descargar archivos desde Kaggle (mdepak/fakenewsnet) usando kagglehub...")
    try:
        base_path = kagglehub.dataset_download("mdepak/fakenewsnet")
        available = os.listdir(base_path)
        for tf in target_files:
            if tf in available:
                try:
                    df_tmp = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "mdepak/fakenewsnet", tf)
                    out_path = os.path.join(CSV_PATH, tf)
                    df_tmp.to_csv(out_path, index=False, encoding='utf-8')
                    downloaded.append(out_path)
                    print(f"Descargado y guardado: {out_path}")
                except Exception as e:
                    print(f"No se pudo cargar/guardar {tf} desde kagglehub: {e}")
    except Exception as e:
        print(f"Error descargando dataset desde kagglehub: {e}")
    return downloaded

def find_candidate_csvs(path):
    candidates = []
    try:
        for fname in os.listdir(path):
            if fname.lower().endswith('.csv'):
                fpath = os.path.join(path, fname)
                try:
                    if os.path.getsize(fpath) > 200:
                        candidates.append(fpath)
                except Exception:
                    candidates.append(fpath)
    except FileNotFoundError:
        pass
    return candidates

def load_fakenewsnet_data_from_csv():
    full_path = os.path.abspath(CSV_PATH)

    expected_files = {
        'PolitiFact_real_news_content.csv': 0,
        'PolitiFact_fake_news_content.csv': 1
    }

    data_frames = []
    found_any = False

    # 1) Intentar cargar archivos locales
    for fname, label in expected_files.items():
        file_path = os.path.join(full_path, fname)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df['label'] = label
                data_frames.append(df)
                found_any = True
                print(f"Cargado: {file_path} (label={label})")
            except Exception as e:
                print(f"Error leyendo {file_path}: {e}")

    # 2) Intentar descargar si no están
    if not found_any:
        downloaded = try_download_from_kagglehub(list(expected_files.keys()))
        if downloaded:
            for p in downloaded:
                try:
                    df = pd.read_csv(p)
                    label = 0 if 'real' in os.path.basename(p).lower() else 1
                    df['label'] = label
                    data_frames.append(df)
                    found_any = True
                except Exception as e:
                    print(f"Error cargando descargado {p}: {e}")

    if not data_frames:
        print("\nNo se encontraron CSVs de entrada. Por favor coloca:")
        print(" - PolitiFact_real_news_content.csv")
        print(" - PolitiFact_fake_news_content.csv")
        print("en la carpeta:", full_path)
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.concat(data_frames, ignore_index=True)
    available_cols = df.columns.tolist()
    
def load_isot_data_from_csv():
    if not os.path.exists(ISOT_CSV_PATH):
        print(f"ISOT dataset no encontrado en {ISOT_CSV_PATH}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ['label'])

    try:
        df = pd.read_csv(ISOT_CSV_PATH)
        print(f"Cargado ISOT Fake News: {len(df)} filas")
    except Exception as e:
        print(f"Error cargando ISOT dataset: {e}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ['label'])

    # Normalizar columnas
    if 'text' not in df.columns:
        raise ValueError("El dataset ISOT debe contener la columna 'text'")

    # Normalizar etiquetas
    if df['label'].dtype == object:
        df['label'] = df['label'].str.lower().map({
            'fake': 1,
            'real': 0
        })

    # Completar columnas requeridas
    df['title'] = ''
    df['canonical_link'] = ''
    df['images'] = ''
    df['id'] = df.index.astype(str)

    df = df[REQUIRED_COLUMNS + ['label']]
    df = df.dropna(subset=['text', 'label'])

    return df

    # ⚙️ Scraping usando 'canonical_link' e 'images'
    critical_missing = [col for col in ['text'] if col not in available_cols]
    if critical_missing and 'canonical_link' in available_cols:
        print(f"\nINICIANDO WEB SCRAPING: Se usará 'canonical_link' e 'images'")
        if 'text' not in df.columns:
            df['text'] = ''

        scraped_results = []
        iterator = tqdm(df.iterrows(), total=len(df), desc="Scraping Artículos")
        for index, row in iterator:
            news_url = row.get('canonical_link', '')
            if not isinstance(news_url, str) or not news_url.startswith('http'):
                scraped_results.append({'index': index, 'text': '', 'images': row.get('images', '')})
                continue
            text, image_url = scrape_article_content(news_url)
            # Si ya hay imágenes en la columna 'images', no sobrescribir
            final_image = row.get('images', image_url)
            scraped_results.append({'index': index, 'text': text, 'images': final_image})
            time.sleep(0.5)

        scraped_df = pd.DataFrame(scraped_results).set_index('index')
        df.update(scraped_df)
        df = df.dropna(subset=['text'])
        df = df[df['text'] != '']

    elif critical_missing:
        print(f"\n¡ERROR! Faltan columnas críticas y no hay 'canonical_link' para hacer scraping.")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    # Asegurar columnas requeridas
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = '' if c != 'label' else 0

    df = df[REQUIRED_COLUMNS + ['label']]
    df = df.reset_index(drop=True)

    # 🔗 Cargar ISOT y fusionar
    df_isot = load_isot_data_from_csv()

    if not df_isot.empty:
        df = pd.concat([df, df_isot], ignore_index=True)
        print(f"ISOT añadido correctamente. Total combinado: {len(df)}")

    print(f"\nSe completó la fase de carga. Total de artículos listos para procesamiento: {len(df)}")
    return df


def extract_text_from_url(image_url):
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

    print("\n*** Diagnóstico rápido ***")
    print(f"DataFrame vacío? {df.empty}")
    print(f"Columnas: {list(df.columns)}")
    try:
        print(f"Primeras filas:\n{df.head(3)}")
    except Exception as e:
        print(f"No se pudo mostrar head(): {e}")
    print(f"Total de filas obtenidas: {len(df)}")
    if not df.empty and 'label' in df.columns:
        print(f"Distribución de etiquetas:\n{df['label'].value_counts()}")

    out_dir = os.path.join('data', 'FakeNewsNet')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fakenewsnet_processed.csv')

    if not df.empty:
        try:
            df.to_csv(out_path, index=False, encoding='utf-8')
            print(f"\nGuardado exitoso en: {os.path.abspath(out_path)}")
        except Exception as e:
            print(f"\nError al guardar CSV: {e}")
    else:
        print("\nNo se guardó CSV porque el DataFrame está vacío. Revisa los mensajes anteriores para más detalles.")
