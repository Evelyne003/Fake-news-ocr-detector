import os
import glob
import json
import pandas as pd

# Ruta donde clonamos el dataset
DATA_PATH = 'data/FakeNewsNet/Data/PolitiFact/'

def load_fakenewsnet_data():
    """
    Carga los datos de PolitiFact (texto, imagen, etiqueta) en un DataFrame.
    """
    print(f"Buscando datos en: {DATA_PATH}")

    # 1. Noticias falsas 
    fake_news_paths = glob.glob(os.path.join(DATA_PATH, 'fake', 'politifact*'))

    # 2. Noticias reales 
    real_news_paths = glob.glob(os.path.join(DATA_PATH, 'real', 'politifact*'))

    all_news_paths = [
        (path, 'fake') for path in fake_news_paths
    ] + [
        (path, 'real') for path in real_news_paths
    ]

    data = []

    for path, label in all_news_paths:
        try:
            # Cargar el texto del artículo
            with open(os.path.join(path, 'news_article.json'), 'r') as f:
                article = json.load(f)
                text = article.get('text', '')

            # Obtener la ruta de la imagen principal
            image_search = glob.glob(os.path.join(path, 'top_img.*'))   # Usamos glob para encontrar la imagen sin importar la extensión

            if text and image_search:
                image_path = image_search[0]
                data.append({
                    'text': text,
                    'image_path': image_path,
                    'label': 1 if label == 'fake' else 0 # 1 = FAKE, 0 = REAL
                })
        except Exception as e:
            print(f"Error cargando {path}: {e}")

    print(f"Se cargaron {len(data)} artículos.")
    return pd.DataFrame(data)

if __name__ == '__main__':
    # Para probar que funciona
    df = load_fakenewsnet_data()
    print("\n--- Muestra de datos ---")
    print(df.head())
    print(f"\nTotal de noticias: {len(df)}")
    print(f"Distribución de etiquetas:\n{df['label'].value_counts()}")