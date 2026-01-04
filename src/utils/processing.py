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

# NLP (NLTK)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import wordpunct_tokenize

    # Descargar recursos necesarios
    for resource in ["stopwords", "punkt", "punkt_tab"]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource, quiet=True)

    STOP_WORDS = set(stopwords.words("english")).union(
        set(stopwords.words("spanish"))
    )
    NLTK_AVAILABLE = True

except Exception as e:
    STOP_WORDS = set()
    NLTK_AVAILABLE = False
    print(f"NLTK no disponible completamente: {e}")

# CONFIGURACIÓN
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# LIMPIEZA DE TEXTO
class TextCleaner:
    def __init__(self, remove_stopwords=True):
        self.remove_stopwords = remove_stopwords and NLTK_AVAILABLE

    def clean(self, text):
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"@\w+|#\w+", "", text)
        text = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if self.remove_stopwords:
            tokens = wordpunct_tokenize(text)
            tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
            text = " ".join(tokens)

        return text

# OCR
class OCRExtractor:
    def __init__(self, timeout=10, cache=True):
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("Tesseract no disponible")

        self.timeout = timeout
        self.cache = cache
        self._cache = {}

    def extract_from_url(self, url):
        if not url or not isinstance(url, str) or not url.startswith("http"):
            return ""

        if self.cache and url in self._cache:
            return self._cache[url]

        try:
            r = requests.get(url, timeout=self.timeout)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            text = pytesseract.image_to_string(img, lang="eng+spa").strip()

            if self.cache:
                self._cache[url] = text

            return text

        except Exception:
            return ""

# DATASET PROCESADOR
class DatasetProcessor:
    def __init__(self, dataset_name, use_ocr=True, clean_text=True):
        self.dataset_name = dataset_name
        self.use_ocr = use_ocr and TESSERACT_AVAILABLE
        self.clean_text = clean_text

        self.processed_dir = DATA_DIR / dataset_name / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.text_cleaner = TextCleaner() if clean_text else None
        self.ocr = OCRExtractor() if self.use_ocr else None

    def load_data(self):
        path = self.processed_dir / "processed.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        print(f"Cargado: {len(df)} filas desde {path}")
        return df

    def extract_image_url(self, images):
        if pd.isna(images):
            return None

        if isinstance(images, str):
            if images.startswith("["):
                try:
                    arr = json.loads(images)
                    if arr:
                        return arr[0]
                except Exception:
                    pass

            m = re.search(r"https?://\S+", images)
            if m:
                return m.group(0)

        return None

    def process(self):
        print(f"\nProcesando: {self.dataset_name.upper()}")
        df = self.load_data()

        if self.dataset_name.lower() == "fakeddit":
            SAMPLE_SIZE = 8000
            RANDOM_SEED = 42

            if len(df) > SAMPLE_SIZE:
                print(f"\nUsando muestra estratificada de {SAMPLE_SIZE} ejemplos de Fakeddit "
                    f"(total original: {len(df)})")

                df = (
                    df.groupby("label", group_keys=False)
                    .apply(
                        lambda x: x.sample(
                            n=max(1, int(SAMPLE_SIZE * len(x) / len(df))),
                            random_state=RANDOM_SEED
                        )
                    )
                    .reset_index(drop=True)
                )         

        print("Distribución de clases en la muestra:")
        print(df["label"].value_counts())


        for col in ["title", "text", "images"]:
            if col not in df:
                df[col] = ""

        if self.clean_text:
            print("Limpiando texto...")
            df["title_clean"] = df["title"].apply(self.text_cleaner.clean)
            df["text_clean"] = df["text"].apply(self.text_cleaner.clean)

        df["X_baseline"] = (
            df["title_clean"].fillna("") + " " + df["text_clean"].fillna("")
        ).str.strip()

        print("Extrayendo URLs de imágenes...")
        df["image_url"] = df["images"].apply(self.extract_image_url)

        if self.use_ocr:
            print("Aplicando OCR...")
            ocr_texts = []
            for url in tqdm(df["image_url"]):
                text = self.ocr.extract_from_url(url)
                text = self.text_cleaner.clean(text) if self.clean_text else text
                ocr_texts.append(text)
                time.sleep(0.1)

            df["ocr_text"] = ocr_texts
        else:
            df["ocr_text"] = ""

        df["X_multimodal"] = (
            df["X_baseline"] + " " + df["ocr_text"]
        ).str.strip()

        out = self.processed_dir / f"{self.dataset_name}_with_ocr.csv"
        df.to_csv(out, index=False)
        print(f"Guardado en {out}")

        return df

# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["FakeNewsNet", "Fakeddit", "all"], default="all")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    datasets = ["FakeNewsNet", "Fakeddit"] if args.dataset == "all" else [args.dataset]

    for d in datasets:
        try:
            DatasetProcessor(
                d,
                use_ocr=not args.no_ocr,
                clean_text=not args.no_clean
            ).process()
        except Exception as e:
            print(f"Error procesando {d}: {e}")

if __name__ == "__main__":
    main()