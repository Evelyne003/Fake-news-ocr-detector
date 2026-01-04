import os
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import kagglehub

# CONFIGURACIÓN DE RUTAS
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

FAKEDDIT_ZIP_PATH = Path(
    r"C:/Users/evely/OneDrive/Documentos/multimodal_only_samples-20260103T021236Z-3-001.zip"
)

# CONFIGURACIÓN DE DATASETS
DATASETS_CONFIG = {
    "fakenewsnet": {
        "type": "kaggle",
        "kaggle_id": "mdepak/fakenewsnet",
        "target_files": [
            "PolitiFact_real_news_content.csv",
            "PolitiFact_fake_news_content.csv",
        ],
        "label_mapping": {
            "PolitiFact_real_news_content.csv": 0,
            "PolitiFact_fake_news_content.csv": 1,
        },
        "output_dir": DATA_DIR / "FakeNewsNet",
        "required_columns": ["id", "title", "text", "images"],
        "description": "FakeNewsNet (PolitiFact)",
    },
    "fakeddit": {
        "type": "local_zip",
        "zip_path": FAKEDDIT_ZIP_PATH,
        "target_files": [
            "multimodal_train.tsv",
            "multimodal_validate.tsv",
            "multimodal_test_public.tsv",
        ],
        "output_dir": DATA_DIR / "Fakeddit",
        "required_columns": ["id", "clean_title", "image_url", "2_way_label"],
        "description": "Fakeddit v2.0 multimodal",
        "separator": "\t",
    },
}

# DATASET LOADER
class DatasetLoader:
    def __init__(self, dataset_name: str):
        if dataset_name not in DATASETS_CONFIG:
            raise ValueError(f"Dataset no soportado: {dataset_name}")

        self.name = dataset_name
        self.config = DATASETS_CONFIG[dataset_name]
        self.raw_dir = self.config["output_dir"] / "raw"
        self.processed_dir = self.config["output_dir"] / "processed"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # DESCARGA
    def download(self, force=False):
        print(f"\nDescargando: {self.name.upper()}")
        print(f"Descripción: {self.config['description']}")

        if not force and self._files_exist():
            print("Archivos ya existen")
            return True

        try:
            if self.config["type"] == "kaggle":
                self._download_from_kaggle()
            elif self.config["type"] == "local_zip":
                self._extract_from_zip()
            return True
        except Exception as e:
            print(f"Error descargando {self.name}: {e}")
            return False

    def _files_exist(self):
        return all((self.raw_dir / f).exists() for f in self.config["target_files"])

    def _download_from_kaggle(self):
        path = kagglehub.dataset_download(self.config["kaggle_id"])
        path = Path(path)

        for fname in self.config["target_files"]:
            shutil.copy2(path / fname, self.raw_dir / fname)
            print(f"✓ {fname}")

    def _extract_from_zip(self):
        if not self.config["zip_path"].exists():
            raise FileNotFoundError("ZIP de Fakeddit no encontrado")

        print("Extrayendo ZIP de Fakeddit...")
        with zipfile.ZipFile(self.config["zip_path"], "r") as z:
            for member in z.namelist():
                if member.endswith(".tsv"):
                    target = self.raw_dir / Path(member).name
                    with z.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"{target.name}")

    # CARGA
    def load(self):
        print(f"\nCargando: {self.name.upper()}")

        if self.name == "fakenewsnet":
            return self._load_fakenewsnet()
        return self._load_fakeddit()

    def _load_fakenewsnet(self):
        dfs = []

        for fname, label in self.config["label_mapping"].items():
            df = pd.read_csv(self.raw_dir / fname)
            df["label"] = label
            df["source_file"] = fname
            dfs.append(df)
            print(f"✓ {fname} ({len(df)} filas)")

        df = pd.concat(dfs, ignore_index=True)
        df["text"] = df["text"].fillna("")
        df["title"] = df["title"].fillna("")
        df["images"] = df["images"].fillna("")

        return df[["id", "title", "text", "images", "label", "source_file"]]

    def _load_fakeddit(self):
        dfs = []
        sep = self.config["separator"]

        for fname in self.config["target_files"]:
            path = self.raw_dir / fname
            if not path.exists():
                print(f"{fname} no encontrado")
                continue

            df = pd.read_csv(path, sep=sep, low_memory=False)
            df["source_file"] = fname
            dfs.append(df)
            print(f"{fname} ({len(df)} filas)")

        if not dfs:
            raise RuntimeError("Fakeddit no disponible")

        df = pd.concat(dfs, ignore_index=True)

        df["title"] = df["clean_title"]
        df["text"] = ""
        df["images"] = df["image_url"]
        df["label"] = df["2_way_label"].astype(int)

        return df[["id", "title", "text", "images", "label", "source_file"]]

    # GUARDADO
    def save_processed(self, df, filename="processed.csv"):
        path = self.processed_dir / filename
        df.to_csv(path, index=False)
        print(f"Guardado en {path}")

# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fakenewsnet", "fakeddit", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    datasets = DATASETS_CONFIG.keys() if args.dataset == "all" else [args.dataset]

    for name in datasets:
        loader = DatasetLoader(name)
        loader.download(force=args.force)
        if args.load:
            df = loader.load()
            loader.save_processed(df)


if __name__ == "__main__":
    main()