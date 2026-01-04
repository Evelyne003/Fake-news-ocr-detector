from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Dataset y DataLoader para entrenamiento multimodal
class MultimodalDataset(Dataset):
    def _init_(self, text_features, visual_features, labels):
        assert len(text_features) == len(visual_features) == len(labels)
        self.text = torch.tensor(text_features, dtype=torch.float32)
        self.visual = torch.tensor(visual_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def _len_(self):
        return len(self.labels)

    def _getitem_(self, idx):
        return {
            "text": self.text[idx],
            "visual": self.visual[idx],
            "label": self.labels[idx],
        }


class TextOnlyDataset(Dataset):
    def _init_(self, text_features, labels):
        self.text = torch.tensor(text_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def _len_(self):
        return len(self.labels)

    def _getitem_(self, idx):
        return {
            "text": self.text[idx],
            "label": self.labels[idx],
        }


class VisualOnlyDataset(Dataset):
    def _init_(self, visual_features, labels):
        self.visual = torch.tensor(visual_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def _len_(self):
        return len(self.labels)

    def _getitem_(self, idx):
        return {
            "visual": self.visual[idx],
            "label": self.labels[idx],
        }


def load_processed_data(dataset_name: str):
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / dataset_name / "processed"
    csv_path = data_dir / f"{dataset_name}_with_ocr.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró {csv_path}. Ejecuta primero el procesamiento."
        )

    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError("El CSV debe contener la columna 'label'")

    labels = df["label"].values

    texts = None
    if "X_multimodal" in df.columns:
        texts = df["X_multimodal"].fillna("").tolist()
    elif "X_baseline" in df.columns:
        texts = df["X_baseline"].fillna("").tolist()

    image_urls = df["image_url"].fillna("").tolist() if "image_url" in df.columns else None

    metadata = {
        "dataset": dataset_name,
        "num_samples": len(df),
        "num_classes": int(len(np.unique(labels))),
        "class_distribution": dict(pd.Series(labels).value_counts()),
        "has_text": texts is not None,
        "has_images": image_urls is not None,
    }

    return texts, image_urls, labels, metadata


def prepare_features(
    texts=None,
    image_urls=None,
    text_encoder=None,
    visual_extractor=None,
    max_samples=None,
):
    if max_samples is not None:
        texts = texts[:max_samples] if texts is not None else None
        image_urls = image_urls[:max_samples] if image_urls is not None else None

    text_features = None
    visual_features = None

    if texts is not None and text_encoder is not None:
        text_features = text_encoder.transform(texts)

    if image_urls is not None and visual_extractor is not None:
        features = []
        for url in image_urls:
            vec = visual_extractor.extract_from_url(url)
            if vec is None:
                vec = np.zeros(4096, dtype=np.float32)
            features.append(vec)
        visual_features = np.vstack(features)

    return text_features, visual_features


def split_data(
    text_features,
    visual_features,
    labels,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    stratify=True,
):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    indices = np.arange(len(labels))
    stratify_labels = labels if stratify else None

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_size + test_size),
        random_state=random_state,
        stratify=stratify_labels,
    )

    val_ratio = val_size / (val_size + test_size)
    val_labels = labels[temp_idx] if stratify else None

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_ratio),
        random_state=random_state,
        stratify=val_labels,
    )

    def pack(idx):
        return {
            "text": text_features[idx] if text_features is not None else None,
            "visual": visual_features[idx] if visual_features is not None else None,
            "labels": labels[idx],
        }

    return {
        "train": pack(train_idx),
        "val": pack(val_idx),
        "test": pack(test_idx),
    }


def create_dataloaders(
    splits,
    dataset_type="multimodal",
    batch_size=32,
    num_workers=4,
):
    loaders = {}

    for split_name, split in splits.items():
        if dataset_type == "multimodal":
            dataset = MultimodalDataset(
                split["text"], split["visual"], split["labels"]
            )
        elif dataset_type == "text_only":
            dataset = TextOnlyDataset(split["text"], split["labels"])
        elif dataset_type == "visual_only":
            dataset = VisualOnlyDataset(split["visual"], split["labels"])
        else:
            raise ValueError(f"Tipo de dataset no soportado: {dataset_type}")

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders


def save_features(output_dir, labels, text_features=None, visual_features=None, prefix=""):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if text_features is not None:
        np.save(output_dir / f"{prefix}text.npy", text_features)

    if visual_features is not None:
        np.save(output_dir / f"{prefix}visual.npy", visual_features)

    np.save(output_dir / f"{prefix}labels.npy", labels)


def load_features(input_dir, prefix=""):
    input_dir = Path(input_dir)

    labels_path = input_dir / f"{prefix}labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(f"No se encontró {labels_path}")

    text_path = input_dir / f"{prefix}text.npy"
    visual_path = input_dir / f"{prefix}visual.npy"

    text_features = np.load(text_path) if text_path.exists() else None
    visual_features = np.load(visual_path) if visual_path.exists() else None
    labels = np.load(labels_path)

    return text_features, visual_features, labels