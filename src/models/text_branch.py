import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from pathlib import Path
from typing import List, Tuple

class TFIDFTextEncoder: # Codificador de texto basado en TF-IDF (scikit-learn)
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            strip_accents="unicode",
            lowercase=True,
            token_pattern=r"\b\w+\b"
        )
        self.is_fitted = False

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]):
        if not self.is_fitted:
            raise RuntimeError("TF-IDF encoder no ha sido entrenado")
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts: List[str]):
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: str | Path):
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True

    def __repr__(self):
        return f"TFIDFTextEncoder(max_features={self.vectorizer.max_features})"

class TextBranchMLP(nn.Module): # Rama textual para modelo multimodal
    def __init__(
        self,
        input_dim: int = 5000,
        hidden_dims: List[int] = [512, 256],
        output_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ]
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def __repr__(self):
        return f"TextBranchMLP(output_dim={self.network[-1].out_features})"

class TextOnlyClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 5000,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ]
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# BASELINE: TF-IDF + LOGISTIC REGRESSION
class BaselineTextClassifier:
    def __init__(self, max_features: int = 5000, ngram_range=(1, 2)):
        self.encoder = TFIDFTextEncoder(
            max_features=max_features,
            ngram_range=ngram_range
        )

        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
        self.is_fitted = False

    def fit(self, texts: List[str], labels):
        X = self.encoder.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]):
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado")
        X = self.encoder.transform(texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts: List[str]):
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado")
        X = self.encoder.transform(texts)
        return self.classifier.predict_proba(X)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.encoder.vectorizer,
                    "classifier": self.classifier
                },
                f
            )

    def load(self, path: str | Path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.encoder.vectorizer = data["vectorizer"]
        self.encoder.is_fitted = True
        self.classifier = data["classifier"]
        self.is_fitted = True

def create_text_model(model_type: str = "baseline", **kwargs): # Factory de modelos textuales
    
    if model_type == "baseline":
        return BaselineTextClassifier(**kwargs)

    if model_type == "mlp":
        return TextOnlyClassifier(**kwargs)

    if model_type == "branch":
        return TextBranchMLP(**kwargs)

    raise ValueError(f"Modelo textual no soportado: {model_type}")