from pathlib import Path
from typing import List
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import VGG16_Weights
from PIL import Image
import numpy as np
import requests
from io import BytesIO

class VGG16FeatureExtractor: # Extractor de características visuales basado en VGG16

    def __init__(
        self,
        freeze_backbone: bool = True,
        device: str | None = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Cargar modelo con pesos oficiales
        weights = VGG16_Weights.IMAGENET1K_V1
        vgg16 = models.vgg16(weights=weights)

        self.features = vgg16.features
        self.avgpool = vgg16.avgpool

        # FC hasta fc7
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # Copiar pesos preentrenados
        self.fc[0].load_state_dict(vgg16.classifier[0].state_dict())
        self.fc[3].load_state_dict(vgg16.classifier[3].state_dict())

        if freeze_backbone:
            for module in [self.features, self.fc]:
                for p in module.parameters():
                    p.requires_grad = False

        self.features.to(self.device).eval()
        self.avgpool.to(self.device).eval()
        self.fc.to(self.device).eval()

        self.transform = weights.transforms()

    @torch.no_grad()
    def extract_tensor(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def extract_pil(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        features = self.extract_tensor(tensor)
        return features.cpu().numpy().squeeze()

    def extract_url(self, url: str, timeout: int = 10) -> np.ndarray:
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return self.extract_pil(img)
        except Exception:
            return np.zeros(4096, dtype=np.float32)

    def extract_batch_urls(self, urls: List[str]) -> np.ndarray:
        return np.vstack([self.extract_url(url) for url in urls])

    def __repr__(self):
        return "VGG16FeatureExtractor(output_dim=4096)"

# Rama visual para modelo multimodal
class VisualBranchMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dims: List[int] = [512, 256],
        output_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ]
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# Clasificador profundo usando features visuales (VGG16)
class VisualOnlyClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ]
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# VGG16 
class VGG16EndToEnd(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        freeze_until_layer: int = 15
    ):
        super().__init__()

        weights = VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights)

        self.features = vgg.features
        self.avgpool = vgg.avgpool

        for i, layer in enumerate(self.features):
            if i < freeze_until_layer:
                for p in layer.parameters():
                    p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# Factory de modelos visuales
def create_visual_model(model_type: str = "extractor", **kwargs):
    if model_type == "extractor":
        return VGG16FeatureExtractor(**kwargs)

    if model_type == "branch":
        return VisualBranchMLP(**kwargs)

    if model_type == "mlp":
        return VisualOnlyClassifier(**kwargs)

    if model_type == "end2end":
        return VGG16EndToEnd(**kwargs)

    raise ValueError(f"Modelo visual no soportado: {model_type}")
