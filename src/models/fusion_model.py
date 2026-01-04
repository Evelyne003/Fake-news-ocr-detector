import torch
import torch.nn as nn
from .text_branch import TextBranchMLP
from .visual_branch import VisualBranchMLP

class MultimodalFusionModel(nn.Module):
    SUPPORTED_FUSIONS = {"concatenation", "sum", "multiply", "average", "attention"}

    def __init__(
        self,
        text_input_dim=5000,
        visual_input_dim=4096,
        text_hidden_dims=[512, 256],
        visual_hidden_dims=[512, 256],
        fusion_embedding_dim=128,
        fusion_hidden_dims=[128, 64],
        num_classes=2,
        dropout=0.3,
        fusion_method="concatenation",
    ):
        super().__init__()

        if fusion_method not in self.SUPPORTED_FUSIONS:
            raise ValueError(
                f"Método de fusión '{fusion_method}' no soportado. "
                f"Opciones válidas: {self.SUPPORTED_FUSIONS}"
            )

        self.fusion_method = fusion_method
        self.num_classes = num_classes

        # Rama Textual 
        self.text_branch = TextBranchMLP(
            input_dim=text_input_dim,
            hidden_dims=text_hidden_dims,
            output_dim=fusion_embedding_dim,
            dropout=dropout,
        )

        # Rama Visual 
        self.visual_branch = VisualBranchMLP(
            input_dim=visual_input_dim,
            hidden_dims=visual_hidden_dims,
            output_dim=fusion_embedding_dim,
            dropout=dropout,
        )

        # Normalización previa a fusión 
        self.text_norm = nn.LayerNorm(fusion_embedding_dim)
        self.visual_norm = nn.LayerNorm(fusion_embedding_dim)

        # Atención tardía 
        if fusion_method == "attention":
            self.attention = nn.Sequential(
                nn.Linear(fusion_embedding_dim * 2, fusion_embedding_dim),
                nn.Tanh(),
                nn.Linear(fusion_embedding_dim, 2),
                nn.Softmax(dim=1),
            )
            fusion_input_dim = fusion_embedding_dim
        elif fusion_method == "concatenation":
            fusion_input_dim = fusion_embedding_dim * 2
        else:
            fusion_input_dim = fusion_embedding_dim

        # Clasificador 
        layers = []
        prev_dim = fusion_input_dim

        for hidden_dim in fusion_hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(hidden_dim),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def _fuse_embeddings(self, text_emb, visual_emb): # Aplica la estrategia de fusión seleccionada
        if self.fusion_method == "concatenation":
            return torch.cat([text_emb, visual_emb], dim=1)

        if self.fusion_method == "sum":
            return text_emb + visual_emb

        if self.fusion_method == "multiply":
            return text_emb * visual_emb

        if self.fusion_method == "average":
            return (text_emb + visual_emb) / 2

        if self.fusion_method == "attention":
            combined = torch.cat([text_emb, visual_emb], dim=1)
            weights = self.attention(combined)
            return (
                text_emb * weights[:, 0:1] + visual_emb * weights[:, 1:2]
            )

        raise RuntimeError("Método de fusión inválido")

    def forward(self, text_features, visual_features):
        text_emb = self.text_norm(self.text_branch(text_features))
        visual_emb = self.visual_norm(self.visual_branch(visual_features))

        fused = self._fuse_embeddings(text_emb, visual_emb)
        return self.classifier(fused)

    def get_embeddings(self, text_features, visual_features): # Devuelve embeddings individuales y fusionados (sin gradientes)
        with torch.no_grad():
            text_emb = self.text_norm(self.text_branch(text_features))
            visual_emb = self.visual_norm(self.visual_branch(visual_features))
            fused = self._fuse_embeddings(text_emb, visual_emb)

        return text_emb, visual_emb, fused