import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Script de entrenamiento para modelos de detección de noticias falsas

from models import (
    TFIDFTextEncoder,
    create_text_model,
    create_visual_model,
    create_multimodal_model,
    VGG16FeatureExtractor,
)
from training.dataset import (
    load_processed_data,
    prepare_features,
    split_data,
    create_dataloaders,
)
from utils import Config, compute_metrics, print_metrics


class Trainer:
    def _init_(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        lr = config.get("training.learning_rate", 1e-3)
        optimizer_name = config.get("training.optimizer", "adam").lower()

        if optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=config.get("training.momentum", 0.9),
            )
        else:
            raise ValueError(f"Optimizador no soportado: {optimizer_name}")

        self.criterion = nn.CrossEntropyLoss()

        self.patience = config.get("training.early_stopping.patience", 10)
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def _forward(self, batch, model_type):
        labels = batch["label"].to(self.device)

        if model_type == "multimodal":
            outputs = self.model(
                batch["text"].to(self.device),
                batch["visual"].to(self.device),
            )
        elif model_type == "text_only":
            outputs = self.model(batch["text"].to(self.device))
        elif model_type == "visual_only":
            outputs = self.model(batch["visual"].to(self.device))
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")

        return outputs, labels

    def train_epoch(self, dataloader, model_type):
        self.model.train()

        total_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(dataloader, desc="Train", leave=False):
            outputs, labels = self._forward(batch, model_type)

            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(dataloader), correct / total

    def evaluate(self, dataloader, model_type):
        self.model.eval()

        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                outputs, labels = self._forward(batch, model_type)

                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return (
            total_loss / len(dataloader),
            correct / total,
            np.array(all_preds),
            np.array(all_labels),
        )

    def fit(self, train_loader, val_loader, epochs, model_type):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, model_type)
            val_loss, val_acc, _, _ = self.evaluate(val_loader, model_type)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    break

        return self.history


def prepare_experiment_data(config):
    dataset_name = config.get("experiment.dataset", "FakeNewsNet")
    model_type = config.get("model.type", "multimodal")

    texts, image_urls, labels, metadata = load_processed_data(dataset_name)

    text_encoder = None
    visual_extractor = None

    if model_type in {"multimodal", "text_only"}:
        text_encoder = TFIDFTextEncoder(
            max_features=config.get("model.text_features", 5000)
        )
        text_encoder.fit(texts)

    if model_type in {"multimodal", "visual_only"}:
        visual_extractor = VGG16FeatureExtractor(
            freeze_backbone=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    text_features, visual_features = prepare_features(
        texts=texts,
        image_urls=image_urls,
        text_encoder=text_encoder,
        visual_extractor=visual_extractor,
    )

    splits = split_data(
        text_features,
        visual_features,
        labels,
        train_size=config.get("data.train_split", 0.7),
        val_size=config.get("data.val_split", 0.15),
        test_size=config.get("data.test_split", 0.15),
        random_state=config.get("experiment.seed", 42),
    )

    dataloaders = create_dataloaders(
        splits,
        dataset_type=model_type,
        batch_size=config.get("data.batch_size", 32),
        num_workers=config.get("data.num_workers", 4),
    )

    return dataloaders, metadata


def create_model_from_config(config):
    model_type = config.get("model.type", "multimodal")

    if model_type == "text_only":
        return create_text_model(
            max_features=config.get("model.text_features", 5000),
            hidden_dims=config.get("model.hidden_layers", [512, 256]),
            dropout=config.get("model.dropout", 0.3),
        )

    if model_type == "visual_only":
        return create_visual_model(
            input_dim=config.get("model.visual_features", 4096),
            hidden_dims=config.get("model.hidden_layers", [512, 256]),
            dropout=config.get("model.dropout", 0.3),
        )

    if model_type == "multimodal":
        return create_multimodal_model(
            fusion_type=config.get("model.fusion_type", "late"),
            fusion_method=config.get("model.fusion_method", "concatenation"),
            dropout=config.get("model.dropout", 0.3),
        )

    raise ValueError(f"Modelo no reconocido: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = Config(args.config)

    exp_name = config.get("experiment.name", "experiment")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"experiments/results/{exp_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloaders, _ = prepare_experiment_data(config)
    model = create_model_from_config(config)

    trainer = Trainer(model, config, args.device)

    history = trainer.fit(
        dataloaders["train"],
        dataloaders["val"],
        epochs=config.get("training.epochs", 50),
        model_type=config.get("model.type"),
    )

    test_loss, test_acc, preds, labels = trainer.evaluate(
        dataloaders["test"],
        config.get("model.type"),
    )

    metrics = compute_metrics(labels, preds)
    print_metrics(metrics)

    torch.save(model.state_dict(), output_dir / "model.pth")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "_main_":
    main()