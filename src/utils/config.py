import yaml
import json
import hashlib
from pathlib import Path
from datetime import datetime
from copy import deepcopy


class Config: # Clase para manejar configuraciones de experimentos
    def __init__(self, config_path=None, config_dict=None, frozen=False):
        if config_path:
            self.config = self._load_from_file(config_path)
            self.config_path = Path(config_path)
        elif config_dict:
            self.config = deepcopy(config_dict)
            self.config_path = None
        else:
            self.config = self._default_config()
            self.config_path = None

        self.config["meta"] = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "version": "1.0"
        }

        self._frozen = frozen
        self._validate()

    # CARGA Y DEFAULTS
    def _load_from_file(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _default_config(self):
        return {
            "experiment": {
                "name": "default_experiment",
                "description": "",
                "dataset": "FakeNewsNet",
                "seed": 42
            },
            "paths": {
                "data_dir": "data/",
                "artifacts_dir": "artifacts/",
                "logs_dir": "logs/",
                "results_dir": "results/"
            },
            "data": {
                "train_split": 0.7,
                "val_split": 0.15,
                "test_split": 0.15,
                "batch_size": 32,
                "shuffle": True,
                "sample_size": None,  
                "random_seed": 42
            },
            "model": {
                "type": "multimodal",
                "text_features": 5000,
                "visual_features": 4096,
                "dropout": 0.3
            },
            "training": {
                "epochs": 50,
                "learning_rate": 1e-3,
                "optimizer": "adam",
                "loss": "binary_crossentropy",
                "early_stopping": {
                    "patience": 10,
                    "monitor": "val_loss"
                }
            },
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1_score"]
            }
        }

    # VALIDACIÓN
    def _validate(self):
        splits = self.get("data.train_split") + \
                 self.get("data.val_split") + \
                 self.get("data.test_split")

        if not abs(splits - 1.0) < 1e-6:
            raise ValueError("train/val/test splits deben sumar 1.0")

        if self.get("training.epochs") <= 0:
            raise ValueError("epochs debe ser > 0")

        if self.get("training.learning_rate") <= 0:
            raise ValueError("learning_rate debe ser > 0")

    # GET / SET
    def get(self, key, default=None):
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key, value):
        if self._frozen:
            raise RuntimeError("La configuración está congelada")

        keys = key.split(".")
        cfg = self.config
        for k in keys[:-1]:
            cfg.setdefault(k, {})
            cfg = cfg[k]
        cfg[keys[-1]] = value

    def freeze(self):
        self._frozen = True

    # EXPORTACIÓN
    def save_yaml(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)
        print(f"✓ Configuración YAML guardada en {path}")

    def save_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"Configuración JSON guardada en {path}")

    def to_dict(self):
        return deepcopy(self.config)

    def experiment_hash(self):
        cfg_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(cfg_str.encode()).hexdigest()[:10]

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)

    def __repr__(self):
        return yaml.dump(self.config, allow_unicode=True, sort_keys=False)

# TEMPLATES
EXPERIMENT_TEMPLATES = {
    "baseline_text": {
        "model": {
            "type": "text_only",
            "text_features": 5000
        }
    },
    "baseline_visual": {
        "model": {
            "type": "visual_only",
            "visual_features": 4096
        }
    },
    "multimodal": {
        "model": {
            "type": "multimodal",
            "fusion_method": "concatenation"
        }
    },
    "fakeddit_sampled": {
        "experiment": {
            "dataset": "Fakeddit"
        },
        "data": {
            "sample_size": 8000
        }
    }
}


def load_experiment_template(template_name):
    if template_name not in EXPERIMENT_TEMPLATES:
        raise ValueError(
            f"Template '{template_name}' no existe. "
            f"Opciones: {list(EXPERIMENT_TEMPLATES.keys())}"
        )

    base = Config()
    template = EXPERIMENT_TEMPLATES[template_name]

    for key, value in template.items():
        if isinstance(value, dict):
            for subk, subv in value.items():
                base.set(f"{key}.{subk}", subv)
        else:
            base.set(key, value)

    return base