import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from pathlib import Path

# MÉTRICAS BÁSICAS
def compute_metrics(y_true, y_pred, y_proba=None, average='binary'): # Calcula métricas estándar de evaluación
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }

    # AUC-ROC 
    if y_proba is not None:
        try:
            y_proba = np.asarray(y_proba)
            if y_proba.ndim > 1:
                y_proba = y_proba[:, 1]  # clase positiva
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics['auc_roc'] = None

    return metrics


def print_metrics(metrics, title="Métricas de Evaluación"): # Imprime métricas de forma legible
    print(f"{title}")

    for name, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, float):
            print(f"  {name.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"  {name.replace('_', ' ').title()}: {value}")

    print(f"{'='*60}\n")

# MATRIZ DE CONFUSIÓN
def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    save_path=None,
    title='Matriz de Confusión',
    normalize=False
):

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=labels or ['Real', 'Falso'],
        yticklabels=labels or ['Real', 'Falso'],
        ax=ax,
        cbar_kws={'label': 'Proporción' if normalize else 'Cantidad'}
    )

    ax.set_xlabel('Predicción')
    ax.set_ylabel('Etiqueta Real')
    ax.set_title(title, fontweight='bold')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Matriz de confusión guardada en: {save_path}")

    return fig

# CURVA ROC
def plot_roc_curve(y_true, y_proba, save_path=None, title='Curva ROC'): # Genera curva ROC
    
    y_proba = np.asarray(y_proba)
    if y_proba.ndim > 1:
        y_proba = y_proba[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Aleatorio')

    ax.set_xlabel('Tasa de Falsos Positivos')
    ax.set_ylabel('Tasa de Verdaderos Positivos')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Curva ROC guardada en: {save_path}")

    return fig

# HISTORIA DE ENTRENAMIENTO
def plot_training_history(history, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    for key, values in history.items():
        ax.plot(values, label=key)

    ax.set_xlabel('Época')
    ax.set_ylabel('Valor')
    ax.set_title('Historia de Entrenamiento')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Historia de entrenamiento guardada en: {save_path}")

    return fig

# REPORTE DE CLASIFICACIÓN
def generate_classification_report(y_true, y_pred, labels=None, save_path=None): # Genera reporte de clasificación detallado
    
    target_names = labels or ['Real', 'Falso']
    report = classification_report(y_true, y_pred, target_names=target_names)

    print(report)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Reporte guardado en: {save_path}")

    return report

# COMPARACIÓN DE MODELOS
def compare_models(results_dict, metrics=('accuracy', 'f1_score'), save_path=None):
    models = list(results_dict.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        values = [results_dict[m].get(metric, 0) for m in models]

        axes[idx].bar(models, values, alpha=0.7)
        axes[idx].set_title(metric.replace('_', ' ').title())
        axes[idx].set_ylim(0, 1)
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparación guardada en: {save_path}")

    return fig