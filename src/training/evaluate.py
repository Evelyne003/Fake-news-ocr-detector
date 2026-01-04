import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    Config,
    compute_metrics,
    print_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    generate_classification_report
)
from training.train import create_model_from_config, prepare_experiment_data


class ModelEvaluator: # Clase para evaluar modelos entrenados
    
    def _init_(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(self, dataloader, model_type='multimodal', return_proba=True):
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluando"):
                labels = batch['label'].to(self.device)
                
                # Forward
                if model_type == 'multimodal':
                    text = batch['text'].to(self.device)
                    visual = batch['visual'].to(self.device)
                    outputs = self.model(text, visual)
                elif model_type == 'text_only':
                    text = batch['text'].to(self.device)
                    outputs = self.model(text)
                elif model_type == 'visual_only':
                    visual = batch['visual'].to(self.device)
                    outputs = self.model(visual)
                
                # Softmax para probabilidades
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if return_proba:
                    all_probs.extend(probs.cpu().numpy())
        
        predictions = np.array(all_preds)
        labels = np.array(all_labels)
        probabilities = np.array(all_probs) if return_proba else None
        
        return predictions, labels, probabilities
    
    def evaluate(self, dataloader, model_type='multimodal', output_dir=None):
        # Predicciones
        predictions, labels, probabilities = self.predict(
            dataloader, 
            model_type=model_type,
            return_proba=True
        )
        
        # Calcular métricas
        proba_positive = probabilities[:, 1] if probabilities is not None else None
        metrics = compute_metrics(
            y_true=labels,
            y_pred=predictions,
            y_proba=proba_positive
        )
        
        # Imprimir métricas
        print_metrics(metrics, title="Métricas de Evaluación")
        
        # Generar reportes si hay directorio de salida
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Matriz de confusión
            plot_confusion_matrix(
                y_true=labels,
                y_pred=predictions,
                save_path=output_dir / 'confusion_matrix.png',
                title='Matriz de Confusión'
            )
            
            # Curva ROC
            if proba_positive is not None:
                plot_roc_curve(
                    y_true=labels,
                    y_proba=proba_positive,
                    save_path=output_dir / 'roc_curve.png'
                )
            
            # Reporte de clasificación
            generate_classification_report(
                y_true=labels,
                y_pred=predictions,
                save_path=output_dir / 'classification_report.txt'
            )
            
            # Guardar métricas en JSON
            with open(output_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"\nReportes guardados en: {output_dir}")
        
        # Análisis de errores
        error_analysis = self.analyze_errors(labels, predictions, probabilities)
        
        if output_dir:
            with open(output_dir / 'error_analysis.json', 'w') as f:
                json.dump(error_analysis, f, indent=2)
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities,
            'error_analysis': error_analysis
        }
    
    def analyze_errors(self, labels, predictions, probabilities):
        # Identificar errores
        errors = predictions != labels
        n_errors = errors.sum()
        
        # Falsos positivos y falsos negativos
        false_positives = ((predictions == 1) & (labels == 0)).sum()
        false_negatives = ((predictions == 0) & (labels == 1)).sum()
        
        # Confianza en errores
        if probabilities is not None:
            error_confidences = probabilities[errors].max(axis=1)
            avg_error_confidence = error_confidences.mean()
        else:
            avg_error_confidence = None
        
        analysis = {
            'total_errors': int(n_errors),
            'error_rate': float(n_errors / len(labels)),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'avg_error_confidence': float(avg_error_confidence) if avg_error_confidence else None
        }
        
        print("ANÁLISIS DE ERRORES")
        print(f"Total de errores: {n_errors} ({n_errors/len(labels)*100:.2f}%)")
        print(f"Falsos positivos: {false_positives}")
        print(f"Falsos negativos: {false_negatives}")
        if avg_error_confidence:
            print(f"Confianza promedio en errores: {avg_error_confidence:.4f}")
        
        return analysis


def load_trained_model(model_path, config_path, device='cuda'):
    # Cargar configuración
    config = Config(config_path=config_path)
    
    # Crear arquitectura
    model = create_model_from_config(config)
    
    # Cargar pesos
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    print(f"Modelo cargado desde: {model_path}")
    
    return model, config


def main():
    parser = argparse.ArgumentParser(
        description='Evaluación de modelos entrenados'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Ruta al modelo entrenado (.pth)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Ruta a la configuración del experimento'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directorio para guardar resultados'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset a evaluar (sobrescribe config)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device para evaluación'
    )
    
    args = parser.parse_args()
    
    # Cargar modelo
    model, config = load_trained_model(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )
    
    # Sobrescribir dataset si se especifica
    if args.dataset:
        config.set('experiment.dataset', args.dataset)
    
    # Preparar datos
    print("\nPreparando datos de evaluación...")
    data = prepare_experiment_data(config)
    
    # Determinar directorio de salida
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_dir = Path(args.model).parent
        output_dir = model_dir / 'evaluation'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear evaluador
    evaluator = ModelEvaluator(model, device=args.device)
    
    # Evaluar
    results = evaluator.evaluate(
        dataloader=data['dataloaders']['test'],
        model_type=config.get('model.type', 'multimodal'),
        output_dir=output_dir
    )
    
    print("EVALUACIÓN COMPLETADA")
    print(f"Resultados en: {output_dir}")


if __name__ == '_main_':
    main()