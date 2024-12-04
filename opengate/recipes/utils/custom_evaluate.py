import os.path
import sys

from opengate.recipes.utils.metrics import BUILTIN_ANOMALY_RECIPE_METRICS

from pandas import DataFrame
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from mlflow.sklearn import load_model
from mlflow.models import EvaluationResult, EvaluationMetric
from mlflow import log_metric, log_artifact

from importlib import import_module
import importlib.util as importlib_utils
from typing import List
import matplotlib.pyplot as plt

import logging

_logger = logging.getLogger(__name__)

def evaluate_anomaly_model(
        model_uri: str,
        dataset:DataFrame,
        label_column: str,
        dataset_name: str,
        artifacts_path: str,
        root_recipe: str,
        extra_metrics: List[EvaluationMetric],
        threshold: float = None
) -> EvaluationResult:
    # Load Isolation Forest model
    model = load_model(model_uri)
    labels = dataset[label_column]
    data = dataset.drop(columns=[label_column])
    raw_predictions = model.predict(data)
    predictions = np.where(raw_predictions == -1, 1, 0)
    artifacts = {}
    calculated_metrics = {}

    sklearn_metrics = import_module("sklearn.metrics")
    for metric in BUILTIN_ANOMALY_RECIPE_METRICS:
        try:
            imported_metric = getattr(sklearn_metrics, metric.name)
            score = imported_metric(labels, predictions)
            calculated_metrics[metric.name] = score
            log_metric(f"{dataset_name}_{metric.name}", score)
            if metric.name == "roc_auc_score":
                # Negating as higher scores indicate normal in IF
                scores = -model.decision_function(data)
                normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
                optimal_threshold = save_roc_curve(labels=labels, prediction_scores=normalized_scores,
                                                   roc_path=artifacts_path, dataset_name=dataset_name)
                optimal_threshold = threshold if threshold is not None else optimal_threshold
                save_cm(labels=labels, scores=normalized_scores, threshold=optimal_threshold,
                        artifact_path=artifacts_path, dataset_name=dataset_name)
        except AttributeError as ae:
            _logger.error(f"Metric '{metric.name}' not found in sklearn.metrics: {ae}")
        except Exception as e:
            _logger.error(f"Error calculating metric '{metric.name}': {e}")

    calculated_extra_metrics = evaluate_custom_metrics(root_recipe=root_recipe, metrics=extra_metrics, labels=labels,
                                                       predictions=predictions, dataset_name=dataset_name)
    calculated_metrics.update(calculated_extra_metrics)
    return EvaluationResult(artifacts=artifacts, metrics=calculated_metrics)

# TODO: ROC curve is not looking so well. optimal threshold is not marked on the right location
def save_roc_curve(labels, prediction_scores, roc_path: str, dataset_name: str):
    os.makedirs(roc_path, exist_ok=True)
    try:
        fpr, tpr, thresholds = roc_curve(labels, prediction_scores)
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))
        optimal_threshold = thresholds[optimal_idx]
        log_metric(f"Optimal_threshold_for_{dataset_name}", optimal_threshold)

        # Plot ROC curve
        plt.figure(figsize=(14, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid()
        roc_full_path = os.path.join(roc_path, dataset_name + "_roc.png")
        plt.savefig(roc_full_path)
        log_artifact(roc_full_path)
        return optimal_threshold
    except Exception as e:
        _logger.error(f"Error occurred while generating ROC Curve: {e}")

def save_cm(labels, scores, threshold, artifact_path: str, dataset_name: str):
    try:
        # Predict using the optimal threshold
        predicted_labels = (scores >= threshold).astype(int)
        cm = confusion_matrix(labels, predicted_labels, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malign"])
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f"{dataset_name} Confusion Matrix")
        cf_path = os.path.join(artifact_path, dataset_name + "_cf.png")
        is_sub_train_folder = True if artifact_path.split("/")[-3] == "train" else False # for identify training step
        if dataset_name == "training" or is_sub_train_folder:
            filename = artifact_path.split("/")[-1].split("_")[-1]
            cf_path = os.path.join(artifact_path, "train_" + filename + "_cf.png")
        plt.savefig(cf_path)
        log_artifact(cf_path)
    except Exception as e:
        _logger.error(f"Error occurred while generating Confusion Matrix: {e}")

def evaluate_custom_metrics(root_recipe: str, metrics: List[EvaluationMetric], labels, predictions, dataset_name: str):
    module_path = os.path.join(root_recipe, "steps", "custom_metrics.py")
    sys.path.append(module_path)
    extra_calculated_metrics = {}
    for metric in metrics:
        spec = importlib_utils.spec_from_file_location(metric.name, module_path)
        module = importlib_utils.module_from_spec(spec)
        spec.loader.exec_module(module)
        metric_function = getattr(module, metric.name)
        extra_metric_score = metric_function(labels, predictions)
        log_metric(f"{dataset_name}_{metric.name}", extra_metric_score)
        extra_calculated_metrics[metric.name] = extra_metric_score

    return extra_calculated_metrics