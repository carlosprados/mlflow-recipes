import os.path

from opengate.recipes.dataset_split import DatasetSplit
from opengate.recipes.utils.metrics import BUILTIN_ANOMALY_RECIPE_METRICS

from pandas import DataFrame
import numpy as np
from sklearn.metrics import roc_curve, auc
from mlflow.sklearn import load_model
from mlflow.models import EvaluationResult
from mlflow import log_metric, log_artifact

import importlib
import matplotlib.pyplot as plt

import logging

_logger = logging.getLogger(__name__)

def evaluate_anomaly_model(
        model_uri: str,
        dataset:DataFrame,
        label_column: str,
        data_prefix: str,
        artifacts_path: str
) -> EvaluationResult:
    # Load Isolation Forest model
    model = load_model(model_uri)
    labels = dataset[label_column]
    data = dataset.drop(columns=[label_column])
    raw_predictions = model.predict(data)
    predictions = np.where(raw_predictions == -1, 1, 0)
    artifacts = {}
    calculated_metrics = {}

    sklearn_metrics = importlib.import_module("sklearn.metrics")
    for metric in BUILTIN_ANOMALY_RECIPE_METRICS:
        try:
            imported_metric = getattr(sklearn_metrics, metric.name)
            score = imported_metric(labels, predictions)
            calculated_metrics[metric.name] = score
            log_metric(data_prefix + metric.name, score)
            if metric.name == "roc_auc_score":
                pred_scores = model.decision_function(data)
                # Normalize probabilities between 0 and 1
                y_pred_prob = (pred_scores - pred_scores.min()) / (pred_scores.max() - pred_scores.min())
                save_roc_curve(labels=labels, prediction_scores=y_pred_prob, roc_path=artifacts_path,
                               data_prefix=data_prefix)

        except AttributeError as ae:
            _logger.error(f"Metric '{metric.name}' not found in sklearn.metrics: {ae}")
        except Exception as e:
            _logger.error(f"Error calculating metric '{metric.name}': {e}")

    return EvaluationResult(artifacts=artifacts, metrics=calculated_metrics)

def find_optimum_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray):
    distances = np.sqrt((fpr - 0) ** 2 + (tpr - 1) ** 2)
    optimal_idx = np.argmin(distances)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, optimal_idx

def save_roc_curve(labels, prediction_scores, roc_path: str, data_prefix: str) -> str:
    fpr, tpr, thresholds = roc_curve(labels, prediction_scores)
    optimal_threshold, optimal_idx = find_optimum_threshold(fpr=fpr, tpr=tpr, thresholds=thresholds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(14, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')

    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100,
                label=f'Optimal Threshold = {optimal_threshold:.2f}')
    plt.annotate(f'{optimal_threshold:.2f}', (fpr[optimal_idx], tpr[optimal_idx]),
                 textcoords="offset points", xytext=(10, -10), ha='center', color='red')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid()
    roc_full_path = os.path.join(roc_path, data_prefix + "roc.png")
    plt.savefig(roc_full_path)
    log_artifact(roc_full_path)
    return roc_path

# TODO: I haven't finished this yet. Fix it
def save_cf(cm: np.ndarray, cf_path: str = "dist/confusion_matrix.png") -> str:
    plt.figure(figsize=(8, 6))
    classes = ["Benign", "Malign"]
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")

    plt.savefig(cf_path)
    return cf_path