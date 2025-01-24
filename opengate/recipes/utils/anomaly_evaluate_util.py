import os.path
import sys

import pandas as pd
from mlflow.exceptions import BAD_REQUEST, MlflowException

from opengate.recipes.custom_models_enum import CustomModels
from opengate.recipes.utils.metrics import BUILTIN_ANOMALY_RECIPE_METRICS

from pandas import DataFrame
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from mlflow.sklearn import load_model
from mlflow.models import EvaluationResult, EvaluationMetric
from mlflow import log_metric, log_artifact

from importlib import import_module
import importlib.util as importlib_utils
from typing import List, Union
import matplotlib.pyplot as plt

import logging

_logger = logging.getLogger(__name__)

class EvaluateAnomalyModel:
    def __init__(self, model_uri: str, dataset: DataFrame, label_column:str, dataset_name: str, artifacts_path:str,
                 root_recipe: str, extra_metrics: List[EvaluationMetric], model_type: str, threshold: float = None):
        self.model_uri = model_uri
        self.dataset = dataset
        self.label_column = label_column
        self.dataset_name = dataset_name
        self.artifacts_path = artifacts_path
        self.root_recipe = root_recipe
        self.extra_metrics = extra_metrics
        self.threshold: float = threshold
        self.model_type = model_type
        self.labels = dataset[label_column] if label_column is not None else []

    def evaluate_anomaly_model(self):
        # Load model
        model = load_model(self.model_uri)
        column_to_drop = [self.label_column] if self.label_column is not None else []
        raw_data = self.dataset.drop(columns=column_to_drop)
        raw_predictions = model.predict(raw_data)
        predictions = process_predictions(threshold=self.threshold, model_input=raw_data, raw_predictions=raw_predictions,
                                          model_type=self.model_type)
        artifacts = {}
        calculated_metrics = {}

        sklearn_metrics = import_module("sklearn.metrics")
        for metric in BUILTIN_ANOMALY_RECIPE_METRICS:
            try:
                imported_metric = getattr(sklearn_metrics, metric.name)
                score = imported_metric(self.labels, predictions)
                calculated_metrics[metric.name] = score
                log_metric(f"{self.dataset_name}_{metric.name}", score)
                if metric.name == "roc_auc_score":
                    match self.model_type:
                        case CustomModels.AUTOENCODER.model_name:
                            normalized_scores = np.mean(np.power(raw_data - raw_predictions, 2), axis=1)
                        case CustomModels.ISOLATION_FOREST.model_name:
                            # Negating as higher scores indicate normal in IF
                            scores = -model.decision_function(raw_data)
                            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
                        case _:
                            raise MlflowException(
                                f"Unsupported model selected: {self.model_type}",
                                error_code=BAD_REQUEST,
                            )
                    optimal_threshold = self.save_roc_curve(prediction_scores=normalized_scores)
                    self.threshold = self.threshold if self.threshold is not None else optimal_threshold
                    self.save_cm(scores=normalized_scores)
            except AttributeError as ae:
                _logger.error(f"Metric '{metric.name}' not found in sklearn.metrics: {ae}")
            except Exception as e:
                _logger.error(f"Error calculating metric '{metric.name}': {e}")

        if self.extra_metrics is not None:
            calculated_extra_metrics = self.evaluate_custom_metrics(metrics=self.extra_metrics, predictions=predictions)
            calculated_metrics.update(calculated_extra_metrics)
        return EvaluationResult(artifacts=artifacts, metrics=calculated_metrics)

    def save_roc_curve(self, prediction_scores):
        os.makedirs(self.artifacts_path, exist_ok=True)
        try:
            fpr, tpr, thresholds = roc_curve(self.labels, prediction_scores)
            roc_auc = auc(fpr, tpr)
            optimal_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))
            optimal_threshold = thresholds[optimal_idx]
            log_metric(f"Optimal_threshold_for_{self.dataset_name}", optimal_threshold)

            # Plot ROC curve
            plt.figure(figsize=(14, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random chance')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid()
            roc_full_path = os.path.join(self.artifacts_path, self.dataset_name + "_roc.png")
            plt.savefig(roc_full_path)
            log_artifact(roc_full_path)
            return optimal_threshold
        except Exception as e:
            _logger.error(f"Error occurred while generating ROC Curve: {e}")

    def save_cm(self, scores):
        try:
            # Autoencoder threshold calculation different. Therefore, added below if statement. Info below link
            # https://github.com/Gradiant/SYP-SecBluRed-PR-1465-UEBA/blob/04c36060c944cd1ce75b06cf5552be002ec65886/src/secblured/models/autoencoder.py#L163
            if CustomModels.AUTOENCODER.model_name == self.model_type:
                self.threshold = np.quantile(scores, self.threshold)
            predicted_labels = (scores > self.threshold).astype(int).to_numpy()
            cm = confusion_matrix(self.labels, predicted_labels, labels=[0, 1])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malign"])
            disp.plot(cmap=plt.cm.Blues, values_format='d')
            plt.title(f"{self.dataset_name} Confusion Matrix")
            cf_path = os.path.join(self.artifacts_path, self.dataset_name + "_cf.png")
            is_sub_train_folder = True if self.artifacts_path.split("/")[-3] == "train" else False  # for identify training step
            if self.dataset_name == "training" or is_sub_train_folder:
                filename = self.artifacts_path.split("/")[-1].split("_")[-1]
                cf_path = os.path.join(self.artifacts_path, "train_" + filename + "_cf.png")
            plt.savefig(cf_path)
            log_artifact(cf_path)
        except Exception as e:
            _logger.error(f"Error occurred while generating Confusion Matrix: {e}")

    def evaluate_custom_metrics(self, metrics: List[EvaluationMetric], predictions):
        module_path = os.path.join(self.root_recipe, "steps", "custom_metrics.py")
        sys.path.append(module_path)
        extra_calculated_metrics = {}
        for metric in metrics:
            spec = importlib_utils.spec_from_file_location(metric.name, module_path)
            module = importlib_utils.module_from_spec(spec)
            spec.loader.exec_module(module)
            metric_function = getattr(module, metric.name)
            extra_metric_score = metric_function(self.labels, predictions)
            log_metric(f"{self.dataset_name}_{metric.name}", extra_metric_score)
            extra_calculated_metrics[metric.name] = extra_metric_score

        return extra_calculated_metrics


def preprocess_anomaly_data(dataset: pd.DataFrame, recipe_root: str, target_col: str, is_train_step=False) -> pd.DataFrame:
    from opengate.recipes.utils.execution import get_step_output_path
    import joblib
    transformer_path = get_step_output_path(
        recipe_root_path=recipe_root,
        step_name="transform",
        relative_path="transformer.pkl",
    )
    transformers = joblib.load(transformer_path)
    transformed_array = transformers.transform(dataset)
    transformed_data = pd.DataFrame(transformed_array, columns=dataset.columns, index=dataset.index)

    return transformed_data

def process_predictions(
    threshold: Union[float, np.floating],
    model_input: pd.DataFrame,
    raw_predictions: pd.DataFrame,
    model_type: str
) -> np.ndarray:
    """
    Process raw predictions based on model type and return final labels.

    Args:
        threshold (float): Quantile threshold for anomaly detection.
        model_input (pd.DataFrame): Input data used for the model.
        raw_predictions (pd.DataFrame): Raw predictions from the model.
        model_type (str): Type of model (e.g., AUTOENCODER).

    Returns:
        np.ndarray: Binary labels indicating anomalies (1) or normal (0).
    """
    if model_type == CustomModels.AUTOENCODER.model_name:
        mse = compute_mean_squared_error(model_input, raw_predictions)
        threshold_value = np.quantile(mse, threshold)
        return (mse > threshold_value).astype(int)

    # For isolation forest models, map raw_predictions to binary labels directly.
    return np.where(raw_predictions == -1, 1, 0)

def calculate_threshold_quantile(
    x_train: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold: float
) -> np.float64:
    """
    Calculate the quantile threshold based on Mean Squared Error (MSE).

    Args:
        x_train (pd.DataFrame): Training data used to generate predictions.
        predictions (pd.DataFrame): Predicted values corresponding to the training data.
        threshold (float): Desired quantile threshold.

    Returns:
        float: Calculated quantile threshold.
    """
    mse = compute_mean_squared_error(x_train, predictions)
    return np.quantile(mse, threshold)

def compute_mean_squared_error(input_data: pd.DataFrame, predictions: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Compute the Mean Squared Error (MSE) between input data and predictions.

    Args:
        input_data (pd.DataFrame): Original input data.
        predictions (Union[pd.DataFrame, np.ndarray]): Predicted values.

    Returns:
        np.ndarray: Array of MSE values for each input row.
    """
    return np.mean(np.power(input_data - predictions, 2), axis=1)