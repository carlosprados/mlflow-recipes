import os.path
import sys

import pandas as pd

from opengate.recipes.custom_models_enum import CustomModels
from opengate.recipes.utils.metrics import BUILTIN_ANOMALY_RECIPE_METRICS

from pandas import DataFrame
import numpy as np

import mlflow
from mlflow.models import EvaluationResult, EvaluationMetric

from importlib import import_module
import importlib.util as importlib_utils
from typing import List, Union

import logging

_logger = logging.getLogger(__name__)

class EvaluateAnomalyModel:
    def __init__(self, model_uri: str, dataset: DataFrame, dataset_name: str, root_recipe: str,
                 extra_metrics: List[EvaluationMetric]):
        self.model_uri = model_uri
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.root_recipe = root_recipe
        self.extra_metrics = extra_metrics


    def evaluate_anomaly_model(self, model_type: str, threshold: np.floating):
        # Load model
        if model_type == CustomModels.ISOLATION_FOREST.model_name:
            model = mlflow.sklearn.load_model(self.model_uri)
            scores = -model.score_samples(self.dataset)
            raw_predictions = (scores > threshold).astype(int)
        else:
            model = mlflow.pyfunc.load_model(self.model_uri)
            raw_predictions = model.predict(self.dataset)
        artifacts = {}
        calculated_metrics = {}

        sklearn_metrics = import_module("sklearn.metrics")
        for metric in BUILTIN_ANOMALY_RECIPE_METRICS:
            try:
                imported_metric = getattr(sklearn_metrics, metric.name)
                score = imported_metric(self.dataset, raw_predictions)
                calculated_metrics[metric.name] = score
                mlflow.log_metric(f"{self.dataset_name}_{metric.name}", score)
            except AttributeError as ae:
                _logger.error(f"Metric '{metric.name}' not found in sklearn.metrics: {ae}")
            except Exception as e:
                _logger.error(f"Error calculating metric '{metric.name}': {e}")

        if self.extra_metrics is not None:
            calculated_extra_metrics = self.evaluate_custom_metrics(metrics=self.extra_metrics, predictions=raw_predictions)
            calculated_metrics.update(calculated_extra_metrics)
        return EvaluationResult(artifacts=artifacts, metrics=calculated_metrics)

    def evaluate_custom_metrics(self, metrics: List[EvaluationMetric], predictions):
        module_path = os.path.join(self.root_recipe, "steps", "custom_metrics.py")
        sys.path.append(module_path)
        extra_calculated_metrics = {}
        for metric in metrics:
            spec = importlib_utils.spec_from_file_location(metric.name, module_path)
            module = importlib_utils.module_from_spec(spec)
            spec.loader.exec_module(module)
            metric_function = getattr(module, metric.name)
            extra_metric_score = metric_function(self.dataset, predictions)
            mlflow.log_metric(f"{self.dataset_name}_{metric.name}", extra_metric_score)
            extra_calculated_metrics[metric.name] = extra_metric_score

        return extra_calculated_metrics


def preprocess_anomaly_data(dataset: pd.DataFrame, recipe_root: str) -> pd.DataFrame:
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