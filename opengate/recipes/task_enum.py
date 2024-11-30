from enum import Enum

class MLTask(Enum):
    """
    Represents the allowed ML tasks
    """
    REGRESSION = "regression/v1"
    CLASSIFICATION = "classification/v1"
    ANOMALY = "anomaly/v1"