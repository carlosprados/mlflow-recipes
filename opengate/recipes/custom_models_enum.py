from enum import Enum

class CustomModels(Enum):
    AUTOENCODER = ("anomaly", "autoencoder")
    ISOLATION_FOREST = ("anomaly","isolation_forest")

    def __init__(self, category:str, model_name:str):
        self._category = category
        self._model_name = model_name

    @property
    def category(self):
        return self._category

    @property
    def model_name(self):
        return self._model_name