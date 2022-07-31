import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator


class MockEstimator(BaseEstimator):

    def __init__(self):
        self.n_classes_ = 2
        self.classes_ = np.array([0, 1])
        self.fit_call_list = []
        self.predict_call_list = []

    def fit(self, features, target, sample_weight=None):
        self.fit_call_list.append([features, target, sample_weight])
        return self

    def predict(self, features: DataFrame) -> np.ndarray:
        self.predict_call_list.append(features)
        return np.where(features.values[:, 1] > 0, 1, 0)
