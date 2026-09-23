from pathlib import Path

import joblib
import numpy as np


MODEL_FILE = Path(
    "models/world_model/attack_category_estimation.joblib"
)


class AttackCategoryEstimator:
    """Nearest-centroid category estimator for observed training classes."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.classes_ = None
        self.centroids_ = None

    def fit(self, features, labels):
        labels = np.asarray(labels)
        self.classes_ = np.array(sorted(np.unique(labels)))
        self.centroids_ = np.asarray(
            [
                features[labels == label].mean(axis=0)
                for label in self.classes_
            ]
        )
        return self

    def _distances(self, features):
        differences = (
            features[:, None, :]
            - self.centroids_[None, :, :]
        )
        return np.linalg.norm(differences, axis=2)

    def predict_proba(self, features):
        distances = self._distances(features)
        scores = np.exp(-distances / self.temperature)
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, features):
        probabilities = self.predict_proba(features)
        return self.classes_[probabilities.argmax(axis=1)]

    def save(self, path=MODEL_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path=MODEL_FILE):
        return joblib.load(path)
