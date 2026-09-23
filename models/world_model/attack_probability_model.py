from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression


MODEL_FILE = Path(
    "models/world_model/attack_probability_model.joblib"
)


class AttackProbabilityModel:
    """Binary attack-probability model for predicted network states."""

    def __init__(self):
        self.model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )

    def fit(self, features, targets):
        self.model.fit(features, targets)
        return self

    def predict_probability(self, features):
        return self.model.predict_proba(features)[:, 1]

    def save(self, path=MODEL_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path=MODEL_FILE):
        instance = cls()
        instance.model = joblib.load(path)
        return instance
