import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(features: np.ndarray, labels: np.ndarray, seed: int) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed)
    model.fit(features, labels)
    return model