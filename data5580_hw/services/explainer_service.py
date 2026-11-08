import numpy as np
import pandas as pd

from data5580_hw.models.prediction import Prediction, Shap


class ExplainerService:

    @staticmethod
    def create_explanation(prediction: Prediction) -> Prediction:
        if prediction.model is None or prediction.model._explainer is None:
            model_name = prediction.model.name if prediction.model else "unknown"
            model_version = prediction.model.version if prediction.model else "unknown"
            raise ValueError(
                "SHAP computation failed: explainer is not configured for "
                f"model '{model_name}' version '{model_version}'"
            )

        features = prediction.features
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        elif not isinstance(features, pd.DataFrame):
            features = pd.DataFrame([list(features)])

        try:
            raw = prediction.model._explainer.predict(features)
        except Exception as error:
            raise ValueError(f"SHAP computation failed: {error}") from error

        values = _one_row(raw, features.shape[1])
        columns = [str(name) for name in features.columns]
        if len(values) != len(columns):
            raise ValueError(
                f"SHAP computation failed: expected {len(columns)} values, got {len(values)}"
            )

        prediction.shap_values = [
            Shap(
                label=None,
                shap_values={name: float(value) for name, value in zip(columns, values)},
            )
        ]
        return prediction


def _one_row(raw, n_features: int) -> np.ndarray:
    if isinstance(raw, list):
        raw = raw[-1]
    values = np.asarray(raw, dtype=float)
    if values.ndim == 3:
        values = values[0]
    if values.ndim == 2:
        values = values[0] if values.shape[0] == 1 or values.shape[1] == n_features else values.reshape(-1)
    else:
        values = values.reshape(-1)
    return values


explainer_service = ExplainerService()
