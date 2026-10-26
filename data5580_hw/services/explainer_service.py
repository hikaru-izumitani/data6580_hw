import numpy as np
import pandas as pd

from data5580_hw.models.prediction import Shap, Prediction


class ExplainerService:

    @staticmethod
    def create_explanation(prediction: Prediction) -> Prediction:
        #shap_values_ = prediction.model._explainer.predict(prediction.features, )
        #
        #shape_values = []
        #
        ## Assign feature names to SHAP Values
        #shape_values.append(
        #    Shap(
        #        label=None
        #        ,shap_values=shap_values_
        #    )
        #)
        #
        #prediction.shap_values = shape_values
        #
        #return prediction

        if prediction.model is None or prediction.model._explainer is None:
            model_name = prediction.model.name if prediction.model else "unknown"
            model_version = prediction.model.version if prediction.model else "unknown"
            raise ValueError(
                f"SHAP explainer is not configured for model {model_name} version {model_version}"
            )

        features = prediction.features
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            feature_names = list(getattr(prediction.model.model, "feature_names_in_", []))
            features_df = pd.DataFrame([features], columns=feature_names or None)

        try:
            raw = prediction.model._explainer.predict(features_df)
        except Exception as e:
            raise ValueError(f"SHAP computation failed: {e}") from e

        values = np.asarray(raw).reshape(-1)
        columns = [str(name) for name in features_df.columns]
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


explainer_service: ExplainerService = ExplainerService()
