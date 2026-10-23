from data5580_hw.models.prediction import Shap, Prediction


class ExplainerService:

    @staticmethod
    def create_explanation(prediction: Prediction) -> Prediction:

        shap_values_ = prediction.model._explainer.predict(prediction.features, )

        shape_values = []

        # Assign feature names to SHAP Values
        shape_values.append(
            Shap(
                label=None
                ,shap_values=shap_values_
            )
        )

        prediction.shap_values = shape_values

        return prediction


explainer_service: ExplainerService = ExplainerService()