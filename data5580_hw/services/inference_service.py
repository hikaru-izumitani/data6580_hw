from data5580_hw.models.prediction import Prediction


class InferenceService:
    def __init__app__(self):
        ...

    @staticmethod
    def create_inference(prediction: Prediction) -> Prediction:
        # config stores "REGRESSION". An Arize enum's name is the same string.
        model_type = getattr(prediction.model.type, "name", prediction.model.type)

        if model_type == "REGRESSION":
            label = prediction.model._model.predict(prediction.features)
            prediction.label_numeric = float(label[0])
            prediction.score = prediction.label_numeric

        elif model_type == "SCORE_CATEGORICAL":
            proba = prediction.model._model.predict_proba(prediction.features)
            score = float(proba[0][1]) if len(getattr(proba, "shape", ())) == 2 else float(proba[0])
            threshold = prediction.model.threshold if isinstance(prediction.model.threshold, dict) else {}
            cutoff = threshold.get("threshold")
            if score >= cutoff:
                label = threshold["ABOVE_THRESHOLD"]
            else:
                label = threshold["BELOW_THRESHOLD"]
            prediction.threshold = cutoff
            prediction.score = score
            prediction.label_str = label

        else:
            raise ValueError(f"Unsupported model type: {prediction.model.type}")

        return prediction


inference_service = InferenceService()
