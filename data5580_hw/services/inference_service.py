from data5580_hw.models.prediction import Prediction

#from arize.utils.types import ModelTypes


class InferenceService:

    def __init__app__(self):
        ...

    @staticmethod
    def create_inference(prediction: Prediction) -> Prediction:

        #if prediction.model.type == ModelTypes.REGRESSION:
        if prediction.model.type == "REGRESSION":

            label = prediction.model._model.predict(prediction.features, )
            label = label[0]
            prediction.label_numeric = label
        
        #elif prediction.model.type == ModelTypes.SCORE_CATEGORICAL:
        elif prediction.model.type == "SCORE_CATEGORICAL":
            proba = prediction.model._model.predict_proba(prediction.features)

            if proba >= prediction.model.threshold["threshold"]:
                label = prediction.model.threshold["ABOVE_THRESHOLD"]
            else:
                label = prediction.model.threshold["BELOW_THRESHOLD"]

            prediction.threshold = prediction.model.threshold["threshold"]
            prediction.score = proba

            prediction.label_str = label


        return prediction


inference_service = InferenceService()