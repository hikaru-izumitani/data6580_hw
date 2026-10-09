from data5580_hw.models.prediction import Prediction

class InferenceService:
    def __init__app__(self):
        pass
    @staticmethod
    def create_inference(self, prediction: Prediction) -> Prediction:
        label = prediction.model._model.predict(prediction.features)
        
        prediction.label_numeric = label
        return prediction

inference_service = InferenceService()