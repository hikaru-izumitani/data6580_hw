from data5580_hw.models.prediction import Prediction
from data5580_hw.services.prediction_service import PredictionService

def test_create_prediction(app):
    with app_.app_context():

        prediction = Prediction.generate_test_record()
        prediction_service = PredictionService()

        prediction = prediction_service.create_prediction(prediction)
        assert int(prediction.label_numeric) == 500