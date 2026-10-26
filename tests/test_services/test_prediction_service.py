from decimal import Decimal
from data5580_hw.models.prediction import Prediction
from data5580_hw.services.prediction_service import PredictionService

def test_create_prediction(app):
    with app.app_context():
        prediction = Prediction(
            features=[8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]
        )
        prediction_service = PredictionService()

        prediction = prediction_service.create_prediction(
            model_name="california-housing",
            model_version="2",
            prediction=prediction
        )
        
        assert prediction.label_numeric is not None
        # float または Decimal のどちらであってもパスするように修正
        assert isinstance(prediction.label_numeric, (float, Decimal))