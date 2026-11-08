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
            model_version="5",
            prediction=prediction
        )
        
        assert prediction.label_numeric is not None
        # Accept either float or Decimal.
        assert isinstance(prediction.label_numeric, (float, Decimal))
        assert float(prediction.label_numeric) == 1.5
        shap_values = prediction.shap_values[0].shap_values
        assert shap_values["MedInc"] == 0.4
        assert set(shap_values) == {
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        }
        stored = prediction.embedding_values.embeddings
        assert stored == [0.24, -0.38, 0.4]
        assert prediction.to_dict()["embeddings"] == {
            "kind": "inputs",
            "x": 0.24,
            "y": -0.38,
            "z": 0.4,
        }
        again = prediction_service.create_prediction(
            model_name="california-housing",
            model_version="5",
            prediction=Prediction(
                features=[8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]
            ),
        )
        assert again.embedding_values.embeddings == stored