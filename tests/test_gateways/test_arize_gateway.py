from datetime import datetime
from unittest.mock import MagicMock

from arize.ml.types import Environments, ModelTypes

from data5580_hw.gateways.arize_gateway import ArizeGateway
from data5580_hw.models.prediction import Embedding, Model, Prediction, Shap


def test_send_prediction_maps_regression_record():
    gateway = ArizeGateway()
    gateway.space_id = "space"
    gateway.environment = "development"
    gateway.arize_client = MagicMock()
    response = MagicMock(status_code=200, text="ok")
    gateway.arize_client.ml.log_stream.return_value.result.return_value = response

    model = Model(name="california-housing", version="5", type="REGRESSION")
    model._model = MagicMock()
    model._model.feature_names_in_ = ["MedInc", "HouseAge"]
    created = datetime(2026, 11, 11, 12, 0, 0)
    prediction = Prediction(
        id="prediction-1",
        model=model,
        features=[8.3, 41.0],
        label_numeric=1.5,
        shap_values=[Shap(shap_values={"MedInc": 0.4, "HouseAge": -0.1})],
        embedding_values=Embedding(embeddings=[0.24, -0.38, 0.4]),
        created=created,
    )

    gateway.send_prediction(prediction)

    kwargs = gateway.arize_client.ml.log_stream.call_args.kwargs
    assert kwargs["space_id"] == "space"
    assert kwargs["prediction_id"] == "prediction-1"
    assert kwargs["model_name"] == "california-housing"
    assert kwargs["model_version"] == "5"
    assert kwargs["model_type"] == ModelTypes.REGRESSION
    assert kwargs["environment"] == Environments.PRODUCTION
    assert kwargs["features"] == {"MedInc": 8.3, "HouseAge": 41.0}
    assert kwargs["prediction_label"] == 1.5
    assert kwargs["shap_values"] == {"MedInc": 0.4, "HouseAge": -0.1}
    assert kwargs["prediction_timestamp"] == int(created.timestamp())
    assert kwargs["tags"] == {"deployment": "development"}
    assert list(kwargs["embedding_features"]["embedding"].vector) == [0.24, -0.38, 0.4]


def test_send_prediction_does_nothing_without_credentials():
    gateway = ArizeGateway()
    prediction = Prediction()

    gateway.send_prediction(prediction)
