import json
from decimal import Decimal

import pytest

from data5580_hw.models.prediction import Prediction
from data5580_hw.services.prediction_service import PredictionService

FEATURES = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.02381,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


def test_create_prediction(app):
    with app.app_context():
        prediction = Prediction(features=FEATURES)
        prediction_service = PredictionService()

        prediction = prediction_service.create_prediction(
            model_name="california-housing",
            model_version="4",
            prediction=prediction,
        )

        assert float(prediction.label_numeric) == pytest.approx(3.822, abs=0.001)
        assert isinstance(prediction.label_numeric, (float, Decimal))

        shap_values = prediction.shap_values[0].shap_values
        assert set(shap_values) == set(FEATURES)
        assert all(isinstance(value, float) for value in shap_values.values())
        assert shap_values["MedInc"] > 0
        assert shap_values["Latitude"] < 0

        stored = prediction.embeddings[0]
        assert stored.kind == "inputs"
        assert len(stored.values) == 3
        assert all(isinstance(value, float) for value in stored.values)

        again = prediction_service.create_prediction(
            model_name="california-housing",
            model_version="4",
            prediction=Prediction(features=FEATURES),
        )
        assert again.embeddings[0].values == stored.values

        body = prediction.to_dict()
        assert body["label"] == pytest.approx(3.822, abs=0.001)
        assert body["shap_values"] == shap_values
        assert body["embeddings"]["kind"] == "inputs"
        assert body["embeddings"]["x"] == stored.values[0]
        assert body["embeddings"]["y"] == stored.values[1]
        assert body["embeddings"]["z"] == stored.values[2]


def test_unknown_model_version_returns_error(app):
    response = app.test_client().post(
        "/model/california-housing/version/99",
        json={"features": FEATURES},
    )

    assert response.status_code == 404
    assert response.json["error"] == "Model 'california-housing' (version '99') was not found"


def test_shap_failure_returns_descriptive_error(app, monkeypatch):
    def fail(prediction):
        raise ValueError("SHAP computation failed: unsupported model type")

    monkeypatch.setattr(
        "data5580_hw.services.prediction_service.explainer_service.create_explanation",
        fail,
    )

    response = app.test_client().post(
        "/model/california-housing/version/4",
        data=json.dumps({"features": FEATURES}),
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.json["error"] == "SHAP computation failed: unsupported model type"


def test_umap_failure_returns_descriptive_error(app, monkeypatch):
    def fail(prediction):
        raise ValueError("UMAP computation failed: invalid input data")

    monkeypatch.setattr(
        "data5580_hw.services.prediction_service.embedding_service.create_embedding",
        fail,
    )

    response = app.test_client().post(
        "/model/california-housing/version/4",
        data=json.dumps({"features": FEATURES}),
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.json["error"] == "UMAP computation failed: invalid input data"
