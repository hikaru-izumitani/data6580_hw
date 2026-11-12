from decimal import Decimal

from sqlalchemy.exc import IntegrityError

from data5580_hw.controllers.prediction_controller import prediction_controller
from data5580_hw.models.prediction import Embedding, Prediction
from data5580_hw.services.prediction_service import PredictionService, _features_frame

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


def test_dict_features_are_accepted(app):
    named = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.02381,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }
    response = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": named},
    )
    assert response.status_code == 200
    assert float(response.json["label"]) == 1.5


def test_arize_failure_is_saved_and_prediction_still_returns(app, monkeypatch):
    saved = []

    def fail(prediction):
        raise RuntimeError("down")

    monkeypatch.setattr(
        "data5580_hw.services.prediction_service.arize_gateway.send_prediction",
        fail,
    )
    monkeypatch.setattr(
        "data5580_hw.services.prediction_service.save_for_later",
        lambda *args: saved.append(args),
    )
    app.config["TESTING"] = False
    response = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": [8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]},
    )
    assert response.status_code == 200
    assert saved[0][0] == "prediction"


def test_database_error_returns_500(app, monkeypatch):
    def fail():
        raise IntegrityError("insert", {}, Exception("duplicate"))

    monkeypatch.setattr(
        "data5580_hw.services.prediction_service.db.session.commit",
        fail,
    )
    response = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": [8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]},
    )
    assert response.status_code == 500


def test_missing_explainer_or_embedding_is_rejected(app, monkeypatch):
    from data5580_hw.gateways.mlflow_gateway import mlflow_gateway

    original = mlflow_gateway.get_model

    def without_explainer(name, version):
        model = original(name, version)
        model._explainer = None
        return model

    monkeypatch.setattr(mlflow_gateway, "get_model", without_explainer)
    response = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": [8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]},
    )
    assert response.status_code == 400
    assert "explainer" in response.json["error"]

    def without_embedding(name, version):
        model = original(name, version)
        model._embedding = None
        return model

    monkeypatch.setattr(mlflow_gateway, "get_model", without_embedding)
    response = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": [8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]},
    )
    assert response.status_code == 400
    assert "embedding" in response.json["error"]


def test_features_frame_handles_names_and_mismatches():
    frame = _features_frame({"MedInc": 1}, object())
    assert list(frame.columns) == ["MedInc"]

    model = type("Model", (), {"feature_names_in_": 5})()
    unnamed = _features_frame([1.0, 2.0], model)
    assert list(unnamed.columns) == [0, 1]

    named = type("Model", (), {"feature_names_in_": ["MedInc", "HouseAge"]})()
    matched = _features_frame([8.3, 41.0], named)
    assert list(matched.columns) == ["MedInc", "HouseAge"]


def test_prediction_dictionary_covers_edge_values():
    parsed = Prediction(features='{"MedInc": 1}').to_dict()
    assert parsed["features"]["MedInc"] == 1

    invalid = Prediction(features="not-json")
    invalid.shap_values = "bad"
    body = invalid.to_dict()
    assert body["features"] == "not-json"
    assert body["shap_values"] == {}

    short = Prediction(embedding_values=Embedding(embeddings=[1.0, 2.0]))
    assert short.to_dict()["embeddings"] is None

    generated = Prediction.generate_test_record()
    assert "feature_1" in generated.features


def test_get_prediction_sql_returns_the_row(app):
    created = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": [8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]},
    )
    row = prediction_controller.get_prediction_sql(created.json["id"])
    assert row.id == created.json["id"]