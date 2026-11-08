FEATURES = [8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]

def test_predict_returns_label(app):
    response = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": FEATURES},
    )
    assert response.status_code == 200
    assert float(response.json["label"]) == 1.5
    assert response.json["shap_values"]["MedInc"] == 0.4
    assert response.json["shap_values"]["Latitude"] == -0.2
    assert set(response.json["shap_values"]) == {
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    }

def test_predict_missing_features_returns_400(app):
    response = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={},
    )

    assert response.status_code == 400
    assert response.json["error"] == "Invalid input data: features are required."

def test_predict_unknown_model_returns_404(app):
    response = app.test_client().post(
        "/v1/missing/version/1/predict",
        json={"features": FEATURES},
    )

    assert response.status_code == 404
    assert "was not found" in response.json["error"]


def test_prediction_runs_inference_service(app, monkeypatch):
    calls = []
    from data5580_hw.services.inference_service import inference_service

    original = inference_service.create_inference

    def wrapped(prediction):
        calls.append(prediction)
        return original(prediction)

    monkeypatch.setattr(
        "data5580_hw.services.prediction_service.inference_service.create_inference",
        wrapped,
    )
    response = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": FEATURES},
    )
    assert response.status_code == 200
    assert calls
    assert float(response.json["label"]) == 1.5


def test_shap_failure_returns_400(app, monkeypatch):
    def fail(prediction):
        raise ValueError("SHAP computation failed: unsupported model type")

    monkeypatch.setattr(
        "data5580_hw.services.prediction_service.explainer_service.create_explanation",
        fail,
    )
    response = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": FEATURES},
    )
    assert response.status_code == 400
    assert response.json["error"] == "SHAP computation failed: unsupported model type"


def test_get_prediction_returns_stored_shap(app):
    created = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": FEATURES},
    )
    stored = app.test_client().get(f"/prediction/{created.json['id']}")
    assert stored.status_code == 200
    assert float(stored.json["label"]) == 1.5
    assert stored.json["shap_values"] == created.json["shap_values"]
    assert stored.json["embeddings"] == created.json["embeddings"]

def test_umap_failure_returns_400(app, monkeypatch):
    def fail(prediction):
        raise ValueError("UMAP computation failed: invalid input data")

    monkeypatch.setattr(
        "data5580_hw.services.prediction_service.embedding_service.create_embedding",
        fail,
    )
    response = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": FEATURES},
    )
    assert response.status_code == 400
    assert response.json["error"] == "UMAP computation failed: invalid input data"