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

def test_predict_without_blocking_returns_task_ids(app):
    response = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": FEATURES, "block": False},
    )
    assert response.status_code == 202
    assert response.json["shap_values"] == {}
    assert response.json["embeddings"] is None
    assert response.json["tasks"]["explanation_task_id"]
    assert response.json["tasks"]["embedding_task_id"]


def test_finished_tasks_are_visible_on_the_prediction(app):
    created = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": FEATURES, "block": False},
    )
    assert created.status_code == 202
    stored = app.test_client().get(f"/prediction/{created.json['id']}")
    assert stored.status_code == 200
    assert stored.json["shap_values"]["MedInc"] == 0.4
    assert stored.json["shap_values"]["Latitude"] == -0.2
    assert stored.json["embeddings"] == {
        "kind": "inputs",
        "x": 0.24,
        "y": -0.38,
        "z": 0.4,
    }


def test_task_status_reports_celery_state(app):
    class Result:
        def __init__(self, state, info):
            self.state = state
            self.info = info

    class FakeCelery:
        def __init__(self, state, info):
            self._state = state
            self._info = info

        def AsyncResult(self, task_id):
            assert task_id == "task-1"
            return Result(self._state, self._info)

    cases = [
        ("PENDING", None, 200, {"task_id": "task-1", "state": "PENDING", "status": "Pending..."}),
        ("STARTED", None, 200, {"task_id": "task-1", "state": "STARTED", "status": "Started..."}),
        ("SUCCESS", "prediction-id", 200, {"task_id": "task-1", "state": "SUCCESS", "result": "prediction-id"}),
        ("FAILURE", "boom", 500, {"task_id": "task-1", "state": "FAILURE", "error": "boom"}),
    ]
    for state, info, code, body in cases:
        app.extensions["celery"] = FakeCelery(state, info)
        status = app.test_client().get("/tasks/task-1")
        assert status.status_code == code
        assert status.json == body


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
        "data5580_hw.services.tasks.explainer_service.create_explanation",
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
        "data5580_hw.services.tasks.embedding_service.create_embedding",
        fail,
    )
    response = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": FEATURES},
    )
    assert response.status_code == 400
    assert response.json["error"] == "UMAP computation failed: invalid input data"


def test_predict_logs_inference_to_arize(app, monkeypatch):
    calls = []

    def record(prediction):
        calls.append(prediction)

    monkeypatch.setattr(
        "data5580_hw.services.prediction_service.arize_gateway.send_prediction",
        record,
    )
    app.config["TESTING"] = False
    response = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": FEATURES},
    )
    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0].id == response.json["id"]


def test_record_actual_logs_to_arize(app, monkeypatch):
    created = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": FEATURES},
    )
    calls = []
    monkeypatch.setattr(
        "data5580_hw.controllers.prediction_controller.arize_gateway.send_actual",
        lambda prediction: calls.append(prediction),
    )
    response = app.test_client().post(
        f"/prediction/{created.json['id']}/actual",
        json={"actual": 4.5},
    )
    assert response.status_code == 200
    assert response.json["actual"] == 4.5
    assert len(calls) == 1
    assert calls[0].actual_numeric == 4.5


def test_record_actual_requires_a_value(app):
    created = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": FEATURES},
    )
    response = app.test_client().post(
        f"/prediction/{created.json['id']}/actual",
        json={},
    )
    assert response.status_code == 400
    assert response.json["error"] == "actual is required"


def test_record_actual_unknown_prediction_returns_404(app):
    response = app.test_client().post(
        "/prediction/missing/actual",
        json={"actual": 4.5},
    )
    assert response.status_code == 404


def test_record_actual_keeps_success_when_arize_fails(app, monkeypatch):
    created = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": FEATURES},
    )
    saved = []

    def fail(prediction):
        raise RuntimeError("network down")

    monkeypatch.setattr(
        "data5580_hw.controllers.prediction_controller.arize_gateway.send_actual",
        fail,
    )
    monkeypatch.setattr(
        "data5580_hw.controllers.prediction_controller.save_for_later",
        lambda *args: saved.append(args),
    )
    response = app.test_client().post(
        f"/prediction/{created.json['id']}/actual",
        json={"actual": 4.5},
    )
    assert response.status_code == 200
    assert saved[0][0] == "actual"


def test_get_missing_prediction_returns_404(app, monkeypatch):
    monkeypatch.setattr(
        "data5580_hw.controllers.prediction_controller.Prediction.from_prediction_sql",
        lambda prediction_id: None,
    )
    response = app.test_client().get("/prediction/missing")
    assert response.status_code == 404


def test_unexpected_error_returns_500(app, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "data5580_hw.controllers.prediction_controller.prediction_service.create_prediction",
        boom,
    )
    response = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": FEATURES},
    )
    assert response.status_code == 500


def test_non_numeric_features_return_400(app):
    response = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": [True, 41.0, 6.9, 1.0, 322.0, 2.5, 37.8, -122.2]},
    )
    assert response.status_code == 400


def test_explainer_adds_summary(app, monkeypatch):
    created = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": FEATURES},
    )
    monkeypatch.setattr(
        "data5580_hw.controllers.prediction_controller.llm_gateway.generate_explanation",
        lambda prediction: "Median income raises the predicted price.",
    )
    monkeypatch.setattr(
        "data5580_hw.controllers.prediction_controller.embedding_service.nearest_predictions",
        lambda prediction: [{"id": "other", "similarity": 0.9}],
    )
    response = app.test_client().get(f"/prediction/{created.json['id']}/explainer")
    assert response.status_code == 200
    assert response.json["label"] == created.json["label"]
    assert response.json["summary"] == "Median income raises the predicted price."


def test_explainer_reports_when_no_neighbors_exist(app, monkeypatch):
    created = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": FEATURES},
    )
    monkeypatch.setattr(
        "data5580_hw.controllers.prediction_controller.llm_gateway.generate_explanation",
        lambda prediction: "The predicted price is 1.5.",
    )
    monkeypatch.setattr(
        "data5580_hw.controllers.prediction_controller.embedding_service.nearest_predictions",
        lambda prediction: [],
    )
    response = app.test_client().get(f"/prediction/{created.json['id']}/explainer")
    assert response.status_code == 200
    assert "No relevant predictions were found for comparison." in response.json["summary"]


def test_explainer_returns_a_friendly_error_when_the_model_call_fails(app, monkeypatch):
    created = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": FEATURES},
    )

    def fail(prediction):
        raise RuntimeError("The explanation could not be generated. Please try again.")

    monkeypatch.setattr(
        "data5580_hw.controllers.prediction_controller.llm_gateway.generate_explanation",
        fail,
    )
    response = app.test_client().get(f"/prediction/{created.json['id']}/explainer")
    assert response.status_code == 502
    assert response.json["error"] == "The explanation could not be generated. Please try again."


def test_explainer_missing_prediction_returns_404(app):
    response = app.test_client().get("/prediction/missing/explainer")
    assert response.status_code == 404
    assert response.json["error"] == "Prediction not found"


def test_wrong_feature_count_returns_400(app):
    response = app.test_client().post(
        "/v1/california-housing/version/5/predict",
        json={"features": [1.0, 2.0]},
    )
    assert response.status_code == 400