FEATURES = [8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]

def test_predict_returns_label(app):
    response = app.test_client().post(
        "/v1/california-housing/version/2/predict",
        json={"features": FEATURES},
    )
    assert response.status_code == 200
    assert float(response.json["label"] )== 1.5

def test_predict_missing_features_returns_400(app):
    response = app.test_client().post(
        "/v1/california-housing/version/2/predict",
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