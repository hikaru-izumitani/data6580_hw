import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from data5580_hw.models.prediction import Model, Prediction
from data5580_hw.services.embedding_service import embedding_service
from data5580_hw.services.explainer_service import _one_row, explainer_service
from data5580_hw.services.inference_service import inference_service


def _categorical_model():
    model = Model(
        name="classifier",
        version="1",
        type="SCORE_CATEGORICAL",
        threshold={
            "threshold": 0.5,
            "ABOVE_THRESHOLD": "yes",
            "BELOW_THRESHOLD": "no",
        },
    )
    model._model = MagicMock()
    return model


def test_inference_assigns_categorical_labels():
    model = _categorical_model()
    model._model.predict_proba.return_value = np.array([[0.2, 0.8]])
    above = inference_service.create_inference(Prediction(model=model, features=[[1]]))
    assert above.label_str == "yes"
    assert above.score == 0.8

    model._model.predict_proba.return_value = np.array([[0.8, 0.2]])
    below = inference_service.create_inference(Prediction(model=model, features=[[1]]))
    assert below.label_str == "no"

    model._model.predict_proba.return_value = np.array([0.9])
    direct = inference_service.create_inference(Prediction(model=model, features=[[1]]))
    assert direct.label_str == "yes"


def test_inference_rejects_an_unknown_model_type():
    model = Model(name="m", version="1", type="UNKNOWN")
    with pytest.raises(ValueError, match="Unsupported model type"):
        inference_service.create_inference(Prediction(model=model, features=[1]))


def test_explainer_requires_a_configured_model():
    with pytest.raises(ValueError, match="not configured"):
        explainer_service.create_explanation(Prediction(features=[1, 2]))


def test_explainer_accepts_dict_list_and_frame():
    model = Model(name="m", version="1", type="REGRESSION")
    model._explainer = MagicMock()
    model._explainer.predict.return_value = [[0.1, 0.2]]
    explained = explainer_service.create_explanation(
        Prediction(model=model, features={"a": 1, "b": 2})
    )
    assert explained.shap_values[0].shap_values == {"a": 0.1, "b": 0.2}

    again = explainer_service.create_explanation(
        Prediction(model=model, features=[1, 2])
    )
    assert again.shap_values[0].shap_values["0"] == 0.1


def test_explainer_wraps_prediction_errors():
    model = Model(name="m", version="1", type="REGRESSION")
    model._explainer = MagicMock()
    model._explainer.predict.side_effect = RuntimeError("boom")
    with pytest.raises(ValueError, match="boom"):
        explainer_service.create_explanation(Prediction(model=model, features=[1, 2]))


def test_explainer_rejects_the_wrong_number_of_values():
    model = Model(name="m", version="1", type="REGRESSION")
    model._explainer = MagicMock()
    model._explainer.predict.return_value = [[0.1]]
    with pytest.raises(ValueError, match="expected 2"):
        explainer_service.create_explanation(Prediction(model=model, features=[1, 2]))


def test_one_row_flattens_nested_arrays():
    assert list(_one_row(np.ones((1, 1, 2)), 2)) == [1.0, 1.0]
    assert len(_one_row(np.ones((2, 3)), 4)) == 6
    assert list(_one_row(np.array([1.0, 2.0]), 2)) == [1.0, 2.0]


def test_embedding_requires_a_configured_model():
    with pytest.raises(ValueError, match="not configured"):
        embedding_service.create_embedding(Prediction(features=[1]))


def test_embedding_wraps_prediction_errors_and_checks_length():
    model = Model(name="m", version="1", type="REGRESSION")
    model._embedding = MagicMock()
    model._embedding.predict.side_effect = RuntimeError("bad")
    with pytest.raises(ValueError, match="bad"):
        embedding_service.create_embedding(Prediction(model=model, features=[1]))

    model._embedding.predict.side_effect = None
    model._embedding.predict.return_value = [1.0, 2.0]
    with pytest.raises(ValueError, match="expected 3"):
        embedding_service.create_embedding(Prediction(model=model, features=[1]))


def test_mlflow_gateway_loads_each_flavor(monkeypatch):
    source = Path("data5580_hw/gateways/mlflow_gateway.py").resolve()
    spec = importlib.util.spec_from_file_location("mlflow_gateway_fresh", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.mlflow.set_tracking_uri = lambda uri: None
    module.mlflow.pyfunc.load_model = lambda *args, **kwargs: MagicMock()
    module.mlflow.sklearn.load_model = lambda *args, **kwargs: MagicMock()
    gateway = module.MLFlowGateway()
    app = MagicMock()
    app.config = {
        "MLFLOW_TRACKING_URI": "http://localhost:8080",
        "MODELS": {
            "plain": {
                "1": {"model_type": "REGRESSION", "mlflow_flavor": "pyfunc", "threshold": None}
            },
            "tree": {
                "2": {
                    "model_type": "REGRESSION",
                    "flavor": "sklearn",
                    "explainer_path": "exp/1",
                    "embedding_path": "emb/1",
                    "threshold": None,
                }
            },
        },
    }
    gateway.init_app(app)
    loaded = gateway.get_model("plain", "1")
    assert loaded._explainer is None
    assert loaded._embedding is None
    explained = gateway.get_model("tree", "2")
    assert explained._explainer is not None
    assert explained._embedding is not None

    app.config["MODELS"] = {
        "bad": {"3": {"model_type": "REGRESSION", "mlflow_flavor": "xgboost", "threshold": None}}
    }
    with pytest.raises(ValueError, match="Invalid flavor"):
        gateway.init_app(app)
