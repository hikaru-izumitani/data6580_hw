from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from arize.ml.types import Environments, ModelTypes

from data5580_hw.gateways.arize_gateway import ArizeGateway, save_for_later
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


def _config(values):
    app = MagicMock()

    def get(key, default=None):
        return values.get(key, default)

    app.config.get.side_effect = get
    return app


def _regression(features, **kwargs):
    model = Model(name="california-housing", version="5", type="REGRESSION")
    model._model = MagicMock()
    model._model.feature_names_in_ = ["MedInc", "HouseAge"]
    return Prediction(
        id="prediction-1",
        model=model,
        features=features,
        label_numeric=1.5,
        created=datetime(2026, 11, 11, 12, 0, 0),
        **kwargs,
    )


def test_init_app_can_be_disabled():
    gateway = ArizeGateway()
    gateway.init_app(_config({"ARIZE_ENVIRONMENT": "staging", "ARIZE_ENABLED": False}))
    assert gateway.environment == "staging"
    assert gateway.arize_client is None


def test_init_app_skips_client_without_credentials():
    gateway = ArizeGateway()
    gateway.init_app(_config({"ARIZE_ENABLED": True, "SPACE_ID": None, "ARIZE_API_KEY": None}))
    assert gateway.arize_client is None


def test_init_app_builds_client(monkeypatch):
    monkeypatch.setattr(
        "data5580_hw.gateways.arize_gateway.ArizeClient",
        lambda api_key: MagicMock(api_key=api_key),
    )
    gateway = ArizeGateway()
    gateway.init_app(
        _config({"ARIZE_ENABLED": True, "SPACE_ID": "space", "ARIZE_API_KEY": "key"})
    )
    assert gateway.arize_client.api_key == "key"


def test_send_actual_maps_regression_value():
    gateway = ArizeGateway()
    gateway.space_id = "space"
    gateway.arize_client = MagicMock()
    response = MagicMock(status_code=200, text="ok")
    gateway.arize_client.ml.log_stream.return_value.result.return_value = response
    prediction = _regression([8.3, 41.0], actual_numeric=4.5)

    gateway.send_actual(prediction)

    assert gateway.arize_client.ml.log_stream.call_args.kwargs["actual_label"] == 4.5


def test_send_actual_maps_categorical_value():
    gateway = ArizeGateway()
    gateway.space_id = "space"
    gateway.arize_client = MagicMock()
    response = MagicMock(status_code=200, text="ok")
    gateway.arize_client.ml.log_stream.return_value.result.return_value = response
    model = Model(name="classifier", version="1", type="SCORE_CATEGORICAL")
    prediction = Prediction(
        id="prediction-2",
        model=model,
        actual_str="yes",
        score=0.8,
        created=datetime(2026, 11, 11, 12, 0, 0),
    )

    gateway.send_actual(prediction)

    assert gateway.arize_client.ml.log_stream.call_args.kwargs["actual_label"] == ("yes", 0.8)


def test_send_actual_rejects_unsupported_type():
    gateway = ArizeGateway()
    gateway.arize_client = MagicMock()
    other = next(
        item
        for item in ModelTypes
        if item not in (ModelTypes.REGRESSION, ModelTypes.SCORE_CATEGORICAL)
    )
    prediction = Prediction(model=Model(name="m", version="1", type=other))

    with pytest.raises(ValueError, match="Unsupported model type"):
        gateway.send_actual(prediction)


def test_send_actual_does_nothing_without_credentials():
    gateway = ArizeGateway()
    gateway.send_actual(Prediction())


def test_failed_response_is_saved_for_later(tmp_path, monkeypatch):
    class QueuePath:
        def __init__(self, path):
            self._path = path

        def resolve(self):
            return self

        @property
        def parent(self):
            return self

        def __truediv__(self, other):
            return QueuePath(self._path / other)

        def open(self, *args, **kwargs):
            return self._path.open(*args, **kwargs)

    monkeypatch.setattr(
        "data5580_hw.gateways.arize_gateway.Path",
        lambda *_args, **_kwargs: QueuePath(tmp_path),
    )
    gateway = ArizeGateway()
    gateway.space_id = "space"
    gateway.environment = "development"
    gateway.arize_client = MagicMock()
    response = MagicMock(status_code=503, text="unavailable")
    gateway.arize_client.ml.log_stream.return_value.result.return_value = response

    gateway.send_prediction(_regression([8.3, 41.0]))

    saved = (tmp_path / "arize_queue.jsonl").read_text(encoding="utf-8")
    assert "prediction-1" in saved
    assert "503" in saved


def test_send_prediction_accepts_dict_features_and_tags():
    gateway = ArizeGateway()
    gateway.space_id = "space"
    gateway.environment = "production"
    gateway.arize_client = MagicMock()
    response = MagicMock(status_code=200, text="ok")
    gateway.arize_client.ml.log_stream.return_value.result.return_value = response
    prediction = _regression({"MedInc": Decimal("8.3"), "ok": True})

    class Tags:
        def to_dict(self):
            return {"region": "west"}

    prediction.tags = Tags()
    prediction.shap_values = {"MedInc": 0.4, "ok": 0.1}
    vector = MagicMock()
    vector.tolist.return_value = [0.1, 0.2]
    prediction.embedding_values = Embedding(embeddings=vector)

    gateway.send_prediction(prediction)

    kwargs = gateway.arize_client.ml.log_stream.call_args.kwargs
    assert kwargs["features"] == {"MedInc": 8.3, "ok": True}
    assert kwargs["tags"]["region"] == "west"
    assert kwargs["tags"]["deployment"] == "production"
    assert list(kwargs["embedding_features"]["embedding"].vector) == [0.1, 0.2]


def test_categorical_shap_uses_the_predicted_label():
    gateway = ArizeGateway()
    gateway.space_id = "space"
    gateway.environment = "development"
    gateway.arize_client = MagicMock()
    response = MagicMock(status_code=200, text="ok")
    gateway.arize_client.ml.log_stream.return_value.result.return_value = response
    model = Model(name="classifier", version="1", type=ModelTypes.SCORE_CATEGORICAL)
    prediction = Prediction(
        id="prediction-3",
        model=model,
        features=[1, 2],
        label_str="yes",
        score=0.8,
        shap_values={"yes": {"a": 0.4}, "no": {"a": -0.4}},
        created=datetime(2026, 11, 11, 12, 0, 0),
    )

    gateway.send_prediction(prediction)

    kwargs = gateway.arize_client.ml.log_stream.call_args.kwargs
    assert kwargs["prediction_label"] == ("yes", 0.8)
    assert kwargs["shap_values"] == {"a": 0.4}


def test_categorical_shap_list_selects_matching_label():
    gateway = ArizeGateway()
    gateway.space_id = "space"
    gateway.environment = "development"
    gateway.arize_client = MagicMock()
    response = MagicMock(status_code=200, text="ok")
    gateway.arize_client.ml.log_stream.return_value.result.return_value = response
    model = Model(name="classifier", version="1", type="SCORE_CATEGORICAL")
    model._model = MagicMock()
    model._model.feature_names_in_ = 5
    prediction = Prediction(
        id="prediction-4",
        model=model,
        features=[1, 2],
        label_str="yes",
        score=0.7,
        shap_values=[
            Shap(label="no", shap_values={"a": 0.1}),
            Shap(label="yes", shap_values={"a": 0.9}),
        ],
        created=datetime(2026, 11, 11, 12, 0, 0),
    )

    gateway.send_prediction(prediction)

    kwargs = gateway.arize_client.ml.log_stream.call_args.kwargs
    assert kwargs["features"] == {"feature_0": 1, "feature_1": 2}
    assert kwargs["shap_values"] == {"a": 0.9}


def test_non_dict_shap_and_non_regression_label_are_rejected():
    gateway = ArizeGateway()
    gateway.environment = "development"
    gateway.arize_client = MagicMock()
    response = MagicMock(status_code=200, text="ok")
    gateway.arize_client.ml.log_stream.return_value.result.return_value = response
    prediction = _regression([8.3, 41.0], shap_values=[Shap(shap_values=None)])

    gateway.send_prediction(prediction)

    assert gateway.arize_client.ml.log_stream.call_args.kwargs["shap_values"] is None

    other = next(
        item
        for item in ModelTypes
        if item not in (ModelTypes.REGRESSION, ModelTypes.SCORE_CATEGORICAL)
    )
    prediction.model.type = other
    with pytest.raises(ValueError, match="Unsupported model type"):
        gateway.send_prediction(prediction)


def test_missing_or_unknown_model_type_is_rejected():
    gateway = ArizeGateway()
    gateway.environment = "development"
    gateway.arize_client = MagicMock()
    missing = Prediction(model=Model(name="m", version="1", type=None), features={})
    with pytest.raises(ValueError, match="missing"):
        gateway.send_prediction(missing)

    unknown = Prediction(model=Model(name="m", version="1", type="NOT_A_TYPE"), features={})
    with pytest.raises(ValueError, match="Unsupported model type"):
        gateway.send_prediction(unknown)


def test_save_for_later_writes_one_json_line(tmp_path, monkeypatch):
    class QueuePath:
        def __init__(self, path):
            self._path = path

        def resolve(self):
            return self

        @property
        def parent(self):
            return self

        def __truediv__(self, other):
            return QueuePath(self._path / other)

        def open(self, *args, **kwargs):
            return self._path.open(*args, **kwargs)

    monkeypatch.setattr(
        "data5580_hw.gateways.arize_gateway.Path",
        lambda *_args, **_kwargs: QueuePath(tmp_path),
    )
    save_for_later("actual", "prediction-9", "network down")
    assert "prediction-9" in (tmp_path / "arize_queue.jsonl").read_text(encoding="utf-8")
