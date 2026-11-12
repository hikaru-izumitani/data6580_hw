import logging
import json
from pathlib import Path
from decimal import Decimal

from arize import ArizeClient
from arize.ml.types import Embedding, Environments, ModelTypes

from data5580_hw.models.prediction import Prediction


logger = logging.getLogger(__name__)


class ArizeGateway:

    def __init__(self):
        self.space_id = None
        self.arize_client = None

    def init_app(self, app):
        self.environment = app.config.get("ARIZE_ENVIRONMENT", "production")
        if not app.config.get("ARIZE_ENABLED", True):
            self.space_id = None
            self.arize_client = None
            logger.warning("Arize monitoring is disabled by ARIZE_ENABLED.")
            return
        self.space_id = app.config.get("SPACE_ID")
        api_key = app.config.get("ARIZE_API_KEY")
        self.arize_client = None
        if not api_key or not self.space_id:
            logger.warning(
                "Arize monitoring is off because ARIZE_API_KEY or ARIZE_SPACE_ID is missing."
            )
            return
        self.arize_client = ArizeClient(api_key=api_key)

    def send_prediction(self, prediction: Prediction) -> None:
        if self.arize_client is None:
            return

        model_type = _model_type(prediction)
        embedding_payload = _embedding_payload(prediction)
        shap_values_payload = _shap_payload(prediction, model_type)
        prediction_label = _prediction_label(prediction, model_type)
        tags = _tags(prediction) or {}
        tags["deployment"] = self.environment

        response = self.arize_client.ml.log_stream(
            space_id=self.space_id,
            prediction_id=prediction.id,
            model_name=prediction.model.name,
            model_type=model_type,
            environment=Environments.PRODUCTION,
            shap_values=shap_values_payload,
            model_version=prediction.model.version,
            prediction_timestamp=_timestamp(prediction),
            features=_features(prediction),
            embedding_features=embedding_payload,
            prediction_label=prediction_label,
            tags=tags,
        )
        _log_result(response, "prediction", prediction.id)

    def send_actual(self, prediction: Prediction) -> None:
        if self.arize_client is None:
            return

        model_type = _model_type(prediction)
        if model_type == ModelTypes.SCORE_CATEGORICAL:
            actual_label = (str(prediction.actual_str), float(prediction.score))
        elif model_type == ModelTypes.REGRESSION:
            actual_label = float(prediction.actual_numeric)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        response = self.arize_client.ml.log_stream(
            space_id=self.space_id,
            prediction_id=prediction.id,
            model_name=prediction.model.name,
            model_type=model_type,
            environment=Environments.PRODUCTION,
            actual_label=actual_label,
        )
        _log_result(response, "actual", prediction.id)


def _model_type(prediction: Prediction) -> ModelTypes:
    raw = prediction.model.type
    if isinstance(raw, ModelTypes):
        return raw
    name = getattr(raw, "name", raw)
    if name is None:
        raise ValueError("Model type is missing.")
    try:
        return ModelTypes[str(name)]
    except KeyError as error:
        raise ValueError(f"Unsupported model type: {name}") from error


def _features(prediction: Prediction) -> dict:
    features = prediction.features
    if isinstance(features, dict):
        return {str(name): _scalar(value) for name, value in features.items()}

    values = list(features)
    names = getattr(getattr(prediction.model, "_model", None), "feature_names_in_", None)
    try:
        names = [str(name) for name in list(names)] if names is not None else []
    except TypeError:
        names = []
    if len(names) != len(values):
        names = [f"feature_{index}" for index in range(len(values))]
    return {name: _scalar(value) for name, value in zip(names, values)}


def _shap_payload(prediction: Prediction, model_type: ModelTypes):
    raw = prediction.shap_values
    if not raw:
        return None
    if isinstance(raw, dict):
        if (
            model_type == ModelTypes.SCORE_CATEGORICAL
            and isinstance(raw.get(prediction.label_str), dict)
        ):
            raw = raw[prediction.label_str]
        return {str(name): float(value) for name, value in raw.items()}

    item = raw[0]
    if model_type == ModelTypes.SCORE_CATEGORICAL and prediction.label_str is not None:
        item = next(
            (
                candidate
                for candidate in raw
                if getattr(candidate, "label", None) == prediction.label_str
            ),
            item,
        )
    values = getattr(item, "shap_values", item)
    if not isinstance(values, dict):
        return None
    return {str(name): float(value) for name, value in values.items()}


def _prediction_label(prediction: Prediction, model_type: ModelTypes):
    if model_type == ModelTypes.SCORE_CATEGORICAL:
        return (str(prediction.label_str), float(prediction.score))
    if model_type == ModelTypes.REGRESSION:
        return float(prediction.label_numeric)
    raise ValueError(f"Unsupported model type: {model_type}")


def _embedding_payload(prediction: Prediction):
    embedding_values = getattr(prediction, "embedding_values", None)
    vector = getattr(embedding_values, "embeddings", None) if embedding_values else None
    if vector is None:
        return None
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return {"embedding": Embedding(vector=[float(value) for value in vector])}


def _tags(prediction: Prediction):
    tags = getattr(prediction, "tags", None)
    if tags is None:
        return None
    return tags.to_dict()


def _timestamp(prediction: Prediction) -> int:
    return int(prediction.created.timestamp())


def _scalar(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, Decimal):
        return float(value)
    return value

def save_for_later(kind: str, prediction_id: str, reason:str) -> None:
    path = Path(__file__).resolve().parent.parent / "arize_queue.jsonl"
    record = {
        "kind": kind,
        "prediction_id": prediction_id,
        "reason": reason,
    }
    with path.open("a", encoding="utf-8") as retry_file:
        retry_file.write(json.dumps(record) + "\n")
    logger.error(
        "Saved failed Arize %s %s for later retry: %s", kind, prediction_id, reason,
    )



def _log_result(response, kind: str, prediction_id: str) -> None:
    result = response.result()
    if result.status_code == 200:
        logger.info("Sent %s %s to Arize.", kind, prediction_id)
        return
    save_for_later(kind, prediction_id, f"{result.status_code} {result.text}")


arize_gateway = ArizeGateway()
