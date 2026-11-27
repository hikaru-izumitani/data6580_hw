import json
import logging

import pandas as pd
from celery import shared_task

from data5580_hw.gateways import mlflow_gateway
from data5580_hw.models.prediction import Prediction
from data5580_hw.services.embedding_service import embedding_service
from data5580_hw.services.explainer_service import explainer_service

logger = logging.getLogger(__name__)


def _feature_frame(features, model):
    if isinstance(features, str):
        features = json.loads(features)
    if isinstance(features, dict):
        return pd.DataFrame([features])
    if isinstance(features, pd.DataFrame):
        return features
    values = list(features)
    names = getattr(model, "feature_names_in_", None)
    try:
        names = [str(name) for name in list(names)] if names is not None else []
    except TypeError:
        names = []
    if len(names) == len(values):
        return pd.DataFrame([values], columns=names)
    return pd.DataFrame([values])


@shared_task(
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
    soft_time_limit=60,
    time_limit=90,
)
def generate_embedding(prediction_id: str) -> str:
    from data5580_hw.services.database.database_client import db
    from data5580_hw.services.database.prediction_sql import EmbeddingSQL

    logger.info("generate_embedding started for %s", prediction_id)
    try:
        existing = db.session.query(EmbeddingSQL).filter_by(prediction_id=prediction_id).first()
        if existing is not None:
            logger.info("generate_embedding already stored for %s", prediction_id)
            return prediction_id

        prediction = Prediction.from_prediction_sql(prediction_id)
        loaded = mlflow_gateway.mlflow_gateway.get_model(
            prediction.model.name,
            prediction.model.version,
        )
        prediction.model._model = loaded._model
        prediction.model._explainer = loaded._explainer
        prediction.model._embedding = loaded._embedding
        prediction.features = _feature_frame(prediction.features, loaded._model)

        prediction = embedding_service.create_embedding(prediction)

        embedding_sql = prediction.embedding_values.to_sql(prediction_id)

        db.session.add(embedding_sql)
        db.session.commit()
        logger.info("generate_embedding finished for %s", prediction_id)
        return prediction_id
    except Exception:
        db.session.rollback()
        logger.exception("generate_embedding failed for %s", prediction_id)
        raise


@shared_task(
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def generate_explanation(prediction_id: str) -> str:
    from data5580_hw.services.database.database_client import db
    from data5580_hw.services.database.prediction_sql import ShapSQL

    logger.info("generate_explanation started for %s", prediction_id)
    try:
        existing = db.session.query(ShapSQL).filter_by(prediction_id=prediction_id).first()
        if existing is not None:
            logger.info("generate_explanation already stored for %s", prediction_id)
            return prediction_id

        prediction = Prediction.from_prediction_sql(prediction_id)
        loaded = mlflow_gateway.mlflow_gateway.get_model(
            prediction.model.name,
            prediction.model.version,
        )
        prediction.model._model = loaded._model
        prediction.model._explainer = loaded._explainer
        prediction.model._embedding = loaded._embedding
        prediction.features = _feature_frame(prediction.features, loaded._model)

        prediction = explainer_service.create_explanation(prediction)

        explanation_sql = [
            this_shap_value.to_sql(prediction_id) for this_shap_value in prediction.shap_values
        ]

        db.session.add_all(explanation_sql)
        db.session.commit()
        logger.info("generate_explanation finished for %s", prediction_id)
        return prediction_id
    except Exception:
        db.session.rollback()
        logger.exception("generate_explanation failed for %s", prediction_id)
        raise
