from asyncio.log import logger
from typing import Dict
import logging

from sqlalchemy.exc import IntegrityError

import pandas as pd

from data5580_hw.gateways import mlflow_gateway
from data5580_hw.models.prediction import Prediction, Model
from data5580_hw.services.database.prediction_sql import PredictionSQL
from data5580_hw.services.embedding_service import embedding_service
from data5580_hw.services.explainer_service import explainer_service
from data5580_hw.services.inference_service import inference_service
from data5580_hw.services.database.database_client import db

logger = logging.getLogger(__name__)


class PredictionService(object):
    def __init__app__(self,app):
        ...
    def create_prediction(self, model_name: str, model_version: str, prediction: Prediction) -> Prediction:
        #model = Model(
            #name=model_name,
            #version=model_version
        #)
        model = mlflow_gateway.mlflow_gateway.get_model(model_name, model_version)
        prediction.model = model

        values = list(prediction.features.values()) if isinstance(prediction.features, dict) else list(prediction.features)
        expected = model._model.n_features_in_

        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            raise ValueError("Invalid input data: features must be numeric.")
        if len(values) != expected:
            raise ValueError(f"Invalid input data: expected {expected} features, got {len(values)}.")

        original_features = prediction.features
        prediction.features = _features_frame(original_features, model._model)
        prediction = inference_service.create_inference(prediction)
        if prediction.model._explainer:
            prediction = explainer_service.create_explanation(prediction)
        else:
            raise ValueError(
                "SHAP computation failed: explainer is not configured for "
                f"model '{model_name}' version '{model_version}'"
            )
        if prediction.model._embedding:
            prediction = embedding_service.create_embedding(prediction)
        else:
            raise ValueError(
                "UMAP computation failed: embedding model is not configured for "
                f"model '{model_name}' version '{model_version}'"
            )
        prediction.features = original_features

        model_sql = model.to_sql()
        prediction_sql = prediction.to_prediction_sql()
        try:
            db.session.add_all([model_sql, prediction_sql])
            db.session.commit()
        except IntegrityError as e:
            logger.error(str(e))
            raise e
        prediction = Prediction.from_prediction_sql(prediction_sql.id)
        #user_sql = db.session.query(PredictionSQL).filter_by(id=prediction.id).first()
        logger.info(f"Created prediction: {prediction.id} for {prediction.model.name} {prediction.model.version}."
        f"features={original_features} label={prediction.label_numeric}")
        return prediction

def _features_frame(features, model) -> pd.DataFrame:
    if isinstance(features, dict):
        return pd.DataFrame([features])
    values = list(features)
    names = getattr(model, "feature_names_in_", None)
    try:
        names = [str(name) for name in list(names)] if names is not None else []
    except TypeError:
        names = []
    if len(names) == len(values):
        return pd.DataFrame([values], columns=names)
    return pd.DataFrame([values])


prediction_service = PredictionService()